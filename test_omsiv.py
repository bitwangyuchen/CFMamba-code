import torch
import argparse
import torch.backends.cudnn as cudnn
import os
from architecture import *
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import time
import numpy as np
from scipy.io import loadmat, savemat
from utils import cv_imwrite, image_normalization, omsiv_srf_ircut, omsiv_srf_irpass, MAE
from torch.utils.data import Dataset
import h5py
from skimage.measure import compare_psnr, compare_ssim

parser = argparse.ArgumentParser(description="SSR")
parser.add_argument('--method', type=str, default='CFMamba_omsiv')
parser.add_argument('--pretrained_model_path', type=str, default='./pretrain_model/OMSIV/omsiv_cfmamba.pth')
parser.add_argument('--data_root', type=str, default='./data/omsiv')
parser.add_argument('--outf', type=str, default='./output')
parser.add_argument('--ensemble_mode', type=str, default='mean')
parser.add_argument("--seed", type=int, default=3407)
parser.add_argument("--gpu_id", type=str, default='0')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)
seed = opt.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
method = opt.method
current_time = time.strftime("%Y-%m-%dT%H-%M", time.localtime())
visual_dir = os.path.join('./visual/{}'.format(method), current_time)


def main():
    cudnn.benchmark = True
    pretrained_model_path = opt.pretrained_model_path
    method = opt.method
    model = model_generator(method, pretrained_model_path).cuda()
    resume_file = pretrained_model_path
    checkpoint = torch.load(resume_file)
    pretrained_model = checkpoint['state_dict']
    model.load_state_dict(pretrained_model, strict=False)
    test_data = TestDataset(data_root=opt.data_root)
    test_loader = DataLoader(
        dataset=test_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    test(test_loader, model)


def test(test_loader, model):
    model.eval()
    criterion_mae = MAE().cuda()
    imgs_ssim = []
    imgs_psnr = []
    imgs_mae = []

    for i, (input, target, srf1, srf2) in enumerate(test_loader):
        input = input.cuda()
        target = target.cuda()
        srf1 = srf1.cuda()
        srf2 = srf2.cuda()
        with torch.no_grad():
            output = model(input, srf1, srf2)

            result_np = output.cpu().numpy() * 1.0
            gt_np = target.cpu().numpy() * 1.0
            result_np = np.transpose(np.squeeze(result_np), [1, 2, 0])
            gt_np = np.transpose(np.squeeze(gt_np), [1, 2, 0])

            tmp_ssim = compare_ssim(gt_np, result_np, gaussian_weights=True, multichannel=True)
            tmp_psnr = compare_psnr(gt_np, result_np)
            imgs_ssim.append(tmp_ssim)
            imgs_psnr.append(tmp_psnr)

            tmp_mae = criterion_mae(output, target)
            imgs_mae.append(tmp_mae.item())

            input_img = input.cpu().numpy() * 1.0
            input_img = np.transpose(np.squeeze(input_img), [1, 2, 0])
            result = output.cpu().numpy() * 1.0
            gt = target.cpu().numpy() * 1.0
            gt = np.transpose(np.squeeze(gt), [1, 2, 0])
            result = np.transpose(np.squeeze(result), [1, 2, 0])

            mat_dir = opt.outf
            mat_path = os.path.join(mat_dir, f"{i:03d}_omsiv_output.mat")
            savemat(mat_path, {'cube': result.astype(np.float32)})

            if not os.path.exists(visual_dir):
                os.makedirs((visual_dir), exist_ok=True)
            path1 = os.path.join(visual_dir, f"{i}_input.png")
            path2 = os.path.join(visual_dir, f"{i}_ouput.png")
            path3 = os.path.join(visual_dir, f"{i}_gt.png")
            cv_imwrite(np.uint8(image_normalization(input_img[:, :, :3])), path1)
            cv_imwrite(np.uint8(image_normalization(result)), path2)
            cv_imwrite(np.uint8(image_normalization(gt)), path3)

    imgs_psnr = np.array(imgs_psnr)
    imgs_ssim = np.array(imgs_ssim)
    imgs_mae = np.array(imgs_mae)

    print('-------------------------------------------')
    print('Evaluation finished on OMSIV dataset')
    print('PSNR: ', imgs_psnr.mean())
    print('SSIM: ', imgs_ssim.mean())
    print('MAE:  ', imgs_mae.mean())
    print('-------------------------------------------')
    print('finish test')


class TestDataset(Dataset):

    def __init__(self, data_root, arg=True):
        self.dim_w = 580
        self.dim_h = 320
        self.data_name = 'OMSIV'
        self.model_state = 'test'
        self.is_training = False if self.model_state.lower() == 'train' else False
        self.shuffle = self.is_training
        self.dataset_dir = f'{data_root}/'
        self.test_list = os.path.join(os.path.dirname(__file__), "data/split_txt/test_list_omsiv.txt")
        self.data_list = self._build_index()
        self.on_epoch_end()

    def _build_index(self):
        base_dir = self.dataset_dir
        list_name = self.test_list
        file_path = self.test_list

        with open(file_path, 'r') as f:
            file_list = f.readlines()
        file_list = [line.strip() for line in file_list]
        file_list = [line.split(' ') for line in file_list]

        input_path = [os.path.join(base_dir, line[0]) for line in file_list]
        gt_path = [os.path.join(base_dir, line[1]) for line in file_list]
        if not self.is_training:
            self.imgs_name = [os.path.basename(k) for k in input_path]
        sample_indeces = [input_path, gt_path]
        return sample_indeces

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data_list[0]))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        indices = self.indices[index:(index + 1)]
        x_list, y_list = self.data_list
        tmp_x_path = [x_list[k] for k in indices]
        tmp_y_path = [y_list[k] for k in indices]
        x, y = self.__data_generation(tmp_x_path, tmp_y_path)
        srf_input = omsiv_srf_irpass()
        srf_input_ircut = omsiv_srf_ircut()
        return x, y, srf_input, srf_input_ircut

    def __data_generation(self, x_path, y_path):
        x = np.empty((self.dim_h, self.dim_w, 4), dtype="float32")
        y = np.empty((self.dim_h, self.dim_w, 3), dtype="float32")

        for i, tmp_data in enumerate(x_path):
            tmp_x_path = tmp_data
            tmp_y_path = y_path[i]
            tmp_x, tmp_y = self.transformer(tmp_x_path, tmp_y_path)
            x = tmp_x.transpose(2, 0, 1)
            y = tmp_y.transpose(2, 0, 1)
        return x, y

    def transformer(self, x_path, y_path):
        tmp_x = self.__read_h5(x_path)
        tmp_y = self.__read_h5(y_path)
        h, w, _ = tmp_x.shape
        return tmp_x, tmp_y

    def __read_h5(self, file_path):
        with h5py.File(file_path, 'r') as h5f:
            data = np.array(h5f.get('data'))
        return data


if __name__ == '__main__':
    main()