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
from utils import cv_imwrite, image_normalization, sdm_srf_ircut, sdm_srf_irpass,MAE
from torch.utils.data import Dataset
from skimage.measure import compare_psnr, compare_ssim
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="SSR")
parser.add_argument('--method', type=str, default='CFMamba_sdm')
parser.add_argument('--pretrained_model_path', type=str, default='./pretrain_model/SDM/sdm_cfmamba.pth')
parser.add_argument('--data_root', type=str, default='./data/SDM')
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


class TestDataset(Dataset):
    def __init__(self, data_root):
        self.irpass_images = []
        self.ircut_images = []
        irpass_data_path = f'{data_root}/'
        ircut_data_path = f'{data_root}/'

        with open(f'./data/split_txt/test_list_sdm.txt', 'r') as fin:
            irpass_list = [line.replace('\n', '_rgbn_irpass.mat') for line in fin]
            ircut_list = [line.replace('_rgbn_irpass.mat', '_rgb_ircut.mat') for line in irpass_list]

        for i in range(len(irpass_list)):
            irpass_path = irpass_data_path + irpass_list[i]
            if 'mat' not in irpass_path:
                continue
            irpass_data = loadmat(irpass_path)
            irpass = np.float32(np.array(irpass_data['cube']))
            irpass = np.transpose(irpass, [2, 0, 1])
            
            
            
            ircut_path = ircut_data_path + ircut_list[i]
            assert (irpass_list[i].split('.')[0]).replace('_rgbn_irpass', '_rgb_ircut') == ircut_list[i].split('.')[
                0], 'irpass and RGB come from different scenes.'
            ircut_data = loadmat(ircut_path)
            ircut = np.float32(np.array(ircut_data['cube']))
            ircut = np.transpose(ircut, [2, 0, 1])
            
            self.irpass_images.append(irpass)
            self.ircut_images.append(ircut)

    def __getitem__(self, idx):
        irpass = self.irpass_images[idx]
        ircut = self.ircut_images[idx]
        srf_input = sdm_srf_irpass()
        srf_input_ircut = sdm_srf_ircut()
        return np.ascontiguousarray(irpass), np.ascontiguousarray(ircut), np.ascontiguousarray(
            srf_input), np.ascontiguousarray(srf_input_ircut)

    def __len__(self):
        return len(self.irpass_images)


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
            
            tmp_mae = criterion_mae(output, target) # mean_angular_error
            imgs_mae.append(tmp_mae.item())  
            
            input_img = input.cpu().numpy() * 1.0
            input_img = np.transpose(np.squeeze(input_img), [1, 2, 0])
            result = output.cpu().numpy() * 1.0
            gt_img = target.cpu().numpy() * 1.0
            gt_img = np.transpose(np.squeeze(gt_img), [1, 2, 0])
            result = np.transpose(np.squeeze(result), [1, 2, 0])
            
            mat_dir = opt.outf
            mat_path = os.path.join(mat_dir, f"{i:03d}_sdm_output.mat")
            savemat(mat_path, {'cube': result.astype(np.float32)})

            if not os.path.exists(visual_dir):
                os.makedirs((visual_dir), exist_ok=True)
            path1 = os.path.join(visual_dir, f"{i}_input.png")
            path2 = os.path.join(visual_dir, f"{i}_ouput.png")
            path3 = os.path.join(visual_dir, f"{i}_gt.png")
            cv_imwrite(np.uint8(image_normalization(input_img[:, :, :3])), path1)
            cv_imwrite(np.uint8(image_normalization(result)), path2)
            cv_imwrite(np.uint8(image_normalization(gt_img)), path3)

    imgs_psnr = np.array(imgs_psnr)
    imgs_ssim = np.array(imgs_ssim)
    imgs_mae = np.array(imgs_mae)

    print('-------------------------------------------')
    print('Evaluation finished on SDM dataset')
    print('PSNR: ', imgs_psnr.mean())
    print('SSIM: ', imgs_ssim.mean())
    print('MAE:  ', imgs_mae.mean())  # Mean Angular Error
    print('-------------------------------------------')
    print('finish test')


if __name__ == '__main__':
    main()