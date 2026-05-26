import torch
from .CFMamba import CFMamba
def model_generator(method, pretrained_model_path=None):
    if  method == 'CFMamba_sdm':
        model = CFMamba(in_channels=4, out_channels=3, n_feat=44, stage=2,srf_dim=120)
    elif method == 'CFMamba_omsiv':
        model = CFMamba(in_channels=4, out_channels=3, n_feat=44, stage=2,srf_dim=141)
    else:
        print(f'Method {method} is not defined !!!!')

    return model
