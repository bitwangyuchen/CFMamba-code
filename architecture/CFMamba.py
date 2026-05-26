import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
from timm.models.layers import to_2tuple
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
import numpy as np
from einops import rearrange, repeat
from torch.nn import init
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from .scanf_util import mair_ids_generate, mair_ids_scan, mair_ids_inverse, mair_shift_ids_generate



def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):

    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')

def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch,int(in_channel/(r**2)), r * in_height, r * in_width
    x1 = x[:, :out_channel, :, :] / 2
    x2 = x[:,out_channel:out_channel * 2, :, :] / 2
    x3 = x[:,out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:,out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)
        
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class CSR_Embedding(nn.Module):
    def __init__(self,srf_dim):
        super().__init__()
        self.srf_dim=srf_dim
        self.norm = nn.LayerNorm(self.srf_dim)
        self.conv1d = nn.Conv1d(in_channels=self.srf_dim, out_channels=60, kernel_size=2)

    def forward(self, x1, x2):
        stacked = torch.stack([x1, x2], dim=2) 
        x_concat = stacked.view(-1, 8, self.srf_dim)   


        x_concat = x_concat.float()
        x_concat = self.norm(x_concat)
        

        x_grouped = x_concat.view(-1, 4, 2, self.srf_dim) 

        x_conv = x_grouped.permute(0,1,3,2).reshape(-1, self.srf_dim, 2) 
        

        x = self.conv1d(x_conv)  
        x = F.relu(x)
        

        y = x.view(-1, 4, 60)
        return y

class CSR_Channel_Interaction(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            srf_dim, 
            out_srf_dim,
    ):
        super().__init__()
        self.branch_conv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        self.sigmoid = nn.Sigmoid()
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.emb_srf=nn.Linear(srf_dim, out_srf_dim, bias=False)

    def forward(self, x,srf,size):
        srf_embed=self.emb_srf(srf)
        B,K,C,L=x.shape
        H,W=size
        x1 = x.reshape(B, K*C, H, W)

        x1=self.pool(x1)
        x1=x1.reshape(B, K,C, 1, 1).squeeze(-1)
        x2 = torch.einsum('b k c l, b k m -> b m c l', x1, srf_embed)  

        branch_interact = self.branch_conv(x2) 

        attn_weights = self.sigmoid(branch_interact) 

        out = x * attn_weights

        return out
class CSR_Guided_Mamba_Module(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            input_resolution=(128, 128),
            srf_dim=60,    
            attn_dim=16,   
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.input_resolution = input_resolution
        self.scan_count = 4
        self.scan_merge_method = 'concate'
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        
        self.in_proj2=CSR_Linear(self.d_model, self.scan_count, 60,ratio=expand)


        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.in_norm = nn.LayerNorm(self.d_model)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        if self.scan_merge_method == 'concate':
            self.d_inner = self.d_inner // self.scan_count

        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs) 
            for _ in range(self.scan_count)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs) for _ in range(self.scan_count)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs


        self.dt_proj_shared = nn.Linear(attn_dim, self.d_inner, bias=True, **factory_kwargs)
        self.attn_dim = attn_dim
        self.scale = math.sqrt(attn_dim) 

        self.wq = nn.Linear(srf_dim, attn_dim)
        self.wk = nn.Linear(self.d_inner, attn_dim)
        self.wv = nn.Linear(self.d_inner, attn_dim)

        self.dt_softplus = nn.Softplus()
        self.dt_bias = nn.Parameter(torch.ones(self.d_inner) * math.log(math.exp(dt_min) - 1))

        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.scan_count, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=self.scan_count, merge=True)

        self.selective_scan = selective_scan_fn
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.attn=CSR_Channel_Interaction(self.scan_count,self.scan_count,1,60,self.scan_count)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError


        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):

        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):

        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  
        D._no_weight_decay = True
        return D

    def tensor_reorder(self, x, size):

        if self.scan_count == 1:
            return x

        H, W = size
        B, C, L = x.shape
        
        to_stack = []
        sections = torch.split(x, C // self.scan_count, dim=1)

        return torch.stack(sections, dim=1)
    def tensor_restore(self, x, size):

        if self.scan_count == 1:
            return x

        H, W = size
        B, _, C, L = x.shape

        to_stack = []
        sections = torch.split(x, 1, dim=1)

        x = torch.cat(sections, dim=1)

        x = x.view(B, self.scan_count, C, L).reshape(B, self.scan_count * C, L)
        return x


    def CSR_Guided_Step_Size_Modulation(self, x, srf):
        B, K, D, L = x.shape 
        _, _, srf_dim = srf.shape  
        srf_expand = srf.unsqueeze(-1).repeat(1, 1, 1, L)  
        srf_seq = srf_expand.reshape(B*K, srf_dim, L)      
        q = self.wq(srf_seq.transpose(-1, -2)).transpose(-1, -2)  
        x_seq = x.permute(0,1,3,2).reshape(B*K, L, D)
        k = self.wk(x_seq)
        v = self.wv(x_seq)
        attn = torch.matmul(q, k) / self.scale 
        attn = attn.softmax(dim=-1)  
        dt_proj = torch.matmul(attn, v.transpose(-2, -1))
        dt_proj = dt_proj.reshape(B, K, self.attn_dim, L)
        dt = self.dt_proj_shared.weight @ dt_proj
        return dt
    
    def forward_core(self, x: torch.Tensor, 
                     mair_ids,
                     srf,
                     x_proj_bias: torch.Tensor=None,
                     ):

        B, C, H, W = x.shape
        L = H * W
        size=x.shape[-2:]
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        K=self.scan_count


        xs = mair_ids_scan(x, mair_ids[0])
        xs=self.tensor_reorder(xs.reshape(B, -1, L),size)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        
        dts = self.CSR_Guided_Step_Size_Modulation(xs, srf)
        
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L) 
        Cs = Cs.float().view(B, K, -1, L) 
        out_y = self.selective_scan(
            xs, dts,
            -torch.exp(self.A_logs.float()).view(-1, self.d_state), Bs, Cs, self.Ds.float().view(-1), z=None,
            delta_bias=self.dt_projs_bias.float().view(-1),
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        out_y=self.attn(out_y,srf,size)
        out_y=self.tensor_restore(out_y,size)
        return mair_ids_inverse(out_y.unsqueeze(1), mair_ids[1], shape=(B, -1, H, W)) #B, C, L

    def forward(self, x: torch.Tensor, mair_ids,srf):
        B, H, W, C = x.shape

        x = self.in_norm(x)
        x = self.in_proj2(x,srf)


        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        y = self.forward_core(x, mair_ids,srf)
        assert y.dtype == torch.float32

        y = y.permute(0, 2, 3, 1).contiguous()
        
        y = self.out_norm(y)

        y = self.out_proj(y)

        return y
    
class CSR_Linear(nn.Module):
    def __init__(self, in_channels, num_spectral_responses, response_dim, ratio=1):

        super(CSR_Linear, self).__init__()
        

        out_channels = int(in_channels * ratio)

        assert out_channels % num_spectral_responses == 0, "The number of output channels must be divisible by the number of spectral response bands."
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_spectral_responses = num_spectral_responses
        self.channels_per_response = int(out_channels // num_spectral_responses)
        

        self.weight_generator = nn.Linear(response_dim, in_channels * self.channels_per_response)
        self.bias_generator = nn.Linear(response_dim, self.channels_per_response)
        

        self._initialize_weights()
    
    def _initialize_weights(self):

        nn.init.xavier_uniform_(self.weight_generator.weight)
        nn.init.zeros_(self.weight_generator.bias)
        nn.init.xavier_uniform_(self.bias_generator.weight)
        nn.init.zeros_(self.bias_generator.bias)
    
    def forward(self, x, spectral_responses):

        batch_size, height, width,_  = x.size()
        

        x_flat = x.reshape(batch_size,height*width,-1)

        responses_flat = spectral_responses.view(batch_size * self.num_spectral_responses, -1) 
        
  
        weights = self.weight_generator(responses_flat)  
        biases = self.bias_generator(responses_flat)  
        
      
        weights = weights.view(batch_size, self.num_spectral_responses, self.in_channels, self.channels_per_response)
        

        biases = biases.view(batch_size, self.num_spectral_responses, self.channels_per_response)
        
 
        x_expanded = x_flat.unsqueeze(1).expand(-1, self.num_spectral_responses, -1, -1) 
        

        output_parts = torch.matmul(x_expanded, weights) + biases.unsqueeze(2)
        

        output = output_parts.permute(0, 2, 1, 3).contiguous()  
        output = output.view(batch_size, height * width, self.out_channels)  
        

        output = output.view(batch_size, height, width,self.out_channels)
        
        return output
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):

        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)
class Neck(nn.Module):
    def __init__(self, dim, num_blocks,d_state,ssm_ratio,n_l_blocks=1, n_h_blocks=1, expand=2):
        super().__init__()
        self.dwt = DWT()

        self.l_blk = nn.Sequential(*[LFMambaBlock(dim=dim, num_blocks=num_blocks, d_state=d_state, ssm_ratio=ssm_ratio) for _ in range(n_l_blocks)])

        self.iwt = IWT()

        self.h_out_conv=PreNorm(dim*3, FeedForward(dim=dim*3))
    def forward(self, x,mair_ids, x_size,srf):
        x_LL, x_HL, x_LH, x_HH = self.dwt(x)
        b, c, h, w = x_LL.shape

        for l_layer in self.l_blk:
            x_LL = l_layer(x_LL, mair_ids, x_size,srf)


        x_h = torch.cat([x_HL, x_LH, x_HH], dim=1)

        x_h = self.h_out_conv(x_h.permute(0,2,3,1))

        x_l = self.iwt(torch.cat([x_LL, x_h.permute(0,3,1,2)], dim=1))
        return x_l
class Down_processing(nn.Module):
    def __init__(self, dim, num_blocks,d_state,ssm_ratio,n_l_blocks=1, n_h_blocks=1, expand=2):
        super().__init__()
        self.dwt = DWT()
        self.l_blk = nn.Sequential(*[LFMambaBlock(dim=dim, num_blocks=num_blocks, d_state=d_state, ssm_ratio=ssm_ratio) for _ in range(n_l_blocks)])
        self.h_out_conv=PreNorm(dim*3, FeedForward(dim=dim*3))
    def forward(self, x,mair_ids, x_size,srf):
        x_LL, x_HL, x_LH, x_HH = self.dwt(x)
        b, c, h, w = x_LL.shape

        for l_layer in self.l_blk:
            x_LL = l_layer(x_LL, mair_ids, x_size,srf)

        x_h = torch.cat([x_HL, x_LH, x_HH], dim=1)

        x_h = self.h_out_conv(x_h.permute(0,2,3,1))

        return x_LL,x_h.permute(0,3,1,2)
class Up_processing(nn.Module):
    def __init__(self, dim, num_blocks,d_state,ssm_ratio,n_l_blocks=1, n_h_blocks=1, expand=2):
        super().__init__()
        self.iwt = IWT()
        self.l_blk = nn.Sequential(*[LFMambaBlock(dim=dim, num_blocks=num_blocks, d_state=d_state, ssm_ratio=ssm_ratio) for _ in range(n_l_blocks)])

        self.h_out_conv=PreNorm(dim, FeedForward(dim=dim))
    def forward(self, x,x_h,mair_ids, x_size,srf):

        for l_layer in self.l_blk:
            x_LL = l_layer(x, mair_ids, x_size,srf)

        x_l = self.iwt(torch.cat([x_LL, x_h], dim=1))
        return x_l
class LFMambaBlock(nn.Module):
    def __init__(
            self,
            dim,
            d_state,
            ssm_ratio,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                CSR_Guided_Mamba_Module(d_model=dim, d_state=d_state,expand=ssm_ratio),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, mair_ids, x_size,srf):

        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, (mair_ids[0], mair_ids[1]),srf) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class CFMambaBlock(nn.Module):
    def __init__(self, in_dim=36, out_dim=36, dim=36, stage=2, srf_dim=120, d_state=4,ssm_ratio=4,num_blocks=[1,1,1]):
        super(CFMambaBlock, self).__init__()
        self.dim = dim
        self.stage = stage
        self.scan_len=4
        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(
                Down_processing(
                    dim=dim_stage, num_blocks=num_blocks[i], d_state=d_state, ssm_ratio=ssm_ratio),
            )


        # Bottleneck
        self.bottleneck = Neck(
            dim=dim_stage, num_blocks=num_blocks[-1], d_state=d_state, ssm_ratio=ssm_ratio)

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.Conv2d(dim_stage*2, dim_stage, 1, 1, bias=False),
                Up_processing(
                    dim=dim_stage, num_blocks=num_blocks[stage - 1 - i], d_state=d_state, ssm_ratio=ssm_ratio),
            ]))


        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
        img_size_ids = to_2tuple(128)
        self.image_size = img_size_ids
        self._generate_ids((1, 1, img_size_ids[0], img_size_ids[1]))

        self.srffunction=CSR_Embedding(srf_dim=srf_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def _generate_ids(self, inp_shape):
        B,C,H,W = inp_shape

        xs_scan_ids, xs_inverse_ids = mair_ids_generate(inp_shape=(1, 1, H, W), scan_len=self.scan_len)
        xs_scan_ids_1, xs_inverse_ids_1 = mair_ids_generate(inp_shape=(1, 1, H//2, W//2), scan_len=self.scan_len)
        xs_scan_ids_2, xs_inverse_ids_2 = mair_ids_generate(inp_shape=(1, 1, H//4, W//4), scan_len=self.scan_len)
        xs_scan_ids_3, xs_inverse_ids_3 = mair_ids_generate(inp_shape=(1, 1, H//8, W//8), scan_len=self.scan_len)
        if torch.cuda.is_available():
            self.xs_scan_ids = xs_scan_ids.cuda()
            self.xs_scan_ids_1 = xs_scan_ids_1.cuda()
            self.xs_scan_ids_2 = xs_scan_ids_2.cuda()
            self.xs_scan_ids_3 = xs_scan_ids_3.cuda()
            self.xs_inverse_ids = xs_inverse_ids.cuda()
            self.xs_inverse_ids_1 = xs_inverse_ids_1.cuda()
            self.xs_inverse_ids_2 = xs_inverse_ids_2.cuda()
            self.xs_inverse_ids_3 = xs_inverse_ids_3.cuda()
            self.xs_scan_ids_all=[self.xs_scan_ids_1,self.xs_scan_ids_2,self.xs_scan_ids_3]
            self.xs_inverse_ids_all=[self.xs_inverse_ids_1,self.xs_inverse_ids_2,self.xs_inverse_ids_3]
        else:
            self.xs_scan_ids = xs_scan_ids.cuda()
            self.xs_scan_ids_1 = xs_scan_ids_1
            self.xs_scan_ids_2 = xs_scan_ids_2
            self.xs_scan_ids_3 = xs_scan_ids_3
            self.xs_inverse_ids = xs_inverse_ids
            self.xs_inverse_ids_1 = xs_inverse_ids_1
            self.xs_inverse_ids_2 = xs_inverse_ids_2
            self.xs_inverse_ids_3 = xs_inverse_ids_3
            self.xs_scan_ids_all=[self.xs_scan_ids_1,self.xs_scan_ids_2,self.xs_scan_ids_3]
            self.xs_inverse_ids_all=[self.xs_inverse_ids_1,self.xs_inverse_ids_2,self.xs_inverse_ids_3]


        del xs_scan_ids, xs_inverse_ids
    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        srf1=x[1]
        srf2=x[2]
        x=x[0]
        B,C,H,W = x.shape
        x_size = (x.shape[2], x.shape[3])

        srf=self.srffunction(srf1.view(B, 4, -1),srf2.view(B, 4, -1))
        fea = self.embedding(x)
        if self.image_size != (H, W):

            xs_scan_ids, xs_inverse_ids = mair_ids_generate(inp_shape=(1, 1, H, W), scan_len=self.scan_len)
            xs_scan_ids_1, xs_inverse_ids_1 = mair_ids_generate(inp_shape=(1, 1, H//2, W//2), scan_len=self.scan_len)
            xs_scan_ids_2, xs_inverse_ids_2 = mair_ids_generate(inp_shape=(1, 1, H//4, W//4), scan_len=self.scan_len)
            xs_scan_ids_3, xs_inverse_ids_3 = mair_ids_generate(inp_shape=(1, 1, H//8, W//8), scan_len=self.scan_len)

            if torch.cuda.is_available():
                xs_scan_ids, xs_inverse_ids = xs_scan_ids.cuda(), xs_inverse_ids.cuda()
                xs_scan_ids_1, xs_inverse_ids_1 = xs_scan_ids_1.cuda(), xs_inverse_ids_1.cuda()
                xs_scan_ids_2, xs_inverse_ids_2 = xs_scan_ids_2.cuda(), xs_inverse_ids_2.cuda()
                xs_scan_ids_3, xs_inverse_ids_3 = xs_scan_ids_3.cuda(), xs_inverse_ids_3.cuda()
            xs_scan_ids_all=[xs_scan_ids_1,xs_scan_ids_2,xs_scan_ids_3]
            xs_inverse_ids_all=[xs_inverse_ids_1,xs_inverse_ids_2,xs_inverse_ids_3]
        else:
            xs_scan_ids_all=self.xs_scan_ids_all
            xs_inverse_ids_all=self.xs_inverse_ids_all
        # Encoder
        fea_encoder_ll = []
        fea_encoder_h = []
        for i, LFMambaBlock in enumerate(self.encoder_layers):
            fea,fea_h = LFMambaBlock(fea, (xs_scan_ids_all[i], xs_inverse_ids_all[i]), x_size,srf)
            fea_encoder_ll.append(fea)
            fea_encoder_h.append(fea_h)
            x_size=(x_size[0]//2,x_size[1]//2)

        # Bottleneck
        fea = self.bottleneck(fea, (xs_scan_ids_all[self.stage], xs_inverse_ids_all[self.stage]), x_size,srf)

        # Decoder
        for i, (Fution, LeWinBlcok) in enumerate(self.decoder_layers):

            fea = Fution(torch.cat([fea, fea_encoder_ll[self.stage-1-i]], dim=1))
            fea = LeWinBlcok(fea,fea_encoder_h[self.stage-i-1], (xs_scan_ids_all[self.stage-i-1], xs_inverse_ids_all[self.stage-i-1]), x_size,srf)
            x_size=(x_size[0]*2,x_size[1]*2)  

        # Mapping
        out = self.mapping(fea) + x

        return [out,srf1,srf2]

class CFMamba(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, n_feat=44, stage=2,srf_dim=120):
        super().__init__()
        self.stage = stage
        self.n_feat=n_feat
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        modules_body = [CFMambaBlock(in_dim=n_feat, out_dim=n_feat, dim=n_feat, stage=2, srf_dim=srf_dim,d_state=4,ssm_ratio=2,num_blocks=[1,1,1]) for _ in range(stage)]
        self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2,bias=False)

    def forward(self, x,srf1,srf2):
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        x_in = self.conv_in(x)
        h = self.body([x_in,srf1,srf2])
        h = self.conv_out(h[0])
        h += x[:,:3,:,:]
        return h[:, :, :h_inp, :w_inp]






if __name__ == '__main__':
    from thop import profile
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model = CFMamba().cuda()
    input_tensor = torch.randn(1, 4, 128, 128).cuda()
    csr_tensor1 = torch.rand(1, 4, 120).cuda()
    csr_tensor2 = torch.rand(1, 4, 120).cuda()

    flops, params = profile(model, inputs=(input_tensor,csr_tensor1,csr_tensor2))

    print(f"Params: {params/ 1e6}")
    print(f"FLOPs: {flops/ 1e9}")







