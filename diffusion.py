import os
import math

import matplotlib
# import PIL.JpegImagePlugin
import matplotlib.pyplot as plt
matplotlib.use('qt5agg')

import numpy as np
import pandas as pd
import copy
import time
from pathlib import Path
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import gc

import torch
import transformers
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import torch.onnx as onnx

import einops
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm

from Lib.torch_lib import show_objectMemory, show_processMemory

# _________________________________________________

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, y=None, *args, **kwargs):
        if y == None:
            return self.fn(x, *args, **kwargs) + x
        else:
            return self.fn(x, y, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)  # ！！！
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):  # heads=4, dim_head=32
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, dimx, dimy, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.k = nn.Linear(dimy, dimx, bias=False)
        self.v = nn.Linear(dimy, dimx, bias=False)
        self.to_q = nn.Conv2d(dimx, hidden_dim, 1, bias=False)
        self.to_k = nn.Conv2d(dimx, hidden_dim, 1, bias=False)
        self.to_v = nn.Conv2d(dimx, hidden_dim, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dimx, 1),
                                    nn.GroupNorm(1, dimx))

    def forward(self, x, y):
        b, c, h, w = x.shape

        k = self.k(y).repeat(1, c, 1, 1)  # [4,1,77,768]->[4,64,77,64]
        v = self.v(y).repeat(1, c, 1, 1)
        q = self.to_q(x)  # [4,3,64,64]->[4,128,64,64]
        k = self.to_k(k)  # [4,64,77,64]->[4,128,77,64]
        v = self.to_v(v)

        # [4,128,64,64]->[4,4,32,4096]
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), [q, k, v]
        )
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        # v = v.softmax(dim=-1)

        q = q * self.scale
        # [4,4,32,77*64]->[4,4,32,32]
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x, y=None):
        x = self.norm(x)
        if y == None:
            return self.fn(x)
        else:
            return self.fn(x, y)


class Unet(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            self_condition=False,
            resnet_block_groups=4,
            CFG_scale=1.1,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)  # changed to 1 and 0 from 7,3

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        # text embeding
        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, Attention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, Attention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time):
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


timesteps = 1000

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def make_noise_dims_same(x_start):
    b, c, h, w = x_start.shape
    noise = torch.randn((1, 1, h, w))
    noise = noise.repeat(b, c, 1, 1)
    return noise


def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    x = denoise_model.vae.encode(x_start).latent_dist.sample()
    if noise is None:
        noise = torch.randn_like(x)

    x_noisy = q_sample(x_start=x, t=t, noise=noise)
    predicted_noise = denoise_model.unet(x_noisy, t)


    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)

    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)

    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)

    # elif loss_type == "ssim":

    else:
        raise NotImplementedError()

    return loss


@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    # Algorithm 2 (including returning all images)


@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    img = make_noise_dims_same(img)
    img = img.to(device)

    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())

    return imgs


@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# ------------------------------------------------------------------------

def on_epoch_end():
    for param_group in optimizer.param_groups:
        old = param_group["lr"]
        param_group["lr"] /= 10
        new = param_group["lr"]
    print("new lr:", old, "->", new)


def lr_change(loss):
    global lr_patience, loss_patience
    if loss_patience == 0:
        loss_patience = loss
    if np.abs(loss - loss_patience) / loss_patience < 0.05 or loss > loss_patience:
        lr_patience += 1
        if lr_patience >= 25:
            on_epoch_end()
            lr_patience = 0


def save_model(epoch, milestone, models):
    i = 0
    for model in models:
        data = {
            'model': model.state_dict(),
            'opt': optimizer.state_dict(),
        }
        # torch.save(data, 'images/results/save_point/' + f'model{i}-{epoch}-{milestone}.pt')
        torch.save(data, r'F:\save_model\diffusion-1115\\' + rf'model{i}-{epoch}-{milestone}.pt')
        i += 1

    print(f"{epoch} epoch already save")


def get_file_list(dir_path, extend:str ='hdf', recur=False):
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(extend):
                file_list.append(os.path.join(root, file)) # 是否增加'\n' 取决于是否输出
        if recur == False:
            break

    return file_list


def save_hdf(data, file_path):
    Q = hdf.SD(file_path, hdf.SDC.WRITE | hdf.SDC.CREATE)
    BT = Q.create('BT', hdf.SDC.FLOAT32, data.shape)
    BT.set(data)
    BT.endaccess()
    Q.end()


def define_logger(filename=r"C:\Users\wfAdmin\Desktop\论文\experiment\log.txt",
                  str1: str = __name__, console=True, datefmt='%Y-%m-%d %H:%M:%S'):
    import logging
    # 创建一个 logger 对象
    logger = logging.getLogger(str1)
    # 指定日志级别
    logger.setLevel(logging.INFO)
    # 创建一个文件处理器
    file_handler = logging.FileHandler(filename)
    # 是否指定时间格式
    if datefmt != None:
        # 创建一个格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt=datefmt)
    else:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 将格式化器添加到文件处理器中
    file_handler.setFormatter(formatter)
    # 将文件处理器添加到 logger 对象中
    logger.addHandler(file_handler)
    if console:
        # 创建一个流处理器
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

def model_half_load(path):
    # turn model from fp32 to fp16
    params = torch.load(path)['model']
    for key in params.keys():
        params[key] = params[key].half()

    return params


if __name__ == '__main__':
    t0 = time.time()

    # from torchvision import datasets
    # from datasets import load_dataset, load_from_disk
    from torchvision.transforms import ToTensor, Compose
    import pyhdf.SD as hdf
    from Lib import utils_image as util
    from torch.cuda import amp

    image_size = 128
    resource_size = 270
    attn_dim = 64  # 64
    lattent_size = 64
    lattent_channels = 4
    channels = 3
    batch_size = 1
    # save_and_sample_every = 1020//batch_size -1
    checkpoint_stride = 50
    loop_break_flag = 0
    resource_folder = r'F:\Modisdata\Pic'
    results_folder = r'F:\save_model\diffusion-1126\images\\'
    log_file = r"F:\save_model\diffusion-1126/logger\log.txt"

    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"


    from diffusers.models import AutoencoderKL
    # from diffusers.models import vq_model,vae
    from diffusers import DDIMPipeline, DiffusionPipeline
    from torchsummary import summary

    class DDIM(nn.Module):
        def __init__(self, path1=None, path2=r'F:\save_model\diffusion-1115\model0-0-0.pt'):
            super(DDIM, self).__init__()
            self.vae_path = path1
            self.unet_path = path2
            self.vae = AutoencoderKL(
                in_channels=3,
                out_channels=3,
                down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
                up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
                block_out_channels=(64, 128),
                latent_channels=4,
                norm_num_groups=32,
                sample_size=lattent_size,
                scaling_factor=0.18215
            )

            self.unet = Unet(
                dim=attn_dim,
                dim_mults=(1, 2, 4, 8),
                # cond_drop_prob = 0.5
                channels=4,
                resnet_block_groups=4
            )

            # def init_net(vae, unet, vae_path, unet_path):
            #     if vae_path:
            #         vae.load_state_dict(torch.load(r'F:\save_model\diffusion-1125\vae0-98-0.pt')['model'])
            #         vae.training = False
            #         vae.eval()
            #
            #     if unet_path:
            #         unet.load_state_dict(torch.load(r'images\results\save_point\model0-285-0.pt')['model'])
            #
            # init_net(self.vae, self.unet, self.vae_path, self.unet_path)

            # self.unet.load_state_dict(torch.load(r'images\results\save_point\model0-285-0.pt')['model'])

        def forward(self, x, t):
            x0 = self.vae.encode(x).latent_dist.sample()
            x0 = self.unet(x0, t)
            x0 = self.vae.decode(x0).sample
            return x0

    model = DDIM(path2=None).to(device)


    transform = Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.CenterCrop([resource_size, resource_size]),  # modis06_imgsize=(7,406,270)
        transforms.Resize(image_size),
        # transforms.ToTensor(),
        # transforms.Lambda(lambda t: (t * 2) - 1)
    ])


    class getImgDataset(Dataset):
        def __init__(self, file_dir, transform=None, is_half=False):
            hdf_files = get_file_list(file_dir)
            self.dataset = []
            self.labels = []
            self.t = transform
            self.is_half = is_half
            for i in hdf_files:
                F = hdf.SD(i)
                img = F.select('BT').get()
                img = img[2:2 + channels]
                labels = F.select('MM').get()
                self.dataset.append(img)
                self.labels.append(labels)
                F.end()

        def __len__(self):  # 数据集样本数量
            return len(self.dataset)

        # __getitem__会执行 batch_size次，__getitem__返回的数据是给模型的
        def __getitem__(self, idx):  # 图像和标签在当前list的索引，每次调用idx是随机值，一个batch里的数据是随机打乱的
            image = self.dataset[idx]
            if self.is_half == False:
                image = torch.FloatTensor(image)  # 双精度压单精度
            else:
                image = torch.HalfTensor(image)  # 压fp16
            image = transform(image)

            return image


    training_data = getImgDataset(resource_folder, transform, is_half=False)

    dataloader = DataLoader(training_data, pin_memory=False, batch_size=batch_size, shuffle=True, )
    # collate_fn=collate_fn)

    checkpoint_stride = 50
    loop_break_flag = 0
    epoch_length = len(dataloader)
    save_and_sample_every = epoch_length - 1
    logger = define_logger(filename=log_file, console=True)

    # from torch.utils.tensorboard import SummaryWriter
    #
    # # default `log_dir` is "runs" - we'll be more specific here
    # writer = SummaryWriter('images/runs/fashion_mnist_experiment_3')

    # from torchsummary import summary
    # print(summary(model, [(channels,image_size,image_size),],
    #               torch.tensor([1],device='cuda'), batch_size=1, device='cuda'))

    optimizer = torch.optim.Adam(model.unet.parameters(), lr=8e-7, eps=1e-11)  # e-5
    lr_patience, loss_patience = 0, 0
    for param_group in optimizer.param_groups:
        opt_eps = param_group["eps"]

    # scaler = amp.GradScaler(enabled=True)

    epoch_losses = 0
    epoch_model_losses = 0
    # model.load_state_dict(torch.load(r'images\results\save_point\model0-285-0.pt')['model'])
    # model.load_state_dict(model_half_load(r'images\results\save_point\model0-185-0.pt'))
    model.vae.load_state_dict(torch.load(r'F:\save_model\diffusion-1125\vae0-74-0.pt')['model'])  # vae-98-0.pt
    model.vae.training = False
    model.vae.eval()
    # model.unet.load_state_dict(torch.load(r'F:\save_model\diffusion-1115\model0-126-0.pt')['model'])

    epoch_start, epoch_end = 0+125, 250

    for epoch in range(epoch_start, epoch_end):
        for step, x in enumerate(dataloader):

            optimizer.zero_grad()

            x = x.to(device)
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            # with amp.autocast(enabled=True):
            loss = p_losses(model, x, t, loss_type="huber")
            epoch_losses += loss.detach().item()

            loss.backward()
            optimizer.step()

            if step % checkpoint_stride == 0 or step % save_and_sample_every == 0:
                # print(f"[{int(step + 1)}/{epoch_length}]Loss: {loss.detach().item()}")
                logger.info(f"[{int(step + 1)}/{epoch_length}]Loss: {loss.detach().item()}")

            if step != 0 and step % save_and_sample_every == 0:
                # print(f"its {epoch} epoch, costs", time.time() - t0)
                # print(f"it's {epoch} epoch costs {time.time() - t0}s, Loss: {epoch_losses}")
                logger.info(f"it's {epoch} epoch costs {time.time() - t0}s, Loss: {epoch_losses}")

                lr_change(epoch_losses)
                loss_patience = epoch_losses
                for param_group in optimizer.param_groups:
                    opt_lr = param_group["lr"]
                if opt_lr <= opt_eps:
                    loop_break_flag = 1
                    break

                # writer.add_scalar('training loss',
                #                   epoch_losses,
                #                   epoch * epoch_length * batch_size + step * batch_size)

                epoch_losses = 0
                # save generated images
                # milestone = step // save_and_sample_every
                milestone = 0
                # batches = num_to_groups(4, batch_size)
                batches = num_to_groups(1, batch_size)
                all_images_list = list(
                    map(lambda n: sample(model.unet, image_size=lattent_size, batch_size=n, channels=lattent_channels), batches))

                images = torch.tensor(all_images_list[0][-1]).to(device)
                images = model.vae.decode(images).sample
                images = images.detach().cpu().numpy()
                images = einops.rearrange(images, 'a b c d->(a c) d b')
                images = (images - images.min()) / (images.max() - images.min())

                plt.imsave((results_folder + f'sample-{epoch}-{milestone}.png'), images)

                # save_hdf(all_images, file_list_aim[img_save_offset])

                # writer.add_images('prediction images',
                #                   sample(batch_size=4),
                #                   epoch * 60000 + step*128)

                if epoch % 1 == 0:
                    save_model(epoch, milestone, [model.unet])

        torch.cuda.empty_cache()
        gc.collect()

        if loop_break_flag == 1:
            break

    try:
        save_model(epoch, milestone, [model.unet])
    except:
        pass

    # writer.close()

    print('pause')
    print('pause')
    print('pause')
    print('pause')

    from diffusers import StableDiffusionImg2ImgPipeline

    # torch.cuda.memory_allocated()
    # torch.cuda.memory_reserved()

    model.load_state_dict(torch.load(r'images\results\save_point\model0-6-0.pt')['model'])
    batches = num_to_groups(4, batch_size)
    all_images_list = list(map(lambda n: sample(model, image_size, batch_size=n, channels=channels), batches))
    img = einops.rearrange(all_images_list[0][-1], 'a b c d->(a c) d b')
    img = (img - img.min()) / (img.max() - img.min())
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('qt5agg')
    img = img * 255
    img = img.astype('int')
    plt.imshow(img)

    # tensorboard --logdir = D:\MyTools\PyCharm2021.2.1\Admin_wang_2023\Pytorch_Programmar\images\runs\
    # http: // localhost: 6006
    print("it costs:{}s".format(time.time() - t0))

import psutil

print(u'使用：%s Gb' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
