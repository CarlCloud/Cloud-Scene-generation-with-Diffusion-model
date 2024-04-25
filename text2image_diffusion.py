import os
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Qt5Agg')
import PIL.JpegImagePlugin
import numpy as np
import pandas as pd
import copy
import time
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import gc


import torch
import torchvision.datasets
import transformers
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import torch.onnx as onnx
import torchvision.models as models

import einops
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm


def get_trainset(path, w=128, h=128, padding=2):
    img = plt.imread(path)
    _img = []
    nw, nh = img.shape[0] // w, img.shape[1] // h
    n = nw * nh

    for i in range(n):
        curse_x = (padding + w) * (i % nw) + padding - 1
        curse_y = (padding + h) * int(np.floor(i / nw)) + padding - 1
        _img1 = img[curse_x:curse_x + w, curse_y:curse_y + h, :]
        _img.append(_img1)

    return _img


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
        if lr_patience >= 65:
            on_epoch_end()
            lr_patience = 0


def save_model(epoch, milestone, models):
    i = 0
    for model in models:
        data = {
            'model': model.state_dict(),
            'opt': optimizer.state_dict(),
        }
        torch.save(data, 'images/results/save_point/' + f'model{i}-{epoch}-{milestone}.pt')
        i += 1
    print(f"{epoch} epoch already save")


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

    def forward(self, x, y, *args, **kwargs):
        return self.fn(x, y, *args, **kwargs) + x

class Residual_x(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


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
            time_emb = self.mlp(time_emb)   # ！！！
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
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
        self.to_q = nn.Conv2d(dimx, hidden_dim , 1, bias=False)
        self.to_k = nn.Conv2d(dimx, hidden_dim, 1, bias=False)
        self.to_v = nn.Conv2d(dimx, hidden_dim, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dimx, 1),
                                    nn.GroupNorm(1, dimx))

    def forward(self, x, y):
        b, c, h, w = x.shape

        k = self.k(y).repeat(1,c,1,1) # [4,1,77,768]->[4,64,77,64]
        v = self.v(y).repeat(1,c,1,1)
        q = self.to_q(x) # [4,3,64,64]->[4,128,64,64]
        k = self.to_k(k) # [4,64,77,64]->[4,128,77,64]
        v = self.to_v(v)

        # [4,128,64,64]->[4,4,32,4096]
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), [q,k,v]
        )
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        # v = v.softmax(dim=-1)

        q = q * self.scale
        # [4,4,32,77*64]->[4,4,32,32]
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        # [4，4,32,32][4,4,32,4096]->[4,128,64,64]
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x, y):
        x = self.norm(x)
        return self.fn(x, y)

class PreNorm_x(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class encoder(nn.Module):
    def __init__(self, vocabulary_size, dim_in):
        super().__init__()
        self.embeding = nn.Embedding(vocabulary_size, dim_in)
        self.attn1 = LinearAttention(64)
        self.attn2 = LinearAttention(64)
        self.norm = nn.LayerNorm([77,768])

    def forward(self, x):
        # [4,1,77]->[4,1,77,768]
        x = self.embeding(x)
        # [4,1,77,768]->[4,64,77,12]
        x = rearrange(x, "b c h (x y)->b (c x) h y", x=64, y=12)
        x = self.attn1(x)
        x = self.attn2(x)
        # [4,64,77,12]->[4,1,77,768]
        x = rearrange(x, "b (c x) h y->b c h (x y)", x=64, y=12)
        x = self.norm(x)

        return x


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
        self.embeding = nn.Embedding(28996, 768)
        # # self.embeding = nn.ModuleList([
        # #     nn.Embedding(28996, 768),
        # #     Attention(dim=768),
        # # ])
        # # self.embeding = transformers.ClapTextModel()
        # self.embeding.training = False
        # self.embeding = encoder(28996, 768) # 会不会[4,64,77,12]才是正确的？不用repeat

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
                        Residual(PreNorm(dim_in, CrossAttention(dim_in,768))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual_x(PreNorm_x(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, CrossAttention(dim_out,768))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, y, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        y = y * 1.5
        # y = self.embeding(y) * 1.5 # ！！！提示词权重提升，是否有效？
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x, y)
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
            x = attn(x, y)

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


def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_losses(denoise_model, batch, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(batch[0])

    x_noisy = q_sample(x_start=batch[0], t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, batch[1], t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

@torch.no_grad()
def p_sample(model, x, y, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, y, t) / sqrt_one_minus_alphas_cumprod_t
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
def p_sample_loop(model, aim_token,  shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        # 训练unet时：
        # img = p_sample(model, img, aim_token, torch.full((b,), i, device=device, dtype=torch.long), i)
        # 训练embedding时：
        img = p_sample(model, img, aim_token, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs


@torch.no_grad()
def sample(model, tokens,  image_size, batch_size=16, channels=3):
    return p_sample_loop(model, tokens, shape=(batch_size, channels, image_size, image_size))


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# class flower102(torchvision.datasets.Flowers102):
#     def __getitem__(self, idx: int):
#         image_file = self._image_files[idx]
#         # image_file, label = self._image_files[idx], self._labels[idx]
#         image = Image.open(image_file).convert("RGB")
#
#         if self.transform:
#             image = self.transform(image)
#         # if self.target_transform:
#         #     label = self.target_transform(label)
#         # return image, label
#
#         return image



if __name__ == '__main__':
    t0 = time.time()

    from torchvision import datasets
    from datasets import load_dataset, load_from_disk
    # import torchvision.transforms
    from torchvision.transforms import ToTensor, Compose
    from torchvision.utils import save_image

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    image_size = 64
    text_length = 77
    text_embedding_size = 768

    channels = 3
    batch_size = 4
    # save_and_sample_every = 1020//batch_size -1
    checkpoint_stride = 50
    loop_break_flag = 0
    # resource_folder = r'images/diffusion_img'
    results_folder = r'images/results/images/'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    transform = Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.CenterCrop([raw_size,raw_size]), # pokemon数据集是正方形的
        transforms.Resize([image_size,image_size]),
        transforms.ToTensor(),
        # transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    # tokenizer = transformers.BertTokenizer.from_pretrained(
    #     pretrained_model_name_or_path='bert-base-cased',  # 'bert-base-chinese'
    #     cache_dir='data/BertTokenizer',
    #     force_download=False,
    # )

    # tokenizer = transformers.CLIPTokenizer.from_pretrained('CompVis/stable-diffusion-v1-4', cache_dir='/data', subfolder='tokenizer')
    tokenizer = torch.load(r'.\images\results\save_point\CLIPTokenizer.pt')

    # from diffusers import DDIMPipeline
    # diffusion_model pokemon--text2img

    # training_data = flower102(root='D:\MyTools\PyCharm2021.2.1\Admin_wang_2023\Pytorch_Programmar\data',
    #                           transform=transform)
    # training_data = load_dataset('data/lambdalabs/pokemon-blip-captions',cache_dir='/data',split='train')
    # training_data.save_to_disk('lambdalabs/pokemon-blip-captions')
    training_data = load_from_disk('data/lambdalabs/pokemon-blip-captions')

    # training_data = load_dataset(path='RoryCochrane/pokemon-and-fakemon', split='train')

    class getImgDataset(Dataset):
        def __init__(self, dataset):
            def func(data):
                # 应用图像增强
                pixel_values = [transform(i) for i in data['image']]

                # 文字编码
                input_ids = tokenizer.batch_encode_plus(data['text'],
                                                        padding='max_length',
                                                        truncation=True,
                                                        max_length=77).input_ids
                input_ids = [torch.tensor(i) for i in input_ids]
                return {'pixel_values': pixel_values, 'input_ids': input_ids}

            self.dataset = func(dataset)

        def __len__(self):  # 数据集样本数量
            return len(self.dataset[next(iter(self.dataset.keys()))])

        # __getitem__会执行 batch_size次，__getitem__返回的数据是给模型的
        def __getitem__(self, idx):  # 图像和标签在当前list的索引，每次调用idx是随机值，一个batch里的数据是随机打乱的
            image = self.dataset['pixel_values'][idx]
            label = self.dataset['input_ids'][idx]
            return image, label

    training_data = getImgDataset(training_data)

    # tokens use in p_sample images, img use in inspect
    # a = torch.randint(0,832,(4,))
    # print(a)
    a = torch.tensor(range(0,4))
    aim_img = [training_data.__getitem__(i)[0] for i in a]
    aim_token = [training_data.__getitem__(i)[1] for i in a]
    aim_token_stack = torch.stack(aim_token)
    aim_token_stack = aim_token_stack.to(device)
    aim_token = [tokenizer.decode(i) for i in aim_token]
    aim_img = [i.numpy().transpose(1,2,0) for i in aim_img]
    del a

    dataloader = DataLoader(training_data, pin_memory=False, batch_size=batch_size, shuffle=True,)
                            # collate_fn=collate_fn)
    # del training_data

    epoch_length = len(dataloader)
    save_and_sample_every = epoch_length - 1

    from torch.utils.tensorboard import SummaryWriter
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('images/runs/fashion_mnist_experiment_3')

    # embed = transformers.CLIPTextModel.from_pretrained('CompVis/stable-diffusion-v1-4', cache_dir='/data', subfolder='text_encoder').to(device)
    embed = torch.load(r'.\images\results\save_point\CLIPTextModel.pt')

    model = Unet(
        dim=image_size,
        dim_mults=(1, 2, 4, 8),
        # cond_drop_prob = 0.5
        channels=channels,
        resnet_block_groups=4
    )
    # model.load_state_dict(torch.load(r'.\images\results\save_point\model0-550-0.pt')['model'])
    model.to(device)

    model.training = True
    embed.training = False

    aim_tokens = embed(aim_token_stack)[0].to(device)
    aim_tokens = torch.unsqueeze(aim_tokens, dim=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=8e-5)
    lr_patience, loss_patience = 0, 0
    for param_group in optimizer.param_groups:
        opt_eps = param_group["eps"]

    epoch_losses = 0

    epochs = 800

    for epoch in range(0, epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # batch_size = batch["pixel_values"].shape[0]
            # for i in batch.keys():
                # batch[i] = batch[i].to(device)
            x,y = batch
            x = x.to(device)
            y = y.to(device)
            y = embed(y)[0]
            y = torch.unsqueeze(y, dim=1).to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = p_losses(model, (x,y), t, loss_type="huber")
            # loss = diffusion(batch)
            epoch_losses += loss.item()

            loss.backward()
            optimizer.step()

            # gc.collect()
            # torch.cuda.empty_cache()

            if step % checkpoint_stride == 0 or step % save_and_sample_every == 0:
                print(f"[{int(step + 1)}/{epoch_length}]Loss:", loss.item())



            if step != 0 and step % save_and_sample_every == 0:

                # print(f"its {epoch} epoch, costs", time.time() - t0)

                print(f"it's {epoch} epoch costs",time.time() - t0, "Loss:", epoch_losses)
                lr_change(epoch_losses)
                loss_patience = epoch_losses
                for param_group in optimizer.param_groups:
                    opt_lr = param_group["lr"]
                if opt_lr <= opt_eps:
                    loop_break_flag = 1
                    break

                writer.add_scalar('training loss',
                                  epoch_losses,
                                  epoch * epoch_length * batch_size + step * batch_size)

                epoch_losses = 0
                # save generated images
                # milestone = step // save_and_sample_every
                milestone = 0
                batches = num_to_groups(4, batch_size)
                # all_images_list = list(map(lambda n: diffusion.sample(batch_size=n, channels=channels), batches))

                all_images_list = list(
                    map(lambda n: sample(model, aim_tokens, image_size, batch_size=n, channels=channels), batches))
                # all_images = torch.cat(all_images_list, dim=0)

                all_images = einops.rearrange(all_images_list[0][-1], 'a b c d->(a c) d b')
                all_images = (all_images + 1) * 0.5
                all_images = (all_images - all_images.min()) / (all_images.max() - all_images.min())
                # all_images = all_images.astype('int')

                # for img in all_images_list[1:]:
                #     img = einops.rearrange(img[-1], 'a b c d->(a b) c d').reshape(-1, image_size)
                #     all_images = np.append(all_images, img, axis=1)

                # all_images = (all_images + 1) * 0.5

                plt.imsave((results_folder + f'sample-{epoch}-{milestone}.png'), all_images)

                # writer.add_images('prediction images',
                #                   sample(batch_size=4),
                #                   epoch * 60000 + step*128)

                if epoch % 5 == 0:
                    save_model(epoch, milestone, [model])

        if loop_break_flag == 1:
            break

    try:
        save_model(epoch, milestone, [model])
    except:
        pass

    writer.close()

    print('pause')
    print('pause')
    print('pause')
    print('pause')



    from diffusers import StableDiffusionImg2ImgPipeline
    #
    # trainer = Trainer(
    #     diffusion,
    #     # r'D:\MyTools\PyCharm 2021.2.1\Admin_wang_2023\Pytorch_Programmar\images\diffusion_img',
    #     datasets=training_dataset,
    #     train_batch_size=batch_size,
    #     train_lr=8e-5,
    #     train_num_steps=120000,  # total training steps
    #     gradient_accumulate_every=2,  # gradient accumulation steps
    #     ema_decay=0.995,  # exponential moving average decay
    #     amp=True,  # turn on mixed precision
    #     calculate_fid=True  # whether to calculate fid during training
    # )
    #
    #
    #
    # trainer.train()

    # do above for many steps

    # sampled_images = diffusion.sample(
    #     classes = image_classes,
    #     cond_scale = 3.                # condition scaling, anything greater than 1 strengthens the classifier free guidance. reportedly 3-8 is good empirically
    # )

    # import matplotlib
    # import matplotlib.pyplot as plt
    #
    # matplotlib.use('qt5agg')
    # sampled_images = diffusion.sample(
    #     batch_size = 4
    # )
    #
    # print(sampled_images.shape)  # (8, 3, 128, 128)
    # img1 = sampled_images.cpu()
    # img1 = torch.squeeze(img1)
    # # img1 = einops.rearrange(np.array(img1), 'b c h w -> b h w c')
    #
    # for i in range(len(img1)):
    #     plt.figure()
    #     plt.imshow(img1[i])
    #
    #
    # for i in range(len(img1)):
    #     plt.imsave(r'D:\MyTools\PyCharm 2021.2.1\Admin_wang_2023\Pytorch_Programmar\images\results\dm_sample'+f'{i}.jpg', img1[i]
    #     )

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

    # import numpy as np
    # import matplotlib
    # import matplotlib.pyplot as plt
    #
    # matplotlib.use('qt5agg')
    # img1 = sampled_images[0].cpu()
    # img1 = np.array(img1)
    # plt.imshow(img1)
    # plt.show()

    # mnist = pd.read_csv('../images/mnist_784.csv', sep=',',encoding='utf-8', header=True)

    # img = plt.imread(r'D:\MyTools\PyCharm 2021.2.1\Admin_wang_2023\Pytorch_Programmar\images\sample.png')

    # print("it cost {}s".format(time.time()-t0))

    # root = r'..\images\diffusion_img'

    # for i in range(len(_training_images)):
    #     file = str(i)+'.jpg'
    #     path = os.path.join(root, file)
    #     plt.imsave(path, _training_images[i])

    # imgs = training_data[0].numpy()
    # print(imgs.shape)
    # for i in range(15):
    #     print(imgs.shape)
    #     imgs = np.concatenate([imgs, training_data[i+np.random.randint(16,999)].numpy()], axis=0)
    #
    # imgs = imgs.reshape(-1, 3, 64, 64).transpose(0,2,3,1)
    # imgs = imgs.reshape(4,4,64,64,3).transpose(0,2,1,3,4)
    # imgs = imgs.reshape(64 * 4, 64 * 4, 3)
    # plt.imshow(imgs)
