import os
import math

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('qt5agg')
import numpy as np
import pandas as pd
import copy
import time

from multiprocessing import cpu_count
import gc

import torch
import torchvision.datasets
import transformers
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms

import einops
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm

from Lib.torch_lib import *
# from denoising_diffusion_pytorch.denoising_diffusion_pytorch import *


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
        if lr_patience >= 40:
            on_epoch_end()
            lr_patience = 0
    elif lr_patience > 0:
        lr_patience -= 1


def save_model(epoch, milestone, models):
    i = 0
    for model in models:
        data = {
            'model': model.state_dict(),
            'opt': optimizer.state_dict(),
        }
        torch.save(data, r'F:\save_model\diffusion-1115\\' + f'vae{i}-{epoch}-{milestone}.pt')
        i += 1
    print(f"{epoch} epoch already save")

def save_model_accl(epoch, milestone, models):
    i = 0
    for model in models:
        data = {
            'model': model.state_dict(),
            'opt': optimizer.state_dict(),
        }
        torch.save(data, r'F:\save_model\diffusion-1115\\' + f'vae{i}-{epoch}-{milestone}.pt')
        i += 1
    print(f"{epoch} epoch already save")



def get_file_list(dir_path, extend:str ='hdf'):
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(extend):
                file_list.append(os.path.join(root, file)) # 是否增加'\n' 取决于是否输出

    return file_list

def save_hdf(data, file_path):
    Q = hdf.SD(file_path, hdf.SDC.WRITE | hdf.SDC.CREATE)
    BT = Q.create('BT', hdf.SDC.FLOAT32, data.shape)
    BT.set(data)
    BT.endaccess()
    Q.end()

def show_cudaMemoryInfo():
    if torch.cuda.is_available():
        print("GPU", torch.cuda.current_device(), "allocated:", torch.cuda.memory_allocated() / 1024 / 1024, "Mb")
        print("GPU", torch.cuda.current_device(), "reserved:", torch.cuda.memory_reserved() / 1024 / 1024, "Mb")
        print("GPU", torch.cuda.current_device(), "max_memory_allocated:",
              torch.cuda.max_memory_reserved() / 1024 / 1024/ 1024, "Gb")

def show_GPUInfo():
    if torch.cuda.is_available():
        if torch.cuda.device_count() >= 1:  # 检查电脑是否有多块GPU
            print(f"It's using {torch.cuda.device_count()} GPUs!")

            for i in range(torch.cuda.device_count()):
                print("GPU", i, ":", end=' ')
                print("calc capability:", torch.cuda.get_device_capability(i))
                print("memory capability:", torch.cuda.get_device_properties(i))

def get_GPUInfo(text=None):
    import GPUtil
    if text!=None:
        print(text, "处:")
    GPUtil.showUtilization()

def img2tensor(x, mode='png', Unsqueeze=0):
    if mode == 'png':
        x = x[...,:-1]
        x = torch.tensor(x).permute(2, 0, 1)
    elif mode == 'rpg':
        x = torch.tensor(x).permute(2, 0, 1)
    else:
        x = torch.tensor(x)

    if Unsqueeze != None:
        x = x.unsqueeze(Unsqueeze)

    return x

def tensor2img(x):
    if torch.is_tensor(x):
        if x.device.type == 'cuda':
            _x = x.cpu().detach().numpy()  # convert cuda to cpu's ndarray
        else:
            _x = x
    elif isinstance(x, np.ndarray):
        _x = x

    if len(_x.shape) == 4:
        b, c, h, w = _x.shape
        return rearrange(_x, 'b c h w-> b h w c')
    elif len(_x.shape) == 3:
        c, h, w = _x.shape
        return rearrange(_x, 'c h w-> h w c')
    else:
        print("Error length of tensor:",x.shape)

def array_concat(a, b):
    c = np.concatenate((a,b), axis=0)
    return rearrange(c, 'b h w c->(b h) w c')

def img2save(x):
    if len(x.shape) == 4:
        return rearrange(x, 'b h w c-> (b h) w c')
    else:
        return x


@torch.no_grad()
def save_img(model, epoch, milestone, path):
    img = next(iter(dataloader))
    img0 = img[0].to(device)
    img1 = img[1]
    img2 = model(img0).sample.cpu().detach().numpy()
    img1 = tensor2img(img1)
    img2 = tensor2img(img2)

    all_images = array_concat(img1, img2)
    for i in range(3):
        all_images[..., i] = (all_images[..., i] - all_images[..., i].min()) / (all_images[..., i].max() - all_images[..., i].min())

    plt.imsave((path + f'sample-{epoch}-{milestone}.png'), all_images)

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

def test(img_path, img_size=256):
    path = img_experiment_folder + img_path
    a = plt.imread(path)
    a = img2tensor(a)
    resize = transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC)
    a = resize(a)
    b = a
    b = b.to(device)
    c = vae(b).sample
    c = tensor2img(c[0])
    a = tensor2img(a[0])
    return a, c

# class SRCNN(nn.Module):
#     def __init__(self, nChannel=1):
#         super(SRCNN,self).__init__()
#         self.conv1 = nn.Conv2d(nChannel, 64, kernel_size=9, padding=9//2)
#         self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5//2)
#         self.conv3 = nn.Conv2d(32, nChannel, kernel_size=5, padding=5//2)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self,x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.conv3(x)
#         return x


if __name__ == '__main__':
    t0 = time.time()

    from torchvision import datasets
    from datasets import load_dataset, load_from_disk
    from torchvision.transforms import ToTensor, Compose
    import pyhdf.SD as hdf

    batch_size = 4
    channels = 3
    latent_image_size = 64
    image_size = 256
    lr_image_size = 64
    # dim = 8
    # init_conv = nn.Conv2d(channels, dim, 1, padding=0)  # changed to 1 and 0 from 7,3
    # dims = [dim, *map(lambda m: dim * m, (1, 2, 4, 8))]
    # in_out = list(zip(dims[:-1], dims[1:]))

    from diffusers.models import AutoencoderKL,vae
    from accelerate import Accelerator

    from diffusers.image_processor import VaeImageProcessor

    resource_folder = r'F:/Modisdata/Pic'
    results_folder = r'images/results/images_vae/'
    img_experiment_folder = r"images/results/images/"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    accelerator = Accelerator()
    device = accelerator.device

    # from diffusers.image_processor import VaeImageProcessor

    import warnings
    warnings.filterwarnings('ignore') # vae加载报futurewarning

    from diffusers import LDMPipeline

    # load model and scheduler
    # pipe = LDMPipeline.from_pretrained("CompVis/ldm-celebahq-256")

    from torchsummary import summary

    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D",),
        block_out_channels=(64, 128, 256),
        latent_channels=4,
        norm_num_groups=32,
        sample_size=64,
        scaling_factor=0.18215
    )

    # run pipeline in inference (sample random noise and denoise)
    # image = pipe().images[0]
    # vae = SRCNN().to(device)

    transform = Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.CenterCrop([270, 270]), # modis06_imgsize=(7,406,270)
        transforms.Resize(image_size),
        # transforms.ToTensor(),
        # transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    transform1 = Compose([
        # transforms.Resize(lr_image_size,transforms.InterpolationMode.BICUBIC),
        # transforms.Resize(image_size,transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop([image_size,image_size])
    ])

    # class getImgDataset(Dataset):
    #     def __init__(self, file_dir, transform = None):
    #         hdf_files = get_file_list(file_dir)
    #         self.dataset = []
    #         self.labels = []
    #         self.t = transform
    #         for i in hdf_files:
    #             F = hdf.SD(i)
    #             img = F.select('BT').get()
    #             img = img[2:2+channels]
    #             labels = F.select('MM').get()
    #             self.dataset.append(img)
    #             self.labels.append(labels)
    #             F.end()
    #
    #     def __len__(self):
    #         return len(self.dataset)
    #
    #     def __getitem__(self, idx):
    #         image = self.dataset[idx]
    #         image = torch.FloatTensor(image) # 双精度压单精度
    #         image = transform(image)
    #         # return image
    #         lr_image = transform1(image)
    #         return lr_image, image

    class getImgDataset(Dataset):
        def __init__(self, file_dir, transform = None):
            hdf_files = get_file_list(file_dir)
            self.dataset = []
            self.labels = []
            self.t = transform
            for i in hdf_files:
                F = hdf.SD(i)
                img = F.select('BT').get()
                img = img[2:2+channels]
                labels = F.select('MM').get()
                self.dataset.append(img)
                self.labels.append(labels)
                F.end()

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image = self.dataset[idx]
            image = torch.FloatTensor(image) # 双精度压单精度
            image = transform(image)
            return image


    training_data = getImgDataset(resource_folder, transform)

    dataloader = DataLoader(training_data, pin_memory=False, batch_size=batch_size, shuffle=True,)
                            # collate_fn=collate_fn)

    optimizer = torch.optim.Adam(vae.parameters(), lr=8e-5)

    vae, optimizer, dataloader = accelerator.prepare(
        vae, optimizer, dataloader
    )

    lr_patience, loss_patience = 0, 0
    for param_group in optimizer.param_groups:
        opt_eps = param_group["eps"]

    checkpoint_stride = 10
    loop_break_flag = 0
    epoch_length = len(dataloader)
    save_and_sample_every = epoch_length - 1
    logger = define_logger()

    def p_losses(model, x, loss_type="l1"):

        predict = model(x).sample

        if loss_type == 'l1':
            loss = F.l1_loss(y, predict)
        elif loss_type == 'l2':
            loss = F.mse_loss(y, predict)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(y, predict)
        else:
            raise NotImplementedError()

        return loss

    epoch_losses = 0
    epoch_start, epoch_end = 0, 50
    t1 = time.time()

    # vae.load_state_dict(torch.load(r'images\results\save_point_vae\vae0-1-211.pt')['model'])

    for epoch in range(epoch_start, epoch_end):
        for step, (x, y) in enumerate(dataloader):

            optimizer.zero_grad()

            x = x.to(device)

            loss = p_losses(vae, x, loss_type="huber")
            epoch_losses += loss.detach().item()

            # loss.backward()
            accelerator.backward(loss)  # for accelerate
            optimizer.step()


            if step % checkpoint_stride == 0 or step % save_and_sample_every == 0:
                print(f"[{int(step + 1)}/{epoch_length}]Loss:", loss.detach().item())
                logger.info(f"[{int(step + 1)}/{epoch_length}]Loss: "+str(loss.detach().item()))

            if step != 0 and step % 70 == 0:
                # save generated images
                milestone = step
                # if epoch == 1:
                #     milestone += 211
                milestone = 0
                save_img(vae, epoch, milestone, results_folder)
                # save_hdf(all_images, file_list_aim[img_save_offset])

                # writer.add_images('prediction images',
                #                   sample(batch_size=4),
                #                   epoch * 60000 + step*128)

            if step != 0 and step % save_and_sample_every == 0:

                print(f"it's {epoch} epoch costs",time.time() - t1, "Loss:", epoch_losses)
                t1 = time.time()

                lr_change(epoch_losses)
                loss_patience = epoch_losses
                for param_group in optimizer.param_groups:
                    opt_lr = param_group["lr"]
                if opt_lr <= opt_eps:
                    loop_break_flag = 1
                    break

                epoch_losses = 0


                if epoch % 1 == 0:
                    save_model(epoch, 0, [vae])

        torch.cuda.empty_cache()
        gc.collect()

        if loop_break_flag == 1:
            break

    try:
        save_model(epoch, 0, [vae])
    except:
        pass

