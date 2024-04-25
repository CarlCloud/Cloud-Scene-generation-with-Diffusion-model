# Graduation_Project
using Pytorch to training a cloud scene generation diffusion model  
先用ModisRead批量读取指定目录（如：ModisData）内的hdf5文件的指定字段，存储为较小的hdf。  
之后可以使用diffusion.py对读取的hdf文件进行训练，训练产生HxWxN(N是你自己设定的channels)。  
最后使用训练好的VAE.py或Real-ESRGAN.py对\images\results\save_point下的checkpoint导入模型提升分辨率。  
VAE是用huggingface的AutoEncoderKL基于Modis06在2020.3-6夜间的数据集进行训练的，数据量小了产生  
的图片会产生畸变。
![image](https://github.com/CarlCloud/Graduation_Project/)
Real-ESRGAN则是用的论文里的基于世界真实图片集的模型。Real-ESRGAN因为训练数据多对真实图片信息扩充能力较强，  
但是冗余也多会出现超分辨完了图片和Modis的三通道图差距很大的问题。其次用Real-GAN官方的模型数据通道只有  
三通道。  
