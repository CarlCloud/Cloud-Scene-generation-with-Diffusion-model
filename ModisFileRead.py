import os
import time

import cv2
import pyhdf.SD as hdf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# mpl.use('Qt5Agg')
mpl.use('TkAgg')
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
# from mpl_toolkits.basemap import Basemap
from Lib import ccplot
from sklearn.preprocessing import MinMaxScaler

# 按文件夹读所有文件名
def get_file_list(dir_path, extend:str ='hdf'):
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(extend):
                file_list.append(os.path.join(root, file)) # 是否增加'\n' 取决于是否输出

    return file_list

if __name__ == "__main__":
    t0 = time.time()

    dir_path = r"./ModisData"
    file_list = get_file_list(dir_path)
    file_list_aim = [i.replace('./ModisData', 'ModisData/pic') for i in file_list]
    dst_path = dir_path+'Pic'
    data_shape = (7, 406, 270)

    # 用hdfexp查看不同字段对应什么信息，然后按字段.get()矩阵信息
    # F = hdf.SD(file_list[5])
    # long = F.select("Longitude")
    # lat = F.select("Latitude")
    # data = F.select("Brightness_Temperature").get()  # Cloud_Optical_Thickness
    #
    # # a,b = long.max(),long.min()
    # # c,d = lat.max(), lat.min()
    # # plt.axis([a,b,c,d])
    # # plt.scatter(long.get(), lat.get())

    # save normalized data in new hdf5 file
    for i in range(0, file_list.__len__()):
        F = hdf.SD(file_list[i]) # 读取文件
        data = F.select("Brightness_Temperature").get() # 按字段读取矩阵
        data = np.float32(data)
        l = np.zeros((7, 2)) # 记录最大最小值
        for j in range(7): # 逐通道归一化
            l[j] = data[j].max(), data[j].min()
            data[j] = (data[j] - data[j].min()) / (data[j].max() - data[j].min())

        Q = hdf.SD(file_list_aim[i], hdf.SDC.WRITE|hdf.SDC.CREATE) # 读写hdf文件
        BT = Q.create('BT',hdf.SDC.FLOAT64,data_shape)
        BT.set(data)
        L = Q.create('MM', hdf.SDC.FLOAT64, (7,2))
        L.set(l)
        BT.endaccess()
        L.endaccess()
        Q.end()
        F.end()
        print(f'file {i}, save successfully!')

   # def bbb(a, b):
    #     mask = np.ones_like(dft_shift)
    #     filt_x, filt_y = a, b
    #     mask[crow - filt_y:crow + filt_y, 0:filt_x] = 0
    #     mask[crow - filt_y:crow + filt_y, -filt_x:] = 0
    #     dft_shift_back = dft_shift * mask
    #     img_back = cv2.idft(dft_shift_back)
    #     magnitude_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    #     plt.imshow(magnitude_back)