
from torch.utils.data import Dataset
import os
import PIL.Image
from random import *
import random

import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
import math
import torch

import pdb

def unfolding_lenslet_padd(img_lf, padd=0):
    '''
    :param img_lf: V x U x Y x X x C
    '''
    lf_shape = img_lf.shape
    UV_diameter = lf_shape[1]
    UV_diameter_padd = UV_diameter + padd * 2
    if img_lf.ndim == 5:
        lf_len_shape = (lf_shape[0] + padd * 2, \
                        lf_shape[1] + padd * 2, \
                        lf_shape[2], \
                        lf_shape[3], 3)
        img_lf_padd = np.zeros(shape=lf_len_shape, dtype='float32')
        img_lf_padd[padd:UV_diameter_padd - padd, padd:UV_diameter_padd - padd, :, :, :] = img_lf
        lf_x_size = (lf_shape[1] + padd * 2) * lf_shape[3]
        lf_y_size = (lf_shape[0] + padd * 2) * lf_shape[2]
        lenslet_padd = img_lf_padd.transpose([2, 0, 3, 1, 4]).reshape(lf_y_size, lf_x_size, 3)
    elif img_lf.ndim == 4:
        lf_len_shape = (lf_shape[0] + padd * 2, \
                        lf_shape[1] + padd * 2, \
                        lf_shape[2], \
                        lf_shape[3])
        img_lf_padd = np.zeros(shape=lf_len_shape, dtype='float32')
        img_lf_padd[padd:UV_diameter_padd - padd, padd:UV_diameter_padd - padd, :, :] = img_lf
        lf_x_size = (lf_shape[1] + padd * 2) * lf_shape[3]
        lf_y_size = (lf_shape[0] + padd * 2) * lf_shape[2]
        lenslet_padd = img_lf_padd.transpose([2, 0, 3, 1]).reshape(lf_y_size, lf_x_size)
    else:
        print('make_lenslet_padd dimension error')

    return lenslet_padd

def unfolding_subaperture_padd(img_lf, padd=0):
    '''
    :param img_lf: V x U x Y x X x C
    '''
    lf_shape = img_lf.shape

    UV_diameter = lf_shape[1]
    y_size = lf_shape[2]
    x_size = lf_shape[3]

    y_size_padd = y_size + padd * 2
    x_size_padd = x_size + padd * 2

    lf_x_size = UV_diameter * x_size_padd
    lf_y_size = UV_diameter * y_size_padd

    if img_lf.ndim == 5:
        img_lf_padd = np.zeros((UV_diameter, UV_diameter, y_size + padd * 2, x_size + padd * 2, 3), dtype='float32')
        img_lf_padd[:, :, padd:y_size_padd - padd, padd:x_size_padd - padd, :] = img_lf
        subaperture_padd = img_lf_padd.transpose([0, 2, 1, 3, 4]).reshape(lf_y_size, lf_x_size, 3)
    elif img_lf.ndim == 4:
        img_lf_padd = np.zeros((UV_diameter,  UV_diameter, y_size + padd * 2, x_size + padd * 2), dtype='float32')
        img_lf_padd[:, :, padd:y_size_padd - padd, padd:x_size_padd - padd] = img_lf
        subaperture_padd = img_lf_padd.transpose([0, 2, 1, 3]).reshape(lf_y_size, lf_x_size)
    else:
        print('make_subaperture_padd dimension error')

    return subaperture_padd

def folding_lenslet2LF(img_lenslet, UV_diameter, padd=0):
    u_size = UV_diameter + padd * 2
    v_size = UV_diameter + padd * 2
    # print(u_size)
    x_size = int(img_lenslet.shape[1] / u_size)
    y_size = int(img_lenslet.shape[0] / v_size)

    if img_lenslet.ndim == 2:
        img_lf = img_lenslet.reshape((y_size, v_size, x_size, u_size)).transpose([1, 3, 0, 2])
        img_lf = img_lf[padd:v_size -padd, padd:u_size -padd, :, :]

    elif img_lenslet.ndim == 3:
        img_lf = img_lenslet.reshape((y_size, v_size, x_size, u_size, 3)).transpose([1, 3, 0, 2, 4])
        img_lf = img_lf[padd:v_size - padd, padd:u_size - padd, :, :, :]

    return img_lf

def folding_subaperture2LF(img_subaperture, UV_diameter, padd=0):
    u_size = UV_diameter
    v_size = UV_diameter

    x_size = int(img_subaperture.shape[1] / u_size)
    y_size = int(img_subaperture.shape[0] / v_size)

    if img_subaperture.ndim == 2:
        img_lf = (img_subaperture.reshape((v_size, y_size, u_size, x_size))).transpose([0, 2, 1, 3])
        img_lf = img_lf[:, :, padd:y_size -padd, padd:x_size -padd]

    elif img_subaperture.ndim == 3:
        img_lf = (img_subaperture.reshape((v_size, y_size, u_size, x_size, 3))).transpose([0, 2, 1, 3, 4])
        img_lf = img_lf[:, :, padd:y_size - padd, padd:x_size - padd, :]

    return img_lf


class LFDataloader(Dataset):
    def __init__(self, data_dir, x_res, y_res, uv_diameter, uv_dilation, opt_output, opt_scale, mode):
        
        self.data_dir = data_dir
        
        self.x_res = x_res
        self.y_res = y_res
        self.uv_diameter = uv_diameter
        self.uv_radius = uv_diameter // 2

        self.opt_output = opt_output
        self.opt_scale = opt_scale
        self.mode = mode
        self.cnt = 0


        self.path_src_img = self.data_dir + '/src_imgs_' + self.mode + '/'
        self.src_img_lst = self.getDataList(self.path_src_img, '.jpg')
        self.path_occ_img = self.data_dir + '/occ_imgs/'
        self.occ_img_lst = self.getDataList(self.path_occ_img, '.jpg')
        self.path_occ_msk = self.data_dir + '/occ_msks/'
        self.occ_msk_lst = self.getDataList(self.path_occ_msk, '.jpg')


    def __getitem__(self, index):
        src_img_name = self.path_src_img + self.src_img_lst[index]        
        src_img = PIL.Image.open(src_img_name)
        src_img = src_img.resize((int(src_img.size[0] * self.opt_scale), int(src_img.size[1] * self.opt_scale)))
        src_img = np.array(src_img, dtype=np.uint8)
        src_img = src_img.astype(np.float32)/255.0 - 0.5
        
        # if self.mode == 'train':
        idx_rand1 = random.randint(0, len(self.occ_img_lst) - 1)
        idx_rand2 = random.randint(0, len(self.occ_img_lst) - 1)
        idx_rand3 = random.randint(0, len(self.occ_img_lst) - 1)
        # elif self.mode == 'valid':
        #     # idx_rand = index % len(self.occ_img_lst)
        #     idx_rand1 = index % len(self.occ_img_lst)
        #     idx_rand2 = index % len(self.occ_img_lst)
        #     idx_rand3 = index % len(self.occ_img_lst)
        # idx_rand = random.randint(0, len(self.occ_img_lst) - 1)

        resize_x = int(self.x_res * self.opt_scale)
        resize_y = int(self.y_res * self.opt_scale)

        occ_img_name = self.path_occ_img + self.occ_img_lst[idx_rand1]
        occ_img = PIL.Image.open(occ_img_name)
        occ_img = occ_img.resize((resize_x, resize_y))
        occ_img = np.array(occ_img, dtype=np.uint8)
        occ_img1 = occ_img.astype(np.float32)/255.0 -0.5

        occ_msk_name = self.path_occ_msk + self.occ_msk_lst[idx_rand1]
        occ_msk = PIL.Image.open(occ_msk_name)
        occ_msk = occ_msk.resize((resize_x, resize_y))
        occ_msk = np.array(occ_msk, dtype=np.uint8)
        occ_msk = occ_msk.astype(np.float32)/255.0
        occ_msk1 = np.expand_dims(occ_msk, 2)

        occ_img_name = self.path_occ_img + self.occ_img_lst[idx_rand2]
        occ_img = PIL.Image.open(occ_img_name)
        occ_img = occ_img.resize((resize_x, resize_y))
        occ_img = np.array(occ_img, dtype=np.uint8)
        occ_img2 = occ_img.astype(np.float32)/255.0 -0.5

        occ_msk_name = self.path_occ_msk + self.occ_msk_lst[idx_rand2]
        occ_msk = PIL.Image.open(occ_msk_name)
        occ_msk = occ_msk.resize((resize_x, resize_y))
        occ_msk = np.array(occ_msk, dtype=np.uint8)
        occ_msk = occ_msk.astype(np.float32)/255.0
        occ_msk2 = np.expand_dims(occ_msk, 2)

        occ_img_name = self.path_occ_img + self.occ_img_lst[idx_rand3]
        occ_img = PIL.Image.open(occ_img_name)
        occ_img = occ_img.resize((resize_x, resize_y))
        occ_img = np.array(occ_img, dtype=np.uint8)
        occ_img3 = occ_img.astype(np.float32)/255.0 -0.5

        occ_msk_name = self.path_occ_msk + self.occ_msk_lst[idx_rand3]
        occ_msk = PIL.Image.open(occ_msk_name)
        occ_msk = occ_msk.resize((resize_x, resize_y))
        occ_msk = np.array(occ_msk, dtype=np.uint8)
        occ_msk = occ_msk.astype(np.float32)/255.0
        occ_msk3 = np.expand_dims(occ_msk, 2)

        if self.opt_output == '2d_sub':
            if self.uv_diameter != 9:
                src_img = folding_subaperture2LF(src_img, 9)
                src_img = src_img[4-self.uv_radius:4+self.uv_radius+1, 4-self.uv_radius:4+self.uv_radius+1, :, :, :]
                src_img = unfolding_subaperture_padd(src_img)

            src_img = src_img.transpose([2, 0, 1])

        elif self.opt_output == '2d_len':
            src_img = folding_subaperture2LF(src_img, self.uv_diameter)
            src_img = unfolding_lenslet_padd(src_img, padd=0)
            src_img = src_img.transpose([2, 0, 1])

        elif self.opt_output == '3d_sub':
            src_img = folding_subaperture2LF(src_img, self.uv_diameter)
            src_img = src_img.transpose([2, 3, 4, 0, 1]).reshape(src_img.shape[2], src_img.shape[3], 3, self.uv_diameter * self.uv_diameter)
            src_img = src_img.transpose([0, 1, 3, 2]).reshape(src_img.shape[0], src_img.shape[1], 3 * self.uv_diameter * self.uv_diameter)
            src_img = src_img.transpose([2, 0, 1])

        elif self.opt_output == '4d':
            src_img = folding_subaperture2LF(src_img, self.uv_diameter)        
        
        occ_img1 = occ_img1.transpose([2, 0, 1])
        occ_msk1 = occ_msk1.transpose([2, 0, 1])
        if random.random() > 0.5: # add random horizontal flip
            occ_img1 = np.flip(occ_img1, 2).copy()
            occ_msk1 = np.flip(occ_msk1, 2).copy()
        if random.random() > 0.5: # add random vertical flip
            occ_img1 = np.flip(occ_img1, 1).copy()
            occ_msk1 = np.flip(occ_msk1, 1).copy()

        occ_img2 = occ_img2.transpose([2, 0, 1])
        occ_msk2 = occ_msk2.transpose([2, 0, 1])
        if random.random() > 0.5: # add random flip
            occ_img2 = np.flip(occ_img2, 2).copy()
            occ_msk2 = np.flip(occ_msk2, 2).copy()
        if random.random() > 0.5: # add random flip
            occ_img2 = np.flip(occ_img2, 1).copy()
            occ_msk2 = np.flip(occ_msk2, 1).copy()

        occ_img3 = occ_img3.transpose([2, 0, 1])
        occ_msk3 = occ_msk3.transpose([2, 0, 1])
        if random.random() > 0.5: # add random flip
            occ_img3 = np.flip(occ_img3, 2).copy()
            occ_msk3 = np.flip(occ_msk3, 2).copy()
        if random.random() > 0.5: # add random flip
            occ_img3 = np.flip(occ_img3, 1).copy()
            occ_msk3 = np.flip(occ_msk3, 1).copy()

        data = {'src_img': torch.from_numpy(src_img),
                'occ_img1': torch.from_numpy(occ_img1), 
                'occ_msk1': torch.from_numpy(occ_msk1), 
                'occ_img2': torch.from_numpy(occ_img2), 
                'occ_msk2': torch.from_numpy(occ_msk2), 
                'occ_img3': torch.from_numpy(occ_img3), 
                'occ_msk3': torch.from_numpy(occ_msk3), 
                'file_name': self.src_img_lst[index]}

        return data

    def __len__(self):
        return len(self.src_img_lst)


    def getDataList(self, data_dir, ext):
        flist = os.listdir(data_dir)
        flist = [f for f in flist if f.endswith(ext)]
        flist.sort()

        return flist


if __name__ == '__main__':
    data_dir = '/workspace/ssd1/datasets/LFGAN'
    uv_diameter = 5
    fTrain = False
    ds = LFDataloader(data_dir, 600, 400, uv_diameter, '2d_sub', 0.5, 'train')
    data = ds.__getitem__(319)
    pdb.set_trace()
    src_img = data['src_img']
    occ_img = data['occ_img']
    occ_msk = data['occ_msk']
    fname = data['file_name']
    sname = './debug/test_img.png'    
    tt = np.clip(src_img.numpy().transpose(1,2,0)[:, :, 0:3] + 0.5, 0, 1)
    plt.imsave(sname, tt)
    sname = './debug/test_img_occ.png'    
    plt.imsave(sname, np.clip(occ_img.numpy().transpose(1,2,0), 0, 1), cmap='gray')