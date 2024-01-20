from torch.utils.data import Dataset
import os
import PIL.Image
from random import *
import random
from Dataloader.dataloader_LF import *
# from dataloader_LF import *

import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
import math
import torch

import pdb
import h5py

class DeOccNetDataloader(Dataset):
    def __init__(self, data_dir, x_res, y_res, uv_diameter_image, uv_diameter, uv_dilation, opt_output, opt_scale, mode, specific_dir):
        
        self.data_dir = data_dir
        
        self.x_res = x_res
        self.y_res = y_res
        self.uv_diameter_image = uv_diameter_image
        self.uv_diameter = uv_diameter
        self.uv_dilation = uv_dilation
        self.uv_radius = uv_diameter // 2

        self.opt_output = opt_output
        self.opt_scale = opt_scale
        self.mode = mode
        self.specific_dir = specific_dir
        self.cnt = 0
        self.gt_exists = False

        self.path_src_img = os.path.join(self.data_dir, self.specific_dir)
        self.path_gt_img = os.path.join(self.data_dir, self.specific_dir+'_gt')
        self.src_img_lst = self.getDataList(self.path_src_img, '.png')
        if os.path.exists(self.path_gt_img):
            self.gt_img_lst = self.getDataList(self.path_gt_img, '.png')
        else:
            self.gt_img_lst = []

    def __getitem__(self, index):
        src_img_name = os.path.join(self.path_src_img, self.src_img_lst[index])
        src_img = PIL.Image.open(src_img_name)

        if src_img.size[0]==512*5 and src_img.size[1]==512*5:
            self.x_res = 512
            self.y_res = 512

        src_img = src_img.resize((int(src_img.size[0] * self.opt_scale), int(src_img.size[1] * self.opt_scale)))
        src_img = np.array(src_img, dtype=np.uint8)
        src_img = src_img.astype(np.float32)/255.0 - 0.5


        if len(self.gt_img_lst) > 0:
            gt_img_name = os.path.join(self.path_gt_img, self.gt_img_lst[index])
            gt_img = PIL.Image.open(gt_img_name)
            gt_img = gt_img.resize((int(self.x_res * self.opt_scale), int(self.y_res * self.opt_scale))) # src_img에 맞게 scale
            gt_img = np.array(gt_img, dtype=np.uint8)
            gt_img = gt_img.astype(np.float32)/255.0 - 0.5
            self.gt_exists = True

        if self.opt_output == '2d_sub':
            if (self.uv_diameter_image == 9) & (self.uv_diameter == 5):
                src_img = folding_subaperture2LF(src_img, 9)
                if self.uv_dilation==1:
                    src_img = src_img[4-self.uv_radius:4+self.uv_radius+1, 4-self.uv_radius:4+self.uv_radius+1, :, :, :]
                elif self.uv_dilation==2:
                    _, _, w, h, c = src_img.shape
                    src_img_new = np.zeros((5,5,w,h,c))
                    for i in range(5):
                        for j in range(5):
                            src_img_new[i,j,:,:,:] = src_img[2*i, 2*j, :, :, :]
                    src_img = src_img_new
                else:
                    raise NotImplementedError
                src_img = unfolding_subaperture_padd(src_img)
            elif (self.uv_diameter_image == 5) & (self.uv_diameter == 5):
                src_img = folding_subaperture2LF(src_img, 5)
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
        
        if self.gt_exists:
            data = {'src_img': torch.from_numpy(src_img),
                    'gt_img': torch.from_numpy(gt_img),
                    'file_name': self.src_img_lst[index]}

        else:
            data = {'src_img': torch.from_numpy(src_img),
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
    data_dir = '/workspace/ssd1/datasets/DeOccNet'
    uv_diameter = 5
    uv_diameter_crop = 5
    uv_dilation = 1
    ds = DeOccNetDataloader(data_dir, 600, 400, uv_diameter, uv_diameter_crop, uv_dilation, '2d_sub', 0.5, 'test', 'Synscenes9')
    pdb.set_trace()
    data = ds.__getitem__(0)
    src_img = data['src_img']
    fname = data['file_name']
    sname = './debug/test_img.png'    
    tt = np.clip(src_img.numpy().transpose(1,2,0)[:, :, 0:3] + 0.5, 0, 1)
    plt.imsave(sname, tt)
    print(sname)
    # sname = './debug/test_img_occ.png'    
    # plt.imsave(sname, np.clip(occ_img.numpy().transpose(1,2,0), 0, 1), cmap='gray')