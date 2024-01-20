from Model.GFLayers import *
from Model.LFlayer import *
from Utils.utils import rgb2y

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

import pdb


class ConvertLF(nn.Module):
    def __init__(self, uv_diameter, model):
        super(ConvertLF, self).__init__()

        self.uv_diameter = uv_diameter
        self.uv_center = uv_diameter//2
        self.is_pix2pix = 'pix2pix' in model 
        
        self.Unfolding2Lenslet = UnfoldingLensletPadd(self.uv_diameter, padding=0)
        self.FoldingLenslet2LF = FoldingLensletPadd(self.uv_diameter, padding=0)        
        

    def forward(self, src, center_view, res_fbs = None, mask = None, train=True):
        with torch.no_grad():
            src_lf_org = self.FoldingLenslet2LF(src) # --> (b, c, v, u, y, x) (0, 1, 2, 3, 4, 5)
            occ_t = src_lf_org[:, :, self.uv_center, self.uv_center, :, :] # --> (b, c, y, x)
            src_lf = src_lf_org.reshape(src_lf_org.shape[0], src_lf_org.shape[1], src_lf_org.shape[2]*src_lf_org.shape[3], src_lf_org.shape[4], src_lf_org.shape[5])
            src_lf = src_lf.permute(0, 2, 1, 3, 4)
            src_lf = src_lf.reshape(src_lf.shape[0], src_lf.shape[1]*src_lf.shape[2], src_lf.shape[3], src_lf.shape[4])

        # random crop
        _,_,y,x = src_lf.shape
        if y==256 and x==256:
            rand_x = 0
            rand_y = 32
        elif train:
            rand_x = random.randint(0, 43)
            rand_y = random.randint(0, 7)
        else:
            rand_x = 22
            rand_y = 3

        crop_x = 256
        crop_y = 192
        src_lf = src_lf[:, :, rand_y:rand_y+crop_y, rand_x:rand_x+crop_x]
        center_view = center_view[:, :, rand_y:rand_y+crop_y, rand_x:rand_x+crop_x]
        occ_t = occ_t[:, :, rand_y:rand_y+crop_y, rand_x:rand_x+crop_x]
        occ_t += 0.5
        center_view += 0.5
        if res_fbs is not None:
            res_fbs = res_fbs[:, :, rand_y:rand_y+crop_y, rand_x:rand_x+crop_x]
        if mask is not None:
            mask = mask[:, :, rand_y:rand_y+crop_y, rand_x:rand_x+crop_x]

        # lenslet image crop
        with torch.no_grad():
            lenslet_crop = self.Unfolding2Lenslet(src_lf_org[:,:,:,:,rand_y:rand_y+crop_y, rand_x:rand_x+crop_x])

        return src_lf, center_view, occ_t, res_fbs, mask, lenslet_crop

