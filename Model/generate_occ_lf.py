from Model.GFLayers import *
from Model.LFlayer import *

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random

import matplotlib.pyplot as plt

import pdb


class GenOCCLF(nn.Module):
    def __init__(self, x_res, y_res, uv_diameter, resize_scale, alpha_size=1):
        super(GenOCCLF, self).__init__()

        self.x_res = x_res
        self.y_res = y_res
        self.uv_diameter = uv_diameter
        self.center_pos = int(self.uv_diameter // 2)
        self.resize_scale = resize_scale

        self.re_x_res = int(self.x_res * self.resize_scale)
        self.re_y_res = int(self.y_res * self.resize_scale)
        
        self.Unfolding2Subaperture = UnfoldingSubaperturePadd(self.uv_diameter, padding=0)
        self.FoldingSubaperture = FoldingSubaperturePadd(self.uv_diameter, padding=0)
        
        self.Unfolding2Lenslet = UnfoldingLensletPadd(self.uv_diameter, padding=0)
        self.FoldingLenslet2LF = FoldingLensletPadd(self.uv_diameter, padding=0)
        
        self.reparam = LFReparam(self.re_x_res, self.re_y_res, self.uv_diameter)
        self.permutation_list = [(1, 2, 0), (1, 0, 2), (2, 1, 0), (2, 0, 1), (0, 1, 2), (0, 2, 1)]
        
        self.alpha_size=alpha_size

    def forward(self, src, occs=None, msks=None):
        if self.alpha_size > 0:
            alpha1 = random.uniform(-self.alpha_size, -0.1) # occlusion disparity range augmentation
            alpha2 = random.uniform(-self.alpha_size*2, -self.alpha_size) # occlusion disparity range augmentation
            alpha3 = random.uniform(-self.alpha_size*3, -self.alpha_size*2) # occlusion disparity range augmentation
        else:
            alpha1 = random.uniform(-1, -0.1) # occlusion disparity range augmentation
            alpha2 = random.uniform(-4, -1) # occlusion disparity range augmentation
            alpha3 = random.uniform(-9, -4) # occlusion disparity range augmentation
        alphas = [alpha1, alpha2, alpha3]

        largest_alpha = 0
        with torch.no_grad():
            src_lf = self.FoldingSubaperture(src)
            center_view = src_lf[:, :, self.center_pos, self.center_pos, :, :]
            src_len = self.Unfolding2Lenslet(src_lf)
            src_len_reparam = self.reparam(src_len, alpha=0.5)
            mask = torch.zeros_like(center_view)
            #if (occ1 is not None) & (msk1 is not None) & (alpha is not None):
            if occs is not None:
                rand_occ = random.randint(1, 3)
                for i in range(rand_occ):
                    rand_idx = random.randint(0, len(self.permutation_list)-1)
                    
                    occ = torch.zeros_like(occs[i]) # B x C x Y x X
                    occ[:, 0] = occs[i][:, self.permutation_list[rand_idx][0]]
                    occ[:, 1] = occs[i][:, self.permutation_list[rand_idx][1]]
                    occ[:, 2] = occs[i][:, self.permutation_list[rand_idx][2]]
                    msk = msks[i]
                    alpha = alphas[i]
                    
                    occ_lf = torch.zeros_like(src)
                    occ_lf = self.FoldingSubaperture(occ_lf)
                    msk_lf = torch.zeros_like(src)
                    msk_lf = self.FoldingSubaperture(msk_lf)
                    
                    msk_ch3 = torch.zeros(size=(msk.shape[0], 3, msk.shape[2], msk.shape[3]), device=msk.device)
                    msk_ch3[:, 0] = msk.squeeze()
                    msk_ch3[:, 1] = msk.squeeze()
                    msk_ch3[:, 2] = msk.squeeze()
                    mask += msk_ch3
                    
                    for u_idx in range(0, self.uv_diameter):
                        for v_idx in range(0, self.uv_diameter):
                            occ_lf[:, :, v_idx, u_idx, :, :] = occ
                            msk_lf[:, :, v_idx, u_idx, :, :] = msk_ch3
                    
                    occ_len = self.Unfolding2Lenslet(occ_lf)
                    occ_len_reparam = self.reparam(occ_len, alpha)
                    msk_len = self.Unfolding2Lenslet(msk_lf)

                    msk_len_reparam = self.reparam(msk_len, alpha)
                    
                    src_len_reparam[msk_len_reparam > 0.95] = occ_len_reparam[msk_len_reparam > 0.95]

                    # largest_alpha = alpha

            return src_len_reparam, center_view, mask #, largest_alpha
