from Model.GFLayers import *
from Model.LFlayer import *
from Utils.utils import rgb2y

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import matplotlib.pyplot as plt

import pdb


class SeparateFB(nn.Module):
    def __init__(self, x_res, y_res, uv_diameter, resize_scale):
        super(SeparateFB, self).__init__()

        self.x_res = x_res
        self.y_res = y_res
        self.uv_diameter = uv_diameter
        self.resize_scale = resize_scale

        self.re_x_res = int(self.x_res * self.resize_scale)
        self.re_y_res = int(self.y_res * self.resize_scale)
        
        self.Unfolding2Subaperture = UnfoldingSubaperturePadd(self.uv_diameter, padding=0)
        self.FoldingSubaperture = FoldingSubaperturePadd(self.uv_diameter, padding=0)
        
        self.Unfolding2Lenslet = UnfoldingLensletPadd(self.uv_diameter, padding=0)
        self.FoldingLenslet2LF = FoldingLensletPadd(self.uv_diameter, padding=0)
        
        self.fbs = HandcraftFeature_lamb(self.uv_diameter, gf_radius=5, gf_eps=1e-5, fThres=False)


    def forward(self, src, pdb_flag=False):
        with torch.no_grad():
            if pdb_flag:
                pdb.set_trace()
            src_y = rgb2y(src)
            src_y = self.FoldingLenslet2LF(src_y)
            output = self.fbs(src_y)

        return output

