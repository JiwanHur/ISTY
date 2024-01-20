# from Model.LFlayer import UnfoldingLensletPadd

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import pdb

class UnfoldingLensletPadd(nn.Module):
    def __init__(self, UV_diameter, padding):
        super().__init__()
        self.UV_diameter_padd = UV_diameter + padding * 2
        self.UV_diameter = UV_diameter
        self.padding = padding

    def forward(self, x):
        xshape = x.shape

        lf_shape = [xshape[0], xshape[1], self.UV_diameter_padd, self.UV_diameter_padd, xshape[4], xshape[5]]
        lf_padd = x.new_zeros(lf_shape, dtype=torch.float32)

        lf_padd[:, :, self.padding:self.UV_diameter_padd - self.padding,
        self.padding:self.UV_diameter_padd - self.padding, :, :] = x
        lf_padd = lf_padd.permute(0, 1, 4, 2, 5, 3)
        lf_reshape = [xshape[0], xshape[1], self.UV_diameter_padd * xshape[4], self.UV_diameter_padd * xshape[5]]

        return torch.reshape(lf_padd, lf_reshape)

class AngularFE(nn.Module):
    def __init__(self, input_channels):
        super(AngularFE, self).__init__()
        # input: bx3x(192*5)x(256*5) lenslet image
        # self.unfold5 = UnfoldingLensletPadd(5, padding=0)

        self.AFE1 = AM2(3, input_channels, 64, kernel_size=3, stride=5, padding=0, bias=False)
        self.AFE2 = AEB(64, 64, 3, 3, 0, False)

        self.AFconv1 = AEB(64, 64, 4, 2, 1, False) # b x 128 x 96 x 128
        self.AFconv2 = AEB(64, 128, 4, 2, 1, False) # b x 256 x 48 x 64
        self.AFconv3 = AEB(128, 256, 4, 2, 1, False) # b x 512 x 24 x 32
        self.AFconv4 = AEB(256, 256, 4, 2, 1, False) # b x 512 x 12 x 16
        self.AFconv5 = AEB(256, 256, 4, 2, 1, False) # b x 512 x 6 x 8 
        self.AFconv6 = AEB(256, 256, 4, 2, 1, False) # b x 512 x 3 x 4 
        self.AFconv7 = AEB(256, 256, 4, 2, 1, False) # b x 512 x 2 x 2
    def forward(self, LensletImg):
        outLenslet33 = self.AFE1(LensletImg)
        out = self.AFE2(outLenslet33)

        af1 = self.AFconv1(out)
        af2 = self.AFconv2(af1)
        af3 = self.AFconv3(af2)
        af4 = self.AFconv4(af3)
        af5 = self.AFconv5(af4)
        af6 = self.AFconv6(af5)
        af7 = self.AFconv7(af6)
        return af1, af2, af3, af4, af5, af6, af7

class AM(nn.Module):
    def __init__(self, unfold_dim, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(AM, self).__init__()
        self.unfold = UnfoldingLensletPadd(unfold_dim, padding=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_list = []
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
        for i in range(1,10):
            self.conv_list.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),).cuda())
                
    def forward(self, feat):
        b, _, h, w = feat.shape
        outFeat = torch.zeros((b, self.out_channels, 3, 3, h//5, w//5)).to(feat.device) # batch x 64 x 3 x 3 x h x w
        for i in range(3):
            for j in range(3):
                padImg = F.pad(feat, (-i, +i, -j, +j)) # left, right, up, down padding
                outFeat[:,:,i,j,:,:] = self.conv_list[3*i+j](padImg)

        return self.unfold(outFeat)

class AEB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(AEB, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, feat):
        return self.seq(feat)

# no weight sharing for each angular domain
class AM2(nn.Module):
    def __init__(self, unfold_dim, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(AM2, self).__init__()
        self.unfold = UnfoldingLensletPadd(unfold_dim, padding=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, feat):
        b, _, h, w = feat.shape
        outFeat = torch.zeros((b, self.out_channels, 3, 3, h//5, w//5)).to(feat.device) # batch x 64 x 3 x 3 x h x w
        for i in range(3):
            for j in range(3):
                padImg = F.pad(feat, (-i, +i, -j, +j)) # left, right, up, down padding
                outFeat[:,:,i,j,:,:] = self.seq(padImg)

        return self.unfold(outFeat)

if __name__=='__main__':
    input = torch.zeros(4, 3, 3, 3, 64, 64)
    AFE = AngularFE(3)
    src_lf = torch.ones(1, 3, 960, 1280)
    AFE(src_lf)

    import random
    height = 11
    for i in range(height):
        print(' ' * (height - i), end='')
        for j in range((2 * i) + 1):
            if random.random() < 0.1:
                color = random.choice(['\033[1;31m', '\033[33m', '\033[1;34m'])
                print(color, end='')  #  the lights 
            else:
                print('\033[32m', end='')  #  green 
            print('*', end='')
        print()
    print((' ' * height) + '|')