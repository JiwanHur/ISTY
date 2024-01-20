import torch
import torch.nn as nn
import math
import torchvision.utils
import numpy as np
import pdb

from Model.GFLayers import *

class HandcraftFeature_lamb(nn.Module):
    def __init__(self, uv_diameter, gf_radius, gf_eps, fThres):
        super(HandcraftFeature_lamb, self).__init__()
        self.unfolding_len = UnfoldingLensletPadd(UV_diameter=uv_diameter, padding=1)
        self.unfolding_sub = UnfoldingSubaperturePadd(UV_diameter=uv_diameter, padding=1)
        self.folding_len = FoldingLensletPadd(UV_diameter=uv_diameter, padding=1)
        self.folding_sub = FoldingSubaperturePadd(UV_diameter=uv_diameter, padding=1)
        self.ang_pool = LFAngAvgPooling(uv_diameter=uv_diameter)
        self.gf = GuidedFilter(r=gf_radius, eps=gf_eps)
        self.uv_center = int(uv_diameter / 2)
        self.fThres = fThres

        self.grad_x = torch.FloatTensor([[1, 0, -1],
                                         [2, 0, -2],
                                         [1, 0, -1]])
        self.grad_x = self.grad_x.view(1, 1, 3, 3).repeat(1, 1, 1, 1)

        self.grad_y = torch.FloatTensor([[1, 2, 1], \
                                         [0, 0, 0], \
                                         [-1, -2, -1]])
        self.grad_y = self.grad_y.view(1, 1, 3, 3).repeat(1, 1, 1, 1)

    def forward(self, x):
        x_t = x.clone()
        
        x_center = x_t[:, 0, self.uv_center, self.uv_center, :, :].unsqueeze(1)
        x_len = self.unfolding_len(x_t[:, 0, :, :, :, :].unsqueeze(1))
        
        self.grad_x = x.new_tensor(self.grad_x, device=x.device)
        self.grad_y = x.new_tensor(self.grad_y, device=x.device)

        grad_u = F.conv2d(x_len, self.grad_x, stride=1, padding=1)
        grad_v = F.conv2d(x_len, self.grad_y, stride=1, padding=1)
        grad_len = torch.cat((grad_u, grad_v), 1)
        grad_len = self.folding_len(grad_len)

        x_sub = self.unfolding_sub(x[:, 0, :, :, :, :].unsqueeze(1))
        grad_x = F.conv2d(x_sub, self.grad_x, stride=1, padding=1)
        grad_y = F.conv2d(x_sub, self.grad_y, stride=1, padding=1)
        grad_sub = torch.cat((grad_x, grad_y), 1)
        grad_sub = self.folding_sub(grad_sub)

        grad_mul = grad_len * grad_sub
        grad_sgn = torch.sign(grad_mul)
        lamb_t = torch.tensor(grad_sgn[:, 0, :, :, :, :] == grad_sgn[:, 1, :, :, :, :], dtype=torch.float32, device=grad_sgn.device)
        lamb_t = lamb_t.unsqueeze(dim=1)
        grad_sgn_xu = grad_sgn[:, 0, :, :, :, :].unsqueeze(dim=1)
        
        numerator = self.ang_pool(lamb_t * grad_sgn_xu)
        denominator = self.ang_pool(lamb_t)

        costd = numerator / denominator
        costd[denominator == 0] = 0

        sgn_gf = self.gf(x_center, costd)
        if self.fThres == 1:
            sgn_gf = torch.tensor(sgn_gf > 0, dtype=torch.float32, device=sgn_gf.device) - 0.5

        return sgn_gf


# reparameterization
class LFReparam(nn.Module):
    def __init__(self, x_res, y_res, uv_diameter):
        super().__init__()
        self.x_res = x_res
        self.y_res = y_res
        self.uv_diameter = uv_diameter
        self.uv_radius = math.floor(uv_diameter/2)
        self.UnfoldingLF2Lenslet = UnfoldingLensletPadd(uv_diameter, padding=0)
        self.FoldingLenslet2LF = FoldingLensletPadd(uv_diameter, padding=0)

    def forward(self, x, alpha):
        img_lenslet = x.clone()
        img_reparam = x.new_zeros(x.shape, dtype=torch.float32)

        vgrid = torch.tensor(range(0, self.uv_diameter), dtype=torch.int16, device=x.device)
        ugrid = torch.tensor(range(0, self.uv_diameter), dtype=torch.int16, device=x.device)
        ygrid = torch.tensor(range(0, self.y_res), dtype=torch.int16, device=x.device)
        xgrid = torch.tensor(range(0, self.x_res), dtype=torch.int16, device=x.device)
        vgrid, ugrid, ygrid, xgrid = torch.meshgrid([vgrid, ugrid, ygrid, xgrid])
        vgrid = torch.tensor((vgrid.permute(2, 0, 3, 1).reshape(self.y_res * self.uv_diameter, self.x_res * self.uv_diameter) - self.uv_radius), dtype=torch.long, device=x.device)
        ugrid = torch.tensor((ugrid.permute(2, 0, 3, 1).reshape(self.y_res * self.uv_diameter, self.x_res * self.uv_diameter) - self.uv_radius), dtype=torch.long, device=x.device)
        ygrid = torch.tensor((ygrid.permute(2, 0, 3, 1).reshape(self.y_res * self.uv_diameter, self.x_res * self.uv_diameter)), dtype=torch.long, device=x.device)
        xgrid = torch.tensor((xgrid.permute(2, 0, 3, 1).reshape(self.y_res * self.uv_diameter, self.x_res * self.uv_diameter)), dtype=torch.long, device=x.device)

        x_ind = (torch.tensor(ugrid, dtype=torch.float, device=x.device) * - alpha + torch.tensor(xgrid, dtype=torch.float, device=x.device))
        y_ind = (torch.tensor(vgrid, dtype=torch.float, device=x.device) * - alpha + torch.tensor(ygrid, dtype=torch.float, device=x.device))

        x_floor = torch.tensor(torch.floor(x_ind), dtype=torch.float, device=x.device)
        y_floor = torch.tensor(torch.floor(y_ind), dtype=torch.float, device=x.device)

        x_1 = torch.clamp(x_floor, 0, self.x_res - 1)
        y_1 = torch.clamp(y_floor, 0, self.y_res - 1)
        x_2 = torch.clamp(x_floor + 1, 0, self.x_res - 1)
        y_2 = torch.clamp(y_floor + 1, 0, self.y_res - 1)

        x_1_w = (1.0 - (x_ind - x_floor))
        x_2_w = (1.0 - x_1_w)
        y_1_w = (1.0 - (y_ind - y_floor))
        y_2_w = (1.0 - y_1_w)

        x_1_index = torch.tensor(ugrid + self.uv_radius + torch.tensor(x_1, dtype=torch.long, device=ugrid.device) * self.uv_diameter, dtype=torch.long, device=x.device)
        y_1_index = torch.tensor(vgrid + self.uv_radius + torch.tensor(y_1, dtype=torch.long, device=ugrid.device) * self.uv_diameter, dtype=torch.long, device=x.device)
        x_2_index = torch.tensor(ugrid + self.uv_radius + torch.tensor(x_2, dtype=torch.long, device=ugrid.device) * self.uv_diameter, dtype=torch.long, device=x.device)
        y_2_index = torch.tensor(vgrid + self.uv_radius + torch.tensor(y_2, dtype=torch.long, device=ugrid.device) * self.uv_diameter, dtype=torch.long, device=x.device)
        #pdb.set_trace()
        #x_1_index = torch.tensor(ugrid + self.uv_radius + x_1 * self.uv_diameter, dtype=torch.long)
        #y_1_index = torch.tensor(vgrid + self.uv_radius + y_1 * self.uv_diameter, dtype=torch.long)
        #x_2_index = torch.tensor(ugrid + self.uv_radius + x_2 * self.uv_diameter, dtype=torch.long)
        #y_2_index = torch.tensor(vgrid + self.uv_radius + y_2 * self.uv_diameter, dtype=torch.long)

        x_r_index = torch.tensor(ugrid + self.uv_radius + xgrid * self.uv_diameter, dtype=torch.long, device=x.device)
        y_r_index = torch.tensor(vgrid + self.uv_radius + ygrid * self.uv_diameter, dtype=torch.long, device=x.device)

        # del ugrid, vgrid, xgrid, ygrid, x_1, y_1, x_2, y_2, x_floor, y_floor, x_ind, y_ind
        
        img_reparam[:, :, y_r_index, x_r_index] = y_1_w * x_1_w * img_lenslet[:, :, y_1_index, x_1_index]
        img_reparam[:, :, y_r_index, x_r_index] += y_1_w * x_2_w * img_lenslet[:, :, y_1_index, x_2_index]
        img_reparam[:, :, y_r_index, x_r_index] += y_2_w * x_1_w * img_lenslet[:, :, y_2_index, x_1_index]
        img_reparam[:, :, y_r_index, x_r_index] += y_2_w * x_2_w * img_lenslet[:, :, y_2_index, x_2_index]

        #img_reparam = a + b + c + d
        #img_reparam[:, :, y_r_index, x_r_index]
        #img_reparam = self.FoldingLenslet2LF(img_reparam)
        
        return img_reparam


class UnfoldingLensletPadd(nn.Module):
    def __init__(self, UV_diameter, padding):
        super().__init__()
        self.UV_diameter_padd = UV_diameter + padding * 2
        self.UV_diameter = UV_diameter
        self.padding = padding

    def forward(self, x):
        xshape = x.shape

        lf_shape = [xshape[0], xshape[1], self.UV_diameter_padd, self.UV_diameter_padd, xshape[4], xshape[5]]

        # if x.get_device() >= 0:
        #     lf_padd = torch.zeros(lf_shape).to(x.device)
        # else:
        #     lf_padd = torch.zeros(lf_shape)
        lf_padd = x.new_zeros(lf_shape, dtype=torch.float32)

        lf_padd[:, :, self.padding:self.UV_diameter_padd - self.padding,
        self.padding:self.UV_diameter_padd - self.padding, :, :] = x
        lf_padd = lf_padd.permute(0, 1, 4, 2, 5, 3)
        lf_reshape = [xshape[0], xshape[1], self.UV_diameter_padd * xshape[4], self.UV_diameter_padd * xshape[5]]

        #if x.get_device() >= 0:
        #    return torch.reshape(lf_padd, lf_reshape).to(x.device)
        #else:
        #    return torch.reshape(lf_padd, lf_reshape)
        return torch.reshape(lf_padd, lf_reshape)


class UnfoldingSubaperturePadd(nn.Module):
    def __init__(self, UV_diameter, padding):
        super().__init__()
        self.UV_diameter = UV_diameter
        self.padding = padding

    def forward(self, x):
        xshape = x.shape

        y_res_padd = xshape[4] + self.padding * 2
        x_res_padd = xshape[5] + self.padding * 2
        lf_shape = [xshape[0], xshape[1], xshape[2], xshape[3], y_res_padd, x_res_padd]

        # if x.get_device() >= 0:
        #     lf_padd = torch.zeros(lf_shape).to(x.device)
        # else:
        #     lf_padd = torch.zeros(lf_shape)
        lf_padd = x.new_zeros(lf_shape, dtype=torch.float32)

        lf_padd[:, :, :, :, self.padding:y_res_padd - self.padding,
        self.padding:x_res_padd - self.padding] = x
        lf_padd = lf_padd.permute(0, 1, 2, 4, 3, 5)
        lf_reshape = [xshape[0], xshape[1], xshape[2] * y_res_padd, xshape[3] * x_res_padd]

        #if x.get_device() >= 0:
        #    return torch.reshape(lf_padd, lf_reshape).to(x.device)
        #else:
        #    return torch.reshape(lf_padd, lf_reshape)
        return torch.reshape(lf_padd, lf_reshape)


class FoldingLensletPadd(nn.Module):
    def __init__(self, UV_diameter, padding):
        super().__init__()
        self.UV_diameter_padd = UV_diameter + padding * 2
        self.padding = padding

    def forward(self, x):
        xshape = x.shape
        lf_shape = [xshape[0], xshape[1],
                    math.floor(xshape[2] / self.UV_diameter_padd),
                    self.UV_diameter_padd,
                    math.floor(xshape[3] / self.UV_diameter_padd),
                    self.UV_diameter_padd]

        x = torch.reshape(x, lf_shape)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x[:, :, self.padding:self.UV_diameter_padd - self.padding,self.padding:self.UV_diameter_padd - self.padding, :, :]

        #if x.get_device() >= 0:
        #    return x.to(x.device)
        #else:
        #    return x
        return x


class FoldingSubaperturePadd(nn.Module):
    def __init__(self, UV_diameter, padding):
        super().__init__()
        self.UV_diameter = UV_diameter
        self.padding = padding

    def forward(self, x):
        xshape = x.shape
        y_res = math.floor(xshape[2] / self.UV_diameter)
        x_res = math.floor(xshape[3] / self.UV_diameter)
        lf_shape = [xshape[0], xshape[1],
                    self.UV_diameter, y_res,
                    self.UV_diameter, x_res]

        x = torch.reshape(x, lf_shape)
        x = x.permute(0, 1, 2, 4, 3, 5)
        x = x[:, :, :, :, self.padding:y_res - self.padding, self.padding:x_res - self.padding]

        #if x.get_device() >= 0:
        #    return x.to(x.device)
        #else:
        #    return x
        return x


# angular pooling
class LFAngPoolConv(nn.Module):
    def __init__(self, in_channel, out_channel, uv_diameter, fBN=False, fAct=False):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.uv_diameter = uv_diameter
        self.fBN = fBN
        self.fAct = fAct

        self.UnfoldingLen = UnfoldingLensletPadd(uv_diameter, padding=0)
        self.conv_pool = nn.Conv2d(in_channel, out_channel, kernel_size=uv_diameter, stride=uv_diameter, padding=0,
                                   bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.FoldingLen = FoldingLensletPadd(uv_diameter, padding=0)

    def forward(self, x):
        x = self.UnfoldingLen(x)
        x = self.conv_pool(x)
        if self.fBN:
            x = self.bn(x)
        if self.fAct:
            x = self.relu(x)

        return x


# spatial pooling
class LFSpaPooling(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, uv_diameter, type, fBN=False, fAct=False):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.uv_diameter = uv_diameter
        self.kernel_size = kernel_size
        self.type = type
        self.fBN = fBN
        self.fAct = fAct

        self.UnfoldingSub = UnfoldingSubaperturePadd(uv_diameter, padding=0)
        if type == 'conv':
            self.conv_pool = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=kernel_size, padding=0, bias=False)
        elif type == 'avg':
            self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size, padding=0)
        elif type == 'max':
            self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size, padding=0)
            
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.FoldingSub = FoldingSubaperturePadd(uv_diameter, padding=0)

    def forward(self, x):
        x = self.UnfoldingSub(x)
        if self.type == 'conv':
            x = self.conv_pool(x)
        elif self.type == 'avg':
            x = self.avg_pool(x)
        elif self.type == 'max':
            x = self.max_pool(x)
        x = self.FoldingSub(x)
        
        return x


class LFSpaUpscale(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, uv_diameter, type, fBN=False, fAct=False):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.uv_diameter = uv_diameter
        self.kernel_size = kernel_size
        self.type = type
        self.fBN = fBN
        self.fAct = fAct

        self.UnfoldingSub = UnfoldingSubaperturePadd(uv_diameter, padding=0)
        self.FoldingSub = FoldingSubaperturePadd(uv_diameter, padding=0)
        
        if type == 'deconv':
            self.upscale = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=kernel_size, padding=0, bias=False)
        elif type == 'upsample':
            self.upscale = nn.Upsample(scale_factor=kernel_size, mode='bicubic')

        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()        

    def forward(self, x):
        x = self.UnfoldingSub(x)

        x = self.upscale(x)
        
        if self.fBN == True:
            x = self.bn(x)
        if self.fAct == True:
            x = self.relu(x)
        
        x = self.FoldingSub(x)

        return x
    
    
class LFAngAvgPooling(nn.Module):
    def __init__(self, uv_diameter):
        super().__init__()
        self.denominator = uv_diameter * uv_diameter

    def forward(self, x):
        return torch.sum(torch.sum(x, 3), 2) / self.denominator


# activation
class LFSoftSign(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / (0.01 + torch.abs(x))