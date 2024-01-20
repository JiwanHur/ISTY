import torch
import torch.nn.functional as F
from torch.autograd import Variable
from Utils.utils import rgb2y
from math import exp, log10
import time

import pdb


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    # window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

class MetricPSNRSSIM():
    def __init__(self, scope, batch_size):
        self.scope = scope
        self.batch_size = batch_size

        self.window_size = 11
        self.channel = 1
        self.window = create_window(self.window_size, self.channel)

        self.reset()

    def reset(self):
        self.batch_idx = 0
        self.psnr = 0
        self.ssim = 0
        self.len = 0

    def update(self, batch_idx, input, output):
        self.batch_idx = batch_idx
        input_y = rgb2y(input)
        output_y = rgb2y(output)
        mse = torch.sum((input_y - output_y) ** 2) / input_y.numel()

        psnr = 10 * log10(1 / mse.detach())
        self.psnr += psnr
        if input_y.is_cuda:
            self.window = self.window.cuda(input_y.get_device())
        self.window = self.window.type_as(input_y)
        mu1 = F.conv2d(input_y, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(output_y, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(input_y * input_y, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(output_y * output_y, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(input_y * output_y, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim = ssim_map.detach().mean()
        self.ssim += ssim
        self.len += 1

        return {'psnr': psnr, 'ssim': ssim}

    def get_current_status(self):
        if self.len==0:
            data_metric = {'psnr': 0,
                            'ssim': 0}
        else:
            data_metric = {'psnr': self.psnr/self.len,
                            'ssim': self.ssim/self.len}

        return data_metric
    
    def print_metrics(self):
        msg = '%s - ' % (self.scope)
        data_metric = self.get_current_status()
        for key, value in data_metric.items():
            msg += ('%s: %f, ' % (key, value))

        return msg