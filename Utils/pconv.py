import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torch
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable

class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False  

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
            
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in
                        
                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)


        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)


        if self.return_mask:
            return output, self.update_mask
        else:
            return output
if __name__=='__main__':
    pconv = PartialConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, multi_channel=True, return_mask=True)

# class PartialConv2d(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size,
#                  stride=1, padding=0, dilation=1, groups=1, bias=True,
#                  padding_mode='zeros'):
#         # Inherit the parent class (Conv2d)
#         super(PartialConv2d, self).__init__(in_channels, out_channels,
#                                             kernel_size, stride=stride,
#                                             padding=padding, dilation=dilation,
#                                             groups=groups, bias=bias,
#                                             padding_mode=padding_mode)
#         # Define the kernel for updating mask
#         self.mask_kernel = torch.ones(self.out_channels, self.in_channels,
#                                       self.kernel_size[0], self.kernel_size[1])
#         # Define sum1 for renormalization
#         self.sum1 = self.mask_kernel.shape[1] * self.mask_kernel.shape[2] \
#                                               * self.mask_kernel.shape[3]
#         # Define the updated mask
#         self.update_mask = None
#         # Define the mask ratio (sum(1) / sum(M))
#         self.mask_ratio = None
#         # Initialize the weights for image convolution
#         torch.nn.init.xavier_uniform_(self.weight)

#     def forward(self, img, mask):
#         with torch.no_grad():
#             if self.mask_kernel.type() != img.type():
#                 self.mask_kernel = self.mask_kernel.to(img)
#             # Create the updated mask
#             # for calcurating mask ratio (sum(1) / sum(M))
#             self.update_mask = F.conv2d(mask, self.mask_kernel,
#                                         bias=None, stride=self.stride,
#                                         padding=self.padding,
#                                         dilation=self.dilation,
#                                         groups=1)
#             # calcurate mask ratio (sum(1) / sum(M))
#             self.mask_ratio = self.sum1 / (self.update_mask + 1e-8)
#             self.update_mask = torch.clamp(self.update_mask, 0, 1)
#             self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

#         # calcurate WT . (X * M)
#         conved = torch.mul(img, mask)
#         conved = F.conv2d(conved, self.weight, self.bias, self.stride,
#                           self.padding, self.dilation, self.groups)

#         if self.bias is not None:
#             # Maltuply WT . (X * M) and sum(1) / sum(M) and Add the bias
#             bias_view = self.bias.view(1, self.out_channels, 1, 1)
#             output = torch.mul(conved - bias_view, self.mask_ratio) + bias_view
#             # The masked part pixel is updated to 0
#             output = torch.mul(output, self.mask_ratio)
#         else:
#             # Multiply WT . (X * M) and sum(1) / sum(M)
#             output = torch.mul(conved, self.mask_ratio)

#         return output, self.update_mask


# class UpsampleConcat(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Define the upsampling layer with nearest neighbor
#         self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

#     def forward(self, dec_feature, enc_feature, dec_mask, enc_mask):
#         # upsample and concat features
#         out = self.upsample(dec_feature)
#         out = torch.cat([out, enc_feature], dim=1)
#         # upsample and concat masks
#         out_mask = self.upsample(dec_mask)
#         out_mask = torch.cat([out_mask, enc_mask], dim=1)
#         return out, out_mask


# class PConvActiv(nn.Module):
#     def __init__(self, in_ch, out_ch, sample='none-3', dec=False,
#                  bn=True, active='relu', conv_bias=False):
#         super().__init__()
#         # Define the partial conv layer
#         if sample == 'down-7':
#             params = {"kernel_size": 7, "stride": 2, "padding": 3}
#         elif sample == 'down-5':
#             params = {"kernel_size": 5, "stride": 2, "padding": 2}
#         elif sample == 'down-3':
#             params = {"kernel_size": 3, "stride": 2, "padding": 1}
#         else:
#             params = {"kernel_size": 3, "stride": 1, "padding": 1}
#         self.conv = PartialConv2d(in_ch, out_ch,
#                                   params["kernel_size"],
#                                   params["stride"],
#                                   params["padding"],
#                                   bias=conv_bias)

#         # Define other layers
#         if dec:
#             self.upcat = UpsampleConcat()
#         if bn:
#             bn = nn.BatchNorm2d(out_ch)
#         if active == 'relu':
#             self.activation = nn.ReLU()
#         elif active == 'leaky':
#             self.activation = nn.LeakyReLU(negative_slope=0.2)

#     def forward(self, img, mask, enc_img=None, enc_mask=None):
#         if hasattr(self, 'upcat'):
#             out, update_mask = self.upcat(img, enc_img, mask, enc_mask)
#             out, update_mask = self.conv(out, update_mask)
#         else:
#             out, update_mask = self.conv(img, mask)
#         if hasattr(self, 'bn'):
#             out = self.bn(out)
#         if hasattr(self, 'activation'):
#             out = self.activation(out)
#         return out, update_mask