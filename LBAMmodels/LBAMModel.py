import torch
import torch.nn as nn
from torchvision import models
from LBAMmodels.forwardAttentionLayer import ForwardAttention
from LBAMmodels.reverseAttentionLayer import ReverseAttention, ReverseMaskConv
from LBAMmodels.weightInitial import weights_init
import pdb

#VGG16 feature extract
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        # vgg16.load_state_dict(torch.load('./LBAMmodels/vgg16-397923af.pth'))
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

class LBAMModel(nn.Module):
    def __init__(self, inputChannels, outputChannels):
        super(LBAMModel, self).__init__()

        # default kernel is of size 4X4, stride 2, padding 1, 
        # and the use of biases are set false in default ReverseAttention class.
        self.ec1 = ForwardAttention(inputChannels, 64, bn=False)
        self.ec2 = ForwardAttention(64, 128)
        self.ec3 = ForwardAttention(128, 256)
        self.ec4 = ForwardAttention(256, 512)

        for i in range(5, 7):
            name = 'ec{:d}'.format(i)
            setattr(self, name, ForwardAttention(512, 512))
        self.ec7 = ForwardAttention(512,512)
        
        # reverse mask conv
        self.reverseConv1 = ReverseMaskConv(3, 64)
        self.reverseConv2 = ReverseMaskConv(64, 128)
        self.reverseConv3 = ReverseMaskConv(128, 256)
        self.reverseConv4 = ReverseMaskConv(256, 512)
        self.reverseConv5 = ReverseMaskConv(512, 512)
        self.reverseConv6 = ReverseMaskConv(512, 512)

        self.dc1 = ReverseAttention(512, 512, bnChannels=1024, kernelSize=(5,4))
        self.dc2 = ReverseAttention(512 * 2, 512, bnChannels=1024)
        self.dc3 = ReverseAttention(512 * 2, 512, bnChannels=1024)
        self.dc4 = ReverseAttention(512 * 2, 256, bnChannels=512)
        self.dc5 = ReverseAttention(256 * 2, 128, bnChannels=256)
        self.dc6 = ReverseAttention(128 * 2, 64, bnChannels=128)
        self.dc7 = nn.ConvTranspose2d(64 * 2, outputChannels, kernel_size=4, stride=2, padding=1, bias=False)

        self.tanh = nn.Tanh()

    def forward(self, inputImgs, masks):
        ef1, mu1, skipConnect1, forwardMap1 = self.ec1(inputImgs, masks)
        ef2, mu2, skipConnect2, forwardMap2 = self.ec2(ef1, mu1)
        ef3, mu3, skipConnect3, forwardMap3 = self.ec3(ef2, mu2)
        ef4, mu4, skipConnect4, forwardMap4 = self.ec4(ef3, mu3)
        ef5, mu5, skipConnect5, forwardMap5 = self.ec5(ef4, mu4)
        ef6, mu6, skipConnect6, forwardMap6 = self.ec6(ef5, mu5)
        ef7, _, _, _ = self.ec7(ef6, mu6)


        reverseMap1, revMu1 = self.reverseConv1(1 - masks)
        reverseMap2, revMu2 = self.reverseConv2(revMu1)
        reverseMap3, revMu3 = self.reverseConv3(revMu2)
        reverseMap4, revMu4 = self.reverseConv4(revMu3)
        reverseMap5, revMu5 = self.reverseConv5(revMu4)
        reverseMap6, _ = self.reverseConv6(revMu5)

        concatMap6 = torch.cat((forwardMap6, reverseMap6), 1)
        dcFeatures1 = self.dc1(skipConnect6, ef7, concatMap6)

        concatMap5 = torch.cat((forwardMap5, reverseMap5), 1)
        dcFeatures2 = self.dc2(skipConnect5, dcFeatures1, concatMap5)

        concatMap4 = torch.cat((forwardMap4, reverseMap4), 1)
        dcFeatures3 = self.dc3(skipConnect4, dcFeatures2, concatMap4)

        concatMap3 = torch.cat((forwardMap3, reverseMap3), 1)
        dcFeatures4 = self.dc4(skipConnect3, dcFeatures3, concatMap3)

        concatMap2 = torch.cat((forwardMap2, reverseMap2), 1)
        dcFeatures5 = self.dc5(skipConnect2, dcFeatures4, concatMap2)

        concatMap1 = torch.cat((forwardMap1, reverseMap1), 1)
        dcFeatures6 = self.dc6(skipConnect1, dcFeatures5, concatMap1)

        dcFeatures7 = self.dc7(dcFeatures6)

        output = (self.tanh(dcFeatures7) + 1) / 2

        return output


class Mask_Generator(nn.Module):
    def __init__(self):
        super(Mask_Generator, self).__init__()
        self.eb1 = EB(4, 64, 3, 2, 1)
        self.eb2 = EB(64, 128, 3, 2, 1)
        self.eb3 = EB(128, 256, 3, 2, 1)

        self.db3 = DB(256* 3, 128, 3, 2, 1, 1)
        self.db2 = DB(128 * 4, 64, 3, 2, 1, 1)
        self.db1 = DB(64 * 4, 64, 3, 2, 1, 1)
        self.db0 = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, cv_fbs, *args):
        ef1 = self.eb1(cv_fbs) # 128x128
        ef2 = self.eb2(ef1) # 64x64
        ef3 = self.eb3(ef2) # 32x32

        df3 = self.db3(torch.cat((ef3, args[5], args[4]), dim=1)) # 64x64
        df2 = self.db2(torch.cat((df3, ef2, args[3], args[2]), dim=1)) # 128x128
        df1 = self.db1(torch.cat((df2, ef1, args[1], args[0]), dim=1))
        mask = self.db0(df1)
        return torch.cat((mask, mask, mask), dim=1)

class Mask_Generator_features_all(nn.Module):
    def __init__(self):
        super(Mask_Generator, self).__init__()
        self.eb1 = EB(4, 64, 3, 2, 1)
        self.eb2 = EB(64, 128, 3, 2, 1)
        self.eb3 = EB(128, 256, 3, 2, 1)

        self.db3 = DB(256* 3, 128, 3, 2, 1, 1)
        self.db2 = DB(128 * 4, 64, 3, 2, 1, 1)
        self.db1 = DB(64 * 4, 64, 3, 2, 1, 1)
        self.db0 = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, cv, fbs, *args):
        ef1 = self.eb1(torch.cat((cv, fbs), dim=1)) # 128x128
        ef2 = self.eb2(ef1) # 64x64
        ef3 = self.eb3(ef2) # 32x32

        df3 = self.db3(torch.cat((ef3, args[5], args[4]), dim=1)) # 64x64
        df2 = self.db2(torch.cat((df3, ef2, args[3], args[2]), dim=1)) # 128x128
        df1 = self.db1(torch.cat((df2, ef1, args[1], args[0]), dim=1))
        return self.db0(df1)

# class Mask_Generator(nn.Module):
#     def __init__(self, views):
#         super(Mask_Generator, self).__init__()
#         self.views = views
#         self.eb1 = EB(int(self.views*3)+1, 64, 3, 2, 1)
#         self.eb2 = EB(64, 128, 3, 2, 1)
#         self.eb3 = EB(128, 256, 3, 2, 1)

#         self.db3 = DB(256, 128, 3, 2, 1, 1)
#         self.db2 = DB(256, 64, 3, 2, 1, 1)
#         self.db1 = nn.Sequential(
#             nn.ConvTranspose2d(128, 3, 3, 2, 1, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, SAIs, fbs):
#         ef1 = self.eb1(torch.cat((SAIs, fbs), dim=1)) # b x 64 x 96 x 128
#         ef2 = self.eb2(ef1) # b x 128 x 48 x 64
#         ef3 = self.eb3(ef2) # b x 256 x 24 x 32

#         df3 = self.db3(ef3) # 64x64
#         df2 = self.db2(torch.cat((df3, ef2), dim=1)) # 128x128
#         df1 = self.db1(torch.cat((df2, ef1), dim=1))
#         return df1


class EB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(EB, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, feat):
        return self.seq(feat)

class DB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super(DB, self).__init__()
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, feat):
        return self.seq(feat)

class Mask_Generator_Unet(nn.Module):
    def __init__(self):
        super(Mask_Generator_Unet, self).__init__()
        self.eb1 = Down(4, 64)
        self.eb2 = Down(64, 128)
        self.eb3 = Down(128, 256)

        self.db3 = Up(256* 3, 128)
        self.db2 = Up(128 * 4, 64)
        self.db1 = Up(64 * 4, 64)
        self.db0 = nn.Sequential(
            nn.Conv2d(64, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, cv_fbs, *args):
        ef1 = self.eb1(cv_fbs) # 128x128
        ef2 = self.eb2(ef1) # 64x64
        ef3 = self.eb3(ef2) # 32x32

        df3 = self.db3(torch.cat((ef3, args[5], args[4]), dim=1)) # 64x64
        df2 = self.db2(torch.cat((df3, ef2, args[3], args[2]), dim=1)) # 128x128
        df1 = self.db1(torch.cat((df2, ef1, args[1], args[0]), dim=1))
        mask = self.db0(df1)
        return mask

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.seq = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, feat):
        return self.seq(feat)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.seq = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(in_channels, out_channels, in_channels // 2)
            )
    def forward(self, feat):
        return self.seq(feat)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, stride=1, padding=1):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, feat):
        return self.seq(feat)