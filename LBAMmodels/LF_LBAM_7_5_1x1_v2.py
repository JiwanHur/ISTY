from re import X
import torch
import torch.nn as nn
from torchvision import models
from LBAMmodels.ActivationFunction import GaussActivation, MaskUpdate
from LBAMmodels.weightInitial import weights_init
from LBAMmodels.LBAMModel import LBAMModel
import torch.nn.functional as F
from LBAMmodels.MaskGenerator import Mask_Generator_Unet
from LBAMmodels.MaskAttention import ForwardAttention, ReverseAttention

class LF_LBAM(nn.Module):
    def __init__(self, inputChannels, outputChannels, model_path, device):
        super(LF_LBAM, self).__init__()
        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.model_path = model_path
        self.device = device

        # define mask generator
        self.GenMask = Mask_Generator_Unet()

        # define spatial feature extract modules
        views = 25
        self.init_feature = nn.Sequential(
            nn.Conv2d(int(views * 3), 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            ResASPPB(64),
            ResBlock(64),
            ResASPPB(64),
            ResBlock(64),
        )
        self.sec1 = SEB(64, 64, kernel_size=4, stride=2, padding=1, bias=False) # sec: spatial encoder (LFE)
        self.sec2 = SEB_with_Attn(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.sec3 = SEB_with_Attn(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        for i in range(4,8):
            name = 'sec{:d}'.format(i)
            setattr(self, name, SEB_with_Attn(256, 256, kernel_size=4, stride=2, padding=1, bias=False))

        self.fuse_conv1 = Fuse_1x1(512 + 256, 512, 512) # input channel: (dec1 + sec1)
        self.fuse_conv2 = Fuse_1x1(512 + 256, 512, 512)
        self.fuse_conv3 = Fuse_1x1(512 + 256, 512, 512)
        self.fuse_conv4 = Fuse_1x1(512 + 256, 512, 512)
        self.fuse_conv5 = Fuse_1x1(256 + 256, 256, 256)
        self.fuse_conv6 = Fuse_1x1(128 + 128, 128, 128)

    def load_LBAM(self):
        # load pretrained model
        LBAM = LBAMModel(self.inputChannels, self.outputChannels)
        LBAM.load_state_dict(torch.load(self.model_path, map_location=self.device))

        self.ec1 = LBAM.ec1
        self.ec2 = LBAM.ec2
        self.ec3 = LBAM.ec3
        self.ec4 = LBAM.ec4
        self.ec5 = LBAM.ec5
        self.ec6 = LBAM.ec6
        self.ec7 = LBAM.ec7
        
        self.reverseConv1 = LBAM.reverseConv1
        self.reverseConv2 = LBAM.reverseConv2
        self.reverseConv3 = LBAM.reverseConv3
        self.reverseConv4 = LBAM.reverseConv4
        self.reverseConv5 = LBAM.reverseConv5
        self.reverseConv6 = LBAM.reverseConv6

        # replace decoder part
        self.dc1 = ReverseAttention(512, 512, bnChannels=1024, kernelSize=(5,4))
        self.dc2 = ReverseAttention(512*2, 512, bnChannels=1024)
        self.dc3 = ReverseAttention(512*2, 512, bnChannels=1024)
        self.dc4 = ReverseAttention(512*2, 256, bnChannels=512)
        self.dc5 = ReverseAttention(256*2, 128, bnChannels=256)
        self.dc6 = ReverseAttention(128*2, 64, bnChannels=128)
        self.dc7 = nn.ConvTranspose2d(64*2, self.outputChannels, kernel_size=4, stride=2, padding=1, bias=False)

        with torch.no_grad():
            self.dc1.conv.weight[:512] = LBAM.dc1.conv.weight
            self.dc2.conv.weight[:512 * 2] = LBAM.dc2.conv.weight
            self.dc3.conv.weight[:512 * 2] = LBAM.dc3.conv.weight
            self.dc4.conv.weight[:512 * 2] = LBAM.dc4.conv.weight
            self.dc5.conv.weight[:256 * 2] = LBAM.dc5.conv.weight
            self.dc6.conv.weight[:128 * 2] = LBAM.dc6.conv.weight
            self.dc7.weight[:64 * 2] = LBAM.dc7.weight
        self.tanh = nn.Tanh()
        del LBAM

    def forward(self, inputSAImgs, LensletImgs, fbs):
        inputSAImgs += 0.5
        LensletImgs += 0.5
        cv_fbs = torch.cat((inputSAImgs[:,3*12:3*13,:,:], 1-fbs), dim=1)

        # forward spatial feature encoder
        sf0 = self.init_feature(inputSAImgs) # b x 64 x 192 x 256
        sf1 = self.sec1(sf0) # b x 128 x 96 x 128
        sf2 = self.sec2(sf1) # b x 256 x 48 x 64
        sf3 = self.sec3(sf2) # b x 256 x 24 x 32
        sf4 = self.sec4(sf3) # b x 256 x 12 x 16
        sf5 = self.sec5(sf4) # b x 256 x 6 x 8 
        sf6 = self.sec6(sf5) # b x 256 x 3 x 4 
        sf7 = self.sec7(sf6) # b x 256 x 1 x 2

        # mask generation using inputSAI and fbs
        mask = self.GenMask(cv_fbs, sf1, sf2, sf3) # first channel: background, second channel: foreground
        masks = torch.cat((mask[:,1].unsqueeze(1), mask[:,1].unsqueeze(1), mask[:,1].unsqueeze(1)), dim=1).detach()

        # forward LBAM encoder
        input_cv = torch.cat((inputSAImgs[:,3*12:3*13,:,:], mask[:,1].unsqueeze(1)), dim=1)
        ef1, mu1, skipConnect1, forwardMap1 = self.ec1(input_cv, masks) # input the center view
        ef2, mu2, skipConnect2, forwardMap2 = self.ec2(ef1, mu1)
        ef3, mu3, skipConnect3, forwardMap3 = self.ec3(ef2, mu2)
        ef4, mu4, skipConnect4, forwardMap4 = self.ec4(ef3, mu3)
        ef5, mu5, skipConnect5, forwardMap5 = self.ec5(ef4, mu4)
        ef6, mu6, skipConnect6, forwardMap6 = self.ec6(ef5, mu5)
        ef7, _, _, forwardMap7 = self.ec7(ef6, mu6)

        reverseMap1, revMu1 = self.reverseConv1(1 - masks)
        reverseMap2, revMu2 = self.reverseConv2(revMu1)
        reverseMap3, revMu3 = self.reverseConv3(revMu2)
        reverseMap4, revMu4 = self.reverseConv4(revMu3)
        reverseMap5, revMu5 = self.reverseConv5(revMu4)
        reverseMap6, _ = self.reverseConv6(revMu5)

        # forward decoder part
        concatMap6 = torch.cat((forwardMap6, reverseMap6), dim=1)
        fuse7 = self.fuse_conv1(ef7, sf7, forwardMap7)
        dcFeatures_f1, dcFeatures_r1 = self.dc1(skipConnect6, fuse7, concatMap6)

        concatMap5 = torch.cat((forwardMap5, reverseMap5), dim=1)
        fuse6 = self.fuse_conv2(dcFeatures_r1, sf6, forwardMap6)
        dcFeatures1 = torch.cat((dcFeatures_f1, fuse6), dim=1)
        dcFeatures_f2, dcFeatures_r2 = self.dc2(skipConnect5, dcFeatures1, concatMap5)

        concatMap4 = torch.cat((forwardMap4, reverseMap4), dim=1)
        fuse5 = self.fuse_conv3(dcFeatures_r2, sf5, forwardMap5)
        dcFeatures2 = torch.cat((dcFeatures_f2, fuse5), dim=1)
        dcFeatures_f3, dcFeatures_r3 = self.dc3(skipConnect4, dcFeatures2, concatMap4)

        concatMap3 = torch.cat((forwardMap3, reverseMap3), dim=1)
        fuse4 = self.fuse_conv4(dcFeatures_r3, sf4, forwardMap4)
        dcFeatures3 = torch.cat((dcFeatures_f3, fuse4), dim=1)
        dcFeatures_f4, dcFeatures_r4 = self.dc4(skipConnect3, dcFeatures3, concatMap3)

        concatMap2 = torch.cat((forwardMap2, reverseMap2), dim=1)
        fuse3 = self.fuse_conv5(dcFeatures_r4, sf3, forwardMap3)
        dcFeatures4 = torch.cat((dcFeatures_f4, fuse3), dim=1)
        dcFeatures_f5, dcFeatures_r5 = self.dc5(skipConnect2, dcFeatures4, concatMap2)

        concatMap1 = torch.cat((forwardMap1, reverseMap1), dim=1)
        fuse2 = self.fuse_conv6(dcFeatures_r5, sf2, forwardMap2)
        dcFeatures5 = torch.cat((dcFeatures_f5, fuse2), dim=1)
        dcFeatures_f6, dcFeatures_r6 = self.dc6(skipConnect1, dcFeatures5, concatMap1)

        dcFeatures6 = torch.cat((dcFeatures_f6, dcFeatures_r6), dim=1)
        dcFeatures7 = self.dc7(dcFeatures6)

        output = (self.tanh(dcFeatures7) + 1) / 2
        return output, mask

class ResASPPB(nn.Module):
    def __init__(self, channels):
        super(ResASPPB, self).__init__()
        self.conv_1_0 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False),
                                      nn.LeakyReLU(0.1, inplace=True))
        self.conv_1_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 2, 2, bias=False),
                                      nn.LeakyReLU(0.1, inplace=True))
        self.conv_1_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False),
                                      nn.LeakyReLU(0.1, inplace=True))
        self.conv_1_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False),
                                      nn.LeakyReLU(0.1, inplace=True))

        self.conv_1 = nn.Conv2d(channels * 4, channels, 1, 1, 0, bias=False)

    def forward(self, x):
        buffer_1_0 = self.conv_1_0(x)
        buffer_1_1 = self.conv_1_1(x)
        buffer_1_2 = self.conv_1_2(x)
        buffer_1_3 = self.conv_1_3(x)
        buffer_1 = torch.cat((buffer_1_0, buffer_1_1, buffer_1_2, buffer_1_3), 1)
        buffer_1 = self.conv_1(buffer_1)

        return x + buffer_1

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False),
                                      nn.LeakyReLU(0.1, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False),
                                      nn.LeakyReLU(0.1, inplace=True),
                                      nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False))

    def forward(self, x):
        buffer = self.conv_1(x)
        buffer = self.conv_2(buffer)

        return x + buffer

class SEB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(SEB, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, feat):
        return self.seq(feat)

class SEB_with_Attn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(SEB_with_Attn, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.Q_conv = nn.Conv2d(out_channels, out_channels//8, kernel_size=1)
        self.K_conv = nn.Conv2d(out_channels, out_channels//8, kernel_size=1)
        self.V_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.ones(1)*0.25)

    def forward(self, feat):
        x = self.seq(feat)

        batch_size, channels, width, height = x.shape
        proj_Q = self.Q_conv(x).view(batch_size, -1, width*height).permute(0,2,1)
        proj_K = self.K_conv(x).view(batch_size, -1, width*height)
        attention = self.softmax(torch.bmm(proj_Q, proj_K))
        proj_V = self.V_conv(x).view(batch_size, -1, width*height)
        out = torch.bmm(proj_V, attention.permute(0,2,1))
        out = out.view(batch_size, channels, width, height)
        out = self.gamma*out + x

        return out

class Fuse_1x1(nn.Module):
    def __init__(self, in_channels, out_channels, attn_channels):
        super(Fuse_1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1)*0.25)
    def forward(self, ef, sf, Attn):
        x = torch.cat((ef, sf), dim=1)
        out = self.conv(x)
        # fusion method
        out = self.gamma * out + ef
        return out

class SA_Fuse(nn.Module):
    def __init__(self, in_channels, out_channels, attn_channels):
        super(SA_Fuse, self).__init__()
        self.Q_conv = nn.Conv2d(in_channels, out_channels//8, kernel_size=1)
        self.K_conv = nn.Conv2d(attn_channels, out_channels//8, kernel_size=1)
        self.V_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.ones(1)*0.25)
    def forward(self, ef, sf, Attn):
        # self attention
        batch_size, channels, width, height = ef.shape
        x = torch.cat((ef, sf), dim=1)
        proj_Q = self.Q_conv(x).view(batch_size, -1, width*height).permute(0,2,1)
        proj_K = self.K_conv(Attn).view(batch_size, -1, width*height)
        attention = self.softmax(torch.bmm(proj_Q, proj_K))
        proj_V = self.V_conv(x).view(batch_size, -1, width*height)
        out = torch.bmm(proj_V, attention.permute(0,2,1))
        out = out.view(batch_size, channels, width, height)
        # fusion method
        out = self.gamma * out + ef
        return out

if __name__=="__main__":
    model = LF_LBAM(4,3, './LBAMModels/LBAM_NoGAN_500.pth')