import torch
import torch.nn as nn

class Mask_Generator_Unet(nn.Module):
    def __init__(self):
        super(Mask_Generator_Unet, self).__init__()
        self.eb1 = Down(4, 64)
        self.eb2 = Down(64, 128)
        self.eb3 = Down(128, 256)

        self.db3 = Up(256*2 , 128)
        self.db2 = Up(128 * 3 , 64)
        self.db1 = Up(64 * 3 , 64)
        self.db0 = nn.Sequential(
            nn.Conv2d(64, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, cv_fbs, *args):
        ef1 = self.eb1(cv_fbs) # 4 x 96 x 128
        ef2 = self.eb2(ef1) # 64 x 48 x 64
        ef3 = self.eb3(ef2) # 128 x 24 x 32

        df3 = self.db3(torch.cat((ef3, args[2]), dim=1)) # 48 x 64
        df2 = self.db2(torch.cat((df3, ef2, args[1]), dim=1)) # 96 x 128
        df1 = self.db1(torch.cat((df2, ef1, args[0]), dim=1)) # 192 x 256
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