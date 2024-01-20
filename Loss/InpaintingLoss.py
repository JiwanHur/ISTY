import torch
from torch import nn
from torch import autograd
from tensorboardX import SummaryWriter
# from models.discriminator import DiscriminatorDoubleColumn
import pdb
import numpy as np
from Loss.SSL import SSLoss

def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram

#tv loss
def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss

# modified from WGAN-GP
def calc_gradient_penalty(netD, real_data, fake_data, device, Lambda, masks=None):
    BATCH_SIZE = real_data.size()[0]
    DIM_y = real_data.size()[2]
    DIM_x = real_data.size()[3]
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous()
    alpha = alpha.view(BATCH_SIZE, 3, DIM_y, DIM_x)
    alpha = alpha.cuda()
    
    fake_data = fake_data.view(BATCH_SIZE, 3, DIM_y, DIM_x)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    # disc_interpolates = netD(interpolates, masks)
    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * Lambda
    return gradient_penalty.sum().mean()

##discriminator
# two column discriminator
class DiscriminatorDoubleColumn(nn.Module):
    def __init__(self, inputChannels):
        super(DiscriminatorDoubleColumn, self).__init__()

        self.globalConv = nn.Sequential(
            nn.Conv2d(inputChannels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fusionLayer = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=(3,4)),
            nn.Sigmoid()
        )


    def forward(self, batches, masks=None):
        globalFt = self.globalConv(batches)
        output = self.fusionLayer(globalFt)
        return output.view(batches.size()[0], -1)

class InpaintingLossWithGAN(nn.Module):
    def __init__(self, logPath, extractor, Lamda, lr, betasInit=(0.5, 0.9), name_loss='000', device='cpu'):
        super(InpaintingLossWithGAN, self).__init__()
        self.l1 = nn.L1Loss()
        self.loss_mask = nn.MSELoss()

        if (name_loss[1]=='1') or (name_loss[2]=='1'):  
            self.extractor = extractor.to(device)
        if name_loss[0]=='1':
            self.discriminator = DiscriminatorDoubleColumn(3).to(device)
            self.discriminator.load_state_dict(torch.load('./LBAMModels/LBAM_D_500.pth', map_location=device))
            self.D_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betasInit)
        self.cudaAvailable = torch.cuda.is_available()
        self.numOfGPUs = torch.cuda.device_count()
        self.lamda = Lamda

        self.writer = SummaryWriter(logPath)
        self.name_loss = name_loss

    # def forward(self, gt, output, count, log_iter, input=None, mask=None):
    def forward(self, output, gt, count, log_iter, device, input=None, masks=None, D_loss=0):

        D_loss=0
        D_fake=0
        if self.name_loss[0]=='1': # GAN loss
            self.discriminator.zero_grad()
            D_real = self.discriminator(gt, masks[0])
            # D_real = self.discriminator(gt)
            D_real = D_real.mean().sum() * -1
            # D_fake = self.discriminator(output, mask)
            D_fake = self.discriminator(output.detach())
            D_fake = D_fake.mean().sum() * 1
            # gp = calc_gradient_penalty(self.discriminator, gt, output, mask, self.cudaAvailable, self.lamda)
            gp = calc_gradient_penalty(self.discriminator, gt, output, self.cudaAvailable, self.lamda)
            D_loss = D_fake + D_real + gp
            self.D_optimizer.zero_grad()
            D_loss.backward(retain_graph=True)
            self.D_optimizer.step()

            D_fake = -1 * self.discriminator(output).mean().sum() # for generator
        # output_comp = mask * input + (1 - mask) * output
        output_comp = output

        if masks is not None:
            gt_mask = torch.cat((masks[0][:,0,:,:].unsqueeze(1), 1-masks[0][:,0,:,:].unsqueeze(1)),dim=1) # mask, 1-mask
            mask_loss = self.loss_mask(gt_mask, masks[1])
            if count % log_iter == 0:
                self.writer.add_scalar('LossMask', mask_loss.item(), count)
            # calc L1 loss with gt mask
            validAreaLoss = self.l1((1 - masks[0]) * output, (1 - masks[0]) * gt)
            holeLoss = 6 * self.l1(masks[0] * output, masks[0] * gt)
            L1_loss = holeLoss + validAreaLoss
        else:
            L1_loss = self.l1(output, gt)

        if (self.name_loss[1]=='1') or (self.name_loss[2]=='1'):
            if output.shape[1] == 3:
                feat_output_comp = self.extractor(output_comp)
                feat_output = self.extractor(output)
                feat_gt = self.extractor(gt)
            elif output.shape[1] == 1:
                feat_output_comp = self.extractor(torch.cat([output_comp]*3, 1))
                feat_output = self.extractor(torch.cat([output]*3, 1))
                feat_gt = self.extractor(torch.cat([gt]*3, 1))
            else:
                raise ValueError('only gray an')

        prcLoss = 0.0
        if self.name_loss[1]=='1': # perceptual loss
            for i in range(3):
                prcLoss += 0.01 * self.l1(feat_output[i], feat_gt[i])
                prcLoss += 0.01 * self.l1(feat_output_comp[i], feat_gt[i])

        styleLoss = 0.0
        if self.name_loss[2]=='1': # style loss
            for i in range(3):
                styleLoss += 120 * self.l1(gram_matrix(feat_output[i]),
                                            gram_matrix(feat_gt[i]))
                styleLoss += 120 * self.l1(gram_matrix(feat_output_comp[i]),
                                            gram_matrix(feat_gt[i]))
        
        """ if self.numOfGPUs > 1:
            holeLoss = holeLoss.sum() / self.numOfGPUs
            validAreaLoss = validAreaLoss.sum() / self.numOfGPUs
            prcLoss = prcLoss.sum() / self.numOfGPUs
            styleLoss = styleLoss.sum() / self.numOfGPUs """

        # GLoss = holeLoss + validAreaLoss + prcLoss + styleLoss + 0.1 * D_fake
        GLoss = L1_loss + prcLoss + styleLoss + mask_loss #+  0.1 * D_fake
        if count % log_iter == 0:
            self.writer.add_scalar('LossG/L1 loss', L1_loss.item(), count)
            if self.name_loss[0]=='1':
                self.writer.add_scalar('LossD/Discrinimator loss', D_loss, count)
            if self.name_loss[1]=='1':
                self.writer.add_scalar('LossPrc/Perceptual loss', prcLoss.item(), count)    
            if self.name_loss[2]=='1':
                self.writer.add_scalar('LossStyle/style loss', styleLoss.item(), count)    
            self.writer.add_scalar('Generator/Joint loss', GLoss.item(), count)
        return GLoss.sum()