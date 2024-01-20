import torch
import torch.nn as nn
import pdb

class SSLoss(nn.Module):
    def __init__(self):
        super(SSLoss, self).__init__()
        self.ssl_alpha = 0.5
        self.BCE = nn.BCEWithLogitsLoss()

    def forward(self, target, pred):
        ##############################################################################################
        # SSL
        mtarget_true =  target.float()
        mtarget_false = (1-target.float())
        mpred_true = pred[:, 1, :, :].unsqueeze(1)
        mpred_false = pred[:, 0, :, :].unsqueeze(1)

        true_positive = torch.sum(mtarget_true * mpred_true)
        true_negative = torch.sum(mtarget_false * mpred_false)

        sensitivity = (1.0 - true_positive / (torch.sum(mtarget_true) + 0.1))
        sensitivity = torch.clamp(sensitivity, 0.0, 1.0)
        specificity = (1.0 - true_negative / (torch.sum(mtarget_false) + 0.1))
        specificity = torch.clamp(specificity, 0.0, 1.0)

        masked_ssl = self.ssl_alpha * sensitivity + (1 - self.ssl_alpha) * specificity

        ##############################################################################################
        # L1 
        mae = torch.abs(pred[:, 1, :, :].unsqueeze(1) - target)
            
        numerator = torch.mean(mae)
        masked_mae = numerator       
        
        ##############################################################################################
        # image gradient Loss - for occlusion boundary
        gt_gy = torch.gradient(target.float(), dim=2)
        gt_gx = torch.gradient(target.float(), dim=3)
        gt_lap = (torch.abs(gt_gy[0]) + torch.abs(gt_gx[0]))

        pr_gy = torch.gradient(pred[:, 1, :, :].unsqueeze(1), dim=2)
        pr_gx = torch.gradient(pred[:, 1, :, :].unsqueeze(1), dim=3)
        pr_lap = (torch.abs(pr_gy[0]) + torch.abs(pr_gx[0]))

        # BCELoss로 변경(F.BCELoss)
        # eps = 1e-10
        # bce = -1.0 * gt_lap * torch.log(pr_lap + eps) - (1.0 - gt_lap) * torch.log(1.0 - pr_lap + eps)
        bce = self.BCE(pr_lap, gt_lap)
                
        numerator = torch.mean(bce)
        masked_bce = numerator
        ##############################################################################################
        loss_out = masked_mae + masked_ssl + masked_bce

        return loss_out