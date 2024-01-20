
import torch.nn as nn
import pdb
def LossSelector(args):
    if args.loss_mode == 'base':
        fn_loss = None
        if args.name_loss == 'BCE':
            fn_loss = nn.BCELoss()

        elif args.name_loss == 'DBCE':
            from loss.dice_bce_t import DiceBCELoss
            fn_loss = DiceBCELoss()

        elif args.name_loss == 'CE':
            fn_loss = nn.CrossEntropyLoss(reduction='none')

        elif args.name_loss == 'CEB':
            from loss.cross_entropy_balance import CrossEntropyBalanceLoss
            fn_loss = CrossEntropyBalanceLoss()

        elif args.name_loss == 'DCEB':
            from loss.dice_cross_entropy_balance import DiceCrossEntropyBalanceLoss
            fn_loss = DiceCrossEntropyBalanceLoss()

        elif args.name_loss == 'MAE':
            fn_loss = nn.L1Loss()

        elif args.name_loss == 'MSE':
            fn_loss = nn.MSELoss()

        else:
            print('no loss selected')
            return

    elif args.loss_mode == 'Inpainting':
        from Loss.InpaintingLoss import InpaintingLossWithGAN
        from LBAMmodels.LBAMModel import VGG16FeatureExtractor
        fn_loss = InpaintingLossWithGAN(args.dir_log, VGG16FeatureExtractor(), 
            Lamda=10.0, lr=0.0004, betasInit=(0.0, 0.9), name_loss=args.name_loss, device=args.device)

    else:
        print('no loss selected')
        return

    return fn_loss