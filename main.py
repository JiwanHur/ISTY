
import pdb

from Utils.utils import *
import argparse

from train import Train
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser(description='LFDeOccGAN',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
### general options ###
parser.add_argument("--scope",type=str, default="default")
parser.add_argument("--random_seed", type=int, default=1234,
                    help="Random seed to have reproducible results.")
parser.add_argument("--gpu_ids", type=str, default='0',
                    help="choose gpu device.")
parser.add_argument("--mode", type=str, default='trainvalid',
                    help="choose gpu device.")
parser.add_argument('--world_size', type=int, default=1, help='number of nodes for distributed training')
parser.add_argument('--rank', type=int, default=1, help='node rank for distributed training')
parser.add_argument('--multiprocessing_distributed', action='store_true', help='Use ddp')
parser.add_argument('--ngpus_per_node', type=int, default=1, help='number of gpus for training if use ddp')

### dataset options ###
parser.add_argument("--data_directory", type=str, default='/dataset/GTA5', 
                    help="Path to the directory containing the source dataset.")
parser.add_argument("--name_data", type=str, default="DUTLF-V2")
parser.add_argument("--save_data", type=int, default=0)
parser.add_argument("--name_train_data", type=str, default="LFGAN")
parser.add_argument("--x_res", type=int, default=300)
parser.add_argument("--y_res", type=int, default=200)
parser.add_argument("--uv_diameter_image", type=int, default=9)
parser.add_argument("--uv_diameter", type=int, default=9)
parser.add_argument("--uv_dilation", type=int, default=1)
parser.add_argument("--data_output_option", type=str, default='3d_sub', 
                    help="2d_sub, 2d_len, 3d_sub, 4d")
parser.add_argument("--resize_scale", type=float, default=0.5, 
                    help="2d_sub, 2d_len, 3d_sub, 4d")
parser.add_argument("--num_workers", type=int, default=0,
                    help="number of workers for multithread dataloading.")


### train options ###
parser.add_argument("--model", type=str, default='deoccnet', 
                    help="available options : DeepLab")
parser.add_argument("--views", type=int, default=81, 
                    help="available options : DeepLab")
parser.add_argument("--train_continue", type=str, default='on')
parser.add_argument("--batch_size", type=int, default=1,
                    help="Number of images sent to the network in one step.")
parser.add_argument("--num_epoch", type=int, default=100,
                    help="Number of images sent to the network in one step.")
parser.add_argument("--alpha_size", type=int, default=1,
                    help="Occlusion images disparity step size")

### GAN options ###
parser.add_argument('--train_gan', action='store_true', help='only used if netD==n_layers')
parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
parser.add_argument('--lr_G', type=float, default=0.0001, help='initial learning rate for adam') #for TTUR
parser.add_argument('--lr_D', type=float, default=0.0004, help='initial learning rate for adam') 
parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
parser.add_argument("--loss_gan", type=str, default='BASIC', help="mse.")
parser.add_argument("--gd", type=str, default='g', help="used for gan training")
parser.add_argument("--gan_mode", type=str, default='vanilla', help="used for gan training")
parser.add_argument('--anti_alias', action='store_true', help='Use anti-aliased DeOccNet')

## optimizer options ##
parser.add_argument("--learning_rate", type=float, default=2.5e-4,
                    help="Base learning rate for training with polynomial decay.")
parser.add_argument("--scheduler", type=str, default='None',
                    help="type of scheduler, default = None")
parser.add_argument("--scheduler_gamma", type=float, default=0.2)
parser.add_argument("--scheduler_step", type=int, default=200)
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')


## loss options ##
parser.add_argument("--loss_mode", type=str, default='base',
                    help="base.")
parser.add_argument("--name_loss", type=str, default='mse',
                    help="mse.")
parser.add_argument("--debug", action='store_true', help='activate if do not want to save(for debug)')



### log options ###
parser.add_argument("--tensorboard", action='store_true', 
                    help="choose whether to use tensorboard.")
parser.add_argument("--log_dir", type=str, default='./log',
                    help="Path to the directory of log.")
parser.add_argument("--checkpoint_dir", type=str, default='./checkpoint',
                    help="Where to save snapshots of the model.")
parser.add_argument("--results_dir", type=str, default='./results',
                    help="Where to save imgs.")
parser.add_argument("--name_metric", type=str, default='psnrssim',
                    help="Metrics.")
parser.add_argument("--log_iter", type=int, default=100, help="logging iter step")

### test options
parser.add_argument("--eval_epoch", type=int)
parser.add_argument("--st_epoch", type=int)
parser.add_argument("--max_epoch", type=int)
parser.add_argument("--valid_multiple", action='store_true')
parser.add_argument("--specific_dir", type=str, default='', 
                    help = 'specific data root for test. you can sepcify the directory after name data')

PARSER = Parser(parser)

def main():
    ARGS = PARSER.get_arguments()
    PARSER.write_args()
    PARSER.print_args()
            
    TRAINER = Train(ARGS)
    if ARGS.mode == 'train':
        TRAINER.train()
    else:
        TRAINER.valid()

if __name__ == "__main__":
	main()