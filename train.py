from Model.LFlayer import FoldingLensletPadd
from Dataloader.dataloader_selector import *
from Dataloader.MultiEpochDataLoader import MultiEpochDataLoader
from Model.model_selector import *
from Model.generate_occ_lf import GenOCCLF
from Model.separateFB import SeparateFB
from Model.convert_lf import ConvertLF
from Metric.metric_selector import *
from Loss.loss_selector import *
from Utils.utils import *

import time
import csv
import os
import torch
import numpy as np
import random
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torchvision

from torchsummary import summary
from tqdm import tqdm
import torch.nn.functional as F
import torch.cuda.amp as amp
import matplotlib.pyplot as plt

import pdb

class Train:
    def __init__(self, args):
        self.args = args

        if self.args.gpu_ids and torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % self.args.gpu_ids[0])
            self.args.device = self.device
            torch.cuda.set_device(self.args.gpu_ids[0])
        else:
            self.device = torch.device("cpu")

        torch.manual_seed(self.args.random_seed)
        np.random.seed(self.args.random_seed)
        random.seed(self.args.random_seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def save_imgs(self, results_dir, epoch, fname, output, gt, occ=None, fbs=None, masks=None):
        blen = len(fname)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        for idx in range(0, blen):
            save_name = '%s/output_%04d_%s' % (results_dir, epoch, fname[idx])
            output_t = np.clip(output[idx].squeeze().detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
            plt.imsave(save_name, output_t)

            save_name = '%s/gt_%04d_%s' % (results_dir, epoch, fname[idx])
            gt_t = np.clip(gt[idx].squeeze().detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
            plt.imsave(save_name, gt_t)

            if occ is not None:
                save_name = '%s/occ_%04d_%s' % (results_dir, epoch, fname[idx])
                occ_t = np.clip(occ[idx].squeeze().detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
                plt.imsave(save_name, occ_t)
            if fbs is not None:
                save_name = '%s/fbs_%04d_%s' % (results_dir, epoch, fname[idx])
                fbs_t = np.clip(fbs[idx].squeeze().detach().cpu().numpy(), 0, 1)
                plt.imsave(save_name, fbs_t, cmap='gray')

                save_name = '%s/fbs_bin_%04d_%s' % (results_dir, epoch, fname[idx])
                fbs_bin = fbs_t > 0
                fbs_bin = fbs_bin * np.ones_like(fbs_bin, dtype=np.float32)
                plt.imsave(save_name, fbs_bin, cmap='gray')
            if masks is not None:
                save_name = '%s/mask_gt_%04d_%s' % (results_dir, epoch, fname[idx])
                mask_gt_t = np.clip(masks[0][idx].squeeze().detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
                plt.imsave(save_name, mask_gt_t, cmap='gray')

                save_name = '%s/mask_%04d_%s' % (results_dir, epoch, fname[idx])
                mask_t = masks[1][idx,1].unsqueeze(0)
                mask_t = torch.cat((mask_t,mask_t,mask_t), dim=0)
                mask_t = np.clip(mask_t.squeeze().detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
                plt.imsave(save_name, mask_t, cmap='gray')

    def save_model(self, checkpoint_dir, scope, net, optim, epoch, iter):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        torch.save({'net': net.module.state_dict(),
                    'optim': optim.state_dict(),
                    'iter': iter},
                    '%s/%s_epoch%04d.pth' % (checkpoint_dir, scope, epoch))

    def load(self, checkpoint_dir, scope, net, optim=None, epoch=None, iter=0):
        if not os.path.exists(checkpoint_dir):
            epoch = 0
            if optim is None:
                return net, epoch, iter
            else:
                return net, optim, epoch, iter

        if not epoch:
            ckpt = os.listdir(checkpoint_dir)
            if len(ckpt) == 0: #if folder exists but no ckpt file
                epoch = 0
                return net, optim, epoch, iter
            ckpt = [f for f in ckpt if f.startswith(scope)]
            ckpt.sort()

            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/%s_epoch%04d.pth' % (checkpoint_dir, scope, epoch), map_location=self.device)

        print('Loaded %dth network' % epoch)

        if optim is None:
            net.load_state_dict(dict_net['net'])
            return net, epoch
        else:
            net.load_state_dict(dict_net['net'])
            optim.load_state_dict(dict_net['optim'])
            return net, optim, epoch, iter

    def setup_dataset(self, eval_epoch=0):
        #### setup dataset ####
        self.dir_chck = os.path.join(self.args.checkpoint_dir, self.args.scope, self.args.name_train_data)

        self.dir_result = os.path.join(self.args.results_dir, self.args.scope, self.args.name_data, self.args.mode)
        if self.args.mode == 'valid' or self.args.mode == 'test':
            self.dir_result = os.path.join(self.dir_result, '%04d' %eval_epoch)
        
        if not os.path.exists(self.dir_result):
            os.makedirs(self.dir_result)

        self.dir_data = os.path.join(self.args.data_directory, self.args.name_data)
        self.args.dir_data = self.dir_data

        self.dir_log = os.path.join(self.args.log_dir, self.args.scope, self.args.name_data, self.args.mode)
        self.args.dir_log = self.dir_log
        if not os.path.exists(self.dir_log):
            os.makedirs(self.dir_log)
        
        dataset = DataLoaderSelector(self.args)
        self.loader = MultiEpochDataLoader(dataset,
                                            batch_size=self.args.batch_size, 
                                            shuffle=(self.args.mode=='train'),
                                            num_workers=self.args.num_workers,
                                            drop_last=True)
        print(self.dir_data)

    def setup_network(self):
        self.occ = GenOCCLF(self.args.x_res, self.args.y_res, self.args.uv_diameter, self.args.resize_scale, self.args.alpha_size)
        self.fbs = SeparateFB(self.args.x_res, self.args.y_res, self.args.uv_diameter, self.args.resize_scale)
        self.clf = ConvertLF(self.args.uv_diameter, self.args.model)
        
        #add argument gd for the function can notice what model to treat.
        self.net = ModelSelector(self.args)
        init_weights(self.net, init_type='xavier', init_gain=0.02)

        if 'LBAM' in self.args.model: # if LBAM in model, initiate pre-trained LBAM
            self.net.load_LBAM()
            print('LBAM pretrained model is loaded!')

        self.net.to(self.device)
        self.params = self.net.parameters()
        params = self.net.parameters()
        self.optim = torch.optim.Adam(params, lr=self.args.learning_rate, betas=(0.5, 0.9))
        if self.args.scheduler is not 'None':
            if self.args.scheduler == 'step':
                from torch.optim.lr_scheduler import StepLR
                self.scheduler = StepLR(self.optim, step_size=self.args.scheduler_step, gamma=self.args.scheduler_gamma)
                print('scheduler is setted as', self.args.scheduler)
        pp = get_n_params(self.net)
        print(pp)

    def set_requires_grad(self, net, requires_grad):
        for param in net.parameters():
            param.requires_grad = requires_grad

    def train(self):
        self.setup_dataset()
        self.setup_network()

        ###### load from checkpoints ######
        st_epoch = 0
        cur_iter = 0
        if self.args.train_continue == 'on':
            ckpt_dir = os.path.join(self.args.checkpoint_dir, self.args.scope, self.args.name_data)
            self.net, self.optim, st_epoch, cur_iter = self.load(ckpt_dir, 'model', self.net, self.optim)
        if self.args.gpu_ids:
            self.net = torch.nn.DataParallel(self.net, self.args.gpu_ids)  # multi-GPUs

        ####### setup loss & optimization #######
        fn_loss = LossSelector(self.args)
        fn_loss = fn_loss.to(self.device)
        
        ######### train phase #########
        iter=cur_iter
        self.net.train()
        evaluation_meter = MetricSelector(self.args)
        for epoch in range(st_epoch + 1, self.args.num_epoch + 1):
            evaluation_meter.reset()
            dir_result_epoch = os.path.join(self.dir_result, '%04d' % epoch)
            pbar = tqdm(enumerate(self.loader, 1))

            for batch_idx, data in pbar:
                iter+=1
                src_img = data['src_img'].to(self.device)
                occ_img1 = data['occ_img1'].to(self.device)
                occ_msk1 = data['occ_msk1'].to(self.device)
                occ_img2 = data['occ_img2'].to(self.device)
                occ_msk2 = data['occ_msk2'].to(self.device)
                occ_img3 = data['occ_img3'].to(self.device)
                occ_msk3 = data['occ_msk3'].to(self.device)
                fname = data['file_name']

                with torch.no_grad():
                    input_occ, center_view, mask_gt = self.occ(src_img, [occ_img1, occ_img2, occ_img3], [occ_msk1, occ_msk2, occ_msk3])
                    res_fbs = self.fbs(input_occ).to(self.device)
                    input_occ, center_view, occ_t, res_fbs, mask_gt, input_lenslet = self.clf(input_occ, center_view, res_fbs, mask_gt)
                    del src_img, occ_img1, occ_img2, occ_img3, occ_msk1, occ_msk2, occ_msk3

                # forward
                output, mask = self.net(input_occ, input_lenslet, res_fbs)
                masks = [mask_gt, mask]

                # backward
                loss_value = fn_loss(output, center_view, iter, self.args.log_iter, self.device, masks=masks, D_loss = 0)
                
                self.optim.zero_grad()
                loss_value.backward()
                self.optim.step()


                if iter % self.args.log_iter == 0 and not self.args.debug:
                    evaluation_meter.update(batch_idx, output, center_view)
                
                pbar.set_description("%d epoch - %s" % (epoch, evaluation_meter.print_metrics()))
            
            # update scheduler
            if self.args.scheduler == 'step': 
                self.scheduler.step()
                if epoch % 200 == 0:
                    print('current learning rate is', self.scheduler.get_last_lr())

            # save results
            if not self.args.debug:
                if self.args.save_data == 1:
                    self.save_imgs(dir_result_epoch, epoch, fname, output, center_view, occ_t, res_fbs, masks)
                if not os.path.exists(dir_result_epoch):
                    os.makedirs(dir_result_epoch)
                if epoch % 100 == 0: # save model every 100 epoch
                    self.save_model(self.dir_chck, 'model', self.net, self.optim, epoch, iter)
                f = open(os.path.join(dir_result_epoch, 'metric.txt'), 'w')
                data_metric = evaluation_meter.get_current_status()
                print(data_metric, file=f)
                f.close()

    def valid(self):

        epoch = self.args.eval_epoch
        self.setup_dataset(epoch)
        self.setup_network()

        ###### load from checkpoints ######
        ckpt_dir = os.path.join(self.args.checkpoint_dir, self.args.scope, self.args.name_train_data)
        self.net, _, _, _ = self.load(ckpt_dir, 'model', self.net, self.optim, self.args.eval_epoch)

        ######### valid phase #########
        self.net.eval()
        
        evaluation_meter = MetricSelector(self.args)
        evaluation_meter.reset()
        pbar = tqdm(enumerate(self.loader, 1))
        
        cnt=0
        times = []
        for batch_idx, data in pbar:
            src_img = data['src_img'].to(self.device)
            if self.args.mode == 'test':
                occ_img = None
                occ_msk = None
                if 'gt_img' in data:
                    gt_img = data['gt_img'].to(self.device)
                else:
                    gt_img = None
            else: # for validation or dense_occ test
                occ_img1 = data['occ_img1'].to(self.device)
                occ_msk1 = data['occ_msk1'].to(self.device)
                occ_img2 = data['occ_img2'].to(self.device)
                occ_msk2 = data['occ_msk2'].to(self.device)
                occ_img3 = data['occ_img3'].to(self.device)
                occ_msk3 = data['occ_msk3'].to(self.device)
                gt_img = None
            fname = data['file_name']
            with torch.no_grad():
                if self.args.mode == 'test':
                    input_occ, center_view, mask_gt = self.occ(src_img)
                else: # for validation or dense_occ test
                    input_occ, center_view, mask_gt = self.occ(src_img, [occ_img1, occ_img2, occ_img3], [occ_msk1, occ_msk2, occ_msk3])
                if gt_img is not None:
                    center_view = gt_img.transpose(1,3).transpose(2,3) 

                res_fbs = self.fbs(input_occ).to(self.device)
                input_occ, center_view, occ_t, res_fbs, mask_gt, input_lenslet = self.clf(input_occ, center_view, res_fbs, mask_gt, train=False)
            
                t1 = time.time()

                # out, mask = self.net(input_occ, input_lenslet, res_fbs, fname[0])
                out, mask = self.net(input_occ, input_lenslet, res_fbs)
                masks = [mask_gt, mask]

                t2 = time.time()
                times.append(t2-t1)

                evaluation_meter.update(batch_idx, out, center_view)
                cur_metric = evaluation_meter.update(batch_idx, out, center_view)

                cnt+=1

                if not self.args.debug:
                    if self.args.mode == 'test': # test set has no occ imgs
                        dir_final = os.path.join(self.dir_result, self.args.specific_dir)
                        if self.args.save_data == 1:
                            self.save_imgs(dir_final, epoch, fname, out, center_view, fbs=res_fbs, masks=masks, occ=input_occ[:,12*3:13*3])
                        if not os.path.exists(dir_final):
                            os.makedirs(dir_final)
                        f = open(os.path.join(dir_final, 'metric_%s.txt' %fname[0]), 'w')
                        data_metric = cur_metric
                        print(data_metric, file=f)
                        f.close()
                    else: 
                        dir_final = self.dir_result
                        self.save_imgs(dir_final, epoch, fname, out, center_view, occ_t, res_fbs)

                pbar.set_description("%d epoch - %s" % (epoch, evaluation_meter.print_metrics()))

            if not self.args.debug:
                dir_final = os.path.join(self.dir_result, self.args.specific_dir)
                if not os.path.exists(dir_final):
                    os.makedirs(dir_final)
                f = open(os.path.join(dir_final, 'metric.txt'), 'w')
                data_metric = evaluation_meter.get_current_status()
                print(data_metric, file=f)
                f.close()

            if self.args.valid_multiple:
                output_dict={}
                output_dict['psnr']=data_metric['psnr']
                output_dict['ssim']=data_metric['ssim']
                self.args.output_dict = output_dict
        print(sum(times)/len(times))