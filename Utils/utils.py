import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import torch
import numpy as np
import h5py

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG','.bmp'])


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        file_list = os.listdir(dataset_dir)
        item_num = len(file_list)
        self.item_num = item_num

    def __getitem__(self, index):
        dataset_dir = self.dataset_dir
        index = index + 1
        file_name = [dataset_dir + '/Data_%06d' % index + '.h5']
        with h5py.File(file_name[0], 'r') as hf:
            data = np.array(hf.get('data'))
            data = np.transpose(data, (1, 2, 0))
            data = ToTensor()(data.copy())
            label = np.array(hf.get('label'))
            label = np.transpose(label, (1, 2, 0))
            label = ToTensor()(label.copy())
        return data, label

    def __len__(self):
        return self.item_num


class TestSetLoader(Dataset):
    def __init__(self, dataset_dir):
        super(TestSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        file_list = os.listdir(dataset_dir)
        item_num = len(file_list)
        self.item_num = item_num

    def __getitem__(self, index):
        dataset_dir = self.dataset_dir
        index = index + 1
        file_name = [dataset_dir + '/Data_%04d' % index + '.h5']
        with h5py.File(file_name[0], 'r') as hf:
            data = np.array(hf.get('data'))
            data = np.transpose(data, (1, 2, 0))
            data = ToTensor()(data.copy())
        return data

    def __len__(self):
        return self.item_num


def rgb2y(img):
    img_r = torch.round(img[:, 0, :, :] * 255)
    img_g = torch.round(img[:, 1, :, :] * 255)
    img_b = torch.round(img[:, 2, :, :] * 255)
    image_y = torch.round(0.257 * torch.unsqueeze(img_r, 1) + 0.504 * torch.unsqueeze(img_g, 1) + 0.098 * torch.unsqueeze(img_b, 1) + 16) / 255
    return image_y


class Parser:
    def __init__(self, parser):
        self.__parser = parser
        self.__args = parser.parse_args()

        # set gpu ids
        str_ids = self.__args.gpu_ids.split(',')
        self.__args.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.__args.gpu_ids.append(id)

    def get_parser(self):
        return self.__parser

    def get_arguments(self):
        return self.__args

    def write_args(self):
        params_dict = vars(self.__args)

        log_dir = os.path.join(params_dict['log_dir'], params_dict['scope'])
        args_name = os.path.join(log_dir, 'args.txt')

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        with open(args_name, 'wt') as args_fid:
            args_fid.write('----' * 10 + '\n')
            args_fid.write('{0:^40}'.format('PARAMETER TABLES') + '\n')
            args_fid.write('----' * 10 + '\n')
            for k, v in sorted(params_dict.items()):
                args_fid.write('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)) + '\n')
            args_fid.write('----' * 10 + '\n')

    def print_args(self, name='PARAMETER TABLES'):
        params_dict = vars(self.__args)

        print('----' * 10)
        print('{0:^40}'.format(name))
        print('----' * 10)
        for k, v in sorted(params_dict.items()):
            if '__' not in str(k):
                print('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)))
        print('----' * 10)