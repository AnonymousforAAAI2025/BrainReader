import h5py
import scipy.io as spio
import scipy.io
import pickle
import numpy as np
import torch
import nibabel as nib


import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import nibabel as nib
from PIL import Image
from pathlib import Path
import nibabel.gifti as gifti
from timm.models import create_model

import utils
import modeling_pretrain
from datasets import DataAugmentationForMAE

from torchvision.transforms import ToPILImage
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def get_args():
    parser = argparse.ArgumentParser('MAE visualization reconstruction script', add_help=False)
    parser.add_argument('--surface_path', default='/media/test/Cui/NSD/data_01/session01_1.mgh', type=str,
                        help='input surface path')
    parser.add_argument('--save_path', default='./run/test1', type=str, help='save image path')
    parser.add_argument('--model_path', default='./results/Z-Score_OHEM_clip/checkpoint-499.pth', type=str,
                        help='checkpoint path of model')

    parser.add_argument('--vertex_size', default=642, type=int,
                        help='images input size for backbone')
    parser.add_argument('--device', default='cuda:2',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    # Model parameters
    parser.add_argument('--model', default='pretrain_mae_base_patch7_642', type=str, metavar='MODEL',
                        help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    return parser.parse_args()

mean = 0.1937880640322664
var = 5.133784924469524
var = np.sqrt(var)
max_value = 118.40547943115234
min_value = -126.06008911132812

def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    return model

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def read_h5py():
    # 打开HDF5文件
    filename = '/data/member/Cui/nsd_stimuli.hdf5'
    file = h5py.File(filename, 'r')
    # 查看文件中的数据集名称
    dataset_names = list(file.keys())
    print("Dataset names:", dataset_names)

    # 选择一个数据集并读取数据
    dataset = file['imgBrick']
    data = dataset[()]  # 读取数据

    # 查看数据的形状和类型
    print("Data shape:", data.shape)
    print("Data type:", data.dtype)

    # 关闭文件
    file.close()

def read_mat():
    # 加载MAT文件
    filename = '/data/member/Cui/nsd_expdesign.mat'
    stim_order = loadmat(filename)
    ''' Go through all samples to build a dict with keys being their stimulus (image) IDs. '''
    sig = {}
    # for idx in range(MAX_IDX):

    #nsdId = stim_order['masterordering']
    min = 9999999
    max = -9999999
    min_0 = 9999999
    max_0s = -9999999
    sum = 0
    for i in stim_order['masterordering']:
        sum += i
        nsdId = stim_order['subjectim'][0,i-1]
        if min > nsdId:
            min = nsdId
        if max < nsdId:
            max = nsdId
    print(min)
    print(max)
    print(sum)
    mat_data = stim_order
    # 查看MAT文件中的变量名称
    variable_names = list(mat_data.keys())
    print("Variable names:", variable_names)

    # 选择一个变量并查看其内容
    #variable = mat_data['masterordering']
    #print(variable.shape)
    #print("Variable content:", variable)

    # 关闭MAT文件（可选）
    # 如果您只是读取MAT文件中的数据，而不进行任何修改，关闭文件是可选的。

def read_pkl():
    # 读取.pkl文件
    filename = '/data/member/Cui/nsd_stim_info_merged.pkl'
    with open(filename, 'rb') as file:
        try:
            data = pickle.load(file)
        except UnicodeDecodeError:
            # 尝试使用不同的编码方式
            file.seek(0)
            data = pickle.load(file, encoding='latin1')

    # 查看数据类型和内容
    print("Data type:", type(data))
    print("Data content:", data)


def main(args):

#数据集处理
    # 设置参数
    # num_sessions = 40
    # num_trials = 750
    # num_files = num_sessions * num_trials
    # file = '/data/member/Cui/nsd_expdesign.mat'
    # # 加载刺激呈现顺序
    # stim_order = scipy.io.loadmat(file)['masterordering']
    # # 加载图片编号
    # subjectim = scipy.io.loadmat(file)['subjectim']
    #
    # # 打开HDF5文件
    # hdf5_file = h5py.File('/data/member/Cui/nsd_stimuli.hdf5', 'r')
    #
    # # 创建空列表存储输入和目标
    # inputs = []
    # targets = []
    #
    # #遍历所有文件
    # for i in range(num_files):
    #     # 获取文件名
    #     session = (i // num_trials) + 1
    #     trial = (i % num_trials) + 1
    #     filename = f'/media/test/Cui/NSD/data_01/session{session:02d}_{trial:}.mgh'
    #
    #     # 使用nibabel库加载fMRI数据
    #     fmri_data = nib.load(filename)
    #     fmri_data = torch.Tensor(fmri_data.get_fdata())
    #
    #     # 获取对应的刺激编号和目标
    #     stim_idx = stim_order[0, i] - 1
    #     target_idx = subjectim[0, stim_idx] - 1
    #
    #     # 从HDF5文件中读取对应的图片数据
    #     image_data = hdf5_file['imgBrick'][target_idx]  # 替换为HDF5文件中存放图片数据的数据集名称
    #
    #     # 将fMRI数据和图片数据添加到列表
    #     inputs.append(fmri_data)
    #     targets.append(torch.from_numpy(image_data))
    #
    # # 关闭HDF5文件
    # hdf5_file.close()
    #
    # # 将列表转换为PyTorch张量
    # inputs = torch.stack(inputs)
    # targets = torch.stack(targets)
    #
    # # 打印数据集大小
    # print("Inputs shape:", inputs.shape)
    # print("Targets shape:", targets.shape)


    #调用预训练的encoder提取fmri embedding
    #模型部分采用stable diffusion 2-1 base
    print(args)
    device = torch.device(args.device)
    cudnn.benchmark = True
    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.vertex_size // patch_size[0], 7 // patch_size[1])
    args.patch_size = patch_size
    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    #需要修改
    surface = nib.load(args.surface_path)
    data = surface.get_fdata()

    transforms = DataAugmentationForMAE(args)
    surfaces, bool_masked_pos = transforms(data)
    bool_masked_pos = torch.from_numpy(bool_masked_pos)
    surfaces = surfaces.reshape(1, -1, 1)
    with torch.no_grad():
        # surfaces = surfaces[None, :]
        bool_masked_pos = bool_masked_pos[None, :]
        surfaces = surfaces.to(device, non_blocking=True)

        # Z变换
        # surfaces[torch.abs(surfaces) <= 1] = 0

        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        # MinMax
        # input_min = surfaces.min(dim=1,keepdim=True)[0]
        # input_max = surfaces.max(dim=1,keepdim=True)[0]
        # surfaces = 10*((surfaces - input_min) / (input_max - input_min)-1)
        # surfaces = 2 *(surfaces - min_value) / (max_value - min_value) - 1

        # Z-Score
        # mean = surfaces.mean(dim=1, keepdim=True)
        # std = surfaces.std(dim=1, keepdim=True)
        # surfaces = (surfaces - mean) / std
        surfaces = (surfaces - mean) / var

        # 百分比裁剪
        # surfaces = percentile_clipping(surfaces)

        surfaces = surfaces.to(torch.float)
        surfaces_l = surfaces[:, :163842, :]
        surfaces_r = surfaces[:, 163842:, :]

        outputs_l, outputs_r = model(surfaces_l, surfaces_r, bool_masked_pos)
        print(outputs_l.shape)
        print(outputs_r.shape)


if __name__ == '__main__':
    opts = get_args()
    main(opts)
