 # -*- coding: utf-8 -*-
# @Time    : 2021/11/18 22:40
# @Author  : zhao pengfei
# @Email   : zsonghuan@gmail.com
# @File    : run_mae_vis.py
# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

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
import torch.nn as nn
import utils
import modeling_pretrain
from datasets import DataAugmentationForMAE

from torchvision.transforms import ToPILImage
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def get_args():
    parser = argparse.ArgumentParser('MAE visualization reconstruction script', add_help=False)
    parser.add_argument('--surface_path', default='/media/test/Cui/NSD/data_01/session40_666.mgh',type=str, help='input surface path')
    parser.add_argument('--save_path', default='./run/test3',type=str, help='save image path')
    parser.add_argument('--model_path', default='./results/Z-OHEM-SimDec/checkpoint-499.pth',type=str, help='checkpoint path of model')

    parser.add_argument('--vertex_size', default=642, type=int,
                        help='images input size for backbone')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.0, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    # Model parameters
    parser.add_argument('--model', default='pretrain_mae_base_patch7_642', type=str, metavar='MODEL',
                        help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    return parser.parse_args()

def get_model(args):
    #print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    return model


mean = 0.1937880640322664
var = 5.133784924469524
var = np.sqrt(var)
max_value = 118.40547943115234
min_value = -126.06008911132812

def main(args):
    #print(args)

    device = torch.device(args.device)
    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    #print("Patch size = %s" % str(patch_size))
    args.window_size = (args.vertex_size // patch_size[0], 7 // patch_size[1])
    args.patch_size = patch_size

    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
#需要修改
    # with open(args.img_path, 'rb') as f:
    #     img = Image.open(f)
    #     img.convert('RGB')
    #     print("img path:", args.img_path)

    surface = nib.load(args.surface_path)
    data = surface.get_fdata()


    transforms = DataAugmentationForMAE(args)
    surfaces, bool_masked_pos = transforms(data)

    bool_masked_pos = torch.from_numpy(bool_masked_pos)
    surfaces = surfaces.reshape(1, -1, 1)
    empty = torch.zeros(1,327684,1)
    with torch.no_grad():
        #surfaces = surfaces[None, :]
        bool_masked_pos = bool_masked_pos[None, :]
        surfaces = surfaces.to(device, non_blocking=True)
        empty = empty.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        # Z-Score
        surfaces = (surfaces - mean) / var

        surfaces = surfaces.to(torch.float)
        surfaces_l = surfaces[:,:163842,:]
        surfaces_r = surfaces[:,163842:,:]

        empty_l = empty[:,:163842,:]
        empty_r = empty[:,163842:,:]

        
        outputs_l,outputs_r = model(surfaces_l, surfaces_r,bool_masked_pos)
        compare_l,compare_r = model(empty_l, empty_r,bool_masked_pos)



        loss_func = nn.MSELoss()
        loss1 = loss_func(input=outputs_l, target=compare_l)
        loss2 = loss_func(input=outputs_r, target=compare_r)
        loss = loss1 + loss2
        print(loss)
if __name__ == '__main__':
    opts = get_args()
    main(opts)
    # surface = nib.load('./run/orgin_minmax.mgh')
    # data = surface.get_fdata()
    # print(data.shape)
