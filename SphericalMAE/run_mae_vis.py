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

import utils
import modeling_pretrain
from datasets import DataAugmentationForMAE

from torchvision.transforms import ToPILImage
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def get_args():
    parser = argparse.ArgumentParser('MAE visualization reconstruction script', add_help=False)
    parser.add_argument('--surface_path', default='/media/test/Cui/NSD/data_01/session23_333.mgh',type=str, help='input surface path')
    parser.add_argument('--save_path', default='./run/test3',type=str, help='save image path')
    parser.add_argument('--model_path', default='./results/Z-SimDec/checkpoint-499.pth',type=str, help='checkpoint path of model')

    parser.add_argument('--vertex_size', default=642, type=int,
                        help='images input size for backbone')
    parser.add_argument('--device', default='cuda:3',
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


structure_type = ['CORTEX_LEFT', 'CORTEX_RIGHT', 'INVALID']
#head_type = structure_type[1]
LH = structure_type[0]
HD = structure_type[1]

def get_general_meta(structure_type):
    '''

    :param structure_type:  [CORTEX_LEFT, CORTEX_RIGHT, INVALID]
    :return:
    '''
    new_meta = gifti.GiftiMetaData([('AnatomicalStructurePrimary', structure_type)])
    return new_meta




def percentile_clipping(data, percentile=1, replace_with=0):
    temp = data.view(-1)
    lower_bound = temp.kthvalue(int(len(temp) * percentile / 100)).values
    upper_bound = temp.kthvalue(int(len(temp) * (100 - percentile) / 100)).values
    clipped_data = torch.where(temp < lower_bound, replace_with, temp)
    clipped_data = torch.where(temp > upper_bound, replace_with, clipped_data)
    clipped_data = clipped_data.view(-1,327684,1)
    return clipped_data


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    return model

meta_l = get_general_meta('CORTEX_LEFT')
meta_r = get_general_meta('CORTEX_RIGHT')

mean = 0.1937880640322664
var = 5.133784924469524
var = np.sqrt(var)
max_value = 118.40547943115234
min_value = -126.06008911132812

def main(args):
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
    # with open(args.img_path, 'rb') as f:
    #     img = Image.open(f)
    #     img.convert('RGB')
    #     print("img path:", args.img_path)

    surface = nib.load(args.surface_path)
    data = surface.get_fdata()


    transforms = DataAugmentationForMAE(args)
    surfaces, bool_masked_pos = transforms(data)

    # 将原始的数据转换成.gii文件
    origin = surfaces.numpy().astype(np.float32).reshape(-1, 1)
    origin_l = origin[:163842,:]
    origin_r = origin[163842:,:]
    origin_l = origin_l.astype(np.float32).reshape(-1,1)
    origin_r = origin_r.astype(np.float32).reshape(-1,1)
    origin_l_file = f"origin_l.func.gii"
    origin_r_file = f"origin_r.func.gii"
    origin_l_path = os.path.join(args.save_path, origin_l_file)
    origin_r_path = os.path.join(args.save_path, origin_r_file)
    img0_l = nib.gifti.GiftiImage()
    img0_r = nib.gifti.GiftiImage()
    data_l = nib.gifti.GiftiDataArray(data=origin_l, intent='NIFTI_INTENT_NORMAL',meta=meta_l)
    data_r = nib.gifti.GiftiDataArray(data=origin_r, intent='NIFTI_INTENT_NORMAL',meta=meta_r)
    img0_l.add_gifti_data_array(data_l)
    img0_r.add_gifti_data_array(data_r)
    nib.save(img0_l, origin_l_path)
    nib.save(img0_r, origin_r_path)

    bool_masked_pos = torch.from_numpy(bool_masked_pos)
    surfaces = surfaces.reshape(1, -1, 1)
    with torch.no_grad():
        #surfaces = surfaces[None, :]
        bool_masked_pos = bool_masked_pos[None, :]
        surfaces = surfaces.to(device, non_blocking=True)
        
        #Z变换
        #surfaces[torch.abs(surfaces) <= 1] = 0

        
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        #MinMax
        #input_min = surfaces.min(dim=1,keepdim=True)[0]
        #input_max = surfaces.max(dim=1,keepdim=True)[0]
        #surfaces = 10*((surfaces - input_min) / (input_max - input_min)-1)
        #surfaces = 2 *(surfaces - min_value) / (max_value - min_value) - 1
        
        # Z-Score
        #mean = surfaces.mean(dim=1, keepdim=True)
        #std = surfaces.std(dim=1, keepdim=True)
        #surfaces = (surfaces - mean) / std
        surfaces = (surfaces - mean) / var

        # 百分比裁剪
        #surfaces = percentile_clipping(surfaces)

        surfaces = surfaces.to(torch.float)
        surfaces_l = surfaces[:,:163842,:]
        surfaces_r = surfaces[:,163842:,:]
        

        #outputs = model(surfaces, bool_masked_pos)
        #outputs = outputs*(input_max - input_min) + input_min
        
        outputs_l,outputs_r = model(surfaces_l, surfaces_r,bool_masked_pos)
        #print(outputs_l.shape)
        #print(outputs_r.shape)
	
        #surfaces = surfaces.cpu()
        surfaces_l = surfaces_l.cpu()
        surfaces_r = surfaces_r.cpu()
        #outputs = outputs.cpu()
        outputs_l = outputs_l.cpu()
        outputs_r = outputs_r.cpu()

        surfaces_l = surfaces_l.numpy().astype(np.float32)
        surfaces_r = surfaces_r.numpy().astype(np.float32)
        #outputs = outputs.numpy().astype(np.float32)
        outputs_l = outputs_l.numpy().astype(np.float32)
        outputs_r = outputs_r.numpy().astype(np.float32)

        #surfaces = surfaces.reshape(-1,1)
        #outputs = outputs.reshape(-1, 1)
        surfaces_l = surfaces_l.reshape(-1, 1)
        surfaces_r = surfaces_r.reshape(-1, 1)
        outputs_l = outputs_l.reshape(-1, 1)
        outputs_r = outputs_r.reshape(-1, 1)

        #save preprocessed original surface
        #orgin_minmax = nib.MGHImage(surfaces,affine=np.eye(4))

        pred_l_file = f"pred_l.func.gii"
        pred_r_file = f"pred_r.func.gii"
        pred_l_path = os.path.join(args.save_path, pred_l_file)
        pred_r_path = os.path.join(args.save_path, pred_r_file)
        img1_l = nib.gifti.GiftiImage()
        img1_r = nib.gifti.GiftiImage()
        data_l = nib.gifti.GiftiDataArray(data=surfaces_l, intent='NIFTI_INTENT_NORMAL', meta=meta_l)
        data_r = nib.gifti.GiftiDataArray(data=surfaces_r, intent='NIFTI_INTENT_NORMAL', meta=meta_r)
        img1_l.add_gifti_data_array(data_l)
        img1_r.add_gifti_data_array(data_r)
        nib.save(img1_l, pred_l_path)
        nib.save(img1_r, pred_r_path)


        # save output surface
        output_l_file = f"output_l.func.gii"
        output_r_file = f"output_r.func.gii"
        output_l_path = os.path.join(args.save_path, output_l_file)
        output_r_path = os.path.join(args.save_path, output_r_file)
        img2_l = nib.gifti.GiftiImage()
        img2_r = nib.gifti.GiftiImage()
        data_l = nib.gifti.GiftiDataArray(data=outputs_l, intent='NIFTI_INTENT_NORMAL', meta=meta_l)
        data_r = nib.gifti.GiftiDataArray(data=outputs_r, intent='NIFTI_INTENT_NORMAL', meta=meta_r)
        img2_l.add_gifti_data_array(data_l)
        img2_r.add_gifti_data_array(data_r)
        nib.save(img2_l, output_l_path)
        nib.save(img2_r, output_r_path)

if __name__ == '__main__':
    opts = get_args()
    main(opts)
    # surface = nib.load('./run/orgin_minmax.mgh')
    # data = surface.get_fdata()
    # print(data.shape)
