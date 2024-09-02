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
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import nibabel as nib
import nibabel.gifti as gifti
from layers_noCBAM_modifiedDIscrimintor import sperical_unet

def get_args():
    parser = argparse.ArgumentParser('U-Net visualization reconstruction script', add_help=False)
    parser.add_argument('--surface_path', default='/media/amax/Cui/NSD/data_01/session21_134.mgh',type=str, help='input surface path')
    parser.add_argument('--save_path', default='./run/test1',type=str, help='save image path')
    parser.add_argument('--model_path', default='./results/U-Net/best_model_params.pth',type=str, help='checkpoint path of model')
    parser.add_argument('--device', default='cuda:3',
                        help='device to use for training / testing')
    return parser.parse_args()


mean = 0.2875914866986795
var = 4.729840210010396
var = np.sqrt(var)
min_value = -96.1386489868164
max_value = 85.21990203857422
def get_general_meta(structure_type):
    '''

    :param structure_type:  [CORTEX_LEFT, CORTEX_RIGHT, INVALID]
    :return:
    '''
    new_meta = gifti.GiftiMetaData([('AnatomicalStructurePrimary', structure_type)])
    return new_meta


def percentile_cliping(data, lower_percentile = 1,upper_percentile=99):
    lower_cutoff = torch.kthvalue(data.flatten(),int(lower_percentile * len(data.flatten())/100)).values
    upper_cutoff = torch.kthvalue(data.flatten(),int(upper_percentile * len(data.flatten())/100)).values
    clipped_data = torch.clamp(data, lower_cutoff,upper_cutoff)
    return clipped_data

meta_l = get_general_meta('CORTEX_LEFT')
meta_r = get_general_meta('CORTEX_RIGHT')

def main(args):
    print(args)
    device = torch.device(args.device)
    cudnn.benchmark = True

    model = sperical_unet(1)
    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()


    surface = nib.load(args.surface_path)
    data = surface.get_fdata()
    surfaces = torch.tensor(data)

    # 将原始的数据转换成.gii文件
    origin = surfaces.numpy().astype(np.float32).reshape(-1,1)
    origin_l = origin[:10242,:]
    origin_r = origin[163842:174084,:]

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

    surfaces = surfaces.reshape(1, 1, -1)
    x_l = surfaces[:, :, :10242]
    x_r = surfaces[:, :, 163842:174084]
    surfaces = torch.cat((x_l, x_r), dim=-1)
    with torch.no_grad():
        surfaces = surfaces.to(device, non_blocking=True)

        #Z变换
        #surfaces[torch.abs(surfaces) <= 1] = 0

        #百分比裁剪
        #surfaces = percentile_cliping(surfaces,1,99)

        #MinMax
        #surfaces = 2 * (surfaces - min_value) / (max_value - min_value) - 1

        # Z-Score
        #mean = surfaces.mean(dim=-1, keepdim=True)
        #std = surfaces.std(dim=-1, keepdim=True)
        #surfaces = (surfaces - mean) / std
        surfaces = (surfaces - mean) / var
        
        surfaces = surfaces.to(torch.float)
        surfaces_l = surfaces[:,:,:10242]
        surfaces_r = surfaces[:,:,10242:]
        

        #outputs = model(surfaces, bool_masked_pos)
        #outputs = outputs*(input_max - input_min) + input_min
        
        outputs_l,outputs_r = model(surfaces_l, surfaces_r)

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
