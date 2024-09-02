import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
import os
from scipy import interpolate
from einops import rearrange
import json
import csv
import torch
from pathlib import Path
import torchvision.transforms as transforms
from collections import defaultdict
import h5py
import scipy.io as spio
import scipy.io
import pickle
import nibabel as nib
#import albumentations
from PIL import Image


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
from dc_ldm import modeling_pretrain
from datasets import DataAugmentationForMAE

from torchvision.transforms import ToPILImage
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD




def identity(x):
    return x
def pad_to_patch_size(x, patch_size):
    assert x.ndim == 2
    return np.pad(x, ((0,0),(0, patch_size-x.shape[1]%patch_size)), 'wrap')

def pad_to_length(x, length):
    assert x.ndim == 3
    assert x.shape[-1] <= length
    if x.shape[-1] == length:
        return x

    return np.pad(x, ((0,0),(0,0), (0, length - x.shape[-1])), 'wrap')

def pad_to_length_dim2(x, length):
    assert x.ndim == 2
    assert x.shape[-1] <= length
    if x.shape[-1] == length:
        return x

    return np.pad(x, ((0,0), (0, length - x.shape[-1])), 'wrap')

def normalize(x, mean=None, std=None):
    mean = np.mean(x) if mean is None else mean
    std = np.std(x) if std is None else std
    return (x - mean) / (std * 1.0)

def process_voxel_ts(v, p, t=8):
    '''
    v: voxel timeseries of a subject. (1200, num_voxels)
    p: patch size
    t: time step of the averaging window for v. Kamitani used 8 ~ 12s
    return: voxels_reduced. reduced for the alignment of the patch size (num_samples, num_voxels_reduced)

    '''
    # average the time axis first
    num_frames_per_window = t // 0.75 # ~0.75s per frame in HCP
    v_split = np.array_split(v, len(v) // num_frames_per_window, axis=0)
    v_split = np.concatenate([np.mean(f,axis=0).reshape(1,-1) for f in v_split],axis=0)
    # pad the num_voxels
    # v_split = np.concatenate([v_split, np.zeros((v_split.shape[0], p - v_split.shape[1] % p))], axis=-1)
    v_split = pad_to_patch_size(v_split, p)
    v_split = normalize(v_split)
    return v_split

def augmentation(data, aug_times=2, interpolation_ratio=0.5):
    '''
    data: num_samples, num_voxels_padded
    return: data_aug: num_samples*aug_times, num_voxels_padded
    '''
    num_to_generate = int((aug_times-1)*len(data)) 
    if num_to_generate == 0:
        return data
    pairs_idx = np.random.choice(len(data), size=(num_to_generate, 2), replace=True)
    data_aug = []
    for i in pairs_idx:
        z = interpolate_voxels(data[i[0]], data[i[1]], interpolation_ratio)
        data_aug.append(np.expand_dims(z,axis=0))
    data_aug = np.concatenate(data_aug, axis=0)

    return np.concatenate([data, data_aug], axis=0)

def interpolate_voxels(x, y, ratio=0.5):
    ''''
    x, y: one dimension voxels array
    ratio: ratio for interpolation
    return: z same shape as x and y

    '''
    values = np.stack((x,y))
    points = (np.r_[0, 1], np.arange(len(x)))
    xi = np.c_[np.full((len(x)), ratio), np.arange(len(x)).reshape(-1,1)]
    z = interpolate.interpn(points, values, xi)
    return z

def img_norm(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = (img / 255.0) * 2.0 - 1.0 # to -1 ~ 1
    return img

def channel_first(img):
        if img.shape[-1] == 3:
            return rearrange(img, 'h w c -> c h w')
        return img

class base_dataset(Dataset):
    def __init__(self, x, y=None, transform=identity):
        super(base_dataset, self).__init__()
        self.x = x
        self.y = y
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        if self.y is None:
            return self.transform(self.x[index])
        else:
            return self.transform(self.x[index]), self.transform(self.y[index])
    
def remove_repeats(fmri, img_lb):
    assert len(fmri) == len(img_lb), 'len error'
    fmri_dict = {}
    for f, lb in zip(fmri, img_lb):
        if lb in fmri_dict.keys():
            fmri_dict[lb].append(f)
        else:
            fmri_dict[lb] = [f]
    lbs = []
    fmris = []
    for k, v in fmri_dict.items():
        lbs.append(k)
        fmris.append(np.mean(np.stack(v), axis=0))
    return np.stack(fmris), lbs

def get_stimuli_list(root, sub):
    sti_name = []
    path = os.path.join(root, 'Stimuli_Presentation_Lists', sub)
    folders = os.listdir(path)
    folders.sort()
    for folder in folders:
        if not os.path.isdir(os.path.join(path, folder)):
            continue
        files = os.listdir(os.path.join(path, folder))
        files.sort()
        for file in files:
            if file.endswith('.txt'):
                sti_name += list(np.loadtxt(os.path.join(path, folder, file), dtype=str))

    sti_name_to_return = []
    for name in sti_name:
        if name.startswith('rep_'):
            name = name.replace('rep_', '', 1)
        sti_name_to_return.append(name)
    return sti_name_to_return

def list_get_all_index(list, value):
    return [i for i, v in enumerate(list) if v == value]




def create_NSD_dataset(path='/media/test/Cui/NSD/nsddata_stimuli/stimuli/nsd', patch_size=16, fmri_transform=identity,
            image_transform=identity, include_nonavg_test=False):
  
    
    fmri_train = torch.load('/data1/dataset/NSD_surface/fmri_Zscore.pt')["fmri_train"]#[:100]
    fmri_test = torch.load('/data1/dataset/NSD_surface/fmri_Zscore.pt')["fmri_test"]#[:10]
    
    img_train = torch.load('/data1/dataset/NSD_surface/img.pt')["img_train"]#[:100]
    img_test = torch.load('/data1/dataset/NSD_surface/img.pt')["img_test"]#[:10]
   
    if isinstance(image_transform, list):
        return (NSD_dataset(fmri_train, img_train, fmri_transform, image_transform[0]),
                NSD_dataset(fmri_test, img_test, torch.FloatTensor, image_transform[1]))
    else:
        return (NSD_dataset(fmri_train, img_train, fmri_transform, image_transform),
                NSD_dataset(fmri_test, img_test, torch.FloatTensor, image_transform))


class NSD_dataset(Dataset):
    def __init__(self, fmri, image, fmri_transform=identity, image_transform=identity):
        self.fmri = fmri
        self.image = image
        self.fmri_transform = fmri_transform
        self.image_transform = image_transform

    def __len__(self):
        return len(self.fmri)

    def __getitem__(self, index):
        fmri = self.fmri[index]
        img = self.image[index] / 255.0

        return {'fmri': self.fmri_transform(fmri),
                'image': self.image_transform(img)}

def create_BOLD5000_dataset(path='/data1/dataset/BOLD5000', patch_size=16, fmri_transform=identity,
            image_transform=identity, subjects = ['CSI1', 'CSI2', 'CSI3', 'CSI4'], include_nonavg_test=False):
    
    fmri_path = os.path.join(path, 'BOLD5000_surface')
    img_path = os.path.join(path, 'BOLD5000_Stimuli')
    imgs_dict = np.load(os.path.join(img_path, 'Scene_Stimuli/Presented_Stimuli/img_dict.npy'),allow_pickle=True).item()
    repeated_imgs_list = np.loadtxt(os.path.join(img_path, 'Scene_Stimuli', 'repeated_stimuli_113_list.txt'), dtype=str)

    fmri_files = [f for f in os.listdir(fmri_path) if f.endswith('.npy')]
    fmri_files.sort()
    
    fmri_train_major = []
    fmri_test_major = []
    img_train_major = []
    img_test_major = []
    for sub in subjects:
        # load fmri
        fmri_data_sub = []
        for npy in fmri_files:
            if npy.endswith('.npy') and sub in npy:
                fmri_data_sub.append(np.load(os.path.join(fmri_path, npy)))
        #fmri_data_sub = np.concatenate(fmri_data_sub, axis=-1) # concatenate all rois
        fmri_data_sub = np.squeeze(normalize(fmri_data_sub))
        #print(fmri_data_sub.shape)
        # load image
        img_files = get_stimuli_list(img_path, sub)
        img_data_sub = [imgs_dict[name] for name in img_files]
        #print(img_data_sub.shape)
        # split train test
        test_idx = [list_get_all_index(img_files, img) for img in repeated_imgs_list]
        #print(test_idx.shape)
        test_idx = [i for i in test_idx if len(i) > 0] # remove empy list for CSI4
        #print(test_idx.shape)
        test_fmri = np.stack([fmri_data_sub[idx].mean(axis=0) for idx in test_idx])
        test_img = np.stack([img_data_sub[idx[0]] for idx in test_idx])
        
        test_idx_flatten = []
        for idx in test_idx:
            test_idx_flatten += idx # flatten
        if include_nonavg_test:
            test_fmri = np.concatenate([test_fmri, fmri_data_sub[test_idx_flatten]], axis=0)
            test_img = np.concatenate([test_img, np.stack([img_data_sub[idx] for idx in test_idx_flatten])], axis=0)
        #print(test_idx_flatten)
        train_idx = [i for i in range(len(img_files)) if i not in test_idx_flatten]
        train_img = np.stack([img_data_sub[idx] for idx in train_idx])
        train_fmri = fmri_data_sub[train_idx]

        fmri_train_major.append(train_fmri)
        
        fmri_test_major.append(test_fmri)
        img_train_major.append(train_img)
        img_test_major.append(test_img)
    fmri_train_major = np.concatenate(fmri_train_major, axis=0)
    #print(fmri_train_major.shape)
    fmri_test_major = np.concatenate(fmri_test_major, axis=0)
    img_train_major = torch.tensor(np.concatenate(img_train_major, axis=0))
    #print(img_train_major.shape)
    img_test_major = torch.tensor(np.concatenate(img_test_major, axis=0))

    #num_voxels = fmri_train_major.shape[-1]
    if isinstance(image_transform, list):
        return (BOLD5000_dataset(fmri_train_major, img_train_major, fmri_transform, image_transform[0]), 
                BOLD5000_dataset(fmri_test_major, img_test_major, torch.FloatTensor, image_transform[1]))
    else:
        return (BOLD5000_dataset(fmri_train_major, img_train_major, fmri_transform, image_transform), 
                BOLD5000_dataset(fmri_test_major, img_test_major, torch.FloatTensor, image_transform))


class BOLD5000_dataset(Dataset):
    def __init__(self, fmri, image, fmri_transform=identity, image_transform=identity):
        self.fmri = fmri
        self.image = image
        self.fmri_transform = fmri_transform
        self.image_transform = image_transform

    def __len__(self):
        return len(self.fmri)

    def __getitem__(self, index):
        fmri = self.fmri[index]
        img = self.image[index] / 255.0

        return {'fmri': self.fmri_transform(fmri),
                'image': self.image_transform(img)}



