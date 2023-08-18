import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img, Augment_RGB_torch
import torch.nn.functional as F
import random

augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 

##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform

        gt_files = sorted(os.listdir(os.path.join(rgb_dir, 'high')))# groundturth before
        ll_files = sorted(os.listdir(os.path.join(rgb_dir, 'low')))# input before
        
        self.gt_filenames = [os.path.join(rgb_dir, 'high', x) for x in gt_files if is_png_file(x)]
        self.ll_filenames = [os.path.join(rgb_dir, 'low', x)       for x in ll_files if is_png_file(x)]
        
        self.img_options=img_options

        self.tar_size = len(self.gt_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        gt = torch.from_numpy(np.float32(load_img(self.gt_filenames[tar_index])))
        ll = torch.from_numpy(np.float32(load_img(self.ll_filenames[tar_index])))
                
        gt_filename = os.path.split(self.gt_filenames[tar_index])[-1]
        ll_filename = os.path.split(self.ll_filenames[tar_index])[-1]

        gt = gt.permute(2,0,1)
        ll = ll.permute(2,0,1)

        return gt, ll, gt_filename, ll_filename
    
##################################################################################################
class DataLoaderTrainPatch(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain_Patch, self).__init__()

        self.target_transform = target_transform

        gt_files = sorted(os.listdir(os.path.join(rgb_dir, 'high')))# groundturth before
        ll_files = sorted(os.listdir(os.path.join(rgb_dir, 'low')))# input before
        
        self.gt_filenames = [os.path.join(rgb_dir, 'high', x) for x in gt_files if is_png_file(x)]
        self.ll_filenames = [os.path.join(rgb_dir, 'low', x)       for x in ll_files if is_png_file(x)]
        
        self.img_options=img_options

        self.tar_size = len(self.gt_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size
    
    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        gt = torch.from_numpy(np.float32(load_img(self.gt_filenames[tar_index])))
        ll = torch.from_numpy(np.float32(load_img(self.ll_filenames[tar_index])))
        
        gt = gt.permute(2,0,1)
        ll = ll.permute(2,0,1)

        gt_filename = os.path.split(self.gt_filenames[tar_index])[-1]
        ll_filename = os.path.split(self.ll_filenames[tar_index])[-1]

        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = gt.shape[1]
        W = gt.shape[2]
        r = np.random.randint(0, H - ps)
        c = np.random.randint(0, W - ps)
        gt = gt[:, r:r + ps, c:c + ps]
        ll = ll[:, r:r + ps, c:c + ps]

        apply_trans = transforms_aug[random.getrandbits(3)]

        gt = getattr(augment, apply_trans)(gt)
        ll = getattr(augment, apply_trans)(ll)        

        return gt, ll, gt_filename, ll_filename

##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        gt_files = sorted(os.listdir(os.path.join(rgb_dir, 'high')))
        ll_files = sorted(os.listdir(os.path.join(rgb_dir, 'low')))


        self.gt_filenames = [os.path.join(rgb_dir, 'high', x) for x in gt_files if is_png_file(x)]
        self.ll_filenames = [os.path.join(rgb_dir, 'low', x) for x in ll_files if is_png_file(x)]
        

        self.tar_size = len(self.gt_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        gt = torch.from_numpy(np.float32(load_img(self.gt_filenames[tar_index])))
        ll = torch.from_numpy(np.float32(load_img(self.ll_filenames[tar_index])))
                
        gt_filename = os.path.split(self.gt_filenames[tar_index])[-1]
        ll_filename = os.path.split(self.ll_filenames[tar_index])[-1]

        gt = gt.permute(2,0,1)
        ll = ll.permute(2,0,1)

        return gt, ll, gt_filename, ll_filename

##################################################################################################
class DataLoaderValPatch(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderVal_Patch, self).__init__()

        self.target_transform = target_transform

        gt_files = sorted(os.listdir(os.path.join(rgb_dir, 'high')))
        ll_files = sorted(os.listdir(os.path.join(rgb_dir, 'low')))


        self.gt_filenames = [os.path.join(rgb_dir, 'high', x) for x in gt_files if is_png_file(x)]
        self.ll_filenames = [os.path.join(rgb_dir, 'low', x) for x in ll_files if is_png_file(x)]
        
        self.img_options=img_options
        self.tar_size = len(self.gt_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        gt = torch.from_numpy(np.float32(load_img(self.gt_filenames[tar_index])))
        ll = torch.from_numpy(np.float32(load_img(self.ll_filenames[tar_index])))
        
        gt = gt.permute(2,0,1)
        ll = ll.permute(2,0,1)

        gt_filename = os.path.split(self.gt_filenames[tar_index])[-1]
        ll_filename = os.path.split(self.ll_filenames[tar_index])[-1]

        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = gt.shape[1]
        W = gt.shape[2]
        r = np.random.randint(0, H - ps)
        c = np.random.randint(0, W - ps)
        gt = gt[:, r:r + ps, c:c + ps]
        ll = ll[:, r:r + ps, c:c + ps]

        apply_trans = transforms_aug[random.getrandbits(3)]

        gt = getattr(augment, apply_trans)(gt)
        ll = getattr(augment, apply_trans)(ll)        

        return gt, ll, gt_filename, ll_filename
