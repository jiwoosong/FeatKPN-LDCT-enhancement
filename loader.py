import os
import argparse
import jutil

import pickle
import numpy as np
import nibabel as nib

import cv2
from albumentations import (VerticalFlip, HorizontalFlip, Flip, RandomRotate90, Rotate, ShiftScaleRotate,
                            CenterCrop, OpticalDistortion, GridDistortion, ElasticTransform, JpegCompression,
                            HueSaturationValue, RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur,
                            MedianBlur, GaussNoise, CLAHE, ChannelShuffle, InvertImg, RandomGamma, ToGray,
                            PadIfNeeded, OneOf, Compose, PiecewiseAffine, RandomBrightnessContrast, pytorch,
                            Normalize, RandomCrop, Resize, Perspective, grid_distortion, RandomResizedCrop)
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import torch
import torchvision.transforms
from torch.utils.data import Dataset, DataLoader

train_transform = Compose([
    Rotate(limit=90,value=-1024),
    # ElasticTransform(alpha=10,alpha_affine=15, border_mode=0, p=0.5, value=-1024),
    RandomResizedCrop (512, 512, scale=(0.5, 1.0),ratio=(0.75, 1.3333333333333333),interpolation=cv2.INTER_LINEAR,p=0.5),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
], additional_targets={'image0': 'image', 'image1': 'image'})

test_transform = Compose([
    Resize(height=512,width=512)
], additional_targets={'image0': 'image', 'image1': 'image'})

class Denoising_Loader(Dataset):
    def __init__(self, args, mode, subject_list):
        self.args = args
        self.mode = mode
        assert self.mode in ['train', 'val', 'test'], "mode is 'train' or 'test'"
        self.subject_list = subject_list
        self.Name_list = []
        self.NDCT_list = []
        self.LDCT_list = []
        self.ToTensor = torchvision.transforms.ToTensor()
        if self.args.load_memory:
            for idx, subject in enumerate(self.subject_list):
                NDCT = nib.load(os.path.join(subject, 'full.nii.gz'))
                LDCT = nib.load(os.path.join(subject, 'low.nii.gz'))
                if NDCT.shape[2] == LDCT.shape[2]:
                    NDCT_list = list(NDCT.get_fdata().astype(np.float32).transpose(2,0,1))
                    LDCT_list = list(LDCT.get_fdata().astype(np.float32).transpose(2,0,1))

                    self.Name_list += ['%s_%03d'%(subject.split('/')[-1],i) for i in range(len(NDCT_list))]
                    self.NDCT_list += NDCT_list
                    self.LDCT_list += LDCT_list
                    jutil.printProgress(idx+1, len(self.subject_list),'Load Memory(%s)'%(self.mode), subject.split('/')[-1], 1, 50)
                else:
                    print(subject,'Shape Difference %s %s'%(str(NDCT.shape), str(LDCT.shape)))
        else:
            raise
    def __len__(self):
        return len(self.NDCT_list)
    def __getitem__(self, idx):
        if self.args.load_memory:
            # idx=40
            NDCT = self.NDCT_list[idx].copy()[:,:,None]#[None,:]
            LDCT = self.LDCT_list[idx].copy()[:,:,None]#[None,:]
        else:
            raise

        if self.mode =='train':
            transformed_data = train_transform(image=NDCT, image0=LDCT)
            NDCT = transformed_data['image']
            LDCT = transformed_data['image0']
            # Random HU Bias Shift
            bias = np.random.uniform(-20, 20)
            NDCT += bias
            LDCT += bias
            NDCT = self.ToTensor(NDCT)
            LDCT = self.ToTensor(LDCT)

            if self.args.patch_n:
                patch_input_imgs = []
                patch_target_imgs = []
                h, w = NDCT.shape[1:]
                new_h, new_w = self.args.patch_size, self.args.patch_size
                for _ in range(self.args.patch_n):
                    top = np.random.randint(0, h - new_h)
                    left = np.random.randint(0, w - new_w)
                    patch_input_img = LDCT[:, top:top + new_h, left:left + new_w]
                    patch_target_img = NDCT[:, top:top + new_h, left:left + new_w]
                    patch_input_imgs.append(patch_input_img)
                    patch_target_imgs.append(patch_target_img)

                LDCT = torch.stack(patch_input_imgs, dim=0)
                NDCT = torch.stack(patch_target_imgs, dim=0)

        else:
            transformed_data = test_transform(image=NDCT, image0=LDCT)
            NDCT = transformed_data['image']
            LDCT = transformed_data['image0']
            NDCT = self.ToTensor(NDCT)
            LDCT = self.ToTensor(LDCT)

        if len(self.args.window)==1:
            NDCT = self.windowing(NDCT, self.args.window, rand=False)
            LDCT = self.windowing(LDCT, self.args.window, rand=False)
        else:
            NDCT = self.multi_windowing(NDCT, self.args.window, rand=False)
            LDCT = self.multi_windowing(LDCT, self.args.window, rand=False)

        return {'NDCT':NDCT, 'LDCT':LDCT, 'idx': idx}

    def set_window(self, arr, w_min=0, w_max=100):
        arr = arr.clip(w_min,w_max)
        arr = ((1.0 * (arr - w_min) / (w_max - w_min)))
        return arr
    def windowing(self, img, window, rand = False):
        if rand:
            # Random Windowing (for training)
            r = (self.args.window[0][1] - self.args.window[0][0]) * 0.1
            r = np.random.uniform(-r, r)
            img1 = self.set_window(img, window[0][0] + r, window[0][1] + r)
        else:
            img1 = self.set_window(img, window[0][0], window[0][1])
        return img1
    def multi_windowing(self, img, window, rand = False):
        if rand:
            # Random Windowing (for training)
            r = (self.args.window[0][1] - self.args.window[0][0]) * 0.1
            r = np.random.uniform(-r, r)
            img1 = self.set_window(img, window[0][0] + r, window[0][1] + r)

            r = (self.args.window[1][1] - self.args.window[1][0]) * 0.1
            r = np.random.uniform(-r, r)
            img2 = self.set_window(img, window[1][0] + r, window[1][1] + r)

            r = (self.args.window[2][1] - self.args.window[2][0]) * 0.1
            r = np.random.uniform(-r, r)
            img3 = self.set_window(img, window[2][0] + r, window[2][1] + r)
        else:
            img1 = self.set_window(img, window[0][0], window[0][1])
            img2 = self.set_window(img, window[1][0], window[1][1])
            img3 = self.set_window(img, window[2][0], window[2][1])
        return np.concatenate([img1,img2,img3],axis=0)

def get_Full_Denoising_Loader(args):
    brain_path = os.path.join(args.data_path,'2_NIFTI_Data')

    if os.path.isfile(args.subject_path):
        with open(args.subject_path, 'rb') as f:
            train_list, val_list = pickle.load(f)
        print(jutil.str_100('Load Subject Lists'))
    else:
        print(jutil.str_100('No Subject Lists!'))
        subjects_list = sorted(os.listdir(brain_path))
        np.random.seed(42)
        subjects_list = np.random.permutation(subjects_list).tolist()
        print('NP random order : %s' % (str(subjects_list[0:5])))
        train_list = subjects_list[:int(len(subjects_list)*0.8)]
        del subjects_list[:int(len(subjects_list)*0.8)]
        val_list = subjects_list
        del subjects_list
        print('Save in ',args.subject_path)
        with open(args.subject_path, 'wb') as f:
            pickle.dump((train_list, val_list), f)

    train_list = [os.path.join(brain_path, item) for item in train_list if os.path.exists(os.path.join(brain_path, item))]
    val_list = [os.path.join(brain_path, item) for item in val_list if os.path.exists(os.path.join(brain_path, item))]
    print('Train : %d'%(len(train_list)))
    print('Val : %d' % (len(val_list)))

    train_loader = Denoising_Loader(args, mode='train', subject_list = train_list)
    train_loader = torch.utils.data.DataLoader(train_loader,
                                               shuffle=True,
                                               drop_last=True,
                                               pin_memory=args.pin_memory,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers)

    val_loader = Denoising_Loader(args, mode='val', subject_list = val_list)
    val_loader = torch.utils.data.DataLoader(val_loader,
                                               shuffle=False,
                                               drop_last=False,
                                               pin_memory=args.pin_memory,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers)
    return train_loader, val_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=r'Data/')
    parser.add_argument('--subject_path', type=str, default=r'Data/train_val_split.pickle')
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--load_memory', type=bool, default=True)

    parser.add_argument('--window', type=int, default=[(-20, 100)])

    args = parser.parse_args()
    train_loder, val_loader = get_Full_Denoising_Loader(args)

    for idx, batch in enumerate(train_loder):
        NDCT = batch['NDCT'][0]
        LDCT = batch['LDCT'][0]
        jutil.printProgress(idx+1,len(train_loder),'Train','',1,50)
        pass
