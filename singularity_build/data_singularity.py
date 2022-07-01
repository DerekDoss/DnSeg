import sys
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import nibabel as nib
import torch.nn.functional as F
import numpy as np


pd.set_option('display.max_rows', None)

class NBM_dataset(Dataset):
    def __init__(self, raw_imgs_dir, fileNames):

        self.raw_imgs_dir = raw_imgs_dir
        # self.raw_img_files = os.listdir(raw_imgs_dir)
        self.fileNames = fileNames
        # self.label_imgs_dir = label_imgs_dir

    def get_script_filename(self):
        return __file__

    def __len__(self):
        return len(self.fileNames)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        base_name = self.fileNames[idx].replace('img', '')


        img_name = os.path.join(self.raw_imgs_dir, self.fileNames[idx])
        img_nii = nib.load(img_name)
        img_array = img_nii.get_data()

        # Rescale 1-99 percentile values to ~[0-1]
        scale_val01 = np.percentile(img_array, 1)
        img_array = img_array + np.abs(scale_val01)
        scale_val99 = np.percentile(img_array, 99)
        img_array_rescaled = img_array/np.abs(scale_val99)

        img_array_padded = img_array_rescaled
        
        # load in img as a tensor and make sure that it is 4D 
        img_tensor = torch.tensor(img_array_padded, dtype=torch.float32)
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor, base_name  # for dice loss with bonus