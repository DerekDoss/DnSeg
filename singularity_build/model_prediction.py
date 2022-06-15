# from __future__ import print_function, division

import sys

pathList = sys.path
if any("dossdj" in string for string in pathList):
    sys.path.remove('/home/dossdj/.local/lib/python3.6/site-packages')

import nibabel as nib

# print(sys.argv[1])
# print(sys.argv[2])

# script to combine predictions from all 5 models

import torch
import numpy as np
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
# import nibabel as nib
import glob
import pandas as pd

# import torch
# import torch.nn as nn

import cProfile

sys.path.append('/')

from data_singularity import NBM_dataset
from unet3d_adaptive import Unet_adaptive


# import os
# import torch
# import pandas as pd
# from torch.utils.data import Dataset
# import nibabel as nib
# import torch.nn.functional as F
# import numpy as np

# class NBM_dataset(Dataset):
    # def __init__(self, raw_imgs_dir, fileNames):

        # self.raw_imgs_dir = raw_imgs_dir
        # # self.raw_img_files = os.listdir(raw_imgs_dir)
        # self.fileNames = fileNames
        # # self.label_imgs_dir = label_imgs_dir

    # def get_script_filename(self):
        # return __file__

    # def __len__(self):
        # return len(self.fileNames)

    # def __getitem__(self, idx):

        # if torch.is_tensor(idx):
            # idx = idx.tolist()

        # base_name = self.fileNames[idx].replace('img', '')


        # img_name = os.path.join(self.raw_imgs_dir, self.fileNames[idx])
        # img_nii = nib.load(img_name)
        # img_array = img_nii.get_data()

        # # Rescale 1-99 percentile values to ~[0-1]
        # scale_val01 = np.percentile(img_array, 1)
        # img_array = img_array + np.abs(scale_val01)
        # scale_val99 = np.percentile(img_array, 99)
        # img_array_rescaled = img_array/np.abs(scale_val99)

        # # # Pad out to 68x68x68
        # # img_array_padded = np.zeros((68, 68, 68))
        # # img_array_padded[2:66, 2:66, 2:66] = img_array_rescaled
        # img_array_padded = img_array_rescaled
        
        # # load in img as a tensor and make sure that it is 4D 
        # img_tensor = torch.tensor(img_array_padded, dtype=torch.float32)
        # img_tensor = img_tensor.unsqueeze(0)
        
        # return img_tensor, base_name  # for dice loss with bonus


# # Kernel_size must be odd int >= 3
# def double_conv(in_c, out_c, kernel_size):
    # conv = nn.Sequential(
        # nn.Conv3d(in_c, out_c, kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1)/2)),
        # # nn.BatchNorm3d(out_c),
        # nn.ReLU(inplace=True),
        # nn.Conv3d(out_c, out_c, kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1)/2)),
        # nn.ReLU(inplace=True)
    # )
    # return conv


# def crop_img(tensor, target_tensor):
    # target_size = target_tensor.size()[2]
    # tensor_size = tensor.size()[2]
    # delta = tensor_size - target_size
    # delta = delta // 2
    # return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta, delta:tensor_size - delta]


# class Unet_adaptive(nn.Module):

    # def get_script_filename(self):
        # return __file__

    # def __init__(self, num_filters, depth, kernel_size, verbose_bool=False):
        # super(Unet_adaptive, self).__init__()

        # self.num_filters = num_filters
        # self.kernel_size = kernel_size
        # self.depth = depth
        # self.verbose = verbose_bool

        # kernel_size_pool = kernel_size - 1
        # padding_pool = int((kernel_size_pool - 2) / 2)


        # if depth == 1:

            # self.max_pool_2x2 = nn.MaxPool3d(kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            # self.down_conv_1 = double_conv(1, num_filters * 1, kernel_size)
            # self.down_conv_2 = double_conv(num_filters * 1, num_filters * 2, kernel_size)

            # self.up_trans_1 = nn.ConvTranspose3d(in_channels=num_filters * 2, out_channels=num_filters * 1, kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            # self.up_conv_1 = double_conv(num_filters * 2, num_filters * 1, kernel_size)

        # elif depth == 2:

            # self.max_pool_2x2 = nn.MaxPool3d(kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            # self.down_conv_1 = double_conv(1, num_filters * 1, kernel_size)
            # self.down_conv_2 = double_conv(num_filters * 1, num_filters * 2, kernel_size)
            # self.down_conv_3 = double_conv(num_filters * 2, num_filters * 4, kernel_size)

            # self.up_trans_2 = nn.ConvTranspose3d(in_channels=num_filters * 4, out_channels=num_filters * 2, kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            # self.up_conv_2 = double_conv(num_filters * 4, num_filters * 2, kernel_size)
            # self.up_trans_1 = nn.ConvTranspose3d(in_channels=num_filters * 2, out_channels=num_filters * 1, kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            # self.up_conv_1 = double_conv(num_filters * 2, num_filters * 1, kernel_size)

        # elif depth == 3:

            # self.max_pool_2x2 = nn.MaxPool3d(kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            # self.down_conv_1 = double_conv(1, num_filters * 1, kernel_size)
            # self.down_conv_2 = double_conv(num_filters * 1, num_filters * 2, kernel_size)
            # self.down_conv_3 = double_conv(num_filters * 2, num_filters * 4, kernel_size)
            # self.down_conv_4 = double_conv(num_filters * 4, num_filters * 8, kernel_size)

            # self.up_trans_3 = nn.ConvTranspose3d(in_channels=num_filters * 8, out_channels=num_filters * 4, kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            # self.up_conv_3 = double_conv(num_filters * 8, num_filters * 4, kernel_size)
            # self.up_trans_2 = nn.ConvTranspose3d(in_channels=num_filters * 4, out_channels=num_filters * 2, kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            # self.up_conv_2 = double_conv(num_filters * 4, num_filters * 2, kernel_size)
            # self.up_trans_1 = nn.ConvTranspose3d(in_channels=num_filters * 2, out_channels=num_filters * 1, kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            # self.up_conv_1 = double_conv(num_filters * 2, num_filters * 1, kernel_size)

        # elif depth == 4:
            # self.max_pool_2x2 = nn.MaxPool3d(kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            # self.down_conv_1 = double_conv(1, num_filters * 1, kernel_size)
            # self.down_conv_2 = double_conv(num_filters * 1, num_filters * 2, kernel_size)
            # self.down_conv_3 = double_conv(num_filters * 2, num_filters * 4, kernel_size)
            # self.down_conv_4 = double_conv(num_filters * 4, num_filters * 8, kernel_size)
            # self.down_conv_5 = double_conv(num_filters * 8, num_filters * 16, kernel_size)

            # self.up_trans_4 = nn.ConvTranspose3d(in_channels=num_filters * 16, out_channels=num_filters * 8, kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            # self.up_conv_4 = double_conv(num_filters * 16, num_filters * 8, kernel_size)
            # self.up_trans_3 = nn.ConvTranspose3d(in_channels=num_filters * 8, out_channels=num_filters * 4, kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            # self.up_conv_3 = double_conv(num_filters * 8, num_filters * 4, kernel_size)
            # self.up_trans_2 = nn.ConvTranspose3d(in_channels=num_filters * 4, out_channels=num_filters * 2, kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            # self.up_conv_2 = double_conv(num_filters * 4, num_filters * 2, kernel_size)
            # self.up_trans_1 = nn.ConvTranspose3d(in_channels=num_filters * 2, out_channels=num_filters * 1, kernel_size=kernel_size_pool, stride=2, padding=padding_pool)
            # self.up_conv_1 = double_conv(num_filters * 2, num_filters * 1, kernel_size)

        # self.out = nn.Sequential(
            # nn.Conv3d(in_channels=num_filters * 1, out_channels=3, kernel_size=1),
            # nn.Sigmoid()
        # )

    # def forward(self, image):

        # if self.depth == 1:
            # # encoder
            # x1 = self.down_conv_1(image)
            # x2 = self.max_pool_2x2(x1)
            # x3 = self.down_conv_2(x2)

            # y3 = self.up_trans_1(x3)
            # z3 = crop_img(x1, y3)
            # y2 = self.up_conv_1(torch.cat([y3, z3], 1))

            # out = self.out(y2)

            # if self.verbose:
                # print("x1: " + str(x1.size()))
                # print("x2: " + str(x2.size()))
                # print("x3: " + str(x3.size()))

                # print("y3: " + str(y3.size()))
                # print("z3: " + str(z3.size()))
                # print("y2: " + str(y2.size()))
                # print("out: " + str(out.size()))

            # # Build the notes output string
            # notes = "Depth: 1, Kernel size: " + str(self.kernel_size) + ", Num_features: " + str(self.num_filters * 1) + ", " + str(self.num_filters * 2) + \
                    # " Input size: " + str(list(image.size())) + ", Max depth size: " + str(list(x3.size())) + ", Output size: " + str(list(out.size()))

            # return out, notes

        # elif self.depth == 2:
            # # encoder
            # x1 = self.down_conv_1(image)
            # x2 = self.max_pool_2x2(x1)
            # x3 = self.down_conv_2(x2)
            # x4 = self.max_pool_2x2(x3)
            # x5 = self.down_conv_3(x4)

            # y5 = self.up_trans_2(x5)
            # z5 = crop_img(x3, y5)
            # y4 = self.up_conv_2(torch.cat([y5, z5], 1))
            # y3 = self.up_trans_1(y4)
            # z3 = crop_img(x1, y3)
            # y2 = self.up_conv_1(torch.cat([y3, z3], 1))

            # out = self.out(y2)

            # if self.verbose:
                # print("x1: " + str(x1.size()))
                # print("x2: " + str(x2.size()))
                # print("x3: " + str(x3.size()))
                # print("x4: " + str(x4.size()))
                # print("x5: " + str(x5.size()))

                # print("y5: " + str(y5.size()))
                # print("z5: " + str(z5.size()))
                # print("y4: " + str(y4.size()))
                # print("y3: " + str(y3.size()))
                # print("z3: " + str(z3.size()))
                # print("y2: " + str(y2.size()))
                # print("out: " + str(out.size()))

            # # Build the notes output string
            # notes = "Depth: 2, Kernel size: " + str(self.kernel_size) + ", Num_features: " + str(self.num_filters * 1) \
                    # + ", " + str(self.num_filters * 2) + ", " + str(self.num_filters * 4) + \
                    # " Input size: " + str(list(image.size())) + ", Max depth size: " + str(list(x5.size())) + ", Output size: " + str(list(out.size()))

            # return out, notes

        # elif self.depth == 3:
            # # encoder
            # x1 = self.down_conv_1(image)
            # x2 = self.max_pool_2x2(x1)
            # x3 = self.down_conv_2(x2)
            # x4 = self.max_pool_2x2(x3)
            # x5 = self.down_conv_3(x4)
            # x6 = self.max_pool_2x2(x5)
            # x7 = self.down_conv_4(x6)

            # y7 = self.up_trans_3(x7)
            # z7 = crop_img(x5, y7)
            # y6 = self.up_conv_3(torch.cat([y7, z7], 1))
            # y5 = self.up_trans_2(y6)
            # z5 = crop_img(x3, y5)
            # y4 = self.up_conv_2(torch.cat([y5, z5], 1))
            # y3 = self.up_trans_1(y4)
            # z3 = crop_img(x1, y3)
            # y2 = self.up_conv_1(torch.cat([y3, z3], 1))

            # out = self.out(y2)

            # if self.verbose:
                # print("x1: " + str(x1.size()))
                # print("x2: " + str(x2.size()))
                # print("x3: " + str(x3.size()))
                # print("x4: " + str(x4.size()))
                # print("x5: " + str(x5.size()))
                # print("x6: " + str(x6.size()))
                # print("x7: " + str(x7.size()))

                # print("y7: " + str(y7.size()))
                # print("z7: " + str(z7.size()))
                # print("y6: " + str(y6.size()))
                # print("y5: " + str(y5.size()))
                # print("z5: " + str(z5.size()))
                # print("y4: " + str(y4.size()))
                # print("y3: " + str(y3.size()))
                # print("z3: " + str(z3.size()))
                # print("y2: " + str(y2.size()))
                # print("out: " + str(out.size()))

            # # Build the notes output string
            # notes = "Depth: 3, Kernel size: " + str(self.kernel_size) + ", Num_features: " + str(self.num_filters * 1) \
                    # + ", " + str(self.num_filters * 2) + ", " + str(self.num_filters * 4) + ", " + str(self.num_filters * 8) + \
                    # " Input size: " + str(list(image.size())) + ", Max depth size: " + str(list(x7.size())) + ", Output size: " + str(list(out.size()))

            # return out, notes

        # elif self.depth == 4:
            # # encoder
            # x1 = self.down_conv_1(image)
            # x2 = self.max_pool_2x2(x1)
            # x3 = self.down_conv_2(x2)
            # x4 = self.max_pool_2x2(x3)
            # x5 = self.down_conv_3(x4)
            # x6 = self.max_pool_2x2(x5)
            # x7 = self.down_conv_4(x6)
            # x8 = self.max_pool_2x2(x7)
            # x9 = self.down_conv_5(x8)

            # y9 = self.up_trans_4(x9)
            # z9 = crop_img(x7, y9)
            # y8 = self.up_conv_4(torch.cat([y9, z9], 1))
            # y7 = self.up_trans_3(y8)
            # z7 = crop_img(x5, y7)
            # y6 = self.up_conv_3(torch.cat([y7, z7], 1))
            # y5 = self.up_trans_2(y6)
            # z5 = crop_img(x3, y5)
            # y4 = self.up_conv_2(torch.cat([y5, z5], 1))
            # y3 = self.up_trans_1(y4)
            # z3 = crop_img(x1, y3)
            # y2 = self.up_conv_1(torch.cat([y3, z3], 1))

            # out = self.out(y2)

            # if self.verbose:
                # print("x1: " + str(x1.size()))
                # print("x2: " + str(x2.size()))
                # print("x3: " + str(x3.size()))
                # print("x4: " + str(x4.size()))
                # print("x5: " + str(x5.size()))
                # print("x6: " + str(x6.size()))
                # print("x7: " + str(x7.size()))
                # print("x8: " + str(x8.size()))
                # print("x9: " + str(x9.size()))

                # print("y9: " + str(y9.size()))
                # print("z9: " + str(z9.size()))
                # print("y8: " + str(y8.size()))
                # print("y7: " + str(y7.size()))
                # print("z7: " + str(z7.size()))
                # print("y6: " + str(y6.size()))
                # print("y5: " + str(y5.size()))
                # print("z5: " + str(z5.size()))
                # print("y4: " + str(y4.size()))
                # print("y3: " + str(y3.size()))
                # print("z3: " + str(z3.size()))
                # print("y2: " + str(y2.size()))
                # print("out: " + str(out.size()))

            # # Build the notes output string
            # notes = "Depth: 4, Kernel size: " + str(self.kernel_size) + ", Num_features: " + str(self.num_filters * 1) + \
            # ", " + str(self.num_filters * 2) + ", " + str(self.num_filters * 4) + ", " + str(self.num_filters * 8) + ", " + str(self.num_filters * 16) + \
                    # " Input size: " + str(list(image.size())) + ", Max depth size: " + str(list(x9.size())) + ", Output size: " + str(list(out.size()))

            # return out, notes



def runTrainedModel(trainedModelPath, postprocess_raw_folder, out_dir, testFileNames):
    
    model = torch.load(trainedModelPath, map_location=torch.device('cpu'))
    
    # print("Test Files: " + postprocess_raw_folder)
    postprocess_dataset = NBM_dataset(postprocess_raw_folder, testFileNames)
    postprocess_dataloader = DataLoader(postprocess_dataset, 1, shuffle=True, num_workers=1, pin_memory=True)

    with torch.no_grad():
        model.eval()
        count = 0
        len_dataset = postprocess_dataset.__len__()
        
        for batch_index, (data, name) in enumerate(postprocess_dataloader):
            output, model_notes = model(data)

            count = count + 1
            # print("Running model " + str(count) + "/" + str(len_dataset))
            pred = output[0,1:3,:,:,:]
            return pred
            


def getSubjNamesWithXlsx(xlsxPath, all_img_dir, sheetName):
    # 
    foldSheet = pd.read_excel(xlsxPath, header=None, sheet_name=sheetName)
    valSubjs = list(foldSheet.iloc[1].dropna())
    del valSubjs[0] # remove the name of the row
    
    valFileName = []
        
    for valSubjName in valSubjs:
        for fpath in glob.iglob(all_img_dir+valSubjName+"*.nii"):
            fpath = fpath.replace("\\","/")
            fname = fpath.split("/")[-1]
            valFileName.append(fname)
            
    return valFileName

def thresholdWithPercentile(pred_torch, percentile):
    # percentile in decimal format, i.e. 0.5 = 50th percentile
    initialThresh = 0.1
    pred_thresh = pred_torch[pred_torch>initialThresh]
    if len(pred_thresh) == 0:
        pred_thresholded = pred_torch
    else:
        pred_thresh_value = torch.quantile(pred_thresh, percentile)
        pred_thresholded = pred_torch>pred_thresh_value
    return pred_thresholded.type(torch.int32)
    

def saveNiftiFile(raw_img_folder, out_dir, fname, pred_LR):
    # Load in template images
    img_nii = nib.load(raw_img_folder + fname)
    # nib.save(img_nii, out_dir+"Test_RAW_IMG_" + fname)
    
    # print(torch.unique(pred_LR))

    pred_nifti = pred_LR.numpy().squeeze()
    pred_nifti_1_cropped = pred_nifti
    pred_nifti_1_nii = nib.nifti1.Nifti1Image(pred_nifti_1_cropped, img_nii.affine, img_nii.header)
    nib.save(pred_nifti_1_nii, out_dir + fname.replace(".nii","_pred.nii"))

    # pred_nifti = pred_R.numpy().squeeze()
    # pred_nifti_1_cropped  = pred_nifti
    # pred_nifti_1_nii = nib.nifti1.Nifti1Image(pred_nifti_1_cropped, img_nii.affine, img_nii.header)
    # nib.save(pred_nifti_1_nii, out_dir + "Test_RIGHT_pred_" + fname)
    

def main():
   
    # filename that you're running the model on 
    runFileName = sys.argv[1]
    runFolder = sys.argv[2]
    
    runFileName_noDirectory = runFileName.split("/")[-1]
    
    num_models = 5 # how many models are there?
    percentile = 0.5  
    
    # print(torch.get_num_threads())
    # torch.set_num_threads(8)
    # print(torch.get_num_threads())
    
    all_pred_L_torch = torch.zeros(64,64,64)
    all_pred_R_torch = torch.zeros(64,64,64)
    for i in range(num_models):
        #print(i)
        #img_nii = nib.load(runFolder + runFileName_noDirectory)
        
        
        trainedModelPath = "/model"+str(i)+".pt"
        pred = runTrainedModel(trainedModelPath, runFolder, runFolder, [runFileName])
        
        
        
        #pred_nifti_R = pred[1,:,:,:].numpy().squeeze()
        #pred_nifti_1_nii = nib.nifti1.Nifti1Image(pred_nifti_R, img_nii.affine, img_nii.header)
        #nib.save(pred_nifti_1_nii, runFolder + runFileName_noDirectory.replace(".nii","_pred_R"+str(i)+".nii"))
        
        
        # print(torch.size(pred[1,:,:,:]))
        #print(torch.mean(pred[1,:,:,:]))
        #print(torch.median(pred[1,:,:,:]))
        #print(torch.max(pred[1,:,:,:]))
        pred_L_thresholded = thresholdWithPercentile(pred[0,:,:,:], percentile).type(torch.int32)
        pred_R_thresholded = thresholdWithPercentile(pred[1,:,:,:], percentile).type(torch.int32)
        all_pred_L_torch = all_pred_L_torch + pred_L_thresholded
        all_pred_R_torch = all_pred_R_torch + pred_R_thresholded
    all_pred_L_torch_thresh = all_pred_L_torch>=3
    all_pred_L_torch_thresh = all_pred_L_torch_thresh.type(torch.int32)
    all_pred_R_torch_thresh = all_pred_R_torch>=3
    all_pred_R_torch_thresh = all_pred_R_torch_thresh.type(torch.int32)
    
    # print(torch.unique(all_pred_L_torch_thresh))
    # print(torch.unique(all_pred_R_torch_thresh))
    
    pred_LR = all_pred_L_torch_thresh + 2*all_pred_R_torch_thresh
    
    saveNiftiFile(runFolder, runFolder, runFileName_noDirectory, pred_LR)
        

if __name__ == '__main__':
    # cProfile.run('main()')
    main()
