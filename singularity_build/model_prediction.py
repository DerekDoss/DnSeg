# script to combine predictions from all 5 models

import sys
import nibabel as nib
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
import glob
import pandas as pd

sys.path.append('/')

from data_singularity import NBM_dataset
from unet3d_adaptive import Unet_adaptive

def runTrainedModel(trainedModelPath, postprocess_raw_folder, out_dir, testFileNames):
    
    model = torch.load(trainedModelPath, map_location=torch.device('cpu'))
    
    postprocess_dataset = NBM_dataset(postprocess_raw_folder, testFileNames)
    postprocess_dataloader = DataLoader(postprocess_dataset, 1, shuffle=True, num_workers=1, pin_memory=True)

    with torch.no_grad():
        model.eval()
        count = 0
        len_dataset = postprocess_dataset.__len__()
        
        for batch_index, (data, name) in enumerate(postprocess_dataloader):
            output, model_notes = model(data)

            count = count + 1
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

    pred_nifti = pred_LR.numpy().squeeze()
    pred_nifti_1_cropped = pred_nifti
    pred_nifti_1_nii = nib.nifti1.Nifti1Image(pred_nifti_1_cropped, img_nii.affine, img_nii.header)
    nib.save(pred_nifti_1_nii, out_dir + fname.replace(".nii","_pred.nii"))
  

def main():
   
    # filename that you're running the model on 
    runFileName = sys.argv[1]
    runFolder = sys.argv[2]
    
    runFileName_noDirectory = runFileName.split("/")[-1]
    
    num_models = 5 
    percentile = 0.5  
    
    
    all_pred_L_torch = torch.zeros(64,64,64)
    all_pred_R_torch = torch.zeros(64,64,64)
    for i in range(num_models):     
        trainedModelPath = "/dnseg_model"+str(i)+".pt"
        pred = runTrainedModel(trainedModelPath, runFolder, runFolder, [runFileName])
        
        pred_L_thresholded = thresholdWithPercentile(pred[0,:,:,:], percentile).type(torch.int32)
        pred_R_thresholded = thresholdWithPercentile(pred[1,:,:,:], percentile).type(torch.int32)
        all_pred_L_torch = all_pred_L_torch + pred_L_thresholded
        all_pred_R_torch = all_pred_R_torch + pred_R_thresholded
        
    all_pred_L_torch_thresh = all_pred_L_torch>=3
    all_pred_L_torch_thresh = all_pred_L_torch_thresh.type(torch.int32)
    all_pred_R_torch_thresh = all_pred_R_torch>=3
    all_pred_R_torch_thresh = all_pred_R_torch_thresh.type(torch.int32)
      
    pred_LR = all_pred_L_torch_thresh + 2*all_pred_R_torch_thresh
    
    saveNiftiFile(runFolder, runFolder, runFileName_noDirectory, pred_LR)
        

if __name__ == '__main__':
    main()
