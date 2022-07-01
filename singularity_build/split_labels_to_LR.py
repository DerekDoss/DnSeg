import sys
from intensity_normalization.normalize.nyul import NyulNormalize
import nibabel as nib
import numpy as np

bothLabelsFilepath = sys.argv[1]
L_labelFilepath = sys.argv[2]
R_labelFilepath = sys.argv[3]


bothLabels_img = nib.load(bothLabelsFilepath)
bothLabels_np = bothLabels_img.get_fdata()

L_label_np = bothLabels_np==1
L_label_np = L_label_np.astype('int')
R_label_np = bothLabels_np==2
R_label_np = R_label_np.astype('int')

L_label_img = nib.Nifti1Image(L_label_np, bothLabels_img.affine, bothLabels_img.header)
nib.save(L_label_img, L_labelFilepath)

R_label_img = nib.Nifti1Image(R_label_np, bothLabels_img.affine, bothLabels_img.header)
nib.save(R_label_img, R_labelFilepath)