# DnSeg
Deep Learning Segmentation of the Nucleus Basalis of Meynert

# License
This code is licensed under Apache 2.0.
If you use it, please cite the paper: TBD, submitted to preprint and a journal.

# Dependencies (the singlarity can take care of all of the dependencies)
- Singularity 3.9.9 (https://github.com/sylabs/singularity/releases) (problems arise when using singularity 3.10.x)
- ScLimbic (https://surfer.nmr.mgh.harvard.edu/fswiki/ScLimbic)
- Robex v12 (https://www.nitrc.org/projects/robex/)
- python 3.6
- python opencv
- python torchio
- python pytorch
- python intensity-normalization

# How to use
There are two options on how to use DnSeg. You can either build it yourself or download a prebuilt singularity.

Option 1: Download the prebuilt singularity
- Install singularity 3.9.9 (https://github.com/sylabs/singularity/releases)
- Download DnSeg singularity file (www.derekdoss.com/pub/dnseg/DnSeg_v1.0.sif) note: the file is too large for github, thus it is hosted on my personal website
- Place the nifti file you want to segment in a folder on your machine
- Run the following command "singularity exec --bind /path/to/folder/containing/nifti/:/inputs DnSeg_v1.0.sif bash /singularity_DnSeg.sh /inputs/nifti_file.nii"

Option 2: Build the singularity file yourself
- Clone the singularity_build directory from this repo
- Install singularity 3.9.9 (https://github.com/sylabs/singularity/releases)
- Run the following command in the singularity_build directory to build the singularity "sudo singularity build DnSeg.sif DnSeg.def"
- The .def file will automatically download the remaining necessary files
- Place the nifti file you want to segment in a folder on your machine
- Run the following command "singularity exec --bind /path/to/folder/containing/nifti/:/inputs DnSeg_v1.0.sif bash /singularity_DnSeg.sh /inputs/nifti_file.nii"
