#!/bin/sh
# python test_script.py "test1" "test2"

#pip install nibael
#pip install intensity-normalization

# input file, skullStripped[y/n], output file
# outputFile="/mnt/mainErnie/deep_learning_segmentation/nbm/code/singularity_tests/test_output_files/M0164_seg.nii"
# outputFile="/mnt/mainErnie/deep_learning_segmentation/nbm/data/testing_scans/preventad/sub-1000173/ses-NAPBL00/anat/wrsub-1000173_ses-NAPBL00_run-001_T1w_reg_seg.nii"
# inputFile="/mnt/mainErnie/deep_learning_segmentation/nbm/code/singularity_tests/test_input_files/M0164_3T_T1w.nii"
# inputFile="/mnt/mainErnie/deep_learning_segmentation/nbm/data/testing_scans/preventad/sub-1000173/ses-NAPBL00/anat/wrsub-1000173_ses-NAPBL00_run-001_T1w_reg.nii"
# outputDir=${outputFile%/*}"/"

# subjs=("P0309" "P0365" "P0420" "P0440" "P0448" "P0455" "P0490" "P0508" "P0520" "P0522" "P0566" "P0572" "P0646")

# subjs=("P0139-1")

# skullStripInput="n"

# rootDataFolder="/mnt/mainErnie/deep_learning_segmentation/nbm/data/UPennNN/"

# userInputFile="rP0139-1_3T_T1w_reg_sform.nii"
userInputFile=$1
skullStripInput=$2

inputFile=$userInputFile
echo $inputFile
outputFile="${inputFile//.nii/_seg.nii}"
outputDir=${outputFile%/*}"/"


echo "----------------------------------- Reslicing to 1mm isotropic ------------------------"
inputFileName="${inputFile##*/}"
inputFileResliced="${inputFileName//.nii/_resliced.nii}"
inputResliced=$outputDir$inputFileResliced
python3.6 /reslice_to_1mm.py "$inputFile" "$inputResliced"


echo "----------------------------------- Converting To RAS ---------------------------------"
inputFileRAS="${inputFileResliced//.nii/_RAS.nii}"
inputRAS=$outputDir$inputFileRAS
python3.6 /convert_to_RAS.py "$inputResliced" "$inputRAS"


echo "----------------------------------- Skull Stripping -----------------------------------"
if [ "$skullStripInput" == "y" ] || [ "$skullStripInput" == "Y" ]
then
	echo "Skipping skull stripping"
	skullFile=$inputRAS
	brainFile=$outputDir"${inputFileRAS//.nii/_brain.nii}"
	cp $inputFile $brainFile
else
	echo "Performing skull stripping"
	skullFile=$inputRAS
	brainFile=$outputDir"${inputFileRAS//.nii/_brain.nii}"
	/ROBEX/runROBEX.sh $skullFile $brainFile
fi


echo "----------------------------------- Normalizing Image ---------------------------------"
inputNorm="${brainFile//.nii/_norm.nii}"
python3.6 /normalize_images.py "/standard_histogram.npy" "$brainFile" "$inputNorm"


echo "----------------------------------- Running sclimbic ----------------------------------"
inputSclimbic="${inputNorm//_norm.nii/_norm_sclimbic.nii}"
/sclimbic/bin/mri_sclimbic_seg --i $inputNorm --o $inputSclimbic --threads 4


echo "----------------------------------- Extracting Bounding Box ---------------------------"
python3.6 /extract_bounding_box.py "$inputNorm" "$outputDir"


echo "----------------------------------- Running Model -------------------------------------"
inputBox="${inputNorm//.nii/_box.nii}"
python3.6 /model_prediction.py "$inputBox" "$outputDir"


echo "----------------------------------- Converting to full img ----------------------------"
outputPred="${inputBox//.nii/_pred.nii}"
python3.6 /bounding_box_to_full_img.py "$outputPred" "$inputSclimbic" "$outputDir" "$outputFile"


echo "----------------------------------- Converting to L/R labels --------------------------"
outputPred_L="${outputFile//.nii/_L.nii}"
outputPred_R="${outputFile//.nii/_R.nii}"
python3.6 /split_labels_to_LR.py "$outputFile" "$outputPred_L" "$outputPred_R"


# clean up files
if [ -f $inputNorm ]
then
	# echo "removing "$inputNorm
	rm $inputNorm
fi

if [ -f $inputSclimbic ]
then
	# echo "removing "$inputSclimbic
	rm $inputSclimbic
fi

if [ -f $inputBox ]
then
	# echo "removing "$inputBox
	rm $inputBox
fi

# sclimbic log
sclimbicLog=$outputDir"mri_sclimbic.log"
if [ -f $sclimbicLog ]
then
	# echo "removing "$sclimbicLog
	rm $sclimbicLog
fi
