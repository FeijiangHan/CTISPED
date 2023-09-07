# AANet
AANet: Artery-Aware Network for Pulmonary Embolism Detection in CTPA Images, MICCAI2022

# Requirements
pytoch==1.1.0 or 1.3.1 *Newer version may need code adaptation.*

simpleitk==1.2.4

tensorboardX

scikit-image

scikit-learn

tqdm

pandas


# Data Prepare:
1.	The vessel masks and lung masks are already open sourced in:
AANet/PEData/CAD_PE_data/vessel
The vessel masks are named like 001.nii.gz. The lung masks are named like 001_lungmask.nii.gz.
Feel Free to use our vessel and lung annotation for your work. Just remember to quote us!

2.	Download CAD-PE dataset from https://ieee-dataport.org/open-access/cad-pe
Put CTPA images, e.g. 001.nrrd, in ‘AANet/PEData/CAD_PE_data/image’.
Put PE labels, e.g. 001RefStd.nrrd, in ‘AANet/PEData/CAD_PE_data/label’.

3.	Run nifty_preprocess.py to preprocess data. The code will use lung masks to crop the lung region ROI in the CTPA images and labels, and save them back to nifty file in ‘./PEData/processed_itk’. The image size will be smaller for faster loading in training.

# Training and Inference:
Run train_aanet.py for training. Use the argument –unique_name to give a name. At the end of training, the code calls the inference.py automatically, and save the result in ‘AANet/pred_itk/unique_name_sth50’. 
You may want to train several times with different seeds. The official evaluation protocol of CAD-PE is a little bit unreasonable. If two near ground-truth PEs are detected by one connected predicted PE, only one ground-truth is counted as TP, and vice-versa. Therefore, small randomness, that causes two near PEs connected to one or one PE breaked to two, will cause relatively big difference in FROC curve.

# Evaluation:
Run evaluation.py to evaluate and plot FROC curve. Change pred_root in line 279 to the directory of your result, i.e. ‘AANet/pred_itk/unique_name_sth50’.

The guidance and the code may still have some small errors. Feel free to contact me by email or issue.





