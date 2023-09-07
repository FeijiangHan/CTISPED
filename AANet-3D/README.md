# Requirements

pytorch==1.1.0 or 1.3.1 *Newer versions may require code adaptation*

simpleitk==1.2.4

tensorboardX

scikit-image

scikit-learn

tqdm

pandas

# Data Preparation

1. Vessel and lung masks are provided in:
   AANet/PEData/CAD_PE_data/vessel and AANet/PEData/CAD_PE_data/lung
   Filenames correspond to CT scan IDs (e.g. 001.nii.gz). Feel free to use for your research with citation.
2. Download the CAD-PE dataset from [https://ieee-dataport.org/open-access/cad-pe ↗](https://ieee-dataport.org/open-access/cad-pe)
   Place the CTPA scans (e.g. 001.nrrd) in 'AANet/PEData/CAD_PE_data/image'
   Place the PE labels (e.g. 001RefStd.nrrd) in 'AANet/PEData/CAD_PE_data/label'
3. Run nifty_preprocess.py to crop the lung ROI and downsample for faster training.
   Preprocessed data will be saved to './PEData/processed_itk'.

# Training and Inference

Run train_aanet.py to train the model. Use --unique_name to specify an ID. Inference is automatically run after training completes and saves results to 'AANet/pred_itk/unique_name_sth50'.

We recommend training with different random seeds, since the CAD-PE evaluation protocol can be sensitive to minor segmentation variations.

# Evaluation

Run evaluation.py and update pred_root (line 279) to your result directory for FROC curve calculation.
