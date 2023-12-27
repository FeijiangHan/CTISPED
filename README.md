
<div class="column" align="middle">
  <p align="center">
  </p>
  </a>
    <img src="https://img.shields.io/badge/License-Apache%202.0-red.svg" alt="license"/>
    <img src="https://img.shields.io/badge/Language-Python-blue.svg" alt="language"/>
  <img src="https://img.shields.io/badge/platform-MacOS-white.svg" alt="macos"/>
  <img src="https://img.shields.io/badge/platform-Linux-9cf.svg" alt="linux"/>

<h5 align="center">If you are interested in This project, please kindly give Me a triple `Star`, `Fork` and `Watch`, Thanks!</h5>
</div>

# CTISPED: CT Image Segmentation for Pulmonary Embolism DiagnosisCTISPED

Pulmonary embolism is a lung condition most commonly caused by blood clots, also known as thromboembolism. By analyzing CT scans of the lungs, doctors can identify clots in the pulmonary vessels to facilitate timely treatment. The goal of this project is to assist physicians in detecting areas of pulmonary embolism using deep learning, reducing their workload and improving detection accuracy.

Both 2D CT slices (.dcm) and 3D CT volumes (.nii) can be used to identify pulmonary embolism. Segmenting the nii volumes slice-by-slice generates the dcm images, so these two formats can be interconverted. Alternatively, detecting embolism directly from 3D vascular models extracts the key information without needing full volume segmentation.

For image processing, convolutional neural networks were chosen for their efficiency and performance with medical images. For segmentation, UNet architectures are widely used, and various 2D UNet models have been implemented in the Unet Family folder. The 3D AANet model in the AAnet folder focuses on the pulmonary arteries. Additionally, YOLOv4 in the YOLO folder identifies lung region ROIs to reduce the segmentation search space.

**To address class imbalance, the following strategies were employed:**

1. Dice Loss and Focal Loss are combined to improve detection accuracy of small targets.  
2. Lung segmentation and cropping to reduce background and indirectly increase target ratios.
3. Data augmentation via strategic oversampling and synthesis based on target ratio analysis. (concatenating low ratio samples and replicating targets)
4. YOLO for ROI extraction, ensemble Tiny-Attention-Unet for segmentation, and merging the outputs.   
5. Utilizing techniques like lightweight samples, sliding averages, and cosine annealing.

For more information please see the Unets-2D README

- [CTISPED: CT Image Segmentation for Pulmonary Embolism DiagnosisCTISPED](#ctisped-ct-image-segmentation-for-pulmonary-embolism-diagnosisctisped)
- [UNETS-2D](#unets-2d)
  - [Dataset](#dataset)
  - [Features](#features)
  - [Usage](#usage)
  - [Code Structure](#code-structure)
- [AANET-3D](#aanet-3d)
  - [Model Architecture](#model-architecture)
  - [Training Pipeline](#training-pipeline)
  - [Usage](#usage-1)
  - [Output](#output)
- [YOLO Object Detection Python Wrapper](#yolo-object-detection-python-wrapper)
  - [Features](#features-1)
  - [Usage](#usage-2)
  - [Implementation](#implementation)
- [License](#license)


# UNETS-2D
This code implements training and evaluation of various semantic segmentation models on medical CT image datasets using PyTorch. 

## Dataset
* Dataset Description: The current dataset FUMPE (representing Ferdowsi University of Mashhad's PE dataset) consists of computed tomography angiography (CTA) images of pulmonary embolism (PE) from 35 different patients. Two radiologist experts provided the ground truths with the advantage of a semi-automated image processing software tool.


* Data Source: https://tianchi.aliyun.com/dataset/90713, Masoudi, M. et al. A new dataset of computed-tomography angiography images for computer-aided detection of pulmonary embolism. Sci. Data 5:180180 doi: 10.1038/sdata.2018.178 (2018). DOI: https://doi.org/10.6084/m9.figshare.c.4107803.v1  


* Data Processing Logic: Please see https://github.com/FeijiangHan/CTISPED/tree/main/Unets-2D/data_process, where I implemented the processing and augmentation of the original dataset.

## Features
- Provides clear parameter selection interaction: Configure dataset path, model, learning rate, loss function, momentum etc. via command line arguments


- Supports various semantic segmentation network architectures: UNet, Tin Unet, R2UNet, Attention UNet, Nested UNet etc.
- Provides multiple loss functions like cross entropy loss, Dice loss, Focal loss and combined loss.  

- Supports model checkpointing and resuming training.


- Contains complete training and validation loop with metrics visualization.


- Logs training process using TensorBoard.


- Calculates common evaluation metrics for semantic segmentation and visualizes them.


- Enables CUDA multi-GPU training using PyTorch DataParallel.

## Usage

`python train.py`

**Main arguments:**

- `--model`: Model architecture, UNet by default
- `--loss`: Loss function, Dice loss by default
- `--dataset`: Dataset, lung CT by default
- `--epoch`: Number of training epochs, 300 by default
- `--batch_size`: Batch size, 2 by default
- `--lr`: Learning rate, 1e-4 by default
- `--checkpoint`: Path to pretrained model checkpoint
- `--gpu`: GPU 
- `--parallel`: Enable multi-GPU training

The code will automatically log TensorBoard events, save model checkpoints.

## Code Structure

- `models/`: Different model architectures
- `datasets/`: Dataset loaders
- `utils/`: Utility functions like metrics calculation

**Main workflow:**

1. Parse arguments, load data
2. Build model, define optimizer and loss
3. Training loop: forward pass, backprop, optimize
4. Calculate metrics, log TensorBoard
5. Save model checkpoints

# AANET-3D

## Model Architecture

The model architecture is AANet, which contains encoding and decoding blocks with skip connections. Batch normalization is synchronized across GPUs.

## Training Pipeline

- Loads lung CT scans and segmentation masks as training data
- Applies data augmentation like random cropping, flipping, rotation
- Defines AANet model, Dice + Tversky loss, AdamW optimizer
- Training loop with forward/backward passes, optimization, learning rate scheduling
- Evaluates validation metrics like F1 score
- Logs training metrics, validation metrics using TensorBoard
- Saves model checkpoints during training
- Tests trained model on test set and generates lung segmentation

## Usage

Train model:

`python train.py --epochs 500 --batch_size 8`

Main arguments:

- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--learn_rate`: Learning rate
- `--log_path`: TensorBoard log directory

The code supports multi-GPU training via PyTorch DataParallel.

## Output

- Trained model checkpoints saved to `save_models/`
- TensorBoard logs written to `log_path`
- Segmentation results on test set stored as PNGs

# YOLO Object Detection Python Wrapper

This code provides a Python wrapper for performing object detection using YOLO models. It interfaces with the YOLO DLL to run detection on images.

## Features

- Loads YOLO network config, weights and metadata
- Runs detection on images, returns bounding boxes and labels
- Supports batch detection on multiple images
- Displays detection results on images
- Handles CPU or GPU mode automatically

## Usage

Run detection on an image:

```python
from darknet_python import performDetect

boxes = performDetect(imagePath="dog.jpg")
```

Perform batch detection on multiple images:

```python
from darknet_python import performBatchDetect 

batch_boxes = performBatchDetect(img_list=['img1.jpg', 'img2.jpg'])
```

Main parameters:

- `imagePath` - Path to input image
- `configPath`, `weightPath`, `metaPath` - Paths to YOLO files
- `thresh` - Detection threshold
- `showImage` - Show output image

## Implementation

The Python wrapper calls into the YOLO DLL to run detection. Main steps:

- Load network, metadata and image
- Pass image through network for predictions
- Parse predictions into boxes, scores and labels
- Apply NMS thresholding to boxes
- Draw boxes on image


# License
This project is licensed under the [Apache License, Version 2.0](LICENSE).
