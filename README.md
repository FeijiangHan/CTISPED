# CT Image Segmentation for Pulmonary Embolism Diagnosis

Pulmonary embolism is a lung condition most commonly caused by blood clots, also known as thromboembolism. By analyzing CT scans of the lungs, doctors can identify clots in the pulmonary vessels to facilitate timely treatment. The goal of this project is to assist physicians in detecting areas of pulmonary embolism using deep learning, reducing their workload and improving detection accuracy.

Both 2D CT slices (.dcm) and 3D CT volumes (.nii) can be used to identify pulmonary embolism. Segmenting the nii volumes slice-by-slice generates the dcm images, so these two formats can be interconverted. Alternatively, detecting embolism directly from 3D vascular models extracts the key information without needing full volume segmentation.

For image processing, convolutional neural networks were chosen for their efficiency and performance with medical images. For segmentation, UNet architectures are widely used, and various 2D UNet models have been implemented in the Unet Family folder. The 3D AANet model in the AAnet folder focuses on the pulmonary arteries. Additionally, YOLOv4 in the YOLO folder identifies lung region ROIs to reduce the segmentation search space.

**To address class imbalance, the following strategies were employed:**

1. Dice Loss and Focal Loss as objective functions to improve model robustness.
2. Data augmentation via strategic oversampling and synthesis based on target ratio analysis, such as concatenating low ratio samples and replicating targets. This alleviates distribution skew.
3. Lung segmentation and cropping to reduce background and indirectly increase target ratios.

Testing showed these approaches improved small target detection.

Model training incorporates an ensemble with Tiny-Attention-Residual-Unet and Attention UNet, expected to improve accuracy by 8-10%. Convergence is ensured through techniques like little sample, sliding averages, and cosine annealing.

Finally, a segmentation-detection-fusion pipeline was explored by using YOLO for ROI extraction, UNet for segmentation, and merging the outputs. This cascade could synergistically boost performance.

Overall, the techniques aim to showcase skills in medical deep learning, data augmentation, model integration, and pulmonary embolism diagnosis. Please let me know if you would like me to modify or expand any part of the description.

For more information please see the Unets-2D REAMDE

# UNETS-2D

This code implements training and evaluation of various semantic segmentation models on medical image datasets using PyTorch.

## Features

- Supports various semantic segmentation network architectures: UNet, R2UNet, Attention UNet, Nested UNet etc.
- Provides multiple loss functions like cross entropy loss, Dice loss and combined loss.
- Implements lung CT dataset for training and validation.
- Contains complete training and validation loop with metrics visualization.
- Logs training process using TensorBoard.
- Supports model checkpointing and resuming training.
- Calculates common evaluation metrics for semantic segmentation and visualizes them.
- Enables multi-GPU training using PyTorch DataParallel.

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
- `--gpu`: GPU device id
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
