## Requirements

pytoch==1.1.0 or 1.3.1 *Newer version may need code adaptation.*

## reparation

1. The data I used is from Xiangya Hospital affiliated to Central South University of China, so it is private and cannot be shared publicly.
2. Public datasets like the Pulmonary Embolism Detection challenge from Alibaba Cloud Tianchi can be used for validation. The data processing scripts are designed to handle various data types and transform them into formats compatible with UNet models. See: [https://tianchi.aliyun.com/dataset/90713 ↗](https://tianchi.aliyun.com/dataset/90713) for details.

## Training and Inference

1. Run train.py to start training the model. Specify initial model hyperparameters from the command line (see train.py for details).
2. Generate predictions on new data using generate_predict.py after model training is complete.
3. Implementations of various UNet architectures are available in /models, allowing testing of different models.
4. The default learning rate is 0.001 with a warmup schedule. The Adam optimizer with momentum 0.9 is used for training.
5. Checkpoints can be saved during training and loaded later for continued training or inference.

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

Main arguments:

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

Main workflow:

1. Parse arguments, load data

2. Build model, define optimizer and loss

3. Training loop: forward pass, backprop, optimize

4. Calculate metrics, log TensorBoard

5. Save model checkpoints

The code is modular and extensible for new models and datasets. Provides complete training and evaluation framework.