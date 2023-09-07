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
