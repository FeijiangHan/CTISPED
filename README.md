# CT Image Segmentation for Pulmonary Embolism Diagnosis

​	Pulmonary embolism is a lung condition most commonly caused by blood clots, also known as thromboembolism. By analyzing CT scans of the lungs, doctors can identify clots in the pulmonary vessels to facilitate timely treatment. The goal of this project is to assist physicians in detecting areas of pulmonary embolism using deep learning, reducing their workload and improving detection accuracy.

​	Both 2D CT slices (.dcm) and 3D CT volumes (.nii) can be used to identify pulmonary embolism. Segmenting the nii volumes slice-by-slice generates the dcm images, so these two formats can be interconverted. Alternatively, detecting embolism directly from 3D vascular models extracts the key information without needing full volume segmentation.

​	For image processing, convolutional neural networks were chosen for their efficiency and performance with medical images. For segmentation, UNet architectures are widely used, and various 2D UNet models have been implemented in the Unet Family folder. The 3D AANet model in the AAnet folder focuses on the pulmonary arteries. Additionally, YOLOv4 in the YOLO folder identifies lung region ROIs to reduce the segmentation search space.

**To address class imbalance, the following strategies were employed:**

1. Dice Loss and Focal Loss as objective functions to improve model robustness. 

2. Data augmentation via strategic oversampling and synthesis based on target ratio analysis, such as concatenating low ratio samples and replicating targets. This alleviates distribution skew.

3. Lung segmentation and cropping to reduce background and indirectly increase target ratios.

Testing showed these approaches improved small target detection. 

Model training incorporates an ensemble with Tiny-Attention-Residual-Unet and Attention UNet, expected to improve accuracy by 8-10%. Convergence is ensured through techniques like little sample, sliding averages, and cosine annealing.

Finally, a segmentation-detection-fusion pipeline was explored by using YOLO for ROI extraction, UNet for segmentation, and merging the outputs. This cascade could synergistically boost performance.

Overall, the techniques aim to showcase skills in medical deep learning, data augmentation, model integration, and pulmonary embolism diagnosis. Please let me know if you would like me to modify or expand any part of the description.


For more information please see the Unets-2D REAMDE