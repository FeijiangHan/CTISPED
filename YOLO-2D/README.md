# YOLO

On the following figure is visualized the detection of pancreas on a NIH (a) and Decathlon (b) slice, respectively. The green color represents the ground-truth and the purple the YOLO prediction.

![successful_pancreas_detection](https://user-images.githubusercontent.com/30274421/111903314-631d2700-8a4a-11eb-9beb-ffcdfd85c2a4.png)


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

Supports both single image and batched input. Handling for CPU vs GPU mode.

Provides simple interface for running YOLO detection from Python scripts. Builds on YOLO DLL backend.


[1] H. R. Roth, A. Farag, E. B. Turkbey, L. Lu, J. Liu, and R. M. Summers, “Data from pancreas-ct.” The Cancer Imaging Archive, 2016. [Online]. Available: https://doi.org/10.7937/K9/TCIA.2016.TNB1kqBU 

[2] H. R. Roth, L. Lu, A. Farag, H.-C. Shin, J. Liu, E. B. Turkbey, and R. M. Summers, “Deeporgan: Multi-level deep convolutional networks for automated pancreas segmentation,” in International conference on medical image computing and computer-assisted intervention. Springer, 2015, pp. 556–564.

[3] K. Clark, B. Vendt, K. Smith, J. Freymann, J. Kirby, P. Koppel, S. Moore, S. Phillips, D. Maffitt,
M. Pringle et al., “The cancer imaging archive (tcia): maintaining and operating a public information repository,” Journal of digital imaging, vol. 26, no. 6, pp. 1045–1057, 2013. 

[4] B. M. Dawant, R. Li, B. Lennon, and S. Li, “Semi-automatic segmentation of the liver and its
evaluation on the miccai 2007 grand challenge data set,” 3D Segmentation in The Clinic: A Grand
Challenge, pp. 215–221, 2007.
