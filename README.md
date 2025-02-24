
---
license: cc0-1.0
language:
- en
tags:
- pathology
- nuclei
- computer vision
- image detection
- breast cancer
size_categories:
- 1K<n<10K
---


# NuCLS Dataset


## Table of Contents


1. [Overview](#overview)
2. [Accessing the Data](#accessing-the-data)
3. [Dataset Structure](#dataset-structure)
    - [Data Schema](#data-schema)
    - [Data Splits](#data-splits)
4. [Usage Examples](#usage-examples)
5. [Licensing](#licensing)
6. [References](#references)


## Overview

The [comprehensive dataset](https://sites.google.com/view/nucls/home?authuser=0) includes over 220,000 labeled nuclei from breast cancer slides, making it one of the largest datasets for nucleus detection. This extensive collection, annotated by pathologists and medical trainees, supports nuclear detection, classification, segmentation, and interrater analysis research. For this work, I used a subset of approximately 59,500 labeled nuclei from the corrected single-rater data.

![](https://huggingface.co/datasets/minhanhto09/NuCLS_dataset/resolve/main/Images/fig1.PNG)


## Accessing the Data


You can load the NuCLS dataset using the `datasets` library in Python. Depending on your needs, you can choose to load the full dataset or its smaller subset.

To load the full dataset:

```python
from datasets import load_dataset
dataset = load_dataset("minhanhto09/NuCLS_dataset", name="default")
```
To see a smaller subset of the dataset:

```python
from datasets import load_dataset
dataset = load_dataset("minhanhto09/NuCLS_dataset", name="debug")
```

## Dataset Structure


### Data Schema

The [Corrected Single-Rater Dataset](https://sites.google.com/view/nucls/single-rater?authuser=0) is a collection of 1,744 entries, each with an associated Field of View (FOV) image, mask image, visualization image, and a list of nuclei annotation coordinates, comprising 1,744 complete sets. In total, there are 59,485 nuclei annotations. Each image is rendered at a resolution of 0.2 microns-per-pixel, with all annotation coordinates provided in pixel units to correspond with this resolution.

![](https://huggingface.co/datasets/minhanhto09/NuCLS_dataset/resolve/main/Images/fig3.jpeg)

A single dataset entry contains the following details:

- `file_name`: A unique filename that encodes the most relevant information about each example and its associated data.
![](https://huggingface.co/datasets/minhanhto09/NuCLS_dataset/resolve/main/Images/fig2.png)

- `rgb_image`: A high-resolution RGB image of breast cancer tissue.

- `mask_image`: A mask image with each nucleus labeled. Class labels are encoded in the first channel. The second and third channels are used to create a unique identifier for each nucleus. 

- `visualization_image`: A visualization image that overlays the RGB and mask images to assist in interpretability.

- `annotation_coordinates`: Each instance comprises a list of annotations for the nuclei, with each annotation encompassing:

    - `raw_classification`: The base category of the nucleus (13 classes such as 'tumor' or 'lymphocyte').
    
    - `main_classification`: A higher-level category of the nucleus(7 classes including 'tumor_mitotic' and 'nonTILnonMQ_stromal').
    
    - `super_classification`: The broadest category label for the nucleus (4 options including 'sTIL' or 'nonTIL_stromal').
    
    - `type`: The form of annotation used ('rectangle' or 'polyline').
    
    - `xmin`, `ymin`, `xmax`, `ymax`: The bounding box coordinates indicating the extent of the nucleus.
    
    - `coords_x`, `coords_y`: The specific boundary coordinates of the nucleus.

### Data Split

The dataset is divided into six folds, each with its own training and testing set. This division is based on the source hospital to capture variability in medical imaging practices and ensure that models trained on the dataset can generalize well across different institutions. Smaller folds, such as `train_fold_999` and `test_fold_999`, are used specifically for debugging due to their limited number of examples.

## Usage Example

### Introduction

This repository focuses on the task of detecting nuclei in images with high accuracy and efficiency. This task has significant applications in medical imaging, particularly in cancer diagnosis, where identifying and analyzing nuclei is crucial for understanding tumor morphology and guiding treatment decisions.

To achieve this, I conducted a comprehensive review of various object detection models, including **Mask R-CNN**, **Faster R-CNN**, and **YOLOv5**, evaluating their suitability for nuclei detection. While Mask R-CNN excels at instance segmentation and Faster R-CNN offers robust detection, I selected **YOLOv8** for its superior balance of speed and accuracy, as well as its ability to effectively detect small and overlapping objects like nuclei in real-time.

### Model Architecture

The YOLO (You Only Look Once) architecture is a single-stage object detection framework with three main components:

- Backbone: Extracts image features using convolutional layers, capturing both fine and coarse details.
- Neck (FPN): Combines multi-scale feature maps from the backbone to detect objects of varying sizes, which is crucial for identifying small nuclei.
- Head: Predicts bounding boxes, confidence scores, and class probabilities using regression and classification layers.

![](images/image1.png) 

YOLOv8 simplifies detection with anchor-free predictions and uses advanced loss functions, including IoU loss for box accuracy and cross-entropy loss for classification.

In this project, I leveraged YOLOv8’s capabilities to process 1,744 images with 59,373 annotated nuclei, ensuring high efficiency and scalability during both training and inference. 

### Results

![](images/image2.png) 

The YOLOv8 model achieved an overall mAP@0.5 of 56.6% and mAP@0.5-0.95 of 29.2%, demonstrating moderate detection performance across five classes.

- Strengths: The model performed well for AMBIGUOUS (mAP@0.5: 81.6%) and other_nucleus (mAP@0.5: 82.2%), showing its ability to handle overlapping and challenging objects. Its efficient inference speed (10.1 ms per image) highlights its potential for real-time applications.

- Weaknesses: Performance was lower for nonTIL_stromal (mAP@0.5: 26.9%) and TIL (mAP@0.5: 32.4%). 

### Discussion

Upon further investigation, one major challenge is the presence of *multiple visually similar nuclei within a single image*, which makes differentiation difficult. Additionally, *class imbalance* in the training data likely contributed to the lower performance for these underrepresented categories.

To address these challenges, future improvements could include data augmentation to enhance diversity, weighted loss functions to prioritize minority classes, and synthetic data generation to balance the dataset.

Currently, the dataset comprises exclusively the corrected single-rater data. Subsequent releases should expand to incorporate both the uncorrected single-rater and multi-rater datasets.


## Licensing


The dataset is licensed by a [CC0 1.0 license](https://www.google.com/url?q=https%3A%2F%2Fcreativecommons.org%2Fpublicdomain%2Fzero%2F1.0%2F&sa=D&sntz=1&usg=AOvVaw3eAeYgtS7qVsCxTZd1Vltr).


## References

This [repository](https://github.com/PathologyDataScience/BCSS) contains the necessary information about the dataset associated with the following papers:

- Amgad, Mohamed, et al. "Structured Crowdsourcing Enables Convolutional Segmentation of Histology Images." Bioinformatics, vol. 35, no. 18, 2019, pp. 3461-3467, https://doi.org/10.1093/bioinformatics/btz083. Accessed 18 Mar. 2024.

- Amgad, Mohamed, et al. "NuCLS: A Scalable Crowdsourcing Approach and Dataset for Nucleus Classification and Segmentation in Breast Cancer." GigaScience, vol. 11, 2022, https://doi.org/10.1093/gigascience/giac037. Accessed 18 Mar. 2024.

Model References

- The official YOLOv8 GitHub repository: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics).
