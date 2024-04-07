
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
6. [Limitations](#limitations)


## Overview


The [comprehensive dataset](https://sites.google.com/view/nucls/home?authuser=0) comprises over 220,000 labeled nuclei from breast cancer images sourced from [TCGA](https://www.cancer.gov/ccg/research/genome-sequencing/tcga), making it one of the largest datasets for nucleus detection, classification, and segmentation in hematoxylin and eosin-stained digital slides of breast cancer. This extensive labeling effort is the result of a collaboration among pathologists, pathology residents, and medical students, who utilized the Digital Slide Archive for annotation. The dataset serves multiple purposes, including the development and validation of algorithms for nuclear detection, classification, and segmentation. It is also valuable for conducting interrater analysis research. The dataset encompasses annotations from both single-rater and multi-rater evaluations, with this specific collection containing approximately 59,500 labeled nuclei from the corrected single-rater subset.

This [repository](https://github.com/PathologyDataScience/BCSS) contains the necessary information about the dataset associated with the following papers:

- Amgad, Mohamed, et al. "Structured Crowdsourcing Enables Convolutional Segmentation of Histology Images." Bioinformatics, vol. 35, no. 18, 2019, pp. 3461-3467, https://doi.org/10.1093/bioinformatics/btz083. Accessed 18 Mar. 2024.

- Amgad, Mohamed, et al. "NuCLS: A Scalable Crowdsourcing Approach and Dataset for Nucleus Classification and Segmentation in Breast Cancer." GigaScience, vol. 11, 2022, https://doi.org/10.1093/gigascience/giac037. Accessed 18 Mar. 2024.

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
For detailed usage instructions, please refer to [this documentation](https://colab.research.google.com/drive/1d5gEliz8IH06k52OWNWTVDejjKRRqW97?usp=sharing).


## Dataset Structure


### Data Schema

The Corrected Single-Rater Dataset is a collection of 1,744 entries, each with an associated Field of View (FOV) image, mask image, visualization image, and a list of nuclei annotation coordinates, comprising 1,744 complete sets. In total, there are 59,485 nuclei annotations. Each image is rendered at a resolution of 0.2 microns-per-pixel, with all annotation coordinates provided in pixel units to correspond with this resolution.

A single dataset entry contains the following details:

- 'file_name': A unique filename that encodes the most relevant information about each example and its associated data.
![](https://huggingface.co/datasets/minhanhto09/NuCLS_dataset/resolve/main/Images/fig2.png)

- `rgb_image`: A high-resolution RGB image of breast cancer tissue.

- `mask_image`: A mask image with each nucleus labeled. Class labels are encoded in the first channel. The second and third channels are used to create a unique identifier for each nucleus. The field of view (gray area) is marked to delineate the annotated region.

  [This file](hhttps://drive.google.com/file/d/1vT6ZG1s3IQkB9suI21qgzF2N5zM8z0qd/view?usp=sharing) contains the nucleus label encoding, including a special 'fov' code encoding the intended annotation region.

- `visualization_image`: A visualization image that overlays the RGB and mask images to assist in interpretability.

- `annotation_coordinates`: Each instance comprises a list of annotations for the nuclei, with each annotation encompassing:

    - `raw_classification`: The base category of the nucleus, with 13 possible classes such as 'tumor' or 'lymphocyte'.
    
    - `main_classification`: A higher-level category of the nucleus, with 7 classes including 'tumor_mitotic' and 'nonTILnonMQ_stromal'.
    
    - `super_classification`: The broadest category label for the nucleus, with 4 options including 'sTIL' or 'nonTIL_stromal'.
    
    - `type`: The form of annotation used, either 'rectangle' or 'polyline'.
    
    - `xmin`, `ymin`, `xmax`, `ymax`: The bounding box coordinates indicating the extent of the nucleus.
    
    - `coords_x`, `coords_y`: The specific boundary coordinates of the nucleus.

![](https://huggingface.co/datasets/minhanhto09/NuCLS_dataset/resolve/main/Images/fig3.jpeg)

### Data Split


The dataset is divided into six folds, each with its own training and testing set. This division is based on the source hospital to capture the variability in medical imaging practices and ensure that models trained on the dataset can generalize well across different institutions.

The dataset is divided into the following folds:

- `train_fold_1`: 1,481 examples
- `test_fold_1`: 263 examples

- `train_fold_2`: 1,239 examples
- `test_fold_2`: 505 examples

- `train_fold_3`: 1,339 examples
- `test_fold_3`: 405 examples

- `train_fold_4`: 1,450 examples
- `test_fold_4`: 294 examples

- `train_fold_5`: 1,467 examples
- `test_fold_5`: 277 examples

- `train_fold_999`: 21 examples
- `test_fold_999`: 7 examples

Note that the debug configuration utilizes these particular folds `train_fold_999` and `test_fold_999` due to their smaller numbers of examples.

## Usage Example


## Licensing


The dataset is licensed by a [CC0 1.0 license](https://www.google.com/url?q=https%3A%2F%2Fcreativecommons.org%2Fpublicdomain%2Fzero%2F1.0%2F&sa=D&sntz=1&usg=AOvVaw3eAeYgtS7qVsCxTZd1Vltr).


## Limitations


Currently, the dataset comprises exclusively the corrected single-rater data. Subsequent releases should expand to incorporate both the uncorrected single-rater and multi-rater datasets.
