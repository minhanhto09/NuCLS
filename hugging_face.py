#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:13:56 2024

@author: tominhanh
"""

# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
from PIL import Image as PilImage  # Import PIL Image with an alias
import datasets
from datasets import DatasetBuilder, GeneratorBasedBuilder, DownloadManager, DatasetInfo, Features, Image, ClassLabel, Value, Sequence, load_dataset, SplitGenerator, BuilderConfig
import os
import io
from typing import Tuple, Dict, List
import numpy as np
import zipfile
import requests
import random
from io import BytesIO
import csv

_CITATION = """\
https://arxiv.org/abs/2102.09099
"""

_DESCRIPTION = """\
The comprehensive dataset contains over 220,000 single-rater and multi-rater labeled nuclei from breast cancer images
obtained from TCGA, making it one of the largest datasets for nucleus detection, classification, and segmentation in hematoxylin and eosin-stained
digital slides of breast cancer. This version of the dataset is a revised single-rater dataset, featuring over 125,000 nucleus csvs.
These nuclei were annotated through a collaborative effort involving pathologists, pathology residents, and medical students, using the Digital Slide Archive.
"""

_HOMEPAGE = "https://sites.google.com/view/nucls/home?authuser=0"

_LICENSE = "CC0 1.0 license"

_URL = "https://www.dropbox.com/scl/fi/zsm9l3bkwx808wfryv5zm/NuCLS_dataset.zip?rlkey=x3358slgrxt00zpn7zpkpjr2h&dl=1"


class NuCLSDatasetConfig(BuilderConfig):
    def __init__(self, use_fold_999=False, **kwargs):
        super(NuCLSDatasetConfig, self).__init__(**kwargs)
        self.use_fold_999 = use_fold_999

class NuCLSDataset(GeneratorBasedBuilder):
    # Define multiple configurations for your dataset
    BUILDER_CONFIGS = [
        NuCLSDatasetConfig(
            name="default",
            version=datasets.Version("1.1.0"),
            description="Default configuration with the full dataset",
        ),
        NuCLSDatasetConfig(
            name="debug",
            version=datasets.Version("1.1.0"),
            description="Debug configuration which uses fold 999 for quick tests",
            use_fold_999=True
        ),
    ]

    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        """Returns the dataset info."""

        # Define the classes for the classifications
        raw_classification = ClassLabel(names=[
            'apoptotic_body', 'ductal_epithelium', 'eosinophil','fibroblast', 'lymphocyte',
            'macrophage', 'mitotic_figure', 'myoepithelium', 'neutrophil',
            'plasma_cell','tumor', 'unlabeled', 'vascular_endothelium'
        ])
        main_classification = ClassLabel(names=[
            'AMBIGUOUS', 'lymphocyte', 'macrophage', 'nonTILnonMQ_stromal',
            'plasma_cell', 'tumor_mitotic', 'tumor_nonMitotic',
        ])
        super_classification = ClassLabel(names=[
            'AMBIGUOUS','nonTIL_stromal','sTIL', 'tumor_any',
        ])
        type = ClassLabel(names=['rectangle', 'polyline'])

        # Define features
        features = Features({
            'file_name': Value("string"),
            'rgb_image': Image(decode=True),
            'mask_image': Image(decode=True),
            'visualization_image': Image(decode=True),
            'annotation_coordinates': Features({
                'raw_classification': Sequence(Value("string")),
                'main_classification': Sequence(Value("string")),
                'super_classification': Sequence(Value("string")),
                'type': Sequence(Value("string")),
                'xmin': Sequence(Value('int64')),
                'ymin': Sequence(Value('int64')),
                'xmax': Sequence(Value('int64')),
                'ymax': Sequence(Value('int64')),
                'coords_x': Sequence(Sequence(Value('int64'))),  # Lists of integers
                'coords_y': Sequence(Sequence(Value('int64'))),  # Lists of integers
            })
        })
        return DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
          )

    def _split_generators(self, dl_manager: DownloadManager):
        # Download source data
        data_dir = dl_manager.download_and_extract(_URL)

        # Directory paths
        base_dir = os.path.join(data_dir, "NuCLS_dataset")
        rgb_dir = os.path.join(base_dir, "rgb")
        visualization_dir = os.path.join(base_dir, "visualization")
        mask_dir = os.path.join(base_dir, "mask")
        csv_dir = os.path.join(base_dir, "csv")
        split_dir = os.path.join(base_dir, "train_test_splits")

        # Generate a list of unique filenames (without extensions)
        unique_filenames = [os.path.splitext(f)[0] for f in os.listdir(rgb_dir)]

        # Process train/test split files to get slide names for each split and fold
        if self.config.use_fold_999:
            # Generate the split generators for fold 999
            split_slide_names = self._process_train_test_split_files(split_dir, specific_fold = '999')
        else:
            # Generate the split generators for all folds
            split_slide_names = self._process_train_test_split_files(split_dir)

        # Create the split generators for each fold
        split_generators = []
        for fold in split_slide_names:
            train_slide_names, test_slide_names = split_slide_names[fold]

            # Filter unique filenames based on slide names
            train_filenames = [fn for fn in unique_filenames if any(sn in fn for sn in train_slide_names)]
            test_filenames = [fn for fn in unique_filenames if any(sn in fn for sn in test_slide_names)]

            # Map filenames to file paths
            train_filepaths = self._map_filenames_to_paths(train_filenames, rgb_dir, visualization_dir, mask_dir, csv_dir)
            test_filepaths = self._map_filenames_to_paths(test_filenames, rgb_dir, visualization_dir, mask_dir, csv_dir)

            # Add split generators for the fold
            split_generators.append(
                datasets.SplitGenerator(
                    name=f"{datasets.Split.TRAIN}_fold_{fold}",
                    gen_kwargs={"filepaths": train_filepaths}
                )
            )
            split_generators.append(
                datasets.SplitGenerator(
                    name=f"{datasets.Split.TEST}_fold_{fold}",
                    gen_kwargs={"filepaths": test_filepaths}
                )
            )

        return split_generators

    def _process_train_test_split_files(self, split_dir, specific_fold=None):
        """Reads the train/test split CSV files and returns a dictionary with fold numbers as keys and tuple of train/test slide names as values."""
        
        split_slide_names = {}
        for split_file in os.listdir(split_dir):
            fold_number = split_file.split('_')[1]  # Assumes file naming format "fold_X_[train/test].csv"
            # If specific_fold is set, skip all other folds
            if specific_fold is not None and fold_number != specific_fold:
                continue
            file_path = os.path.join(split_dir, split_file)
            fold_number = split_file.split('_')[1]  # Assumes file naming format "fold_X_[train/test].csv"
            with open(file_path, 'r') as f:
                csv_reader = csv.reader(f)
                next(csv_reader)  # Skip header
                for row in csv_reader:
                    slide_name = row[1]  # Assuming slide_name is in the first column
                    if "train" in split_file:
                        split_slide_names.setdefault(fold_number, ([], []))[0].append(slide_name)
                    elif "test" in split_file:
                        split_slide_names.setdefault(fold_number, ([], []))[1].append(slide_name)

        return split_slide_names

    def _map_filenames_to_paths(self, filenames, rgb_dir, visualization_dir, mask_dir, csv_dir):
        """Maps filenames to file paths for each split."""
        filepaths = {}
        for filename in filenames:
            filepaths[filename] = {
                'rgb': os.path.join(rgb_dir, filename + '.png'),
                'visualization': os.path.join(visualization_dir, filename + '.png'),
                'mask': os.path.join(mask_dir, filename + '.png'),
                'csv': os.path.join(csv_dir, filename + '.csv'),
                'file_name': filename
            }
        return filepaths

    def _generate_examples(self, filepaths):
        """Yield examples as (key, example) tuples."""

        for key, paths in filepaths.items():
            
            # Extract the file name
            file_name = paths['file_name']
            
            # Read the images using a method to handle the image files
            rgb_image = self._read_image_file(paths['rgb'])
            mask_image = self._read_image_file(paths['mask'])
            visualization_image = self._read_image_file(paths['visualization'])

            # Read the CSV and format the data as per the defined features
            annotation_coordinates = self._read_csv_file(paths['csv'])

            # Yield the example
            yield key, {
                'file_name': file_name,
                'rgb_image': rgb_image,
                'mask_image': mask_image,
                'visualization_image': visualization_image,
                'annotation_coordinates': annotation_coordinates,
            }

    def _read_image_file(self, file_path: str, ) -> bytes:
        """Reads an image file and returns it as a bytes_like object."""
        try:
            with open(file_path, 'rb') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading image file {file_path}: {e}")
            return None

    def _read_csv_file(self, filepath):
        """Reads the annotation CSV file and formats the data."""

        with open(filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            annotations = {
                'raw_classification': [],
                'main_classification': [],
                'super_classification': [],
                'type': [],
                'xmin': [],
                'ymin': [],
                'xmax': [],
                'ymax': [],
                'coords_x': [],
                'coords_y': []
            }

            for row in reader:
                annotations['raw_classification'].append(row.get('raw_classification', ''))
                annotations['main_classification'].append(row.get('main_classification', ''))
                annotations['super_classification'].append(row.get('super_classification', ''))
                annotations['type'].append(row.get('type', ''))
                annotations['xmin'].append(int(row.get('xmin', 0)))
                annotations['ymin'].append(int(row.get('ymin', 0)))
                annotations['xmax'].append(int(row.get('xmax', 0)))
                annotations['ymax'].append(int(row.get('ymax', 0)))

                # Handle coords_x and coords_y safely
                coords_x = row.get('coords_x', '')
                coords_y = row.get('coords_y', '')
                annotations['coords_x'].append([int(coord) if coord.isdigit() else 0 for coord in coords_x.split(',')])
                annotations['coords_y'].append([int(coord) if coord.isdigit() else 0 for coord in coords_y.split(',')])

            return annotations
