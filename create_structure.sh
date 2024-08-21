#!/bin/bash

mkdir -p text_classification/data
mkdir -p text_classification/model
mkdir -p text_classification/utils

touch text_classification/data/dataset.csv
touch text_classification/model/__init__.py
touch text_classification/model/model.py
touch text_classification/model/train.py
touch text_classification/utils/__init__.py
touch text_classification/utils/data_loader.py
touch text_classification/main.py
touch text_classification/requirements.txt
touch text_classification/config.py

echo "# Configuration settings" > text_classification/config.py