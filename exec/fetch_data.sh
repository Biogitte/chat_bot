#!/bin/bash

echo 'Downloading Kaggle dataset:' "$KAGGLE_DATASET"

python3 "$KAGGLE_SCRIPT" "$KAGGLE_DATASET" "$TRAINING_DATA" "$KAGGLE_NEW_NAME"

echo 'Download completed. The file can be found here:' "$TRAINING_DATA"'/'"$KAGGLE_NEW_NAME"