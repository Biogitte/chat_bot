#!/bin/bash

# Working directory - remember to specify accordingly.
export WORK_DIR=$PWD

# Data directories
export DATA=$WORK_DIR/data
export TRAINING_DATA=$DATA/training
export OUTPUT=$DATA/output

# Source code directory
export SRC=$WORK_DIR/src

# Executables
export EXEC=$WORK_DIR/exec

# Executables directory
export EXECUTE=$WORK_DIR/exec

# Specify inputs, latest output models and related files
export INTENTS=$TRAINING_DATA/first_aid_intents.json # json input data to create the chatbot first aid model
export MODEL="$(ls $OUTPUT/*_chatbot.h5 | sort -V | tail -n1)"
export WORDS="$(ls $OUTPUT/*_chatbot_words.pkl | sort -V | tail -n1)"
export CLASSES="$(ls $OUTPUT/*_chatbot_classes.pkl | sort -V | tail -n1)"

# Scripts and related arguments
export TRAIN_BOT_SCRIPT=$SRC/train_bot.py
export TRAIN_BOT=$EXEC/train_bot.sh

export BOT_SCRIPT=$SRC/chatbot.py
export CHATBOT=$EXEC/run_bot.sh

export KAGGLE_SCRIPT=$SRC/fetch_kaggle_data.py
export KAGGLE_DATASET='therealsampat/intents-for-first-aid-recommendations'
export KAGGLE_NEW_NAME='first_aid_intents.json'
export FETCH_DATA=$EXEC/fetch_data.sh
