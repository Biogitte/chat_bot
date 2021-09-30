#!/bin/bash
echo "Initialising Chatbot. Please wait..."

python3 $BOT_SCRIPT --model_path $MODEL --words_path $WORDS --classes_path $CLASSES --json_file $INTENTS