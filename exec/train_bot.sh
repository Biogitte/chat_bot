#!/bin/bash

echo 'Training Chatbot:' "$INTENTS"

python3 "$TRAIN_BOT_SCRIPT" '--intents' "$INTENTS" '--out_dir' "$OUTPUT"

echo 'Training of Chatbot completed.'
