#!/usr/bin/python3
import time
import random
import pickle
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import argparse
import warnings
warnings.filterwarnings("ignore")


class TrainChatBot:
    """Train and save a Chatbot model and related data."""

    def __init__(self, intents: str, out_dir: str) -> (list, list):
        self.intents = intents
        self.out_dir = out_dir

        if intents.endswith(".json"):
            self.import_json(intents)

        self.lemmatizer = WordNetLemmatizer()
        self.timestr = time.strftime('%Y%m%d')

        self.model = self.train_chatbot()

    def import_json(self, intents: str):
        # TODO: add description of JSON structure used for training data in docstring.
        """Import JSON file containing intents."""
        self.intents = json.loads(open(intents).read())

    def train_chatbot(self):
        # TODO: Add docstring and explain the model development
        self.words = []  # words we are going to have
        self.classes = []  # classes we are going to have
        documents = []  # the belongings/combinations
        characters_to_ignore = ['?', '!', '.', ',', ';']

        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                word = nltk.word_tokenize(pattern)
                self.words.extend(word)
                documents.append((word, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in characters_to_ignore]
        # TODO: refactor the code (e.g., clean up self attributes outside init function)
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))

        pickle.dump(self.words, open(self.out_dir + f"/{self.timestr}_chatbot_words.pkl", 'wb'))
        pickle.dump(self.classes, open(self.out_dir + f"/{self.timestr}_chatbot_classes.pkl", 'wb'))

        training = []
        output_empty = [0] * len(self.classes)

        for document in documents:
            bag = []
            word_patterns = document[0]
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)

            output_row = list(output_empty)
            output_row[self.classes.index(document[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training, dtype=object)
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])

        # TODO: train-test split and model evaluation

        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(train_y[0]), activation='softmax'))
        sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)  # stochastic gradient descent optimizer

        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        history = self.model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

        self.model.save(self.out_dir + f"/{self.timestr}_chatbot.h5", history)

        # TODO: model evaluation and related accuracy and loss plots + CM


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--intents', help='Input JSON file with intents', type=str)
    parser.add_argument('--out_dir', help='Output directory for model and related files', type=str)

    args = parser.parse_args()

    train = TrainChatBot(args.intents, args.out_dir)
    train.train_chatbot()
