#!/usr/bin/python3
import os
import time
import random
import pickle
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

warnings.filterwarnings("ignore")


def prepare_data(intents: str, out_dir: str, timestr: str) -> list:
    """
    Data preparation that involve:
    - Extract a list of classes from input data and save it as a pickled file with the format:
      <date>_chatbot_classes.pkl
    - Create a lemmatized list of vocabulary from input data and save it as a pickled file with the format:
      <date>_chatbot_words.pkl
    - Create a bag-of-words for training, and split it into a training (70%) and test (30%) split that is returned for
      model training and evaluation.
    """
    lemmatizer = WordNetLemmatizer()

    words = []
    classes = []
    documents = []
    characters_to_ignore = ['?', '!', '.', ',', ';']

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word = nltk.word_tokenize(pattern)
            words.extend(word)
            documents.append((word, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in characters_to_ignore]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    pickle.dump(words, open(out_dir + f"/{timestr}_chatbot_words.pkl", 'wb'))
    pickle.dump(classes, open(out_dir + f"/{timestr}_chatbot_classes.pkl", 'wb'))

    training = []
    output_empty = [0] * len(classes)

    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)
    x = list(training[:, 0])
    y = list(training[:, 1])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
    return x_train, x_test, y_train, y_test


def learning_curves(history: tf.keras.callbacks.History, out_dir: str, timestr: str):
    """
    Plot learning curves:
    - Loss vs. Epochs (training and test)
    - Accuracy vs. Epochs (training and test)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.set_title('Model loss')
    ax1.legend(['Train', 'Test'], loc='upper right')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')

    ax2.plot(history.history['accuracy'])
    ax2.plot(history.history['val_accuracy'])
    ax2.set_title('Model accuracy')
    ax2.legend(['Train', 'Test'], loc='upper right')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    plt.savefig(out_dir + f"/{timestr}_learning_curves.png", bbox_inches='tight', dpi=600)
    return plt.show()


class TrainChatBot:
    """ Train and evaluate a Chatbot model. """

    def __init__(self, intents: str, out_dir: str):
        self.intents = intents
        self.out_dir = out_dir
        self.timestr = time.strftime('%Y%m%d')
        if intents.endswith(".json"):
            self.import_json(intents)

        self.train_chatbot()

    def import_json(self, intents: str):
        """
        Import JSON file with the following format:

        {"intents": [{"tag": "<tag/category>",
                      "patterns": ["Input question", "Input question", "Input question"],
                      "repsonses": ["Response", "Response"]},
                    ]}
        """
        self.intents = json.loads(open(intents).read())

    def train_chatbot(self):
        """
        Train a sequential neural network constructed of:
        - An input layer consisting of 128 units using the ReLU activation function.
        - A layer with 50% dropout to minimize overfitting.
        - A hidden fully connected (i.e., dense) layer consisting of 64 units using the ReLU activation function.
        - A layer with 50% dropout to minimize overfitting.
        - An output layer using the Softmax activation function. Softmax scales the results in the output layer,
          so that they all add up to 1.

        The model training configuration consist of:
        - The Stochastic Gradient Descent Optimiser with a learning rate of 0.01, a decay of decay=1e-6, and a Nesterov
          momentum of 0.9.
        - The Categorical Cross-entropy Loss function.
        - The model accuracy metrics will be monitored.

        The training of the model will be performed by slicing the training data into batches of 5 (i.e., `batch_size`)
        and repeatedly iterating over the entire dataset 200 times (i.e., `epochs`).
        """
        x_train, x_test, y_train, y_test = prepare_data(self.intents, self.out_dir, self.timestr)

        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(y_train[0]), activation='softmax'))
        sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)  # stochastic gradient descent optimizer

        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        history_logger = tf.keras.callbacks.CSVLogger(self.out_dir + f"/{self.timestr}_history_logger.csv",
                                                      separator=',', append=False)

        history = self.model.fit(np.array(x_train), np.array(y_train),
                                 epochs=200, batch_size=5, verbose=0,
                                 callbacks=[history_logger],
                                 validation_data=(x_test, y_test))

        learning_curves(history, self.out_dir, self.timestr)
        print(self.model.summary())
        self.model.save(self.out_dir + f"/{self.timestr}_chatbot.h5", history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--intents', help='Input JSON file with intents', type=str)
    parser.add_argument('--out_dir', help='Output directory for model and related files', type=str)
    args = parser.parse_args()
    train = TrainChatBot(args.intents, args.out_dir)
    train.train_chatbot()
