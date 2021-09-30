#!/usr/bin/python3
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import argparse
import json
import warnings
import os

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ChatBot:
    """ Chatbot. """

    def __init__(self, model_path: str, words_path: str, classes_path: str, intents: str):
        # TODO: check and clean-up type hints
        # TODO: Add missing docstrings and improve existing
        self.model_path = model_path
        self.words_path = words_path
        self.classes_path = classes_path

        self.model = load_model(self.model_path)
        self.words = pickle.load(open(self.words_path, 'rb'))
        self.classes = pickle.load(open(self.classes_path, 'rb'))
        self.intents = intents
        if intents.endswith(".json"):
            self.import_json(intents)

        self.lemmatizer = WordNetLemmatizer()

    def import_json(self, intents: str):
        # TODO: add description of JSON structure used for training data in docstring.
        """Import JSON file containing intents."""
        self.intents = json.loads(open(intents).read())

    def clean_sentence(self, sentence: str):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bag_of_words(self, sentence: str, words: list) -> np.array:
        """Convert a sentence into a bag-of-words """
        sentence_words = self.clean_sentence(sentence)
        bag = [0] * len(words)

        for s in sentence_words:
            for i, word in enumerate(words):
                if word == s:
                    bag[i] = 1
        return np.array(bag)

    def predict_category(self, sentence: str):
        """Predict the category/class of a sentence."""
        bow = self.bag_of_words(sentence, self.words)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.1
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]],
                                'probability': str(r[1])})
        return return_list

    def get_response(self, ints, intents_json):
        """Get a response from the chatbot."""
        try:
            tag = ints[0]['intent']
            list_of_intents = intents_json['intents']
            for i in list_of_intents:
                if i['tag'] == tag:
                    result = random.choice(i['responses'])
                    break
        except IndexError:
            result = "Sorry, I do not understand your question."
        return result

    def chatbot(self):
        while True:
            message = input("Please, enter your first aid question (write bye to exit): ")
            if message == 'Bye' or message == 'bye':
                print('Thank you for visiting. Bye.')
                break
            else:
                ints = self.predict_category(message)
                res = self.get_response(ints, self.intents)
                print(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='Path to Chatbot model (i.e., h5) file.', type=str)
    parser.add_argument('--words_path', help='Path to pickle file with Chatbot words.', type=str)
    parser.add_argument('--classes_path', help='Path to pickle file with Chatbot classes.', type=str)
    parser.add_argument('--json_file', help='Input JSON file.', type=str)
    args = parser.parse_args()
    bot = ChatBot(args.model_path, args.words_path, args.classes_path, args.json_file)
    bot.chatbot()
