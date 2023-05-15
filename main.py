"""
API for ex2, implementing the skip-gram model (with negative sampling).

"""

# you can use these packages (uncomment as needed)
import pickle
import numpy as np
import re
from collections import Counter
import time
import sys
import random
import math
import collections
import pandas as pd
import string
import os
import nltk
from nltk import skipgrams
from nltk.corpus import stopwords
from numpy import random
from nltk.tokenize import word_tokenize, sent_tokenize
from itertools import chain


# static functions
def who_am_i():  # this is not a class method
    """Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Etay Lorberboym', 'id': '314977596', 'email': 'etaylor@post.bgu.ac.il'}


def expand_contractions(text):
    contractions_dict = {
        "'s": " is",
        "'re": " are",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "can't": "can not",
        "couldn't": "could not",
        "won't": "will not",
        "wouldn't": "would not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "I'm": "I am",
        "we're": "we are",
        "they're": "they are",
        "let's": "let us",
        "it's": "it is",
        "that's": "that is",
        "what's": "what is",
        "here's": "here is"
        # Add more as needed
    }
    for contraction, expansion in contractions_dict.items():
        text = text.replace(contraction, expansion)
    return text


def normalize_text(fn):
    """ Loading a text file and normalizing it, returning a list of sentences.

    Args:
        fn: full path to the text file to process
    """
    sentences = []

    with open(fn, 'r', encoding='utf-8') as file:
        text = file.read()

    # Tokenize into sentences
    nltk.download('punkt')
    sentences = nltk.sent_tokenize(text)

    # For each sentence, convert to lowercase, remove punctuation, and strip leading/trailing spaces
    sentences = [re.sub(r'[^\w\s]', '', sent.lower()).strip() for sent in sentences]

    return sentences


def sigmoid(x): return 1.0 / (1 + np.exp(-x))


def load_model(fn):
    """ Loads a model pickle and return it.

    Args:
        fn: the full path to the model to load.
    """

    # TODO
    return sg_model


class SkipGram:
    def __init__(self, sentences, d=100, neg_samples=4, context=4, word_count_threshold=5):
        self.sentences = sentences
        self.d = d  # embedding dimension
        self.neg_samples = neg_samples  # num of negative samples for one positive sample
        self.context = context  # the size of the context window (not counting the target word)
        self.word_count_threshold = word_count_threshold  # ignore low frequency words (appearing under the threshold)

        # Create a word:count dictionary
        self.word_count = {}
        counts = Counter()
        for line in sentences:
            counts.update(line.split())

        # Ignore low frequency words and stopwords
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("english"))
        counts = Counter(
            {
                k: v
                for k, v in counts.items()
                if v >= self.word_count_threshold and k not in stop_words
            }
        )
        self.word_count = dict(counts)

        # Word-index map
        self.word_index = {}
        index = 0
        for word in self.word_count.keys():
            self.word_index[word] = index
            index += 1

        # Define the vocabulary size
        self.vocab_size = len(self.word_index)

        # Define embedding matrices
        self.T = []
        self.C = []

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        Calculate the cosine similarity between two vectors.
        """
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

        # To prevent division by zero
        if norm_product == 0:
            return 0
        else:
            return dot_product / norm_product

    def compute_similarity(self, w1, w2):
        """ Returns the cosine similarity (in [0,1]) between the specified words.

        Args:
            w1: a word
            w2: a word
        Retunrns: a float in [0,1]; defaults to 0.0 if one of specified words is OOV.
    """
        sim = 0.0  # default

        # Check if both words are in our word_index map
        if w1 in self.word_index and w2 in self.word_index:
            # Fetch the embeddings of the words
            embedding_w1 = self.T[:, self.word_index[w1]]
            embedding_w2 = self.T[:, self.word_index[w2]]

            # Compute cosine similarity
            sim = SkipGram.cosine_similarity(embedding_w1, embedding_w2)

        return sim

    def get_closest_words(self, w, n=5):
        """Returns a list containing the n words that are the closest to the specified word.

        Args:
            w: the word to find close words to.
            n: the number of words to return. Defaults to 5.
        """

    def create_pos_and_neg_lists(self, sentence):
        """
            Creates lists of positive and negative word pairs for a given sentence.
        """
        pos_lst = list(skipgrams(sentence.split(), int(self.context / 2), 1))
        pos_lst += [(tup[1], tup[0]) for tup in pos_lst]
        neg_lst = []
        for _ in range(self.neg_samples):
            neg_lst += [
                (word, random.choice(list(self.word_count.keys())))
                for word in sentence.split()
            ]

        # merge to key value
        pos = {}
        for x, y in pos_lst:
            if x not in self.word_count or y not in self.word_count:
                continue
            pos.setdefault(x, []).append(y)
        neg = {}
        for x, y in neg_lst:
            if x not in self.word_count or y not in self.word_count:
                continue
            neg.setdefault(x, []).append(y)
        return pos, neg

    def create_learning_vector(self, pos, neg):
        dic = {}
        learning_vector = []
        for key, val in pos.items():
            dic[key] = np.zeros(self.vocab_size, dtype=int)
            for v in val:
                dic[key][self.word_index[v]] += 1
            for v in neg[key]:
                dic[key][self.word_index[v]] -= 1
        learning_vector += dic.items()
        return learning_vector

    def preprocess_sentences(self):
        """
            Preprocesses sentences for the SkipGram model, creating a learning vector with positive and negative examples.
        :return:
            learning_vector: A list of tuples, each with a target word and its corresponding context vector.
        """
        learning_vector = []
        for sentence in self.sentences:
            # create positive and negative lists
            pos, neg = self.create_pos_and_neg_lists(sentence)
            print(f"pos = {pos}")
            print(f"neg = {neg}")
            # create the learning context vector
            learning_vector += self.create_learning_vector(pos, neg)
        return learning_vector

    def learn_embeddings(self, step_size=0.001, epochs=50, early_stopping=3, model_path=None):
        """Returns a trained embedding models and saves it in the specified path

        Args:
            step_size: step size for  the gradient descent. Defaults to 0.0001
            epochs: number or training epochs. Defaults to 50
            early_stopping: stop training if the Loss was not improved for this number of epochs
            model_path: full path (including file name) to save the model pickle at.
        """

        T = np.random.rand(self.d, self.vocab_size)  # embedding matrix of target words
        C = np.random.rand(self.vocab_size, self.d)  # embedding matrix of context words

        # tips:
        # 1. have a flag that allows printing to standard output so you can follow timing, loss change etc.
        # 2. print progress indicators every N (hundreds? thousands? an epoch?) samples
        # 3. save a temp model after every epoch
        # 4.1 before you start - have the training examples ready - both positive and negative samples
        # 4.2. it is recommended to train on word indices and not the strings themselves.

        # create learning vectors
        learning_vector = self.preprocess_sentences()
        print("done preprocessing")
        print("start training")
        for i in range(epochs):
            print(f"epoch {i + 1}")

            # learning:
            for key, val in learning_vector:
                # Input layer x T = Hidden layer
                input_layer_id = word_index[key]
                input_layer = np.zeros(vocab_size, dtype=int)
                input_layer[input_layer_id] = 1
                input_layer = np.vstack(input_layer)
                hidden = T[:, input_layer_id][:, None]
                output_layer = np.dot(C, hidden)
                y = sigmoid(output_layer)

        return T, C

    def combine_vectors(self, T, C, combo=0, model_path=None):
        """Returns a single embedding matrix and saves it to the specified path

        Args:
            T: The learned targets (T) embeddings (as returned from learn_embeddings())
            C: The learned contexts (C) embeddings (as returned from learn_embeddings())
            combo: indicates how wo combine the T and C embeddings (int)
                   0: use only the T embeddings (default)
                   1: use only the C embeddings
                   2: return a pointwise average of C and T
                   3: return the sum of C and T
                   4: concat C and T vectors (effectively doubling the dimention of the embedding space)
            model_path: full path (including file name) to save the model pickle at.
        """

        # TODO

        return V

    def find_analogy(self, w1, w2, w3):
        """Returns a word (string) that matches the analogy test given the three specified words.
           Required analogy: w1 to w2 is like ____ to w3.

        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
        """

        # TODO

        return w

    def test_analogy(self, w1, w2, w3, w4, n=1):
        """Returns True if sim(w1-w2+w3, w4)@n; Otherwise return False.
            That is, returning True if w4 is one of the n closest words to the vector w1-w2+w3.
            Interpretation: 'w1 to w2 is like w4 to w3'

        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
             w4: forth word in the analogy (string)
             n: the distance (work rank) to be accepted as similarity
            """

        # TODO

        return False
