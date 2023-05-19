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


def softmax(x):
    e_x = np.exp(x - np.max(x))  # subtract max(x) for numerical stability
    return e_x / e_x.sum(axis=0)  # return probabilities


def load_model(fn):
    """ Loads a model pickle and return it.

    Args:
        fn: the full path to the model to load.
    """

    with open(fn, "rb") as f:
        sg_model = pickle.load(f)
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

        # Ignore low frequency words
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
        self.vocab_size = len(counts)

        # Define embedding matrices
        self.T = []
        self.C = []

        # create reverse mapping from index to word
        self.index_word = {i: word for word, i in self.word_index.items()}

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
        if w1 in self.word_index and w2 in self.word_index and w1 != w2:
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
        if w not in self.word_index:
            print(f"Word '{w}' not in vocabulary.")
            return []
        # Calculate the similarity between the word and all others
        similarities = [(i, self.compute_similarity(w, self.index_word[i])) for i in range(self.vocab_size)]

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return the top n words, skip the first one because it's the word itself
        n_words = [self.index_word[i] for i, _ in similarities[:n]]
        return n_words

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
        for key, val in pos.items():
            dic[key] = np.zeros(self.vocab_size, dtype=int)
            for v in val:
                dic[key][self.word_index[v]] += 1
            for v in neg[key]:
                dic[key][self.word_index[v]] -= 1
        return dic.items()

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
            # create the learning context vector
            learning_vector += self.create_learning_vector(pos, neg)
        return learning_vector

    def forward_pass(self, word, T, C):
        """
        Performs a forward pass of the neural network for the SkipGram model.
        """
        # One-hot encode the input word
        input_layer = np.zeros(self.vocab_size, dtype=int)
        input_layer[self.word_index[word]] = 1
        input_layer = np.vstack(input_layer)
        # Retrieve the embedding of the current word
        hidden = T[:, self.word_index[word]][:, None]
        # Compute the output layer
        output_layer = np.dot(C, hidden)
        # Scale the predictions to be between 0 and 1
        y = sigmoid(output_layer)

        return input_layer, hidden, y

    def backprop_pass(self, y, val, input_layer, hidden, C, T, step_size):
        # calculate loss
        e = self.calculate_loss(y, val)

        # backprop with stochastic gradient descent
        T, C = SkipGram.update_weights(hidden, e, input_layer, C, T, step_size)
        return T, C, e

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

        # create learning vectors
        learning_vector = self.preprocess_sentences()
        print("done preprocessing")

        print("start training")
        best_loss = np.inf  # initialize the best loss as infinity
        epochs_no_improve = 0  # initialize epochs without improvement

        for i in range(epochs):
            print(f"epoch {i + 1}")

            epoch_loss = 0  # initialize loss for this epoch

            for key, val in learning_vector:
                # forward pass
                input_layer, hidden, y = self.forward_pass(key, T, C)

                # backpropagation pass
                T, C, e = self.backprop_pass(y, val, input_layer, hidden, C, T, step_size)

                epoch_loss += e  # add loss of this example to the epoch loss

            epoch_loss /= len(learning_vector)  # calculate average loss for this epoch
            print(f"Epoch {i + 1}, Loss: {epoch_loss}")

            # Early stopping check
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0  # reset the count
            else:
                epochs_no_improve += 1  # increment the count

            if epochs_no_improve == early_stopping:
                print("Early stopping!")
                break

        # backup the last trained model (the last epoch)
        self.T = T
        self.C = C

        # Save model after finishing training
        with open(model_path, "wb") as file:
            pickle.dump(self, file)

        print(f"Model saved to path: '{model_path}'")

        return T, C

    @staticmethod
    def calculate_loss(y, val):
        """
        Calculates the loss between the target values and the predicted values for Skip-gram with negative sampling.
        """
        loss = 0
        # Positive samples loss
        pos_loss = -np.log(y[val == 1])

        # Negative samples loss
        neg_loss = -np.log(1 - y[val == -1])
        loss = np.sum(pos_loss) + np.sum(neg_loss)

        return loss

    @staticmethod
    def update_weights(hidden, e, input_layer, C, T, step_size):
        """
        Updates the weights of the model based on the calculated error and input values.
        """
        # calculate gradients
        dC = np.dot(hidden, e.T).T
        dT = np.dot(input_layer.T, np.dot(C.T, e).T).T
        # perform update
        C -= step_size * dC
        T -= step_size * dT
        return T, C

    @staticmethod
    def combine_vectors(T, C, combo=0, model_path=None):
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
        if combo == 0:
            V = T
        elif combo == 1:
            V = C.T
        elif combo == 2:
            V = (T + C.T) / 2
        elif combo == 3:
            V = T + C.T
        elif combo == 4:
            V = np.concatenate((T, C.T), axis=1)
        else:
            raise ValueError("Invalid combo option. Choose a number between 0 and 4.")

        if model_path:
            with open(model_path, "wb") as file:
                pickle.dump(V, file)
            print(f"Combined vectors saved to path: '{model_path}'")

        return V

    def find_analogy(self, w1, w2, w3):
        """Returns a word (string) that matches the analogy test given the three specified words.
           Required analogy: w1 to w2 is like ____ to w3.

        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
        """

        if w1 not in self.word_index or w2 not in self.word_index or w3 not in self.word_index:
            print("At least one of the words is not in the vocabulary.")
            return

        # Get the vector representations of the words
        vec_w1 = self.T[:, self.word_index[w1]]
        vec_w2 = self.T[:, self.word_index[w2]]
        vec_w3 = self.T[:, self.word_index[w3]]

        # Compute the target vector
        vec_v = vec_w2 - vec_w1 + vec_w3

        # Find the closest word vector
        closest_word = None
        closest_distance = float('-inf')

        for i in range(self.vocab_size):
            # Compute cosine similarity
            sim = SkipGram.cosine_similarity(vec_v, self.T[:, i])

            if sim > closest_distance:
                closest_distance = sim
                closest_word = self.index_word[i]

        return closest_word

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

        # Check if all words are in the vocabulary
        if w1 not in self.word_index or w2 not in self.word_index or w3 not in self.word_index or w4 not in self.word_index:
            print("At least one of the words is not in the vocabulary.")
            return False

        # Compute the analogy word using the find_analogy function
        analogy_word = self.find_analogy(w1, w2, w3)

        if analogy_word == w4:
            return True
        else:
            # Get the n closest words to the computed analogy vector
            closest_words = self.get_closest_words(analogy_word, n=n)
            # Check if w4 is among the closest words
            return w4 in closest_words
