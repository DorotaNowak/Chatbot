import numpy as np
import nltk
from lemmatizer import *
import matplotlib.pyplot as plt
from stempel import StempelStemmer

stemmer = StempelStemmer.default()


def plot_loss(n, loss):
    plt.plot(n, loss)
    plt.show()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    word = remove_general_ends(word)
    word = remove_diminutive(word)
    word = remove_verbs_ends(word)
    word = remove_nouns(word)
    word = remove_adjective_ends(word)
    word = remove_adverbs_ends(word)
    word = remove_plural_forms(word)
    """
    return stemmer.stem(word)


def set_of_words(tokenized_sentence, words):
    sentence_words = [stem(word.lower()) for word in tokenized_sentence]
    sow = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            sow[idx] = 1

    return sow
