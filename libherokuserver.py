
from sklearn.feature_extraction.text import TfidfVectorizer

import datetime
import requests
import collections as co
import string
import re

import nltk
from sumy.utils import get_stop_words as gsw1
from stop_words import safe_get_stop_words as gsw2


import pandas as pd
import numpy as np
import pickle
from nltk.stem.porter import *

# needs a lot of memory
# from nltk.stem import WordNetLemmatizer


RANDOM_STATE = 42


def load_init(frac=1):
    """Load file
    frac: float, 0..1, sample fraction of used data
    """
    return pd.read_csv('./data/3categories_50.csv').sample(frac=frac, random_state=RANDOM_STATE)


def clear_message(message):
    """Clear message from garbage
    message: string
    """
    if type(message) != type(''):
        return ''
    message = re.sub('\[.*\]', '', message)
    message = re.sub("[!':']", '', message)
    message = re.sub("[^A-Za-z0-9,!.\/'+-=]", ' ', message)
    message = re.sub("\s+", ' ', message)
    message = message.lower()
    return message


def get_sw(languages):
    """Create stop words list
    languages: list of strings, ex. ["english", "russian"]"""
    stopWords = nltk.corpus.stopwords.words()

    sw = {"yeah","zola","don", 'chat', 'transcript'}  | set(stopWords)
    for lang in languages:
        try: 
            sw |= gsw1(lang) | set(gsw2(lang))
        except:
            pass
    return sw


def get_baser(base='stem'):
    """Choose method of word basing
    base: string, 'lem' or 'stem'
    """
    if base == 'stem':
        stemming = PorterStemmer()
        basing_func = stemming.stem
    elif base == 'lem':
        raise Exception("Please, choose stemmer")
        # lemmatizer = WordNetLemmatizer()
        # basing_func = lemmatizer.lemmatize
    
    return basing_func


def basing(message, baser, stop_words=None):
    """Base each word in message and tagged them
    message: string
    baser: nltk.stem.porter.PorterStemmer or nltk.stem.WordNetLemmatizer
    stop_words: set() or list() of string, collection of words to delete from message
    """
    def get_words(sent):
        res = []
        for word in sent.split(): 
            checked = re.search('(\w+)',word)
            if checked:
                res.append(checked.group(0))
        return res
    
    words = get_words(message)
    words_tags = nltk.pos_tag(words)

    def check_word(word, tag):
        not_sw = word not in stop_words 
        not_digits = not re.match('\d+',word)
        nice_tag = tag in ['NN','VB','DT','NNS','VBP','VB']
        return not_sw and not_digits and nice_tag

    clear_words_tags = [word_tag for word_tag in words_tags if check_word(*word_tag)]

    clear_words = [word_tag[0] for word_tag in clear_words_tags]
    based_words = [baser(word) for word in clear_words]
    
    return based_words


def get_top_words(words_list_series, top_N=1500):
    """Select top N words by total count 
    words_list_series: pd.Series of string, based words of message
    top_N: int
    """
    c_text = co.Counter(
        np.concatenate(words_list_series.tolist(), axis=0 ))

    final_words = pd.DataFrame.from_dict(dict(c_text) , orient = 'index').sort_values(by = 0 , ascending = False).head(top_N).index.values    
    return final_words


def get_vectorized(based_words, y_target, top_words_count=1000):
    """Vectorize based words of message
    based_words: list of string
    y_target: pd.Series, targets, copy from original dataframe
    top_words_count: int
    """
    vec = TfidfVectorizer(max_features=top_words_count)
    
    tfidf_mat = vec.fit_transform(based_words).toarray()
    train = pd.DataFrame(tfidf_mat)

    train_columns = train.columns

    target = y_target.name
    train[target] =  y_target

    train.dropna(inplace=True)
    return train, train_columns, target, vec