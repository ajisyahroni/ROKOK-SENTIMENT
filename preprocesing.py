

from sklearn.feature_extraction.text import TfidfTransformer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.pipeline import Pipeline
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import string
import re


# cleaning data
def cleaning(Tweet):
    Tweet = re.sub(r'[^a-zA-Z0-9]', ' ', str(Tweet))
    Tweet = re.sub(r'\b\w[1,2]\b', '', Tweet)
    Tweet = re.sub(r'\s\s+', ' ', Tweet)
    Tweet = re.sub(r'^RT[\s]+', '', Tweet)
    Tweet = re.sub(r'[http|https|ftp|ssh]://', '', Tweet)
    Tweet = re.sub(r'[?|@|&|^|#|=|$|.|%|0-9|!_:")(-+,]', '', Tweet)
    Tweet = re.sub(r'0', r' ', Tweet)
    return Tweet


def tokenization(Tweet):
    Tweet = re.split('\W+', Tweet)
    return Tweet


# stopword
stopword = nltk.corpus.stopwords.words('indonesian')


def sw(Tweet):
    Tweet = [word for word in Tweet if word not in stopword]
    return Tweet


def stemming(Tweet):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    do = []
    for w in Tweet:
        dt = stemmer.stem(w)
        do.append(dt)
    d_clean = []
    d_clean = " ".join(do)
    # print(d_clean)
    return d_clean


def preprocess_runner(df):
    df['Tweet'] = df['Tweet'].apply(lambda x: cleaning(x))
    df['Tweet'] = df['Tweet'].apply(lambda x: tokenization(x.lower()))
    df['Tweet'] = df['Tweet'].apply(lambda x: sw(x))
    df['Tweet'] = df['Tweet'].apply(lambda x: stemming(x))
    df.drop_duplicates(subset="Tweet", keep='first', inplace=True)
    return df


def tfidf_runner(df):
    bow_transformer = CountVectorizer()
    X = bow_transformer.fit_transform(df['Tweet'].astype('U'))
    # TF-IDF
    tf_transform = TfidfTransformer(use_idf=False).fit(X)
    X = tf_transform.transform(X)
    return X.reshape(1, -1)
