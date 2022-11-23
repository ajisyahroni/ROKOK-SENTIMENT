
from preprocesing import preprocess_runner, tfidf_runner
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
import joblib
import pickle
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


df = pd.read_csv('data/dataset.csv', sep=';', encoding='utf-8')
df = df.astype({'Value': 'category'})
df = df.astype({'Tweet': 'string'})


# bow
bow_transformer = CountVectorizer()
X = bow_transformer.fit_transform(df['Tweet'].astype('U'))
# TF-IDF
tf_transform = TfidfTransformer(use_idf=False).fit(X)
X = tf_transform.transform(X)

# pickling process
pickle.dump(bow_transformer, open('models/bow.pkl', 'wb'))
pickle.dump(tf_transform, open('models/tfidf.pkl', 'wb'))


X_train, X_test, y_train, y_test = train_test_split(
    X, df['Value'], test_size=0.2, random_state=42)


nb = MultinomialNB()
nb.fit(X_train, y_train)
preds = nb.predict(X_test)


# custom prediction
# df_testku = pd.read_csv('data/dev.csv')
# df_testku = preprocess_runner(df)

# contoh = ['Rokok tidak baik untuk kesehatan', 'benci rokok']
# jk = tf_transform.transform(bow_transformer.transform(contoh))
# print(nb.predict(jk))

print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))
pickle.dump(nb, open('models/nb.pkl', 'wb'))
