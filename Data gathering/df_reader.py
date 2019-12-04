# Script to read in large dataset and prepare data for basic modeling

import pandas as pd
import numpy as np
from scipy.sparse import hstack

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import stanfordnlp
from textblob import TextBlob

from nltk.tokenize import RegexpTokenizer
from nltk import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split


filepath = '../datasets/news_cleaned_2018_02_13.csv'
nlinesfile = 9408908

cols = ['type', 'title']

nlinesrandomsample = n # choose desired random sample size
lines2skip = np.random.choice(np.arange(1,nlinesfile+1), (nlinesfile-nlinesrandomsample), replace=False, )
df = pd.read_csv(filepath, skiprows=lines2skip, usecols= cols )


print(f"Inital shape : {df.shape}")

df_f = df[df['type'] == 'fake']
df_h = df[df['type'] == 'hate']
df_b = df[df['type'] == 'bias']
df_p = df[df['type'] == 'political']
df_r = df[df['type'] == 'reliable']

df = pd.concat([df_f, df_r, df_h, df_b, df_p], axis = 0)

df.index = range(0, len(df.index))

df.sample(10)

print(f"Subsection df shape : {df.shape}")

# df['type'].value_counts(normalize = True)

df['label'] = df['type'].map({'reliable' : 0, 'political' : 0, 'bias' : 1, 'fake' : 1, 'hate' : 1})

#df['label'].value_counts(normalize = True)

#df.isna().mean()

df.dropna(inplace = True)

def tokenize(x):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(x)

df['tokens'] = df['title'].map(tokenize)

def lemmatize(x):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in x])

df['lemma'] = df['tokens'].map(lemmatize)
    


tf = TfidfVectorizer(max_df=0.8, min_df = 3, stop_words = 'english', ngram_range=(1,2))

X = tf.fit_transform(df['lemma'])

X.shape

y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=42, stratify = y)