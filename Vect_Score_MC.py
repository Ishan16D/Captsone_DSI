#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.sparse import hstack

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import stanfordnlp
from textblob import TextBlob
import textstat

from nltk.tokenize import RegexpTokenizer
from nltk import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB 
from sklearn.naive_bayes import MultinomialNB 


# In[2]:


df = pd.read_csv('df_mc.csv')


# In[6]:


df.sample(10)
print(f'Data shape is {df.shape}')


# In[7]:


df.isna().mean()


# In[8]:


df['type'].value_counts(normalize=True)


# In[9]:


cvec = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_df = .8, min_df = 3)
X = cvec.fit_transform(df['lemma'])

X_full =  hstack((X,np.array(df['sub'])[:,None]))

X_full =  hstack((X,np.array(df['vs'])[:,None]))

X_full =  hstack((X,np.array(df['dc_score'])[:,None]))

X_full =  hstack((X,np.array(df['title_length'])[:,None]))

X_full.shape

y = df['type']

X_train, X_test, y_train, y_test = train_test_split(X_full, y,
                                                    random_state=42, stratify = y)

nb = MultinomialNB()

nb = MultinomialNB()
nb.fit(X_train, y_train)
print(nb.score(X_train, y_train))
print(nb.score(X_test, y_test))


# In[13]:


preds = nb.predict_proba(X_full)


# In[14]:


preds.shape


# In[16]:


nb.classes_


# In[18]:


prob_bias = []
prob_clickbait = []
prob_fake = []
prob_hate = []
prob_junksci = []
prob_political = []
prob_reliable = []

for row in preds:
    prob_bias.append(round(row[0], 4))
    prob_clickbait.append(round(row[1], 4))
    prob_fake.append(round(row[2], 4))
    prob_hate.append(round(row[3], 4))
    prob_junksci.append(round(row[4], 4))
    prob_political.append(round(row[5], 4))
    prob_reliable.append(round(row[6], 4))

df['prob_bias'] = prob_bias
df['prob_clickbait'] = prob_clickbait
df['prob_fake'] = prob_fake
df['prob_hate'] = prob_hate
df['prob_junksci'] = prob_junksci
df['prob_political'] = prob_political
df['prob_reliable'] = prob_reliable

df.sample(10)

     
    


# In[ ]:




