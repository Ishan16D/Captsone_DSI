# Fake news detection with natural language processing

## Problem Statement

Can we classify news articles by credibility using natural language processing and classification algorithms.

Fake news detection is a hot issue in NLP and media discourse. There is so much content being created online and it is difficult to vet what is true and what is fake. Media literacy should be improved but machine learning algorithms can shoulder some of the responsibility.

If it is possible to detect falsehoods using the text in an article then these models can be incredibly valuable to people care about the content they are taking in.

## Executive Summary

### Problem Statement:

Can we classify news articles by credibility using natural language processing and classification algorithms.

### Goals:

The primary goal of this process was to build a classification model that could take in processed news text and output a probability of an article being fake. 

The ideal would excel in recall, meaning that it would catch as many instances of the positive class (fake) as possible.

A secondary goal of the project is to explore natural language processing and find the most effective ways of encoding the meaning of text in news articles.

## Methodology:

### Data: 

The data I used for this project came from two sources. The FakeNewsCorpus and a combination of kaggle data with other observations.

The first, a collection of news articles provided by https://github.com/several27 (more information on the dataset can be found there). It consisted of ~9.5 million article titles pre labeled in 10 labels from credible to fake.

The second includes data provided by https://github.com/payamesfandiari. This repository includes a kaggle dataset, data collected by George McIntyre, and other scraped articles. These observations contained the entire article text. This data was binary in label (fake or true). I combined these sources into one data set for use in the project.

After assessing the quality of available data and after extensive modeling, I decided to go forward with the second dataset for this project. What follows is the work I did around that second dataset.

### Text Processing:

The text data from both datasets were processed in the same way for modelling. The standard tools from NLTK were used alongside custom data specific functions.

I fed the raw text through loops that used regular expressions to tokenize the text and strip it of unwanted characters. These tokens were then lemmatized and joined back into strings.

From this point there were a few different steps I took. In order to use this cleaned text as a feature in my models I fit a TfidfVectorizer. This vectorizer removed stopwords and found ngrams of range 1 and 2 while limiting the number of tokens collected by how many documents they appeared in.

The vectorized text became my primary feature in the scikit-learn models I deployed, as well as in my XGBoost algorithm. 

In addition to the TfidfVectorizer, I also calculated sentiment, subjectivity, and reading difficulty using the Vadersentiment, Textblob, and Textstat libraries respectively. These features proved to have very little effect when included in a model.

For this project I also deployed neural networks and for that the text features were engineered separately.

For use in a recurrent neural network, the text was converted into integer sequences or converted to a matrix using GloVe pre trained word embeddings.

## Results:
Scores provided are on testing data

Models trained on vectorized text features:

#### Multinomial Naive Bayes

- Accuracy: 78% 
- Precision: 78%
- F1 score: 87%
- Recall: 99%

#### Random Forest (max_depth = 8)

- Accuracy: 88%
- Precision: 88%
- F1 Score: 92%
- Recall: 96%

#### XGBoost (max_depth = 8)

- Accuracy: 94%
- Precision: 98%
- F1 Score: 97%
- Recall: 99%

### Neural Networks:

#### RNN with GloVe embeddings

- Testing Accuracy averaging around 90% across 100 epochs
- Ranging between 85% and 95%
- Potential issue with optimizer and loss function

Each model performed well, especially on recall. The XGBoost however performed exceptionally well across the entire dataset.

The models were effective at predicting fake articles but outside of the XGBoost they struggled with higher false positive rates. This means that they were classifying more true articles as fake than desired.

## Insights and Next Steps:

### Drawbacks:

Data availability

- It is difficult to find reliably labeled articles
- Labeling news as fake involved a level of subjectivity that can’t be avoided
- A dataset with exponentially more article titles is available for training, but titles include significantly less information than full articles
- Powerful models such as the XGBoost can predict effectively but are difficult to interpret
- Feature importance for these models did not reveal interesting trends

### Takeaways:

- The XGBoost model and recurrent neural network show that it is possible to effectively predict fake articles.
- Delving deeper into the observations indicates cases where the model’s discretion and the dataset’s original labels can be questioned.
- With more refining, the recurrent neural network can become a more powerful and more reliable tool.
- We need more data!
