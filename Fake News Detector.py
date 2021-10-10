#!/usr/bin/env python
# coding: utf-8

# # **Detecting Fake News**

# In this project I will be attempting to build a machine the learns how to detect fake news from real news based on a specific data set. This is my first crack at machine learning so it should be a good time!

# In[1]:


import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[2]:


# reading the data
face = pd.read_csv('news.csv')

# checking shape
face.shape


# In[3]:


face.head()


# In[4]:


# getting labels from datafrom
labels = face.label
labels.head()


# In[5]:


# now splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(face['text'], labels, test_size = 0.2, random_state = 7)


# In[6]:


# initializing a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7)

# fitting and transforming the vectorizer on the test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)


# In[7]:


# initializing a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter = 50)
pac.fit(tfidf_train, y_train)

# predicting using the test set and calculate accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100, 2)}%')


# In[8]:


# building a Confusion Maxtrix
confusion_matrix(y_test, y_pred, labels = ['FAKE', 'REAL'])


# ### ***Conclusion***

# So it looks like I was able to teach the machine how to detect fake news to the best of its ability. With an accuracy rate of 92.9%, we were able to catch and detect MOST of the news correctly. For my first machine learning project, I feel pretty confident about this one. 
# 
# **Detectigng Fake News with Python and Machine Learning**: https://data-flair.training/blogs/advanced-python-project-detecting-fake-news/

# In[ ]:




