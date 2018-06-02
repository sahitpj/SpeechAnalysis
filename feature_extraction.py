#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 17:29:50 2018

@author: Sahit
"""
import pandas as pd

from datetime import datetime
import dateutil.parser
import nltk



from collections import Counter
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer





from nltk.tokenize import RegexpTokenizer

class Feature_exctraction(object):
    def __init__(self, speeches_data, words):
        self.speech_data = speeches_data
        self.speeches_df = pd.Dataframe(speeches_data)
        self.words = list(set(words))  
        
    
        self.word_freq = Counter(words)
        self.common_words_list = sorted(self.word_freq, key=self.word_freq.get, reverse=True)
        
    def preprocess(self,x):
        x = re.sub('[^a-z\s]', '', x.lower())                  
        return ' '.join(x)       
    
    def simple_tokenizer(self, str_input):
        words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
        return words
        
    def extract(self):
        '''
            We develop our features by using the TFidf tokeniser and by cleaning our text data before that
            
            We can now run our clustering algoirithm on it
            
            The above processes are for normalising it and for bringing it to lower case
        '''
        self.speeches_df['data'] = [" ".join(review) for review in self.speeches_df['data'].values]
        self.speeches_df['clean_data'] =  self.speeches_df['data'].apply(self.preprocess)
        tokenizer = RegexpTokenizer(r'\w+')
        self.speeches_df['tokenised'] = [" ".join(tokenizer.tokenize(review)).lower() for review in self.speeches_df['data'].values]
        
        vectorizer = TfidfVectorizer(
            use_idf=True, tokenizer=self.simple_tokenizer, stop_words='english')
        X = vectorizer.fit_transform(self.speeches_df['tokenised'])
        
        
        return X
            