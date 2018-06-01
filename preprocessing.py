#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 16:28:18 2018

@author: Sahit
"""


import re
import glob
import nltk
from multiprocessing import Process, Queue

folder = nltk.data.find(
    'speeches/manmohansingh/')
paths = glob.glob('speeches/manmohansingh/*')


class preprocessor(object):
    def __init__(self, paths, folder):
        self.paths = paths
        self.speeches_data = []
        self.data_folder = folder
        
    def individual_processor(self, path, q):
        
        corpusReader = nltk.corpus.PlaintextCorpusReader(
                self.data_folder,
                path.split('/')[-1]) 
        
        number_of_sentences = len(corpusReader.sents())
        
        number_of_words = len(
                [word for sentence in corpusReader.sents() for word in sentence])
        
        date = re.search(r'^[0-9]+_(\w+)_(\w+)_(\d+)', path)
        
        city = re.search(r'(\D+).txt$')
        
        filename = path.split('/')[-1]
                 
        data = self.data_extracter(path)
            
        speech = {
                'data': data,
                'filename':filename,
                'city':city,
                'date':date,
                'word_count':number_of_words,
                'sentence_count': number_of_sentences,
                'average_sentence_length': number_of_words / number_of_sentences
                }
        q.put(speech)
            
    def data_extracter(self, path):
        data = ''
        for word in self.get_dictionary_word_list(path):
            if self.language_identifier(word):
                data += word
        return data
        
    def language_identifier(self, line):
        maxchar = ord(max(line))
        if 65 <= maxchar <= 90:
            return 1
        elif 97 <= maxchar <= 122:
            return 1
        else:
            return 0
            
            
          
    def get_dictionary_word_list(self, filepath):
        with open(filepath) as f:
            # return the split results, which is all the words in the file.
            return f.read().split()
            
            
    def Multi_Processor(self, paths):
        for path in paths:
            q = Queue
            z = preprocessor()
            p = Process(target=z.individual_processor, args=(path, q))
            p.start()
            p.join()
            l = q.get()
            self.speeches_data.append(l)
        return self.speeches_data
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            