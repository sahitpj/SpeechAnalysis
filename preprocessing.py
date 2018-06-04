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
        self.words_a= []
        
    def individual_processor(self, path, q):
        
        '''
            We define an individual preprocessor which reads each file separartly
          
            this then extracts the number of sentences, the words and the sentences of the file
            
            We then use the file name to extract the place of the speech, and the date with the help of regex
        '''
        
        corpusReader = nltk.corpus.PlaintextCorpusReader(
                self.data_folder,
                path.split('/')[-1]) 
        
        number_of_sentences = len(corpusReader.sents())
        
        number_of_words = len(
                [word for sentence in corpusReader.sents() for word in sentence])
        
        place = re.search(r'_([a-zA-Z]+)_.txt$', path)
        date = re.search( r'/Users/Sahit/Documents/GitHub/BBC_WorkSpace/PM_Speech_Analysis/Speeches/Speeches_Modi_Demo/\d+_(\d+)_([a-zA-Z]+)_(\d+)', path)
    
        
        #date = re.search(r'^[0-9]+_(\w+)_(\w+)_(\d+)', path)
        #city = re.search(r'(\D+).txt$')
        
        filename = path.split('/')[-1]
                 
        data = self.data_extracter(path)
        
        self.words_a.extend(data.split())
        
        speech = {
                'data': data,
                'filename':filename,
                'word_count':number_of_words,
                'sentence_count': number_of_sentences,
                'average_sentence_length': number_of_words / number_of_sentences
                }
        
        try:
            speech['city'] = place.group(1)
        except:
            speech['city'] = 'NA'  
            
            
        date_ =  ' '.join([date.group(1), date.group(2), date.group(3)])
    
        day = date.group(1)
        month = date.group(2)
        year = date.group(3)
        
            
        speech['date'] = date_
        speech['day'] = day
        speech['month'] = month
        speech['year'] = year
            
        
        q.put(speech)
            
    def data_extracter(self, path):
        data = ''
        for word in self.get_dictionary_word_list(path):
            if self.language_identifier(word):
                data += word
        return data
        
    def language_identifier(self, line):
        '''
            we convert each word into ascii values and see if the following values comes in
            
            the range of the english alphabets and numericals
        '''
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
        '''
            We apply Python's multi processing capability to speed to process
            
            of pre-processing almost 1500 files. We use the Process function which spawns individual
            
            Threads
        '''
        for path in paths:
            q = Queue
            z = preprocessor()
            p = Process(target=z.individual_processor, args=(path, q))
            p.start()
            p.join()
            l = q.get()
            self.speeches_data.append(l)
        return self.speeches_data
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            