#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 16:00:46 2019
live demo (TFIDF, LSTM, Fasttext)
@author: yunsukim
"""
import pickle
import torch 
from string import punctuation
import numpy as np
import csv
import nltk
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
import fasttext


def tfidf(test_input):
    
    with open('TFIDF_MODEL.sav','rb') as fin:
        model = pickle.load(fin)
    
    vectorizer = TfidfVectorizer(decode_error="replace",vocabulary=pickle.load(open("Feature.pkl", "rb")))
    test_tfidf = vectorizer.fit_transform([test_input])
    output = model.predict_proba(test_tfidf)
    
    return "{:.4f}".format(output[0][1])

def lstm(test_input):
    
    def pad_features(example_int, seq_length):

        example_len = len(example_int)        
        if example_len <= seq_length:
            zeroes = list(np.zeros(seq_length-example_len))
            new = zeroes+example_int
        elif example_len > seq_length:
            new = example_int[0:seq_length]
                
        features = np.array([new], dtype = int)    
        return features


    def tokenize(test_input):
        
        vocabulary_size = 8000
        unknown_token = "UNKNOWN_TOKEN"
     
        # Read the data and append SENTENCE_START and SENTENCE_END tokens
        with open('nh_h_all.csv', 'r') as f:
            reader = csv.reader(f, skipinitialspace=True)
        
            labels = []
            sentences = []
            data = []
            
            for x in reader:
                data.append((x[0].lower(),int(x[1])))
                
            for d in data:
                labels.append(d[1])
                sentences.append(d[0])
        
        # Tokenize the sentences into words
        tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
        # Count the word frequencies
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
     
        # Get the most common words and build index_to_word and word_to_index vectors
        vocab = word_freq.most_common(vocabulary_size - 1)
        index_to_word = [x[0] for x in vocab]
        index_to_word.append(unknown_token)
        word_to_index = dict([(w,i+1) for i,w in enumerate(index_to_word)])
     
        test_input = test_input.lower() # lowercase
        test_text = ''.join([c for c in test_input if c not in punctuation])
        test_words = test_text.split()
    
        for i, w in enumerate(test_words):
            test_words[i] = w if w in word_to_index else unknown_token
            
        test_ints = [word_to_index[word] for word in test_words]
    
        return test_ints
    
    net = torch.load('lstm9')
    test_ints = tokenize(test_input)
    
    # test sequence padding
    seq_length=80
    features = pad_features(test_ints, seq_length)
    feature_tensor = torch.from_numpy(features)
    
    batch_size = feature_tensor.size(0)
    h = net.init_hidden(batch_size)
    net.eval()
    output, h = net(feature_tensor, h)
    return "{:.4f}".format(output.item())

def fasttextmodel(test_input):
    model = fasttext.load_model('fastTextModel/fastText/model_fasttext.bin')
    output = model.predict(test_input)
    value = output[1][0]
    output = output[0]
    
    if value > 1:
        value = 1
        
    if output[0] == '__label__1':
        return "{:.4f}".format(value)
    else:
        return "{:.4f}".format(1 - value)
    




