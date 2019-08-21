#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 17:55:28 2019
randomly select 50 from each result set 
BoW, TFIDF, LSTM, GRU
@author: yunsukim
"""
import csv
import random

def bow():
    data = []
    with open('data/clf_result.csv','r') as f:
        reader = csv.reader(f)
        next(reader)
        
        for x in reader:
            data.append((x[0],x[1],x[2]))
            
    data = random.sample(data, 32)
    
    with open('data/clf_result_15.csv','w') as f:
        fieldnames = ['data','prob','index']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        for x in data:
            writer.writerow({'data': x[0], 'prob': x[1], 'index': x[2]})
            
            
def tfidf():
    data = []
    with open('data/tfidf_result.csv','r') as f:
        reader = csv.reader(f)
        next(reader)
        
        for x in reader:
            data.append((x[0],x[1],x[2]))
            
    data = random.sample(data, 34)
    
    with open('data/tfidf_result_15.csv','w') as f:
        fieldnames = ['data','prob','index']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        for x in data:
            writer.writerow({'data': x[0], 'prob': x[1], 'index': x[2]})
            
def lstm():
    data = []
    with open('LSTM_result.csv','r') as f:
        reader = csv.reader(f)
        next(reader)
        
        for x in reader:
            data.append((x[0],x[1],x[2]))
            
    data = random.sample(data, 50)
    
    with open('data/lstm_result_15.csv','w') as f:
        fieldnames = ['data','prob','index']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        for x in data:
            writer.writerow({'data': x[0], 'prob': x[1], 'index': x[2]})
            
def gru():
    data = []
    with open('GRU_result.csv','r') as f:
        reader = csv.reader(f)
        next(reader)
        
        for x in reader:
            data.append((x[0],x[1],x[2]))
            
    data = random.sample(data, 119)
    
    with open('data/gru_result_15.csv','w') as f:
        fieldnames = ['data','prob','index']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        for x in data:
            writer.writerow({'data': x[0], 'prob': x[1], 'index': x[2]})
            
def gru2():
    data = []
    with open('GRU_result_80.csv','r') as f:
        reader = csv.reader(f)
        next(reader)
        
        for x in reader:
            print(x)
            data.append((x[0],x[1],x[2]))
        
            
    data = random.sample(data, 74)
    
    with open('data/gru_result_15_2.csv','w') as f:
        fieldnames = ['data','prob','index']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        for x in data:
            writer.writerow({'data': x[0], 'prob': x[1], 'index': x[2]})