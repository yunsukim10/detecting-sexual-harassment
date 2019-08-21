#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:26:12 2019

@author: yunsukim
"""
import csv

def find_intersection():
    data = []
    index = []
    files = ['../Results/clf_result.csv', '../Results/tfidf_result.csv', '../Results/LSTM_result.csv', '../Results/GRU_result.csv']
    
    for i, f in enumerate(files):
        with open(f,'r') as f:
            reader = csv.reader(f)
            next(reader)
            
            bow = 0
            tfidf = 0
            lstm = 0
            gru = 0
            
            if i == 0:
                bow = 1
            elif i == 1:
                tfidf = 1
            elif i == 2:
                lstm = 1
            elif i == 3:
                gru = 1
                    
            for x in reader:
                
                if x[2] in index:
                    i = index.index(x[2])
                    b = data[i][3] + bow
                    t = data[i][4] + tfidf
                    l = data[i][5] + lstm
                    g = data[i][6] + gru
                    count = data[i][2] + 1
                    data[i] = (x[0], x[2], count, b, t, l, g)
                else:
                    data.append((x[0],x[2],1, bow, tfidf, lstm, gru))
                    index.append(x[2])
            
    
    with open('intersection.csv','w') as f:
        fieldnames = ['data','index','count','bow','tfidf','lstm','gru']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for x in data:
            writer.writerow({'data':x[0],'index':x[1],'count':x[2],'bow': x[3],'tfidf': x[4],'lstm': x[5],'gru': x[6]})
            