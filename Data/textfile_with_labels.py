#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:46:27 2019
to text file with labels for fasttext 
@author: yunsukim
"""

import csv

with open ('../nh_h_train.csv', 'r') as f:
    file = csv.reader(f)
    train = [line for line in file]

with open ('../nh_h_test.csv', 'r') as f:
    file = csv.reader(f)
    test = [line for line in file]  

with open('nh_h_train.txt','w') as f:
    file = csv.writer(f)
    
    for line in train:
        file.writerow(['__label__'+line[1]+' '+line[0]])

with open('nh_h_test.txt','w') as f:
    file = csv.writer(f)
    
    for line in test:
        file.writerow(['__label__'+line[1]+' '+line[0]])