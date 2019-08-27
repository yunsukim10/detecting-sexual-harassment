#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:05:36 2019
test enron with pretrained vectors with fasttext
@author: yunsukim
"""
import fasttext
import csv
import sys

loadname = sys.argv[1]
model = fasttext.load_model(loadname)

print('Testing . . .')
csv.field_size_limit(sys.maxsize)
email = []
index = []
with open('../Data/parsed_email.csv','r') as f:
    file = csv.reader(f)
    for line in file:
        if len(line[0].split()) < 500 and line[0] not in email:
            email.append(line[0])
            index.append(line[1])

output = model.predict(email)
values = output[1]
output = output[0]

count = 0

with open('../Results/fastText_result.csv','w') as f:
    writer = csv.DictWriter(f,fieldnames=['data','index','prob'])
    writer.writeheader()
    for i,out in enumerate(output):
        if out[0] == '__label__1':
            count += 1
            writer.writerow({'data':email[i],'index':index[i],'prob':values[i][0]})

print('Done!')