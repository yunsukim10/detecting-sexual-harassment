#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 17:07:47 2019
Process both harassment and non-harassment data and compile them into one file
and split them into train and test data sets
@author: yunsukim
"""
import re
import csv
import json
import numpy as np
import random

def get_nh_data():

    PATH = 'Non-harassment'
    count = 0
    total = 0

    with open(PATH+'nytimes_news_articles.txt','r') as file:
        articles = file.read()
        
    articles = re.sub("(URL: ).+","",articles)
    article_list = articles.split("\n")

    file = open(PATH+'movielines.txt','rb')
    for line in file:
        line = str(line)
        line = line[4:-3]
        article_list.append(line)
    
    article_list = random.sample(article_list, 20000)

    with open(PATH+'nonharassment_all.txt','w') as file:
        for article in article_list:
            clean = process_text(article)
            if len(clean.split()) < 400 and len(clean.split()) > 4:
                file.write(clean)
                file.write("\n")
                total += len(clean.split())
                count += 1

    print(count)
    avg_length = total / count
    print(avg_length)
    

def get_h_data():
     
    PATH = 'Harassment'
    data = []
    count = 0
    total = 0
    
    with open(PATH+'academia_data.json','r') as file:
        text = json.load(file)
        for id in text:
            data.append(id["event"])
    data = random.sample(data, 300)
    
    with open(PATH+'safecity.json', 'r') as file:
        text = json.load(file)
        for id in text:
            new_text = text[id]['text']
            try: 
                for time in text[id]['records']['Time']: 
                    if any(char.isdigit() for char in time['span']):
                        new_text = new_text.replace(time['span'],'')
                data.append(new_text)
            except:
                data.append(new_text)
            
    with open(PATH+'sexual_harassment_data3.csv') as file:
        csvReader = csv.reader(file)
        for sentence in csvReader:
            row = sentence[0]
            data.append(row)
    
    with open(PATH+'huffpost_data.txt', 'r') as file:
        text = file.read()
        text = text.split("\n")
        for account in text:
            account = process_text(account)
            data.append(account)
    
    with open(PATH+'harassment_all.txt','w') as file:
        for d in data:
            clean = process_text(str(d))
            if len(clean.split()) < 400 and len(clean.split()) > 4:
                file.write(clean)
                file.write("\n")
                total += len(clean.split())
                count += 1
            
    print(count)
    avg_length = total / count
    print(avg_length)


def combine_data():
    whole = []

    with open('Non-harassment/nonharassment_all.txt','r') as file:
        nh = file.read()
        nh = nh.split("\n")
        nh = nh[:-1]
        for item in nh:
            whole.append((item,0))
        
    with open('Harassment/harassment_all.txt','r') as f:
        h = f.read()
        h = h.split("\n")
        h = h[:-1]
        for item in h:
            whole.append((item,1))
    
    np.random.shuffle(whole)
    
    with open('TestTrainData/nh_h_all.csv','w') as csvfile:
        fieldnames = ['text', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for item in whole:
            writer.writerow({'text': item[0], 'label': item[1]})
            
    length = len(whole)
    #split the whole data into 80% train and 20% test
    train = whole[:int(length*0.8)]
    test = whole[int(length*0.8):]

    with open('TestTrainData/nh_h_train.csv','w') as csvfile:
        fieldnames = ['text', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for item in train:
            writer.writerow({'text': item[0], 'label': item[1]})
            
    with open('TestTrainData/nh_h_test.csv','w') as csvfile:
        fieldnames = ['text', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for item in test:
            writer.writerow({'text': item[0], 'label': item[1]})


def process_text(str):
    str = str.replace("’","")
    str = str.replace("'","")
    str = str.replace("-","")
    str = re.sub("[^A-Za-z]", " ", str)
    str = ' '.join(str.split())
    return str.lower()
    
        