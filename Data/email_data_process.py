#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:04:39 2019
Parse enron emails and get each email's subject and body
Output all emails into a csv file
@author: yunsukim
"""
import re
import csv

OUTPUT_PATH = 'parsed_email.csv'

#change this to new email data path if needed
INPUT_PATH = ['../enron_data_100k/labeled/ENG_' + str('%08d'%(int('00000000')+i)) + '.txt' for i in range(212784)]

def email_data_process():
    i = 0
    
    with open(OUTPUT_PATH,'w') as file:
        fieldnames = ['data','index']
        writer = csv.DictWriter(file,fieldnames=fieldnames)
        
        for i, path in enumerate(INPUT_PATH):
            try:
                with open(path,'r') as file:
                    email = file.read()
                    email = re.sub("(meta_subject|Subject:|RE:|Re:|FW:|meta_from.+|meta_date.+|meta_to.+|meta_cc.+|meta_bcc.+|end-of-meta.+|body_greeting|body_content|body_signoff|boilerplate_signature|boilerplate_separator.+|boilerplate_attachment|end-of-attachment|)", "", email)
                    #enron version given to me was formatted with headers; removed all except for subject line and body
                    email = re.sub("\'","",email)
                    email = email.replace("-", '')
                    email = re.sub(r"[^A-Za-z]", " ", email)
                    email = ' '.join(email.split())
                    email = email.lower()
        
                    writer.writerow({'data': email, 'index': i}) 
            except:
                continue
        
        