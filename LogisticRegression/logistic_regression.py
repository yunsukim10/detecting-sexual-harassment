#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:20:41 2019
BoW and TFIDF models, train and test, and test on email data
@author: yunsukim
"""
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import sys
import spacy
import pickle

def main():
    list_corpus = []
    list_labels = []
    
    with open('../Data/TestTrainData/nh_h_all.csv','r') as file:
        data = csv.reader(file)
    
        for row in data:
            list_corpus.append(row[0])
            list_labels.append(row[1])
            
    spacy.load("en_core_web_sm")
    lemmatizer = spacy.lang.en.English()
    
    def token(data):
        tokens = lemmatizer(data)
        return [token.lemma_ for token in tokens]
        
     
    X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2, 
                                                                               random_state=40)

#####bag of words        
    count_vectorizer = CountVectorizer(tokenizer=token, ngram_range=(1,2))
    X_train_counts = count_vectorizer.fit_transform(X_train)
    X_test_counts = count_vectorizer.transform(X_test)
    
    clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 
                         multi_class='multinomial', n_jobs=-1, random_state=40)
    clf.fit(X_train_counts, y_train)

    y_predicted_counts = clf.predict(X_test_counts)
    accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

    for idx_pred, prediction in enumerate(y_predicted_counts):
        if prediction != y_test[idx_pred]:
            print("Pred: {} Actual: {} \n{}".format(prediction, y_test[idx_pred], X_test[idx_pred]))
    
    
    importance = get_most_important_features(count_vectorizer, clf, 10)
    
    top_scores = [a[0] for a in importance[0]['tops']]
    top_words = [a[1] for a in importance[0]['tops']]
    bottom_scores = [a[0] for a in importance[0]['bottom']]
    bottom_words = [a[1] for a in importance[0]['bottom']]
    
    plot_important_words(top_scores, top_words, bottom_scores, bottom_words, "Most important words for relevance")

    cm = confusion_matrix(y_test, y_predicted_counts)
    plt.figure(figsize=(10, 10))
    plot_confusion_matrix(cm, classes=['Not Harassment','Harassment'], normalize=False, title='Confusion matrix')
    plt.show()
    print(cm)
    

#####tfidf
    tfidf_vectorizer = TfidfVectorizer(tokenizer=token, ngram_range=(1,2), stop_words='english')
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    clf_tfidf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 
                         multi_class='multinomial', n_jobs=-1, random_state=40)
    clf_tfidf.fit(X_train_tfidf, y_train)
            
    y_predicted_tfidf = clf_tfidf.predict(X_test_tfidf)
    
    # Check the error
    for idx_pred, prediction in enumerate(y_predicted_tfidf):
        if prediction != y_test[idx_pred]:
            print("Pred: {} Actual: {} \n{}".format(prediction, y_test[idx_pred], X_test[idx_pred]))
    
    accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf = get_metrics(y_test, y_predicted_tfidf)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_tfidf, precision_tfidf, 
                                                                       recall_tfidf, f1_tfidf))
    cm2 = confusion_matrix(y_test, y_predicted_tfidf)
    plt.figure(figsize=(10, 10))
    plot_confusion_matrix(cm2, classes=['Not Harassment','Harassment'], normalize=False, title='Confusion matrix')
    plt.show()
    print("TFIDF confusion matrix")
    print(cm2)

    
    importance_tfidf = get_most_important_features(tfidf_vectorizer, clf_tfidf, 10)
    top_scores = [a[0] for a in importance_tfidf[0]['tops']]
    top_words = [a[1] for a in importance_tfidf[0]['tops']]
    bottom_scores = [a[0] for a in importance_tfidf[0]['bottom']]
    bottom_words = [a[1] for a in importance_tfidf[0]['bottom']]

    plot_important_words(top_scores, top_words, bottom_scores, bottom_words, "Most important words for relevance")

    filename = 'TFIDF_MODEL.sav'
    with open(filename, 'wb') as fout:
        pickle.dump(clf_tfidf, fout)
    
    pickle.dump(tfidf_vectorizer.vocabulary_,open('Feature.pkl','wb'))
    
####predict email data
    PATH = '../Data/parsed_email.csv'
    email_text, email_index = get_email_data(PATH)
    
    
    new_test_count = count_vectorizer.transform(email_text)
    new_predicted_counts = clf.predict(new_test_count)
    prob = clf.predict_proba(new_test_count)
    
    with open('../Results/clf_result.csv','w') as file:
        fieldnames = ['data','prob','index']
        writer = csv.DictWriter(file,fieldnames=fieldnames)
        writer.writeheader()
        for idx_pred, prediction in enumerate(new_predicted_counts):
            if prob[idx_pred][1] > 0.7:
                writer.writerow({'data': email_text[idx_pred], 'prob':prob[idx_pred][1], 'index': email_index[idx_pred]})
                
    new_test_tfidf = tfidf_vectorizer.transform(email_text)
    new_test_predicted = clf_tfidf.predict(new_test_tfidf)
    prob = clf_tfidf.predict_proba(new_test_tfidf)
    
    count = 0
    
    with open('../Results/tfidf_result.csv','w') as file:
        fieldnames = ['data','prob','index']
        writer = csv.DictWriter(file,fieldnames=fieldnames)
        writer.writeheader()
        for idx_pred, prediction in enumerate(new_test_predicted):
            if prob[idx_pred][1] > 0.7:
                writer.writerow({'data': email_text[idx_pred], 'prob':prob[idx_pred][1], 'index': email_index[idx_pred]})
                count += 1
    print(count)
    
def get_email_data(PATH):
    csv.field_size_limit(sys.maxsize)

    text = []
    index = []
    with open(PATH,'r') as file:
        f = csv.reader(file)
        for row in f:
            text.append(row[0])
            index.append(row[1])
            
    return text, index
    
    
def plot_important_words(top_scores, top_words, bottom_scores, bottom_words, name):
    y_pos = np.arange(len(top_words))
    top_pairs = [(a,b) for a,b in zip(top_words, top_scores)]
    top_pairs = sorted(top_pairs, key=lambda x: x[1])
    
    bottom_pairs = [(a,b) for a,b in zip(bottom_words, bottom_scores)]
    bottom_pairs = sorted(bottom_pairs, key=lambda x: x[1], reverse=True)
    
    top_words = [a[0] for a in top_pairs]
    top_scores = [a[1] for a in top_pairs]
    
    bottom_words = [a[0] for a in bottom_pairs]
    bottom_scores = [a[1] for a in bottom_pairs]
    
    plt.figure(figsize=(10, 10))  

    plt.subplot(121)
    plt.barh(y_pos,bottom_scores, align='center', alpha=0.5)
    plt.title('Irrelevant', fontsize=20)
    plt.yticks(y_pos, bottom_words, fontsize=14)
    plt.suptitle('Key words', fontsize=16)
    plt.xlabel('Importance', fontsize=20)
    
    plt.subplot(122)
    plt.barh(y_pos,top_scores, align='center', alpha=0.5)
    plt.title('Harassment', fontsize=20)
    plt.yticks(y_pos, top_words, fontsize=14)
    plt.suptitle(name, fontsize=16)
    plt.xlabel('Importance', fontsize=20)
    
    plt.subplots_adjust(wspace=0.8)
    plt.show()
    
def get_most_important_features(vectorizer, model, n=5):
    
    index_to_word = {v:k for k,v in vectorizer.vocabulary_.items()}
    
    # loop for each class
    classes ={}
    for class_index in range(model.coef_.shape[0]):
        word_importances = [(el, index_to_word[i]) for i,el in enumerate(model.coef_[class_index])]
        sorted_coeff = sorted(word_importances, key = lambda x : x[0], reverse=True)
        tops = sorted(sorted_coeff[:n], key = lambda x : x[0])
        bottom = sorted_coeff[-n:]
        classes[class_index] = {
            'tops':tops,
            'bottom':bottom
        }
    return classes

    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.winter):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)

    return plt

def get_metrics(y_test, y_predicted):  

    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                    average='weighted')             
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                              average='weighted')
    
    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    
    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1

