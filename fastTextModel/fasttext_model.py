#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 11:06:05 2019
fastText train, test, and plot word vectors
@author: yunsukim
"""
import csv
import fasttext
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
import numpy as np
    

def visualize_feature(feature_vectors, words,
                      task_dir, num_classes, method='T-SNE'):
    """Visulize the feature vector that extract from the trained model.

    Args:
        feature_vectors: A numpy array representing N samples in D dimension
        task_dir(str): Name to the task, will be used to save the figures
        method: T-SNE or PCA

    """
    # Choose the dimension reduction technique
    if method == 'PCA':
        pca = PCA(n_components=2)
        X_embedded = pca.fit(feature_vectors).transform(feature_vectors)

        # Percentage of variance explained for each components
        print('explained variance ratio (first two components): %s' % str(
            pca.explained_variance_ratio_))
    else:
        pca = PCA(n_components=100)
        X_embedded = pca.fit(feature_vectors).transform(feature_vectors)
        X_embedded = TSNE(
            n_components=2, learning_rate=250, perplexity=5).fit_transform(X_embedded)

    print("Processing {} data points to {} dimensions".format(
        X_embedded.shape[0], X_embedded.shape[1]))

    title = '{} of dataset'.format(method)
    label_scatter_plotter(X_embedded, words,method,
                          title)


def label_scatter_plotter(data_points,
                          words,model,
                          title=""):
    """Function to draw the scatter plot.

    This function will draw the scatter plot based on the data points and their
    corresponding labels.

    Args:
        data_points(numpy.ndarray): The data points to be plotted,
                                    the shape is (N, D)
    """
    if model == 'PCA':
        for i in range(len(words)):
            fig = plt.gcf()
            fig.set_size_inches(200,200)
            plt.text(data_points[i,0],data_points[i,1],words[i])
            plt.axis([-2.0,1.8,-0.02,0.02])
    
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title(title)
        plt.savefig('word_vectors_4.jpg')
        plt.close()
    
    else:
        for i in range(len(words)):
            fig = plt.gcf()
            fig.set_size_inches(100,100)
            plt.text(data_points[i,0],data_points[i,1],words[i])
            plt.axis([-50.0,50.0,-50.0,50.0])
    
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title(title)
        plt.savefig('word_vectors_4.jpg')
        plt.close()

#test
def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))


#train
model = fasttext.train_supervised('../Data/TestTrainData/train_w_labels.txt',epoch=4,lr=0.4)

#or load model trained on a compiler (this one is trained with option -pretrainedVectors wiki.en.vec)
model2 = fasttext.load_model('model_fasttext.bin')

#test
print_results(*model.test('../Data/TestTrainData/test_w_labels.txt'))

#visualize word vectors
words = model.words
vectors = []

for word in words:
    vector = model.get_word_vector(word)
    vectors.append(vector)

vectors = np.array(vectors)

visualize_feature(vectors, words,'',2)

