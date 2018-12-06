#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 13:24:39 2018

@author: berhe
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances,cosine_similarity
import os
import nltk
import csv
import pandas as pd
import pysrt
from textblob import TextBlob
import matplotlib
from matplotlib import style
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from decimal import *
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import sklearn.cluster as cluster
from sklearn import metrics
import re
from scipy.cluster.hierarchy import ward, dendrogram,complete,fcluster
from gensim.models import Word2Vec

sns.set()
#%matplotlib inline
matplotlib.rcParams['figure.figsize']=(16.0,9.0)
style.use('fivethirtyeight')

"""
######################################### Text Representations ####################################
"""
"""
generate ifidf representation of texts, it returns a tfidf vector of words in the text: we can give ona data or we can split the data after we generate the representation
"""
def tfidf_Representation(training_data):
    documents=training_data


    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_vector = vectorizer.fit_transform(documents).todense()
    terms=vectorizer.get_feature_names()

    return tfidf_vector,terms

"""
The function below genrates vector representation
"""
def count_Vector(training_data):

    vectorizer = CountVectorizer(input='content')
    dtm = vectorizer.fit_transform(training_data)
    vocab = vectorizer.get_feature_names()

    dtm=dtm.toarray()
    vocab=np.array(vocab)

    return dtm,vocab

"""
Compute ecludian and cosine distance between the vectors produced by countvector representation
"""
def distance_Measures(dtm):
    n, _ = dtm.shape
    dist_eclud = np.zeros((n, n))
    dist_eclud = euclidean_distances(dtm)
    cos_sim = 1 - cosine_similarity(dtm)

    return dist_eclud,cos_sim



def heirarchical_dendogram(dtm):
    link_matr=ward(dtm)
    plt.title('Hierarchical Clustering scenes')
    plt.xlabel('scences sample')
    plt.ylabel('distance')
    dendrogram(link_matr,truncate_mode='lastp',leaf_rotation=90,leaf_font_size=12,p=10,show_contracted=True)
    plt.show()



"""
This function below draws the clusters. after clustering is done by given the cluster type as an argument / it draws the clusters and returns the labels
"""
def plot_clusters(data, algorithm, args, kwds):
    #matplotlib inline
    sns.set_context('poster')
    sns.set_color_codes()
    plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}
    #start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    #end_time = time.time()
    #acc=metrics.adjusted_rand_score(labels, tra_lbl)
    palette = sns.color_palette('bright', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.figure(figsize=(10,8))
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    #plt.suptitle('accuracy is {}'.format(str(acc)))
    #plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
    return labels

"""
changes miliseconds to minutes and seconds
"""
def to_min_sec(st_ms,end_ms):
    Ssec=st_ms/1000
    Esec=end_ms/1000
    sm,ss=divmod(Ssec,60)
    em,es=divmod(Esec,60)
    return sm,ss,em,es

"""
this is function to clear a sentence (string of charcters)
"""
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", "", string)
    string = re.sub(r"\'s", "\'s", string)
    string = re.sub(r"\'ve", "\'ve", string)
    string = re.sub(r"n\'t", "n\'t", string)
    string = re.sub(r"\'re", "\'re", string)
    string = re.sub(r"\'d", "\'d", string)
    string = re.sub(r"\'ll", "\'ll", string)
    string = re.sub(r",", ",", string)
    string = re.sub(r"!", "!", string)
    string = re.sub("([\(\[]).*?([\)\]])", "", string)
    string = re.sub(r")", r"", string)
    #string = re.sub(r"\?", " \? ", string)
    #string = re.sub(r"\s{2,}", " ", string)
    return string.strip()#.lower()

"""
here gets the start, end story id of a scene. The format of the file is text file with three raws (start and end time of a scene and its story ID)
"""
def scene_extraction(fileName):
    fileName=open(fileName,'r')
    text=fileName.readlines()
    start_scene=[]
    end_scene=[]
    story_id=[]
    for text_line in text:
        if not text_line in ['\n', '\r\n']:
            splitte_lines=text_line.split()
            try:
                start_scene.append(splitte_lines[0])
                end_scene.append(splitte_lines[1])
                story_id.append(splitte_lines[2])
            except IndexError:
                story_id.append('NA')
        else:
            continue
    return start_scene,end_scene,story_id

"""
extracted the texts of each scene from subtitle files and segmentation files which have the starting and ending time of a scene
"""
def sceneTexts(scenesegementedFiles,subtitleFiles):
    part_st=""
    scene_texts=[]
    scene_lbls=[]
    start,end,story_id=scene_extraction(scenesegementedFiles)
    subs=pysrt.open(subtitleFiles,encoding='iso-8859-1')
    start=[float(i) for i in start]
    end=[float(i) for i in end]
    for i in range(len(start)):
            m,s,em,es=to_min_sec(start[i],end[i])
            part_st=subs.slice(starts_after={'minutes': m, 'seconds':s}, ends_before={'minutes': em, 'seconds': es}).text.encode('utf-8')
            scene_texts.append(part_st)
            scene_lbls.append(story_id[i])
    return scene_texts,scene_lbls ,start,end

"""
splits texts into sentences
"""
def split_into_sentences(text):
    sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)
    cleanSentences=[]
    for i in sentences:
        cleanSentences.append(clean_str(i))

    return cleanSentences
