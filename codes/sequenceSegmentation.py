# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx 
from nltk import texttiling
import nltk
#import uts
from matplotlib import pylab
#from pyannote.core import Segment, Timeline
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
#from pyannote.algorithms.segmentation.sliding_window import SegmentationGaussianDivergence
from sklearn.metrics import recall_score,precision_score,f1_score
#segmenter = SegmentationGaussianDivergence(duration=20, step=1)

shotdir='Desktop/TrecVid/5082189274976367100.shots.json'
threads='Desktop/TrecVid/5082189274976367100.threads.json'
manualScen='Desktop/TrecVid/Eastender_manual_segmentation_inSeconds.json'
#def __init__():
#    shotdir='Desktop/TrecVid/5082189274976367100.shots.json'
#    threads='Desktop/TrecVid/5082189274976367100.threads.json'
#    manualScen='Desktop/TrecVid/Eastender_manual_segmentation_inSeconds1.json'
    
def preprocessingEastender(manualScen=manualScen, threads=threads):
    with open (manualScen,'r') as f:
        data=json.load(f)
  	
    manualBoundry=[]
    for x in data['segment']:
        try:
            manualBoundry.append(x['end time'])
        except:
            continue
#  	for x in data['Segment']:
#  		try:
#  			manualBoundry.append(x['End time'])
#            #print (x['label']+":"),
#            #print(x['End time'])
#   		except:
#   			continue
    with open(threads, 'r') as f:
        data1=json.load(f)
#	with open(threads, 'r') as f:
#		data1 = json.load(f)

    shotSeq=[]
    shotBondry=[]
    for x in data1['content']:
        try:
            shotSeq.append(x['label'])
            shotBondry.append(x['segment']['end'])
	        #print (x['label']+":"),
        except:
            continue

    return	manualBoundry, shotSeq, shotBondry, data1

def preprocessAnnotationFile(fileName):

	SpeakerDF=pd.read_csv(fileName)
	SpeakerDF=SpeakerDF[['Sentence','Speaker','Duration end XML']]
	SpeakerDF['Duration end XML']=[float(x.replace(",", "")) for x in SpeakerDF['Duration end XML']]
	SpeakerDF=SpeakerDF[SpeakerDF.Speaker != '_']
	SpeakerDF=SpeakerDF[SpeakerDF.Speaker != 'ACT']
	SpeakerDF=SpeakerDF[SpeakerDF.Speaker !='nan']
	SpeakerDF.dropna()

	SpeaketrSequence=list(SpeakerDF.Speaker)
	textSent=[]
	SpeakerDF=SpeakerDF.reset_index(drop=True)
	for j in range(len(SpeakerDF)):
		textSent.append(SpeakerDF.Sentence[j])

	lematizedText=[]
	lemmatizer = WordNetLemmatizer()
	for i in textSent:
		tokens = nltk.word_tokenize(i)
		words = [word for word in tokens if word.isalpha()]
		stop_words = set(stopwords.words('english'))
		words = [w for w in words if not w in stop_words]

		lemmaTmp=[]
		for w in words:
			lemmas=lemmatizer.lemmatize(w)
			lemmaTmp.append(lemmas)

		lematizedText.append(lemmaTmp)

	return SpeaketrSequence,SpeakerDF, textSent,lematizedText

"""Takes sequence of speakers or Shots and theur boundry values
    it takes two hyper parameters K and C
    C is number of diffrent speakers or shots
    k window size to slide over the sequence
    it return separation positions of the sequence into scenes and the scene time boundries 
"""
def segmentation_speakers(Sequence,sequenceBoundry,k,C):
    tempList=[]
    sepPosition=[]
    sceneBoundry=[]
    count=0
    tempList.append(Sequence[0:2])
    for i in range(2,len(Sequence)):
        if Sequence[i]==tempList[-1]:
            continue
        if Sequence[i] in tempList:
            tempList.pop(0)
            tempList.append(Sequence[i])
            count=0
        else:
            count=count+1
            if len(tempList)<k:
                tempList.append(Sequence[i])
            if count==C:
                tempList=tempList[C-1:]
                sepPosition.append(i-C)
                sceneBoundry.append(sequenceBoundry[i-C])
                count=0
    return sepPosition,sceneBoundry


"""
    takes the sequence time boundry and scene Boundry as input
    return the truth value of each element of the sequence given 
"""
def sequenceTruthValue(sequenceBoundry,sceneBoundry):
    truthValue=[]
    pos=0
    i=0
    try:
        for x in sequenceBoundry:
            if x<sceneBoundry[pos]:
                truthValue.append(0)
                i=i+1
            else:
                truthValue.append(1)
                i=i+1
                if pos <len(sceneBoundry):
                    pos=pos+1
                    continue
                else:
                    break
    except:
        print('error')
    return truthValue

"""
Evaluation Metrics : It includes all the metrics used to measure the segmentation pk,windowdiff, recall, precision
they must tale atleast two arguments: the ref=the manually annotated scenes truthvalues and 
the computed scene truth values of the sequence
"""
def pk(ref, hyp, k=None, boundary='1'):
   
    if k is None:
        #print(ref.count(boundary))
        k = int(round(len(ref) / (ref.count(boundary) * 2.)))

    err = 0
    for i in range(len(ref)-k +1):
        r =ref[i:i+k].count(boundary) >  0
        h = hyp[i:i+k].count(boundary) > 0
        #print(ref.count(boundary),hyp.count(boundary),boundary)
        #print (h)
        #print (r)
        if r != h:
           err += 1
    return err / (len(ref)-k +1.)

#Evaluation technique windowsDiff: it takes the segments
def windowdiff(ref, hyp, k, boundary="1"):
    
    if k is None:
        #print(ref.count(boundary))
        k = int(round(len(ref) / (ref.count(boundary) * 2.)))

    #if len(seg1) != len(seg2):
       # raise ValueError("Segmentations have unequal length")
    wd = 0
    for i in range(len(ref) - k):
        wd += abs((ref[i:i+k+1].count(boundary)) - (hyp[i:i+k+1].count(boundary)))
    return (wd/(float(len(ref)-k)))

def myRecall(ref, hyp):
    if len(ref)<len(hyp):
        hyp=hyp[0:len(ref)]
    else:
        ref=ref[0:len(ref)]
        

    return recall_score(ref,hyp)

def myPrecision(ref, hyp):
    if len(ref)<len(hyp):
        hyp=hyp[0:len(ref)]
    else:
        ref=ref[0:len(ref)]
        

    return precision_score(ref,hyp)

"""
this function gives the values of the evaluation metrics all in one
ref= referes to the truth values of the ground truth which is the manual segmentation
hyp= the computed truth values using the scene boundries by the algorithm

"""
def evaluateAlgorithm(ref,hyp):
    pkValue=pk(ref,hyp)
    wdValue=windowdiff(ref,hyp)
    recallValue=myRecall(ref,hyp)
    precisionValue=myPrecision(ref,hyp)
    
    return pkValue,wdValue,recallValue,precisionValue

"""
ploting the boundries between the manually annotated scene boundries and the computed scene boundaries
it takes two arguments which are the reference boundary (manually annotated scene boundry) 
and hypothesis boundry (the computed scene boundary)

"""
def plotBoundries(referenceBoundries, hypothesisBoundries, start=0):
		end=referenceBoundries[-1]
		for segment in hypothesisBoundries[start:end]:
			plt.plot([segment, segment], [-10, -0.5], 'r')
		for segment in referenceBoundries[start:end]:
			plt.plot([segment, segment], [0.5, 10], 'g')
		
		plt.ylim(-11, 11);
		plt.xlim(0, segment);
		plt.xlabel('Time (seconds)');




    

