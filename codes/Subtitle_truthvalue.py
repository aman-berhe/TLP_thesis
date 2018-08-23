#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 14:16:21 2018

@author: berhe
"""

from datetime import date, datetime, time, timedelta
#from textblob import TextBlob#text data manipluation and processing kind of Natural Language Processing tool
import csv
import pandas as pd
import numpy as np
import os
import nltk
import re
import pysrt #subtitle manipulation library
from nltk import sent_tokenize

scenes_dir='/home/berhe/Desktop/LIMSI/TLP_thesis/Scenes/'
subtitle_dir='/home/berhe/Desktop/LIMSI/TLP_thesis/subtitles/'
def __init():
    scenes_dir='/home/berhe/Desktop/LIMSI/TLP_thesis/Scenes/'
    subtitle_dir='/home/berhe/Desktop/LIMSI/TLP_thesis/subtitles/'
    
def to_min_sec(st_ms,end_ms):
    Ssec=st_ms/1000
    Esec=end_ms/1000
    sm,ss=divmod(Ssec,60)
    em,es=divmod(Esec,60)
    return sm,ss,em,es

def to_min_sec2(st_ms,end_ms):
    Ssec=st_ms
    Esec=end_ms
    sm,ss=divmod(Ssec,60)
    em,es=divmod(Esec,60)
    return sm,ss,em,es

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
    #string = re.sub(r")", r"", string)
    #string = re.sub(r"\?", " \? ", string)
    #string = re.sub(r"\s{2,}", " ", string)
    return string.strip()#.lower()

def get_sentences(text):
    if(len(text)==0):
        return ' '
    else:
        sentenceList=sent_tokenize(text)
        return sentenceList
"""
extract texts in each scene
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
split a text into sentences and clean the sentence from un necessary charchters 
"""

def split_into_sentences(text):
    sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)
    cleanSentences=[]
    for i in sentences:
        cleanSentences.append(clean_str(i))
    
    return cleanSentences

"""
Extracting the texts in a scene and return the strting and ending time of each scene accordingly each text in a scene
"""
def sceneTexts(scenesegementedFile,subtitleFile):
    part_st=""
    scene_texts=[]
    scene_lbls=[]
    start,end,story_id=scene_extraction(scenesegementedFile)
    subs=pysrt.open(subtitleFile,encoding='iso-8859-1')
    start=[float(i) for i in start]
    end=[float(i) for i in end]
    for i in range(len(start)):
            m,s,em,es=to_min_sec(start[i],end[i])
            part_st=subs.slice(starts_after={'minutes': m, 'seconds':s}, ends_before={'minutes': em, 'seconds': es}).text.encode('utf-8')
            scene_texts.append(part_st)
            scene_lbls.append(story_id[i])
    return scene_texts,scene_lbls,start,end

def get_text_time(subtitleFile):
    DF=pd.Series()
    subs=pysrt.open(subtitleFile)
    start_minute=[]
    start_seconds=[]
    end_minute=[]
    end_second=[]
    textList=[]
    for i in range(len(subs)):
        textList.append(subs[i].text)
        start_minute.append(subs[i].start.minutes)
        start_seconds.append((subs[i].start.seconds+(subs[i].start.minutes*60)))
        end_minute.append(subs[i].end.minutes)
        end_second.append((subs[i].end.seconds+(subs[i].end.minutes*60)))
        
        DF['textList']=textList
        #DF['start_minute']=start_minute
        DF['start_seconds']=start_seconds
        #DF['end_minute']=end_minute
        DF['end_second']=end_second
    return DF

def episodTruthValueMan1(scenesegementedFile,subtitleFile):
    _,_,sc_start,sc_end=sceneTexts(scenesegementedFile,subtitleFile)
    ep_TV=[]
    ep_TVB=[]
    s=0
    sc_end=[int(i/1000) for i in sc_end]
    DF=get_text_time(subtitleFile)
    ki=0
    for j in range(len(DF['end_second'])):
        #for i in range(len(sc_end)): 
        if ki<len(sc_end):
            if (DF['start_seconds'][j]<sc_end[ki]):
                ep_TV.append(0)
                ep_TVB.append(s)
                    #cl_sc=DF['textList'][i]
                    #cl_sc=sent_tokenize(cl_sc)
            else:
                s=s+1
                ep_TV.append(1)
                ep_TVB.append(s)
                ki=ki+1
        else:
            s=s+1
            ep_TV.append(1)
            ep_TVB.append(s)
        
    return ep_TV,ep_TVB,len(DF['end_second'])

def episodTruthValueShot1(scene_start,scene_end, subtitleFile):
    #start=[int(i) for i in scene_start]
    #end=[int(i) for i in scene_end]
    end=scene_end
    ep_TVS=[]
    ep_TVSB=[]
    s=0
    DF=get_text_time(subtitleFile)
    ki=0
    for j in range(len(DF['end_second'])):
        if ki<len(end):
            if (DF['start_seconds'][j]<=end[ki]):
                ep_TVS.append(0)
                ep_TVSB.append(s)
            else:
                #cl_sc=DF['textList'][i] 
                #cl_sc=sent_tokenize(cl_sc)
                s=s+1
                ep_TVS.append(1)
                ep_TVSB.append(s)
                ki=ki+1
        else:
            s=s+1
            ep_TVS.append(1)
            ep_TVSB.append(s)
            
    return ep_TVS,ep_TVSB,len(DF['end_second'])

def episodTruthValueMan(scenesegementedFile,subtitleFile):
    sc_txt,sc_lbl,sc_start,sc_end=sceneTexts(scenesegementedFile,subtitleFile)
    ep_TV=[]
    ep_TVB=[]
    s=0
    for i in sc_txt:
        cl_sc=split_into_sentences(i)
        for j in range(len(cl_sc)):
            ep_TV.append(0)
            ep_TVB.append(s)
        s=s+1
        ep_TV.append(1)
        ep_TVB.append(s)
    return ep_TV,ep_TVB
"""
Takes the start and end time of shot scenes and generate the truth value of sentences in a shot scenes
"""

def episodTruthValueShot(scene_start,scene_end, subtitleFile):
    subs=pysrt.open(subtitle_dir+subtitleFile,encoding='iso-8859-1')
    start=[float(i) for i in scene_start]
    end=[float(i) for i in scene_end]
    part_st=""
    scene_texts=[]
    ep_TVS=[]
    ep_TVSB=[]
    
    for i in range(len(start)):
        m,s,em,es=to_min_sec2(start[i],end[i])
        part_st=subs.slice(starts_after={'minutes': m, 'seconds':s}, ends_before={'minutes': em, 'seconds': es}).text.encode('utf-8')
        scene_texts.append(part_st)

    l=0
    for i in scene_texts:
        cl_sc=split_into_sentences(i)
        for j in range(len(cl_sc)):
            ep_TVS.append(0)
            ep_TVSB.append(l)
        l=l+1
        ep_TVS.append(1)
        ep_TVSB.append(l)
    return ep_TVS,ep_TVSB


    