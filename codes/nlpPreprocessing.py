"""
This is NLP preprocessing of texts
Input--> text (could be part of subtitltes)
functions-->sentenceSplit(text)
                --> return list of sentences
        --->clearText(text)
                --> cleans texts with brackets and some un necessary pancutation
        --->sentimentAnalysis(partSubttile)
                --> takespart of subtitle or a text and returns the sentiment polarity
"""
from nltk import *
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob


class preProcessing:

    def __init__(self, textInput):
        self.textInput=textInput
        self.delta=2
        self.text=""

    """
    gets a text as inpus and returns sequence of sentences using
    textblob library
    """
    def getSentences(self, text):
        blob=TextBlob(text)
        sentences=[]
        for i in blob.sentences:
            sentences.append(i)

        return sentences
    def correctSpelling(sentences):
        corrected=[]
        for s in sentences:
            spell=TextBlob(s)
            spell=spell.correct()
            for sp in spell.sentences:
                corrected.append(sp)
        return corrected




    def removeStopwords(self,text):
        self.text=text
        stopWords = set(stopwords.words('english'))
        words = word_tokenize(text)
        wordsFiltered = []
        for w in words:
            if w not in stopWords:
                wordsFiltered.append(w)
        return wordsFiltered

"""
Compute part or whole subtitle sentiment analysis using interval delta=2
"""
    def sentiment_anal(self,subs,delta=2):
        #subs=pysrt.open(file,encoding='iso-8859-1')
        n=len(subs)
        intervals=[]
        start=time(0,0,0)
        end=subs[-1].end.to_time()
        delta=timedelta(minutes=delta)
        for results in create_intervals(start, end, delta):
            intervals.append(results)
        sentiments=[]
        index=0
        m=len(intervals)
        for i in range(m):
            text=""
            for j in range(index,n):
                if subs[j].end.to_time()<intervals[i]:
                    text+=subs[j].text_without_tags + " "
                else:
                    break
            blob=TextBlob(text)
            pol=blob.sentiment.polarity
            sentiments.append(pol)
            index =j
        intervals.insert(0,time(0,0,0))
        sentiments.insert(0,0.0)
        #for k in range(0,n):
            #print(intervals[k])
            #print(subs[k].text)
        return (intervals, sentiments,text)
