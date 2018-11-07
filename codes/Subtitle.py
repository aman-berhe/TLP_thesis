"""
Input: subtitle file
        start=
        end=endof(subtitle)
functions--> get subtitlext (start,end)
         --> get subSentences (start, end, remove=otional)
                --> remove=binary: to remove texts that are not speech
         --> alignSub(bySeconds)
                --> to forward or backward the subtitle if -bySeconds:
                fasting the subtile other wise delay subtitlext
        -->

outputs-->
"""
import pysrt
#import nlpPreprocessing

Got='/TLP_thesis/subtitles/GoT/English'
BBT='/TLP_thesis/subtitles/BBT/English'
HP='/TLP_thesis/subtitles/HarryPotter/English'

class Subtitle:

    def __init__(self,subFile):
        self.subFile=subFile
        self.subtitleTexts=[]
        self.start=[]
        self.end=[]
        self.subDir=""

    def readSub(self):
        """
            Reads the subtitle file (.srt) and returns all the texts and their starting and endind time as lists
        """
        self.subs=pysub.open(self.subFile)
        for i in self.subs:
            self.start.append(((i.start.hours*60)+i.start.seconds+(i.start.milliseconds/1000)))
            self.end.append(((i.end.hours*60)+i.end.seconds+(i.end.milliseconds/1000)))
            self.subtitleTexts.append(i.texts)
        return self.start,self.end,self.subtitleTexts

    def getsubSentences(self, startTime,endTime):
        """
        Takes the subtitle file with a starting and ending time (in seconds) and returns part of the title text in the interval
        """
        self.subs=pysrt.open(self.subFile)
        texts=self.subs.slice(starts_after={'minutes':(int(startTime/60)),'seconds':(startTime%60)},ends_before={'minutes':(int(endTime/60)),'seconds':(endTime%60)})

        #self.sentences=nlpPreprocessing.sentenceSplit(texts.text)
        if (len(texts.text)==0):
            return 'The Segment Does not have a Text/Dialogue!'
        else:
            return texts.text, texts
