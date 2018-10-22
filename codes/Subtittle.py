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
import pysub
import nlpPreprocessing
class Subtitle:

    def __init__(self,subFile):
        self.subFile=subFile
        self.subtitleTexts=[]
        self.start=[]
        self.end=[]

    def readSub(self):
        self.subs=pysub.read(self.subFile)
        for i in self.subs:
            self.start.append(()(i.start.hours*60)+i.start.seconds+(i.start.milliseconds/1000)))
            self.end.append(()(i.end.hours*60)+i.end.seconds+(i.end.milliseconds/1000)))
            self.subtitleTexts.append(i.texts)
        return self.start,self.end,self.subtitleTexts

    def getsubSentences(self, startTime,endTime):
        #self.subs=pysub.read(self.subFile)
        texts=self.subs.slice(starts_after={'minutes':(int(startTime/60)),'seconds':(startTime%60)},
        end_before={'minutes':(int(endTime/60)),'seconds':(endTime%60))

        self.sentences=nlpPreprocessing.sentenceSplit(texts.text)

        return self.sentences
