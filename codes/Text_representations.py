from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from textblob import TextBlob
import nlpPreprocessing as nlp
import os
import pysrt

file1='/home/berhe/Desktop/Thesis_git/TLP_thesis/Descriptions/AClashOfKings.txt'
file2='/home/berhe/Desktop/Thesis_git/TLP_thesis/Descriptions/AGameOfThrones.txt'
file3='/home/berhe/Desktop/Thesis_git/TLP_thesis/Descriptions/ADanceWithDragons.txt'
file4='/home/berhe/Desktop/Thesis_git/TLP_thesis/Descriptions/AFeastForCrows.txt'


class TextEmbeddings:
    def __init__(self):
        self.fileName=file2
        self.Got='/home/berhe/Desktop/Thesis_git/TLP_thesis/subtitles/GoT/English/'
        self.BBT='/home/berhe/Desktop/Thesis_git/TLP_thesis/subtitles/BBT/English/'
        self.HP='/home/berhe/Desktop/Thesis_git/TLP_thesis/subtitles/HarryPotter/English/'


    def bookW2V(self,fi=file2):
        #self.fileName=fileName

        f=open(fi,'r')
        text=f.read()#.decode('utf-8')
        sentences=[]
        txtblb=TextBlob(text)
        for sent in txtblb.sentences:
            sentences.append(sent.split())
        model=Word2Vec(sentences=sentences,min_count=1,size=300)
        X=model[model.wv.vocab]
        #X is word vectors
        words=list(model.wv.vocab)
        return model,sentences,words

    def getSentencesSubtitle(self):
        letter=input('Enter the beginning letter of the tv-series: Game of Thrones(GoT), Bing bang theory(BBT), Harry Potter(HP)')
        if letter == 'G' or letter == 'g':
            subDir=self.Got
        elif  letter == 'B' or letter == 'b':
            subDir=self.BBT
        elif letter == 'H' or letter == 'h':
            subDir=self.HP

        text=""
        for fl in os.listdir(subDir):
            if '.en.srt' in fl:
                print(fl)
                file=subDir+fl
                subs=pysrt.open(file,encoding='iso-8859-1')
                for s in subs:
                    text=text+s.text
        sentences=[]
        txtblb=TextBlob(text)
        for sent in txtblb.sentences:
            sentences.append(sent.split())

        model=Word2Vec(sentences=sentences,min_count=1,size=300)
        X=model[model.wv.vocab]
        #X is word vectors
        words=list(model.wv.vocab)
        return model,sentences,words


    def plotW2V(model):
        X=model[model.wv.vocab]
        pca=PCA(n_components=2)
        result=pca.fit_transform(X)
        pyplot.scatter(result[:, 0],result[:, 1])
        for i, word in enumerate(words):
            pyplot.annotate(word, xy=(result[i,0], result[i,1]))
        pyplot.show()
