from nltk.corpus.reader.wordnet import Synset
from nltk.corpus.reader.tagged import TaggedCorpusReader
from nltk.corpus.reader.semcor import SemcorCorpusReader
from nltk.corpus import stopwords as stopWordsReader
from nltk.corpus import treebank,wordnet,brown,semcor
from nltk.tag import BigramTagger,UnigramTagger,DefaultTagger
from nltk.stem import WordNetLemmatizer
import json
import nltk;import re
import pickle as pkl
import numpy as np
from gensim.models import Word2Vec
from typing import List, Tuple
W2VLEN = 10
OVERLAPMS = [
    'matmul',
    'mean.cos',
    'mean.dot',
    'matmul.thres.count',
]
OVERLAP_METHOD = 3
semtagwordsep = '**'
nountag = 'NOUN'
class WSD:
    def setupTagger(self,taggedSents):
        self.stopwords = stopWordsReader.words('english')
        self.stopwords.append('.')
        self.defaultTagger = DefaultTagger('NOUN')
        self.unigramTagger = UnigramTagger(taggedSents,backoff=self.defaultTagger)
        self.tagger = BigramTagger(taggedSents,backoff=self.unigramTagger)
        self.lemmatizer = WordNetLemmatizer()
        self.lemmaPOS = {
            'NOUN':'n',
            'VERB':'v',
            'ADJ':'a',
            'ADV':'r',
            'X':'n'
        }
        self.specialChar = re.compile("[!@#$%^&*()[]{};:,./<>?\|`~-=_+]")
        
    def __init__(self):
        self.corpus = brown
        self.setupTagger(self.corpus.tagged_sents(tagset="universal"))
        self.trainWord2Vec(self.corpus)
        self.unksense = 'None'
    def stopWordsFilter(self,seq:List[str]):
        return list(filter(lambda x:not self.stopwords.__contains__(x),seq))
    def lemmatize(self,seq:List[str]):
        #TODO: test out lemmatization with POS
        return list(map(lambda x:self.lemmatizer.lemmatize(x),seq))
    def trainWord2Vec(self,corpus:TaggedCorpusReader):
        sents = list(corpus.sents())
        self.word2vec = Word2Vec(sents,min_count = 1, vector_size = W2VLEN, window = 5)
        self.wordVectors = self.word2vec.wv.vectors
        self.word2vecinds = self.word2vec.wv.key_to_index
        self.unknownwordvect = np.mean(self.wordVectors,0)
    def getWordVecMatrix(self,seq:List[str]):
        vecs = []
        for word in seq:
            if(self.word2vecinds.__contains__(word)):
                vecs.append(self.wordVectors[self.word2vecinds[word]].tolist())
            else:
                vecs.append(self.unknownwordvect.tolist())
        # print(vecs)
        return np.array(vecs).reshape((-1,W2VLEN))
    def getSignature(self,sense:Synset):
        wdef:str = sense.definition()
        wex:List[str] = sense.examples()
        while(re.match(self.specialChar,wdef)!=None):
            wdef = re.sub(self.specialChar, " ", wdef)
        words = wdef.split()
        for ex in wex:
            words.extend(ex.split())
        words = self.stopWordsFilter(words)
        return self.getWordVecMatrix(words)
    def computeOverlap(self,signatMat:np.ndarray,contextMat:np.ndarray,method=0):
        """
        Tune this to one of:
        1. a.b
        2. a.b/|a||b|
        3. (a-b)^d
        """
        res = None
        if(method==0):
            thres = 0.1
            matprod = np.matmul(signatMat,(contextMat.T))
            sgnorm = np.linalg.norm(signatMat,axis=1).reshape((-1,1))
            ctxnorm = np.linalg.norm(contextMat,axis=1).reshape((1,-1))
            normmat = sgnorm*ctxnorm
            matprod = np.abs(matprod)/normmat
            res = np.mean(matprod[matprod>thres])
        elif(method==1):
            signatvect = np.mean(signatMat,axis=0)
            ctxvect = np.mean(contextMat,axis=0)
            res = np.dot(signatvect,ctxvect)/(np.linalg.norm(signatvect)*np.linalg.norm(ctxvect))
            res = np.abs(res)
        elif(method==2):
            signatvect = np.mean(signatMat,axis=0)
            ctxvect = np.mean(contextMat,axis=0)
            res = np.dot(signatvect,ctxvect)
            res = np.abs(res)
        elif(method==3):
            thres = 0.8
            matprod = np.matmul(signatMat,(contextMat.T))
            sgnorm = np.linalg.norm(signatMat,axis=1).reshape((-1,1))
            ctxnorm = np.linalg.norm(contextMat,axis=1).reshape((1,-1))
            normmat = sgnorm*ctxnorm
            matprod = np.abs(matprod)/normmat
            res = np.count_nonzero(matprod>thres)
        # print(matprod)
        return res
    def simplifiedLesk(self,wordtag:Tuple[str],seq:List[str]):
        senses:List[Synset] = wordnet.synsets(wordtag[0],self.lemmaPOS[wordtag[1]])
        if(len(senses)==0):
            return self.unksense
        bestSense = senses[0]
        if(len(senses)==1):
            return bestSense.name()
        maxoverlap = 0
        contxt = self.getWordVecMatrix(seq)
        for sense in senses:
            signat = self.getSignature(sense)
            overlap = self.computeOverlap(signat,contxt,OVERLAP_METHOD)
            if overlap > maxoverlap:
                maxoverlap = overlap
                bestSense = sense
        return bestSense.name()
    def pageRank(self,wordtag:Tuple[str],seq:List[str]):
        pass
    def tokenize(self,seq:str):
        return nltk.word_tokenize(seq)
    def attachSensesTo(self,sent:str,useLesk=True):
        sent = sent.lower()
        tkns = self.tokenize(sent)
        tagged_tkns = self.tagger.tag(tkns)
        lemmatkns = self.lemmatize(tkns)
        for i in range(len(tagged_tkns)):
            tagged_tkns[i] = list(tagged_tkns[i])
            tagged_tkns[i][0] = lemmatkns[i]
        ctxwordsLemmas = self.stopWordsFilter(lemmatkns)
        senses = []
        defdict = {}
        for tgtkn in tagged_tkns:
            if(tgtkn[1]=='NOUN'):
                if(useLesk):
                    bestSense = self.simplifiedLesk(tgtkn,ctxwordsLemmas)
                else:
                    bestSense = self.pageRank(tgtkn,ctxwordsLemmas)
                senses.append(bestSense)
                defdict[tgtkn[0]] = bestSense
        return defdict
    def wfs(self,wordtag:Tuple[str],seq):
        senses = wordnet.synsets(wordtag[0],self.lemmaPOS[wordtag[1]])
        if(len(senses)==0):
            return self.unksense
        return senses[0].name()
    def setupSenseFreqTable(self,sents:List[List[List[str]]]):
        self.sensedict = {}
        for sent in sents:
            for wordtagsent in sent:
                if(wordtagsent[1]=='NOUN' and len(wordtagsent)==3):
                    word,_,syn = wordtagsent
                    if(word in self.sensedict):
                        if(syn in self.sensedict[word]):
                            self.sensedict[word][syn]+=1
                        else:
                            self.sensedict[word][syn] = 1
                    else:
                        self.sensedict[word] = {}
                        self.sensedict[word][syn] = 1
        with open('data/sensefreq.txt','w') as f:
            f.write(json.dumps(self.sensedict,indent=4))
    def mfs(self,wordtag:Tuple[str],seq):
        word,tag = wordtag
        if word in self.sensedict:
            syn = max(self.sensedict[word],key=lambda x:self.sensedict[word][x]) 
            return syn
        else:
            return self.unksense
    def getBaselines(self):
        sents = self.loadSentsFromCorpus()
        self.setupSenseFreqTable(sents)
        print('WFS accuracy: ',end='')
        self.applyMethodOnCorpus(self.wfs)
        print('MFS accuracy: ',end='')
        self.applyMethodOnCorpus(self.mfs)
    def testOnCorpus(self):
        self.applyMethodOnCorpus(self.simplifiedLesk)
    def loadSentsFromCorpus(self):
        content = None
        with open('data/tagsemsents.txt','r') as f:
            content = f.read()
        sents = json.loads(content)
        for i in range(len(sents)):
            sents[i] = list(map(lambda x:x.split(semtagwordsep),sents[i]))
        return sents
    def applyMethodOnCorpus(self,func):
        sents = self.loadSentsFromCorpus()
        senses = []
        for sent in sents:
            seq = list(map(lambda x:x[0].lower(),sent))
            sent_senses = []
            for j in range(len(sent)):
                wordtuple = sent[j]
                if(wordtuple[1]==nountag and len(wordtuple)==3):
                    sense = func(wordtuple[:2],seq)
                    sent_senses.append([j,sense])
            senses.append(sent_senses)
        self.evaluate(sents,senses)
    def evaluate(self,sents,senses):
        accuracy = 0
        numtests = 0
        for i in range(len(senses)):
            sensesent = senses[i]
            realsent = sents[i]
            for ind,sense in sensesent:
                numtests+=1
                if(realsent[ind][-1]==sense):
                    accuracy+=1
        accuracy = round(100*accuracy/numtests,2)
        print(accuracy)
if __name__=='__main__':
    w = WSD()
    w.getBaselines()