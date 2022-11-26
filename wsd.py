from nltk.corpus.reader.wordnet import Synset
from nltk.corpus.reader.tagged import TaggedCorpusReader
from nltk.corpus.reader.semcor import SemcorCorpusReader
from nltk.corpus import stopwords as stopWordsReader
from nltk.corpus import treebank, wordnet, brown, semcor
from nltk.tag import BigramTagger, UnigramTagger, DefaultTagger
from nltk.stem import WordNetLemmatizer
import json
import nltk
import re
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

    def __init__(self):
        self.corpus = brown
        self.setupTagger(self.corpus.tagged_sents(tagset="universal"))
        self.trainWord2Vec(self.corpus)
        self.unksense = 'None'

    def setupTagger(self, taggedSents):
        self.stopwords = stopWordsReader.words('english')
        self.stopwords.append('.')
        self.defaultTagger = DefaultTagger('NOUN')
        self.unigramTagger = UnigramTagger(
            taggedSents, backoff=self.defaultTagger)
        self.tagger = BigramTagger(taggedSents, backoff=self.unigramTagger)
        self.lemmatizer = WordNetLemmatizer()
        self.lemmaPOS = {
            'NOUN': 'n',
            'VERB': 'v',
            'ADJ': 'a',
            'ADV': 'r',
            'X': 'n'
        }
        self.specialChar = re.compile("[!@#$%^&*()[]{};:,./<>?\|`~-=_+]")
        
    def __init__(self,demo=False):
        self.corpus = brown
        self.setupTagger(self.corpus.tagged_sents(tagset="universal"))
        self.trainWord2Vec(self.corpus)
        if(demo):
            self.setupSenseFreqTable(self.loadSentsFromCorpus())
        self.unksense = 'None'
    def stopWordsFilter(self,seq:List[str]):
        return list(filter(lambda x:not self.stopwords.__contains__(x),seq))
    def lemmatize(self,seq:List[str]):
        #TODO: test out lemmatization with POS
        return list(map(lambda x:self.lemmatizer.lemmatize(x),seq))
    def trainWord2Vec(self,corpus:TaggedCorpusReader):
        sents = list(corpus.sents())
        self.word2vec = Word2Vec(
            sents, min_count=1, vector_size=W2VLEN, window=5)
        self.wordVectors = self.word2vec.wv.vectors
        self.word2vecinds = self.word2vec.wv.key_to_index
        self.unknownwordvect = np.mean(self.wordVectors, 0)

    def stopWordsFilter(self, seq: List[str]):
        return list(filter(lambda x: not self.stopwords.__contains__(x), seq))

    def lemmatize(self, seq: List[str]):
        # TODO: test out lemmatization with POS
        return list(map(lambda x: self.lemmatizer.lemmatize(x), seq))

    def getWordVecMatrix(self, seq: List[str]):
        vecs = []
        for word in seq:
            if (self.word2vecinds.__contains__(word)):
                vecs.append(self.wordVectors[self.word2vecinds[word]].tolist())
            else:
                vecs.append(self.unknownwordvect.tolist())
        # print(vecs)
        return np.array(vecs).reshape((-1, W2VLEN))

    def getSignature(self, sense: Synset):
        wdef: str = sense.definition()
        wex: List[str] = sense.examples()
        while (re.match(self.specialChar, wdef) != None):
            wdef = re.sub(self.specialChar, " ", wdef)
        words = wdef.split()
        for ex in wex:
            words.extend(ex.split())
        words = self.stopWordsFilter(words)
        return self.getWordVecMatrix(words)

    def computeOverlap(self, signatMat: np.ndarray, contextMat: np.ndarray, method=0):
        """
        Tune this to one of:
        1. a.b
        2. a.b/|a||b|
        3. (a-b)^d
        """
        res = None
        # print(signatMat.shape,contextMat.shape)
        if (method == 0):
            thres = 0.1
            matprod = np.matmul(signatMat, (contextMat.T))
            sgnorm = np.linalg.norm(signatMat, axis=1).reshape((-1, 1))
            ctxnorm = np.linalg.norm(contextMat, axis=1).reshape((1, -1))
            normmat = sgnorm*ctxnorm
            matprod = np.abs(matprod)/normmat
            res = np.mean(matprod[matprod > thres])
        elif (method == 1):
            signatvect = np.mean(signatMat, axis=0)
            ctxvect = np.mean(contextMat, axis=0)
            res = np.dot(signatvect, ctxvect) / \
                (np.linalg.norm(signatvect)*np.linalg.norm(ctxvect))
            res = np.abs(res)
        elif (method == 2):
            signatvect = np.mean(signatMat, axis=0)
            ctxvect = np.mean(contextMat, axis=0)
            res = np.dot(signatvect, ctxvect)
            res = np.abs(res)
        elif (method == 3):
            thres = 0.8
            matprod = np.matmul(signatMat, (contextMat.T))
            sgnorm = np.linalg.norm(signatMat, axis=1).reshape((-1, 1))
            ctxnorm = np.linalg.norm(contextMat, axis=1).reshape((1, -1))
            normmat = sgnorm*ctxnorm
            matprod = np.abs(matprod)/normmat
            res = np.count_nonzero(matprod > thres)
        # print(matprod)
        return res

    def simplifiedLeskOnSent(self, sent: List[Tuple[str]]):
        func = self.simplifiedLesk
        seq = list(map(lambda x: x[0].lower(), sent))
        sent_senses = []
        for j in range(len(sent)):
            wordtuple = sent[j]
            if(wordtuple[1]==nountag):
                sense = func(wordtuple[:2],seq)
                sent_senses.append([j,sense])
        return sent_senses

    def simplifiedLesk(self, wordtag: Tuple[str], seq: List[str]):
        senses: List[Synset] = wordnet.synsets(
            wordtag[0], self.lemmaPOS[wordtag[1]])
        if (len(senses) == 0):
            return self.unksense
        bestSense = senses[0]
        if (len(senses) == 1):
            return bestSense.name()
        maxoverlap = 0
        contxt = self.getWordVecMatrix(seq)
        for sense in senses:
            signat = self.getSignature(sense)
            overlap = self.computeOverlap(signat, contxt, OVERLAP_METHOD)
            if overlap > maxoverlap:
                maxoverlap = overlap
                bestSense = sense
        return bestSense.name()

    def pageRank(self, seq: List[Tuple[str]], d=0.8):
        ambg_word_tuple = []  # (word, tag)
        indices = []
        # print(seq)

        for j in range(len(seq)):
            wordtuple = seq[j]
            #TODO: change wordtuple length
            if (len(wordtuple) == 3 and (wordtuple[1]==nountag or wordtuple[1]=='VERB')):
                ambg_word_tuple.append(wordtuple[:2])
                indices.append(j)

        if ambg_word_tuple == []:
            return []

        senses = []
        for tup in ambg_word_tuple:
            senses.append(
                (wordnet.synsets(tup[0], self.lemmaPOS[tup[1]]))
            )
            if (len(senses[-1]) == 0):
                senses[-1] = [self.unksense]
        
        scores = []
        for i in range(len(senses)):
            l = len(senses[i])
            scores.append([1/(l)]*l)

        edge_wts = []
        for i in range(len(senses) - 1):
            layer_wts = []
            for j in range(len(senses[i])):
                node_wts = []
                for k in range(len(senses[i+1])):
                    if (type(senses[i][j]) == str):
                        signat1 = self.unknownwordvect
                    else:
                        signat1 = self.getSignature(senses[i][j])
                    if (len(signat1.shape) == 1):
                        signat1 = signat1.reshape((1, -1))
                    if (type(senses[i+1][k]) == str):
                        signat2 = self.unknownwordvect
                    else:
                        signat2 = self.getSignature(senses[i+1][k])
                    if (len(signat2.shape) == 1):
                        signat2 = signat2.reshape((1, -1))
                    overlap = self.computeOverlap(signat1, signat2)
                    node_wts.append(overlap)
                if (type(senses[i][j]) != str):
                    senses[i][j] = senses[i][j].name()

                layer_wts.append(node_wts)
            # print('Printing sense[i]',senses[i])
            edge_wts.append(layer_wts)
        # if(len(senses)>2):
        #     exit()
        for j in range(len(senses[-1])):
            if (type(senses[-1][j]) != str):
                senses[-1][j] = senses[-1][j].name()
        
        for iter in range(10):
            for i in range(len(senses)):
                for j in range(len(senses[i])):
                    score = 0
                    if i > 0:
                        for k in range(len(senses[i-1])):
                            score += edge_wts[i-1][k][j] / \
                                (sum(edge_wts[i-1][k])) * scores[i-1][k]
                    if i < len(senses) -1 :
                        for k in range(len(senses[i+1])):
                            denom = 0
                            for f in range(len(senses[i])):
                                denom += edge_wts[i][f][k]
                            score += edge_wts[i][j][k] / \
                                denom * scores[i+1][k]
                    scores[i][j] = (1-d) * scores[i][j] + d * score

        # for i in range(len(senses)):
        #     for j in range(len(senses[i])):
        #         print(senses[i][j], scores[i][j])
        #     print()

        answer = []
        for i in range(len(senses)):
            if ambg_word_tuple[i][1] == nountag:
                argm = np.argmax(np.asarray(scores[i]))
                answer.append([indices[i], senses[i][argm]])

        return answer

    def tokenize(self, seq: str):
        return nltk.word_tokenize(seq)
    def attachSensesTo(self,sent:str,algo):
        sent = sent.lower()
        tkns = self.tokenize(sent)
        tagged_tkns = self.tagger.tag(tkns)
        lemmatkns = self.lemmatize(tkns)

        for i in range(len(tagged_tkns)):
            tagged_tkns[i] = list(tagged_tkns[i])
            tagged_tkns[i][0] = lemmatkns[i]
        senses = []
        defdict = {}
        if(algo=='wfs'):
            senses = self.wfs(tagged_tkns)
        elif(algo=='mfs'):
            senses = self.mfs(tagged_tkns)
        elif(algo=='elesk'):
            senses = self.simplifiedLeskOnSent(tagged_tkns)
        elif(algo=='pr'):
            senses = self.pageRank(tagged_tkns)
        else:
            return None
        j = 0
        # print(senses)
        for i,tgtkn in enumerate(tagged_tkns):
            if(tgtkn[1]==nountag):
                defdict[tgtkn[0]+'@'+str(i)] = senses[j][1]
                j+=1
        return defdict
    def expandSenseDict(self,defdict):
        for key in defdict.keys():
            if(defdict[key]!=self.unksense):
                synset:Synset = wordnet.synset(defdict[key])
                defdict[key] = {
                    'synset':synset.name(),
                    'def':synset.definition(),
                    'examples':synset.examples(),   
                }
            else:
                defdict[key] = {
                    'synset':'unknown',
                    'def':'unclear',
                    'examples':'unspeakable'
                }
        return defdict

    def smallwfs(self, wordtag):
        senses = wordnet.synsets(wordtag[0], self.lemmaPOS[wordtag[1]])
        if (len(senses) == 0):
            return self.unksense
        return senses[0].name()

    def wfs(self, sent):
        sent_senses = []
        for j in range(len(sent)):
            wordtuple = sent[j]
            if(wordtuple[1]==nountag):
                sense = self.smallwfs(wordtuple[:2])
                sent_senses.append([j, sense])
        return sent_senses

    def setupSenseFreqTable(self, sents: List[List[List[str]]]):
        self.sensedict = {}
        for sent in sents:
            for wordtagsent in sent:
                if (wordtagsent[1] == nountag and len(wordtagsent) == 3):
                    word, _, syn = wordtagsent
                    if (word in self.sensedict):
                        if (syn in self.sensedict[word]):
                            self.sensedict[word][syn] += 1
                        else:
                            self.sensedict[word][syn] = 1
                    else:
                        self.sensedict[word] = {}
                        self.sensedict[word][syn] = 1
        with open('data/sensefreq.txt', 'w') as f:
            f.write(json.dumps(self.sensedict, indent=4))

    def mfs(self, sent):
        def smallmfs(wordtag):
            word, tag = wordtag
            if word in self.sensedict:
                syn = max(self.sensedict[word],
                          key=lambda x: self.sensedict[word][x])
                return syn
            else:
                return self.smallwfs(wordtag)
        sent_senses = []
        for j in range(len(sent)):
            wordtuple = sent[j]
            if(wordtuple[1]==nountag):
                sense = smallmfs(wordtuple[:2])
                sent_senses.append([j, sense])
        return sent_senses

    def getBaselines(self):
        print('WFS accuracies: ', end='')
        self.kfoldeval(lambda x: x, self.wfs)
        print('\nMFS accuracies: ', end='')
        self.kfoldeval(self.setupSenseFreqTable, self.mfs)
        print()

    def testOnCorpus(self):
        # print('Lesk accuracies: ',end='')
        # self.kfoldeval(lambda x:x,self.simplifiedLeskOnSent)
        # print()
        print('Page Rank accuracies: ', end='')
        self.kfoldeval(lambda x: x, self.pageRank)
        print()

    def loadSentsFromCorpus(self):
        content = None
        with open('data/tagsemsents.txt', 'r') as f:
            content = f.read()

        sents = json.loads(content)
        for i in range(len(sents)):
            sents[i] = list(map(lambda x: x.split(semtagwordsep), sents[i]))
        return sents

    def kfoldeval(self, trainfunc, func, k=5):
        sents = self.loadSentsFromCorpus()
        sents = np.asanyarray(sents, dtype=object)
        perm = np.random.permutation(len(sents))
        step = int(len(sents)/k)
        sents = sents[perm].tolist()
        for i in range(k):
            # no training in lesk?
            trainsents = sents[:i*step]+sents[i*step+step:]
            testsents = sents[i*step:i*step+step]
            trainfunc(trainsents)
            self.applyMethodOnCorpus(func, testsents)

    def applyMethodOnCorpus(self, func, sents):
        senses = []
        for sent in sents:
            sent_senses = func(sent)
            senses.append(sent_senses)
        self.evaluate(sents, senses)

    def evaluate(self, sents, senses):
        accuracy = 0
        numtests = 0
        failures = []
        for i in range(len(senses)):
            sensesent = senses[i]
            realsent = sents[i]
            for ind, sense in sensesent:
                numtests += 1
                if (realsent[ind][-1] == sense):
                    accuracy += 1
                else:
                    failures.append([realsent, sense, ])
        accuracy = round(100*accuracy/numtests, 2)
        print(accuracy, end=' ')


if __name__ == '__main__':
    np.random.seed(0)
    w = WSD()
    w.testOnCorpus()
    # w.getBaselines()
