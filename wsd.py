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
import numpy as np
from gensim.models import Word2Vec
from typing import List, Tuple

W2VLEN = 10
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

    def trainWord2Vec(self, corpus: TaggedCorpusReader):
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

    def computeOverlap(self, signatMat: np.ndarray, contextMat: np.ndarray):
        """
        Tune this to one of:
        1. a.b
        2. a.b/|a||b|
        3. (a-b)^d
        """
        thres = 0.1
        matprod = np.matmul(signatMat, (contextMat.T))
        sgnorm = np.linalg.norm(signatMat, axis=1).reshape((-1, 1))
        ctxnorm = np.linalg.norm(contextMat, axis=1).reshape((1, -1))
        normmat = sgnorm*ctxnorm
        matprod = np.abs(matprod)/normmat
        # print(matprod)
        return np.mean(matprod[matprod > thres])

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
            overlap = self.computeOverlap(signat, contxt)
            if overlap > maxoverlap:
                maxoverlap = overlap
                bestSense = sense
        return bestSense.name()

    def overlapTestMat(self):
        a = self.getWordVecMatrix(
            ['boy', 'man', 'lion', 'jungle', 'forest', 'hunt'])
        print(self.computeOverlap(a, a))

    def pageRank(self, seq: List[str], d=0.8):

        ambg_word_tuple = []  # (word, tag)
        indices = []
        # print(seq)
        for j in range(len(seq)):
            wordtuple = seq[j]
            if (len(wordtuple) == 3):
                ambg_word_tuple.append(wordtuple[:2])
                indices.append(j)

        if ambg_word_tuple == []:
            return []

        senses = []
        for tup in ambg_word_tuple:
            senses.append(wordnet.synsets(tup[0], self.lemmaPOS[tup[1]]))

        scores = []
        for i in range(len(senses)):
            scores.append([1/len(senses[i])]*len(senses[i]))

        edge_wts = []
        for i in range(len(senses) - 1):
            layer_wts = []
            for j in range(len(senses[i])):
                node_wts = []
                for k in range(len(senses[i+1])):
                    signat1 = self.getSignature(senses[i][j])
                    signat2 = self.getSignature(senses[i+1][k])
                    overlap = self.computeOverlap(signat1, signat2)
                    node_wts.append(overlap)
                layer_wts.append(node_wts)
            edge_wts.append(layer_wts)

        for i in range(1, len(senses)):
            for j in range(len(senses[i])):
                score = 0
                for k in range(len(senses[i-1])):
                    score += edge_wts[i-1][k][j] / \
                        (sum(edge_wts[i-1][k])) * scores[i-1][k]
                scores[i][j] = (1-d) * scores[i][j] + d * score
        
        for i in range(len(senses)):
            for j in range(len(senses[i])):
                print(senses[i][j].name(), scores[i][j])
            print()

        answer = []
        for i in range(len(senses)):
            if ambg_word_tuple[i][1] == nountag:
                argm = np.argmax(np.asarray(scores[i]))
                answer.append([indices[i], senses[i][argm].name()])

        return answer

    def tokenize(self, seq: str):
        return nltk.word_tokenize(seq)

    def     attachSensesTo(self, sent: str, useLesk=True):
        sent = sent.lower()
        tkns = self.tokenize(sent)
        tagged_tkns = self.tagger.tag(tkns)
        lemmatkns = self.lemmatize(tkns)

        for i in range(len(tagged_tkns)):
            tagged_tkns[i] = list(tagged_tkns[i])
            tagged_tkns[i][0] = lemmatkns[i]

        ctxwordsLemmas = self.stopWordsFilter(lemmatkns)
        print(ctxwordsLemmas)
        senses = []
        defdict = {}

        for tgtkn in tagged_tkns:
            if (tgtkn[1] == 'NOUN'):
                if (useLesk):
                    bestSense = self.simplifiedLesk(tgtkn, ctxwordsLemmas)
                else:
                    bestSense = self.pageRank(tgtkn, ctxwordsLemmas)

                senses.append(bestSense)
                defdict[tgtkn[0]] = bestSense

        return defdict

    def testOnCorpus(self, useLesk=False):
        content = None
        with open('data/tagsemsents.txt', 'r') as f:
            content = f.read()

        sents = json.loads(content)
        for i in range(len(sents)):
            sents[i] = list(map(lambda x: x.split(semtagwordsep), sents[i]))

        senses = []
        for sent in sents[:1]:
            seq = list(map(lambda x: x[0].lower(), sent))
            sent_senses = []

            if useLesk:
                for j in range(len(sent)):
                    wordtuple = sent[j]

                    if (wordtuple[1] == nountag and len(wordtuple) == 3):
                        sense = self.simplifiedLesk(wordtuple[:2], seq)
                        sent_senses.append([j, sense])

                senses.append(sent_senses)

            else:
                sent_senses = self.pageRank(sent)
                senses.append(sent_senses)

        self.evaluate(sents, senses)

    def evaluate(self, sents, senses):
        accuracy = 0
        numtests = 0
        for i in range(len(senses)):
            sensesent = senses[i]
            realsent = sents[i]
            for ind, sense in sensesent:
                numtests += 1
                if (realsent[ind][-1] == sense):
                    accuracy += 1
        accuracy = round(100*accuracy/numtests, 2)
        print(accuracy)


if __name__ == '__main__':
    w = WSD()
    w.testOnCorpus()
