from typing import List
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import Synset,CorpusReader,WordNetCorpusReader
from wsd import WSD
wordnet:WordNetCorpusReader
meanings = wordnet.synsets('dog')
m:Synset = meanings[0]
print([x.lemma_names() for x in m.hypernyms()])