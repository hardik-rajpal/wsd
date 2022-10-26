from typing import List
from nltk.corpus import semcor
from nltk.corpus.reader import SemcorCorpusReader
from nltk.tree import Tree
corpus:SemcorCorpusReader = semcor
sents = list(corpus.tagged_sents(tag="sem"))
s:List[Tree] = sents[0]
trees = list(filter(lambda x:str(type(x)).split("'")[1].split(".")[-1]=='Tree',s))
senses = list(map(lambda x:str(x.label()).split("Lemma('")[1].split("')")[0],trees))
print('here')
# print(corpus.xml())
print(s[1])
s[1].node