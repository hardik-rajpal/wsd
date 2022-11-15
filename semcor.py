from typing import List
from nltk.corpus import semcor
from nltk.corpus.reader import SemcorCorpusReader
from nltk.tree import Tree
corpus:SemcorCorpusReader = semcor
# sents1 = list(corpus.chunk_sents())
sents = list(corpus.tagged_sents(fileids=[corpus._fileids[0]],tag="both"))[:2]
for i in range(len(sents)):
    s = sents[i]
    for tk in s:
        tk:Tree
        print(tk,tk.label(),tk.draw(),tk.leaves())
    exit()
# print(sents[:2])
# s:List[Tree] = sents[0]
# trees = list(filter(lambda x:str(type(x)).split("'")[1].split(".")[-1]=='Tree',s))
# senses = list(map(lambda x:str(x.label()).split("Lemma('")[1].split("')")[0],trees))
# # print(corpus.xml())
# print(s[1])
# s[1].node