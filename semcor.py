import json
from typing import List

from nltk.corpus import semcor
from nltk.corpus.reader import SemcorCorpusReader
from nltk.tag.mapping import tagset_mapping
from nltk.tree import Tree

corpus:SemcorCorpusReader = semcor
# sents1 = list(corpus.chunk_sents())
semtagwordsep = '**'
mapping = tagset_mapping('en-ptb','universal')
globsents = []
for j in range(len(corpus._fileids)):
    fileid = corpus._fileids[j]
    sents = list(corpus.tagged_sents(fileids=[fileid],tag="both"))[:2]
    nountag = 'NN'
    for i in range(len(sents)):
        s = sents[i]
        new_sent = []
        for tk in s:
            tk:Tree
            subtrees = list(tk.subtrees())
            labels = []
            for subtree in subtrees:
                labels.append(subtree.label())
            labels.append('_'.join(subtree.leaves()))
            # print(labels)
            if(None in labels):continue
            if(nountag in labels):
                new_sent.append(semtagwordsep.join([labels[-1],mapping[nountag],labels[0]]))
            else:
                new_sent.append(semtagwordsep.join([labels[-1],mapping[labels[-2]]]))
        sents[i] = new_sent
    globsents.extend(sents)
    print(f'Done: ',round(100*j/len(corpus._fileids),2))
fname = 'tagsemsents'
with open(f'data/{fname}.txt','w+') as f:
    f.write(json.dumps(globsents))