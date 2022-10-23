from typing import List
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import Synset
from wsd import WSD
#Input: A sentence of paragraph
#Output: Sentence/paragraph, with the NOUNS tagged with their senses from wordnet.
#Processing steps:
#MFS most used sense in semcor
#WFS = first sense in wordnet.synsets
senses:List[Synset] = wordnet.synsets('gay')
mfs = senses[0]
for sense in senses[1:]:
    print(sense,sense.definition(),sense.examples(),'\n')
"""
0. get_signature(def,examples)
    wordvecs = []
    for nonstopword in combined(def,examples):
        getwordvec(nonstopword)
    return wordvecs
1. compute_overlap(list[vectors],list[vectors]):
    return mean(list1 x list2)
1. POS Tagging
senselist = []
3. function SIMPLIFIED_LESK(word,sentence) returns best sense of word
    best-sense <- most frequent sense for word
    max-overlap <- 0
    context  = getcontext(sequence)
    for each sense in senses of word do
        signature getSignature(sense.definition(),sense.examples())
        overlap <- COMPUTEOVERLAP (signature,context)
        if overlap > max-overlap then
            max-overlap <- overlap
            best-sense <- sense
    end return (best-sense)
2. context = filter(sequence,not_a_stop_word)
   for word in filter(sequence,tag=NOUN):
        best_sense = SimplifiedLesk(word,context)
        senselist.append(best_sense)

"""

#get vector of target word 