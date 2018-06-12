
# coding: utf-8

# In[ ]:

#Part three attempt 
from __future__ import absolute_import, division, print_function

#nomral os stuff 
import os
import pprint
import re
import logging
import codecs
import glob
import multiprocessing


# In[278]:

#nltk is the best 
import nltk
#Couldnt get Glove working sooooo i tried ahah
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[279]:

data_filenames = sorted(glob.glob("query/queryall.trec"))


# In[280]:

print("Found all the things:")
data_filenames


# In[281]:

nltk.download("stopwords")


#formating or trying to anyway 
corpus_raw = u""
for data_filename in data_filenames:
    print("Reading '{0}'...".format(data_filename))
    with codecs.open(data_filename, "r", "utf-8") as data_file:
        corpus_raw += data_file.read()
    print("Corpus is now {0} characters long".format(len(corpus_raw)))
    print()


# In[282]:

#nltk is life 
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# In[283]:

raw_data = tokenizer.tokenize(corpus_raw)


# In[284]:

#Trying to remove all the shit 
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words


# In[285]:

sentences = []
for raw_sentence in raw_data:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))


# In[286]:

#kinda working, need to remove top, num etc
print(sentence_to_wordlist(raw_data[0]))


# In[287]:

token_count = sum([len(sentence) for sentence in sentences])
print("The data corpus contains {0:,} tokens of stuff".format(token_count))


# In[288]:

#features can change for hopefully better results 
num_features = 300

# min word count 4 is my cut of could be 2 or 3 
min_word_count = 3
num_workers = multiprocessing.cpu_count()
#context size 
context_size = 10

downsampling = 1e-3
# Seed for the RNG, to make the results reproducible.
#random number generator
#deterministic, good for debugging
seed = 1


# In[293]:

data2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)


# In[234]:

#Build im so confused
data2vec.build_vocab(sentences)


# In[235]:

#well thats wrong 
print("Word2Vec vocabulary length:", len(data2vec.wv.vocab))


# In[245]:

data2vec.train(sentences, total_words=token_count, epochs=10)


# In[246]:

data2vec.save(os.path.join("trained", "data2.w2v"))


# In[247]:

data2vec = glove2word2vec.Word2Vec.load(os.path.join("trained", "data2.w2v"))


# In[305]:

print(data2vec.wv.vocab)


# In[248]:

def nearest_similarity_cosmul(start1, end1, end2):
    similarities = data2vec.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2


# In[254]:

data2vec.most_similar("TRADING")


# In[256]:

# could be used for query expansion 
nearest_similarity_cosmul("COMPUTER", "CONTROL", "EQUIPMENT")


# In[2]:

#Part four 
# need both query and raw_data
# raw_data = all the data 
# need the vectors
import numpy as np

#get doc id will need to compare later
def get_doc_id(raw):
    clean = re.sub("[^1-9]"," ", raw)
    words = clean.split()
    return words

#turn all the word into vectors
vec1 = np.vectorize(raw_data)
#get doc id 
vec2 = get_doc_id(raw_data)

#Query Expansion
#compare terms in the vectors with word embedding in the word vector with the docid, get top k and store it
scores = []

for term in vec1:
        ##Compare this terms with words
        for word in queryTerms:
            word=word.lower()
            if word in vectors:
                termScore += (1 - nearest_similarity_cosmul((vectors[term.lower()], vectors[word]))
        termQscores.append((term,termScore))
##similarity:
termQscores = sorted(termQscores, key=lambda tup: tup[1])

##Add the j best candidate terms to the query.
for t in scores[0:j]:
    queryTerms.append(t)
expandedQueries.append(queryTerms)


# In[ ]:




# In[ ]:



