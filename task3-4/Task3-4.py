
# coding: utf-8

# In[1]:


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


# In[2]:


#nltk is the best 
import nltk
#Couldnt get Glove working sooooo i tried ahah
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[ ]:


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


# In[302]:


#put different vectors files here
with open('vectors_ap8889_cbow_s300_w5_neg20_hs0_sam1e-4_iter5.txt', 'r') as myfile:
    data=myfile.read()

#Split data into list by newlines
data = re.split('\n', data)

#make it a dictionary for ease of use
datadict = {}
for i in range(0,len(data)):
   data[i] = re.split('\s', data[i])
   del data[i][-1]
   datadict[data[i][0]] = data[i]
   del datadict[data[i][0]][0]
   
#numbers ?
for word in datadict:
    for k in range(0,len(datadict[word])):
        datadict[word][k] = float(datadict[word][k])


            
with open('queries.txt', 'r') as myfile:
    querydata=myfile.read()
#Split data into 2-D array by spaces
for i in range(0,len(querydata)):
    querydata[i] = re.split('\s', querydata[i])
    #remove all the empties
    querydata[i] = list(filter(lambda a: a != '', querydata[i]))
del querydata[0]


for i in range(0,len(querydata)):
    querybest = ''
    #we'll only accept query extensions with a similarity above 0
    querysimil = 0
    dataminus = dict(datadict)
    for j in range(0,len(querydata[i])):
        #can set to lowercase here
        querydata[i][j] = querydata[i][j].lower()
        #remove because we don't want to evaluate any terms against terms already
        #in the query
        dataminus.pop(querydata[i][j], None)
    for j in range(0,len(querydata[i])):
        if querydata[i][j] in datadict:
            for word in dataminus:
                topside = 0
                bottomleft = 0
                bottomright = 0
                for k in range(0,len(dataminus[word])):
                    topside += datadict[querydata[i][j]][k] * dataminus[word][k]
                    bottomleft += datadict[querydata[i][j]][k] ** 2
                    bottomright += dataminus[word][k] ** 2
                cosine = topside / math.sqrt(bottomleft*bottomright)
                if (cosine > querysimil):
                    #if cosine similarity is higher, make it the new record
                    querybest = word
                    querysimil = cosine
    print(querybest)
    querydata[i].append(querybest)

