#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: dmytrenko.o
"""

from __modules__ import packagesInstaller
packages = ['nltk', 'pandas', 'math', 'numpy']
packagesInstaller.setup_packeges(packages)

import math
from nltk import FreqDist
import pandas as pd
import numpy as np


def get_key(d, value):
    for k, v in d.items():
        if v == value:
            return k

def time_series(doc, vectorTerms):
    # get vector of all unique words from all docs
    fdist = FreqDist(term for term in doc)
    #norm = sum(freq for termtag, freq in fdist)
    #print (fdist.most_common())
    Tseries = []
    for term in doc:
        if (term in vectorTerms):
            Tseries.append(fdist[term])
    return Tseries


def get_vector_terms(doc):
    fdist = FreqDist(term for term in doc)
    print (fdist.most_common(20)[:])
    return fdist.most_common(20)[:]

# get vector of GTF that corresponds to vectorWords for each doc
def GTF_vector(docs, vectorWords):
    # create dataframe of TF-IDF where indeces are vectorWords and colomns are number of documents
    df = pd.DataFrame(index=[word[0] for (word,freq) in vectorWords])
    column = 0
    # calculate number of all words in all documents
    norm = sum(len(doc) for doc in docs)
    
    with pd.ExcelWriter('GTF.xlsx') as output:
        for doc in docs:
            GTFs = []
            column += 1
            for (word, freq) in vectorWords:
                if word in doc:
                    GTFs.append(freq/norm)
                else:
                    GTFs.append(0)
            # add verctor of GTF to dataframe 
            df[str(column)] = GTFs
            df2 = pd.DataFrame(GTFs, index=[word[0] for (word,freq) in vectorWords], columns=["GTF"]) 
            df2 = df2.sort_values(by=["GTF"],ascending=False)
            df2.to_excel(output, sheet_name=str(column))
        
        df.to_excel(output, sheet_name='Matrix')
        print (df.to_numpy)
    return

# get vector of TF-IDF that corresponds to vectorWords for each doc
def TFIDF_vector(docs, vectorWords):
    # create dataframe of TF-IDF where indeces are vectorWords and colomns are number of documents
    df = pd.DataFrame(index=[word for (word,freq) in vectorWords])#, columns=list(range(1, len(docs))))
    column = 0
    
    for doc in docs:
        TFIDFs = []
        column += 1
        for (word, freq) in vectorWords:
            TF = 1.0*doc.count(word) / len(doc)
            IDF = math.log(1.0 * len(docs) / (sum(1 for doc in docs if word in doc)))
            TFIDF = TF*IDF
            TFIDFs.append(TFIDF)
        # add verctor of TF-IDF to dataframe 
        df[str(column)] = TFIDFs
        df.to_excel('TF-IDF.xlsx')
    return
    
def matrix_positions2(doc):
    vectorTerms = get_vector_terms(doc)
    keyTerms = [term for (term, freq) in vectorTerms] 
    
    # build new document from only keyTerms
    newdoc = []
    for term in doc:
        if term in keyTerms:
            newdoc.append(term)
            
    Tseries = []
    for termtag in keyTerms:
        tserie = [0.0]*len(newdoc) # tserie for key termtag 
        # return all indexes (positions) of "termtag" in "newdoc"
        indexes = [index for index, value in enumerate(newdoc) if value == termtag]
        for i in indexes:
            tserie[i] = 1
        Tseries.append(np.array(tserie)) # list of tseries
    return np.array(Tseries)
            