#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Олег Дмитренко

"""
import sys
from __modules__ import packagesInstaller
from __modules__ import modelsLoader, stopwordsLoader
packages = ['fasttext', 'nltk']
packagesInstaller.setup_packeges(packages)

import fasttext
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

stdOutput = open("outlog.log", "a")
sys.stderr = stdOutput
sys.stdout = stdOutput

def remove_terms(TermsTags, num):
    TermsTags = [i for i in TermsTags if i[1].count('~') != num]
    return TermsTags

def stanza_built_words(sent, TermsTags, stopWords):
    WordsTags = []
    for word in sent.words:
        nword = word.lemma
        tag = word.upos
        if tag == 'PROPN':
            tag = 'NOUN'
        WordsTags.append((nword,tag))
        if (nword not in stopWords) and (tag == 'NOUN'): 
            TermsTags.append((nword, tag))
    return TermsTags, WordsTags

def stanza_built_bigrams(WordsTags, TermsTags, stopWords):
    for i in range(1, len(WordsTags)):
        w1 = WordsTags[i-1][0] 
        w2 = WordsTags[i][0]
        t1 = WordsTags[i-1][1]
        t2 = WordsTags[i][1]
        if (t1 == 'ADJ') and (t2 == 'NOUN') and (w1 not in stopWords) and (w2 not in stopWords):
            TermsTags.insert([index for index, value in enumerate(TermsTags) if value == (w2,t2)][-1], (w1+'~'+w2, t1+'~'+t2))
    return TermsTags

def stanza_built_threegrams(WordsTags, TermsTags, stopWords):
    for i in range(2, len(WordsTags)):
        w1 = WordsTags[i-2][0]
        w2 = WordsTags[i-1][0]
        w3 = WordsTags[i][0]
        t1 = WordsTags[i-2][1]
        t2 = WordsTags[i-1][1]
        t3 = WordsTags[i][1]
        if (t1 == 'NOUN') and ((t2 == 'CCONJ') or (t2 == 'ADP')) and (t3 == 'NOUN') and (w1 not in stopWords) and (w3 not in stopWords):
            TermsTags.insert([index for index, value in enumerate(TermsTags) if value == (w1,t1)][-1], (w1+'~'+w2+'~'+w3, t1+'~'+t2+'~'+t3))
        elif (t1 == 'ADJ') and (t2 == 'ADJ') and (t3 == 'NOUN') and (w1 not in stopWords) and (w2 not in stopWords) and (w3 not in stopWords):
            TermsTags.insert([index for index, value in enumerate(TermsTags) if value == (w2+'~'+w3,t2+'~'+t3)][-1], (w1+'~'+w2+'~'+w3, t1+'~'+t2+'~'+t3))
    return TermsTags

def stanza_nlp(text, nlpModel, stopWords, nGrams):
    TermsTags = []
    sentsTermsTags = [] #list of targed sentenses (list of lists of targed words)
    docTermsTags = [] #list of all targed words without division into sentences (only NOUN) 
    doc = nlpModel(text)
    sents = doc.sentences
    for sent in sents:
        TermsTags, WordsTags = stanza_built_words(sent, TermsTags, stopWords)
        if (len(WordsTags)>2):
            TermsTags = stanza_built_bigrams(WordsTags, TermsTags, stopWords)
        if (len(WordsTags)>3):
            TermsTags = stanza_built_threegrams(WordsTags, TermsTags, stopWords)
        if ('Words' not in nGrams.values()):
            TermsTags = remove_terms(TermsTags, 0)
        if ('Bigrams' not in nGrams.values()):
            TermsTags = remove_terms(TermsTags, 1)
        if ('Threegrams' not in nGrams.values()):
            TermsTags = remove_terms(TermsTags, 2)
        docTermsTags = docTermsTags + [wt for wt in TermsTags] 
        sentsTermsTags.append(TermsTags)
        TermsTags = []
    return docTermsTags, sentsTermsTags

def pymorphy2_built_words(sent, TermsTags, nlpModel, stopWords):
    WordsTags = []
    words = word_tokenize(sent)
    for word in words:
        try:
            nword = nlpModel.normal_forms(word)[0]
            tag = str((nlpModel.parse(word)[0]).tag.POS)
        except:
            continue
        if tag == 'PROPN':
            tag = 'NOUN'
        WordsTags.append((nword,tag))
        if (nword not in stopWords) and (tag == 'NOUN'): 
            TermsTags.append((nword, tag))
    return TermsTags, WordsTags

def pymorphy2_built_bigrams(WordsTags, TermsTags, stopWords):
    for i in range(1, len(WordsTags)):
        nw1 = WordsTags[i-1][0] 
        nw2 = WordsTags[i][0]
        t1 = WordsTags[i-1][1]
        t2 = WordsTags[i][1]
        if (t1 == 'ADJF') and (t2 == 'NOUN') and (nw1 not in stopWords) and (nw2 not in stopWords):
            TermsTags.insert([index for index, value in enumerate(TermsTags) if value == (nw2,t2)][-1], (nw1+'~'+nw2, t1+'~'+t2))
    return TermsTags

def pymorphy2_built_threegrams(WordsTags, TermsTags, stopWords):
    for i in range(2, len(WordsTags)):
        nw1 = WordsTags[i-2][0]
        nw2 = WordsTags[i-1][0]
        nw3 = WordsTags[i][0]
        t1 = WordsTags[i-2][1]
        t2 = WordsTags[i-1][1]
        t3 = WordsTags[i][1]
        if (t1 == 'NOUN') and ((t2 == 'CCONJ') or (t2 == 'PREP')) and (t3 == 'NOUN') and (nw1 not in stopWords) and (nw3 not in stopWords):
            TermsTags.insert([index for index, value in enumerate(TermsTags) if value == (nw1,t1)][-1], (nw1+'~'+nw2+'~'+nw3, t1+'~'+t2+'~'+t3))
        elif (t1 == 'ADJF') and (t2 == 'ADJF') and (t3 == 'NOUN') and (nw1 not in stopWords) and (nw2 not in stopWords) and (nw3 not in stopWords):
            TermsTags.insert([index for index, value in enumerate(TermsTags) if value == (nw2+'~'+nw3,t2+'~'+t3)][-1], (nw1+'~'+nw2+'~'+nw3, t1+'~'+t2+'~'+t3))
    return TermsTags

def pymorphy2_nlp(text, nlpModel, stopWords, nGrams):
    TermsTags = []
    sentsTermsTags = [] #list of targed sentenses (list of lists of targed words)
    allTermsTags = [] #list of all targed words without division into sentences (only NOUN) 
    sents = sent_tokenize(text)
    for sent in sents:
        TermsTags, WordsTags = pymorphy2_built_words(sent, TermsTags, nlpModel, stopWords)
        if (len(WordsTags)>2):
            TermsTags = pymorphy2_built_bigrams(WordsTags, TermsTags, stopWords)
        if (len(WordsTags)>3):
            TermsTags = pymorphy2_built_threegrams(WordsTags, TermsTags, stopWords)
        if ('Words' not in nGrams.values()):
            TermsTags = remove_terms(TermsTags, 0)
        if ('Bigrams' not in nGrams.values()):
            TermsTags = remove_terms(TermsTags, 1)
        if ('Threegrams' not in nGrams.values()):
            TermsTags = remove_terms(TermsTags, 2)
        allTermsTags = allTermsTags + [tt for tt in TermsTags] 
        sentsTermsTags.append(TermsTags)
        TermsTags = []
    return allTermsTags, sentsTermsTags

def append_lang(defaultLangs, lang, package):
    try:
        defaultLangs[lang] = package
        #with open(dir_below()+"/config.json", "w") as configFile:
        #    try:
        #    except:
        #        pass
        #    configFile.close()
    except:
        print ('Unexpected Error while adding new languade to default list <defaultLangs>!')
    return defaultLangs

def lang_detect(message, defaultLangs, nlpModels, stopWords):
    lidModel = fasttext.load_model('lid.176.ftz')
    message = message.replace("\n", " ")
    #Check if all the characters in the text are whitespaces
    if message.isspace():
        return 'uk'
    else:
        try:
            # get first item of the prediction tuple, then split by "__label__" and return only language code
            lang = lidModel.predict(message)[0][0].split("__label__")[1]
        except:
            return "uk"
    if (lang not in defaultLangs):
        try:
            nlpModels = modelsLoader.stanza_model_loader(defaultLangs, nlpModels, lang)
        except:
            return "uk"
        stopwordsLoader.load_stop_words(defaultLangs, stopWords, lang)
    return lang
