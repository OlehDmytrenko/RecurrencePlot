# -*- coding: utf-8 -*-
"""

@author: Олег Дмитренко

"""
import sys, os
from __modules__ import configLoader, modelsLoader, stopwordsLoader, textProcessor
from __modules__ import recurrencePlots#, termsWeigher, matrixClustering

import time
t0 = time.time()

if __name__ == "__main__":
    inputFilePath = sys.argv[1]
    
    stdOutput = open("outlog.log", "w")
    sys.stderr = stdOutput
    sys.stdout = stdOutput

    defaultLangs = configLoader.load_default_languages(os.getcwd())
    docSize = configLoader.int_value(os.getcwd(),'docSize')
    numKeyterms = configLoader.int_value(os.getcwd(),'numKeyterms')
    startKeyterm = configLoader.int_value(os.getcwd(),'startKeyterm')
    endKeyterm = configLoader.int_value(os.getcwd(),'endKeyterm')
    defaultSWs = stopwordsLoader.load_default_stop_words(defaultLangs)
    nlpModels = modelsLoader.load_default_models(defaultLangs)
    nGrams = configLoader.load_default_ngrams(os.getcwd())
    
    with open(inputFilePath, "r", encoding="utf-8") as inputFlow:
        document = ""
        docs = []
        docsSentences = []
        
        lines = (inputFlow.read()).splitlines()
        for line in lines:
            document += (line + '\n')
        inputFlow.close()
     
    print ("Кількість символів у документі: ",len(document))
    print (time.time() - t0)
    
    document = document[:docSize].lower()            
    lang = textProcessor.lang_detect(document, defaultLangs, nlpModels, defaultSWs)

    
    if (defaultLangs[lang] == 'pymorphy2'):
        docTermsTags, sentsTermsTags = textProcessor.pymorphy2_nlp(document, nlpModels[lang], defaultSWs[lang], nGrams)
        
    elif (defaultLangs[lang] == 'stanza'):
        docTermsTags, sentsTermsTags  = textProcessor.stanza_nlp(document, nlpModels[lang], defaultSWs[lang], nGrams)
        
    elif (not defaultLangs[lang]):
       docTermsTags, sentsTermsTags  = textProcessor.stanza_nlp(document, nlpModels[lang], defaultSWs[lang], nGrams)
    
    print (time.time() - t0)   
    
    
    # get all unique words from primary doc
    #keyterms = termsWeigher.get_vector_terms(docs[0]).keys()
    #wordsAssessment.TFIDF_vector(docs, vectorWords)
    #termsWeigher.GTF_vector(docs, vectorTerms)
    
    #Tseries = []
    #for i in range(0, len(docs)-30):
    #    # 1D Tseries and 1D Tuples for entire document
    #    Tseries.append(termsWeigher.time_series(docs[0], keyterms))
   
    # для цілого документа 
    #Tseries = termsWeigher.matrix_positions2(docTermsTags)
    #recurrencePlots.JointRecPlot([Tseries])
    
    recPlotMatrix = recurrencePlots.matrix_of_positions(docTermsTags, numKeyterms, startKeyterm, endKeyterm)
    print (time.time() - t0)
    recurrencePlots.visualization(recPlotMatrix)
    recurrencePlots.write_to_csv(recPlotMatrix)
    #matrixClustering.KMeansMethod(matrx)
    print (time.time() - t0)
    