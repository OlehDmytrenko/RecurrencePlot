#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 19:46:00 2022

@author: dmytrenko.o

"""
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

def KMeansMethod(matrx):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(matrx, interpolation='nearest', cmap='Greys')
    plt.colorbar()
    plt.show()
    plt.savefig('general_graphic_.png', format='png', dpi=100)
    
    kmeans = KMeans(n_clusters=36, random_state=0).fit(matrx)
    print (kmeans.predict(matrx))
    print (kmeans.cluster_centers_)
    
    label = kmeans.fit_predict(matrx)
    
    centroids = kmeans.cluster_centers_
    
    #Getting unique labels
    u_labels = np.unique(label)
     
    #plotting the results:
 
    for i in u_labels:
        plt.scatter(matrx[label == i , 0] , matrx[label == i , 1] , label = i)
    plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
    plt.colorbar()
    plt.show()