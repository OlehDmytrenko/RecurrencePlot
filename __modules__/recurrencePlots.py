#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 13:03:57 2022

@author: dmytrenko.o
"""
from __modules__ import packagesInstaller
packages = ['matplotlib', 'mpl_toolkits', 'pyts', 'numpy', 'nltk', 'pandas']
packagesInstaller.setup_packeges(packages)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import RecurrencePlot
from pyts.multivariate.image import JointRecurrencePlot
import numpy as np
from nltk import FreqDist
import pandas as pd


def RecPlot(X):
    # Get the recurrence plots for all the time series
    rp = RecurrencePlot(threshold='point', percentage=20)
    X_rp = rp.fit_transform(X)
    
    # Plot the 50 recurrence plots
    fig = plt.figure(figsize=(6, 6))
    
    grid = ImageGrid(fig, 111, nrows_ncols=(6, 6), axes_pad=0.1, share_all=True)
    for i, ax in enumerate(grid):
        ax.imshow(X_rp[i], cmap='binary', origin='lower')
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])
    
    fig.suptitle(
        "Recurrence plots for the 36 time series in the 'Torah' dataset",
        y=0.92
    )
    
    plt.show()
    
def JointRecPlot(X):
    # Recurrence plot transformation
    jrp = JointRecurrencePlot(threshold='point', percentage=50)
    X_jrp = jrp.fit_transform(X)
    # Show the results for the first time series
    plt.figure(figsize=(5, 5))
    plt.imshow(X_jrp[0], cmap='binary', origin='lower')
    plt.title('Joint Recurrence Plot', fontsize=18)
    plt.tight_layout()
    plt.show()

def matrix_of_positions(doc, numKeyterms, startKeyterm, endKeyterm):
    vectorTerms = (FreqDist(term for term in doc)).most_common(numKeyterms)[startKeyterm:endKeyterm]
    keyTerms = [term for (term, freq) in vectorTerms] 
    print (vectorTerms)
    newdoc = []
    for term in doc:
        if term in keyTerms:
            newdoc.append(term)
            
    recPlotMatrix = np.zeros((len(newdoc),len(newdoc)), dtype=int)
    j = 0
    for termtag in newdoc:
        # return all indexes (positions) of "termtag" in "newdoc"
        indexes = [index for index, value in enumerate(newdoc) if value == termtag]
        for i in indexes:
            recPlotMatrix[j][i] = 1
        j = j + 1 
    return recPlotMatrix
    
def visualization(recPlotMatrix):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(recPlotMatrix, interpolation='nearest', cmap='Greys')
    plt.colorbar()
    plt.show()
    
def write_to_csv(recPlotMatrix):
    pd.DataFrame(recPlotMatrix).to_csv("reccurencePlot.csv")

    
