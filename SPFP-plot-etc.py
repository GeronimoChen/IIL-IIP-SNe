# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 14:10:33 2018

@author: Geronimo Estellar Chen
"""

'''
This program contains many functions
Plot the spectra and major components
Compare the results with DASH, etc
'''
#########################################################
'''
Step 1 Import everything we need.
'''
import dash
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import savgol_filter
import sklearn
from sklearn import svm
from keras.models import Sequential
from keras.layers import Dense



#########################################################
'''
Step 2 Read the csv file.
What we need are:
    Principal functions, Components, Tags, Raw data.
Where:
    EigFuns.csv, FPCS.csv, Category.csv, Rdata.csv
'''
fpcs=pd.read_csv('FPCS.csv')
category=pd.read_csv('Category.csv')
EigFuns=np.transpose(np.array(pd.read_csv('EigFuns.csv')))
raw=pd.read_csv('Rdata.csv')
grid=np.array(pd.read_csv('GRID.csv')).flatten()
[[ShortWave,LongWave,Shorten]]=np.array(pd.read_csv('WaveRescaller.csv')).tolist()
muest=pd.read_csv('muest.csv')
muest=np.array(muest).flatten()



#########################################################
'''
Step 3 Extract the pictures for a preview.
We will plot:
    1. Functional Principal Components --TwoDimShowing, AllPlotIterater
    2. Estimated Spectra, Choose Different FPC level --EstimatedSpectra
    3. Real Spectra, 500 gridpoints --RealSpectra
'''
def TwoDimShowing(v1,v2):
    IIPdata=fpcs[category['Type']=='SNIIP']
    IIndata=fpcs[category['Type']=='SNIIn']
    IIbdata=fpcs[category['Type']=='SNIIb']
    IIdata=fpcs[category['Type']=='SNII']
    plt.scatter(IIndata['V'+str(v1)],IIndata['V'+str(v2)])
    plt.scatter(IIPdata['V'+str(v1)],IIPdata['V'+str(v2)])
    plt.scatter(IIbdata['V'+str(v1)],IIbdata['V'+str(v2)])
    plt.scatter(IIdata['V'+str(v1)],IIdata['V'+str(v2)])
    plt.xlim(fpcs['V'+str(v1)].min(),fpcs['V'+str(v1)].max())
    plt.ylim(fpcs['V'+str(v2)].min(),fpcs['V'+str(v2)].max())
    return
def AllPlotIterater():
    for i in range(fpcs.shape[1]):
        for j in np.arange(i+1,fpcs.shape[1]):
            TwoDimShowing(i+1,j+1)
            plt.savefig('twodimplots/'+str(i+1)+'and'+str(j+1)+'.png')
            plt.close('all')
        
    return
def RealSpectra(a):
    pldata=raw[raw['ID']==a]
    plt.plot(pldata['Wavelength'],pldata['Flux'])
    return
def EstimatedSpectra(a):
    pldata=np.dot(fpcs.iloc[a],EigFuns.transpose())
    pldata=pldata+muest
    plt.plot(grid,pldata)
    return



#########################################################
'''
Step 4 DASH?
'''







