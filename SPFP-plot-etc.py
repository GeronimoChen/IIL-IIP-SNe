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






