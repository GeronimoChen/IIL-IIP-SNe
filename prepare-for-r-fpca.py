# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:51:41 2018

@author: lenovo
"""
'''
This Program is designed to integrate the data for R-fpca
We need a table
type-name rest-frame-wavelength normalized-intensity
'''
#########################################################
'''
Step 1 import packages
'''
import dash
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from scipy.interpolate import interp1d, UnivariateSpline



#########################################################
'''
Step 2 read the position of spectra
'''
wdir=os.getcwd()+'\SNII'
for i in os.walk(wdir):
    i
loc_spec=i[2]
print(wdir)



#########################################################
'''
Step 3 Define the de-redshift function
'''
def fuck_z(wavelength,z):
    return wavelength/(1+z)



#########################################################
'''
Step 4 Define the Spline function
CAVEAT!!!
WILL BE PERFORMED LATER
'''
def spl():
    return



#########################################################
'''
Step 5 Read the redshift and type from filenames
'''
SNtype=[]
z=[]
for i in loc_spec:
    tran=i
    tran=tran.split('_',2)
    SNtype.append(tran[1])
    tran=tran[2].split('.ascii')
    z.append(float(tran[0]))


#########################################################
'''
Step 6 Read files and de-redshift them all 
Spline will be added later
Here I choose a simple normalization method let average=1
'''
flux=np.zeros(0)
res_wav=np.zeros(0)
abbr=[]
count=0
while count<np.size(loc_spec):
    a=np.genfromtxt(wdir+'/'+loc_spec[count])
    flux=np.concatenate([flux,a[:,1]/np.average(a[:,1])])
    res_wav=np.concatenate([res_wav,fuck_z(a[:,0],z[count])])
    abbr.extend([SNtype[count]+str(count) for i in range(a[:,1].size)])
    count=count+1



#########################################################
'''
Step 7 Save everything into files
'''
Rdata={'Name':abbr,'Flux':flux,'Wavelength':res_wav}
Rdata=pd.DataFrame(Rdata)
Rdata.to_csv('Rdata.csv', encoding='utf-8', index=False)
#########################################################
