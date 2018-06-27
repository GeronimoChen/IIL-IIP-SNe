# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 23:26:23 2018

@author: lenovo
"""

'''
Read the spectra and categorize the time, coordination, etc.
'''
import pandas as pd
import numpy as np
import os
from bs4 import BeautifulSoup
from urllib.request import urlopen

DIR=os.getcwd()
for i in os.walk(DIR+'/WEB/SP/',topdown=False):
    i
loc_web=i[2]
for i in loc_web:
    data=pd.read_html(DIR+'/WEB/SP/'+i,match='Id')
    spdta=data[2]
    for j in range(spdta.loc[1].size):
        if pd.isnull(spdta.loc[1,j]):
            spdta=spdta.drop(j,axis=1)
    for j in range(spdta.loc[:,0].size):
        if pd.isnull(spdta.loc[j,0]):
            spdta=spdta.drop(j,axis=0)
    spdta=pd.DataFrame(data=np.array(spdta.iloc[1:]),columns=np.array(spdta.iloc[0]))
    spdta.to_csv(i+'.csv',encoding='utf-8',index=False)

    
