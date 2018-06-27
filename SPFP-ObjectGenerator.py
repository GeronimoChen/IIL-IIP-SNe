# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 23:12:07 2018

@author: lenovo
"""

'''
Read the object and acquire the dataframe
'''

import pandas as pd
import numpy as np
import os


DIR=os.getcwd()
for i in os.walk(DIR+'/WEB/OB/',topdown=False):
    i
loc_web=i[2]
for i in loc_web:
    data=pd.read_html(DIR+'/WEB/OB/'+i,match='Redshift')
    obdta=data[1]
    for j in range(obdta.loc[1].size):
        if pd.isnull(obdta.loc[1,j]):
            obdta=obdta.drop(j,axis=1)
    for j in range(obdta.loc[:,0].size):
        if pd.isnull(obdta.loc[j,0]):
            obdta=obdta.drop(j,axis=0)
    obdta=pd.DataFrame(data=np.array(obdta.iloc[1:]),columns=np.array(obdta.iloc[0]))
    obdta.to_csv(i+'.csv',encoding='utf-8',index=False)
            
