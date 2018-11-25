# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 23:38:05 2018

@author: Geronimo Estellar Chen
"""

'''
This is a brand-new preprocessing script.
Here, we will face all type II SNe' spectra.
So, please assure the compatibility of the program.
1. Read the spectra from files.
2. De-redshift.
3. Filter, only the spectra in the visible wavelength.
4. Smoothing and interpolating.
5. Seperate the data for fpca-optimize, training and testing.
6. Outputs are two csv files for R-fpca.
    one is for solving eigen functions.
    another is for training and testing in neuro-network.
An Extreame Important Notice!!!!!!!!!!!!!!!!!
Please Do Not Strip The Continuum Spectra While Doing Small-Scale
Wavelength Window!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
###############################################################################
'''
Step 1 Import the packages.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d,UnivariateSpline
from scipy.signal import savgol_filter
from sklearn import preprocessing
from sklearn.model_selection import train_test_split



###############################################################################
'''
Step 2 Read the csv file of objects and spectra.
To nocite, here I also defined a shorten parameter,
which will be used to truncate the wavelength.
'''
DIR=os.getcwd()
OBDIR=DIR+'\\CAT\\OB\\'
SPDIR=DIR+'\\CAT\\SP\\'
obii=pd.read_csv(OBDIR+'OBII.csv')
spii=pd.read_csv(SPDIR+'SPII.csv')
obiib=pd.read_csv(OBDIR+'OBIIb.csv')
spiib=pd.read_csv(SPDIR+'SPIIb.csv')
obiil=pd.read_csv(OBDIR+'OBIIL.csv')
spiil=pd.read_csv(SPDIR+'SPIIL.csv')
obiip=pd.read_csv(OBDIR+'OBIIP.csv')
spiip=pd.read_csv(SPDIR+'SPIIP.csv')
obiin=pd.read_csv(OBDIR+'OBIIn.csv')
spiin=pd.read_csv(SPDIR+'SPIIn.csv')
sp={'II':spii,'IIb':spiib,'IIL':spiil,'IIP':spiip,'IIn':spiin}
ob={'II':obii,'IIb':obiib,'IIL':obiil,'IIP':obiip,'IIn':obiin}
Shorten=0.001



###############################################################################
'''
Step 3 Well, this function is reserved for other functions.
    1. De-redshift function.
    2. Smoothen, Interpolate and cut the invisible light.
    3. Read the directory where stores spectra.
'''
def strictly_increasing(L):
    return all(x<y for x,y in zip(L,L[1:]))
def fuck_z(wavelength,z):
    return wavelength/(1+z)
def sg(a,window,order,number):
    a[:,1]=savgol_filter(a[:,1],window,order)
    f1=interp1d(a[:,0],a[:,1])
    wavn=np.linspace(a[:,0].min(),a[:,0].max(),num=number)
    anew=np.array([wavn,f1(wavn)])
    return np.transpose(anew)
def spl(a,number):
    f1=interp1d(a[:,0],a[:,1])
    f2=UnivariateSpline(a[:,0],a[:,1])
    wavn=np.linspace(a[:,0].min(),a[:,0].max(),num=number)
    anew=np.array([wavn,f1(wavn)-f2(wavn)])
    return np.transpose(anew)
def walker(specdir):
    i=''
    for i in os.walk(DIR+specdir):
        i
    loc_spec=i[2]
    return loc_spec
def rescaller():
    #Why did I define this function?
    return



###############################################################################
'''
Step 4 Read the spectra of a specific type.
(because I stored a certain type of SNe into one directory)
    1. Read the spectra.
    2. If the spectrum doesn't cover the whole visible
        wavelength, discard it.
    3. Filter and interploate the spectra.
    4. Seperate the spectra into two category, one for
        fpca eigenfunctions, the other for neuro-network,
        they are denoted as fpca_data and rest_data.
    5. specdir: the relative-path. '/'
    6. fpca_category and rest_category records the selected
        spectra, name and spectral Id.
'''
def read_spectra(specdir,obloc,percent,shortwave,longwave):
    loc_spec=walker(specdir)
    obdata=ob[obloc]
    spectra=[]
    chosen_name=[]
    i=0
    for i in loc_spec:
        SNname=i.split('_',1)[1].split('.ascii')[0]
        print(i)
        a=np.genfromtxt(DIR+specdir+i)
        if a.shape[1]==3 or a.shape[1]==7 or a.shape[1]==6:
            a=a[:,0:2]
        assert a.shape[1]==2
        z=float(obdata[obdata['Obj. Name']==SNname]['Redshift'])
        if np.isnan(z):
            continue
        a[:,0]=fuck_z(a[:,0],z)
        if a[:,0].max()<=longwave or a[:,0].min()>=shortwave:
            continue
        if strictly_increasing(a[:,0])==False:continue
        a=spl(a,5000)
        a=pd.DataFrame({'Wavelength':a[:,0],'Flux':a[:,1]})
        a=a[a['Wavelength']>shortwave]
        a=a[a['Wavelength']<longwave]
        a=np.array(a)
        if a.shape[0]<40:continue
        a=sg(a,21,1,500)
        a[:,1]=preprocessing.scale(a[:,1])
        spectra.append(a)
        chosen_name.append(i)
        
    spectra_fpca,spectra_rest,name_fpca,name_rest\
    =train_test_split(spectra,chosen_name,test_size=percent)
    
    fpca_flux=np.zeros(0)
    fpca_wave=np.zeros(0)
    fpca_Id=[]
    fpca_category=[]
    for i in range(len(spectra_fpca)):
        fpca_flux=np.concatenate([fpca_flux,spectra_fpca[i][:,1]])
        fpca_wave=np.concatenate([fpca_wave,spectra_fpca[i][:,0]])
        fpca_Id.extend([name_fpca[i].split('_')[0] \
                        for j in range(np.shape(spectra_fpca[i])[0])])
        fpca_category.append(name_fpca[i])
        print(name_fpca[i])
        #if the program fails, sometimes the last line of a file is the troublemaker.
    fpca_data=pd.DataFrame({'ID':fpca_Id,'Flux':fpca_flux,'Wavelength':fpca_wave})
    rest_flux=np.zeros(0)
    rest_wave=np.zeros(0)
    rest_Id=[]
    rest_category=[]
    for i in range(len(spectra_rest)):
        rest_flux=np.concatenate([rest_flux,spectra_rest[i][:,1]])
        rest_wave=np.concatenate([rest_wave,spectra_rest[i][:,0]])
        rest_Id.extend([name_rest[i].split('_')[0] \
                        for j in range(np.shape(spectra_rest[i])[0])])
        rest_category.append(name_rest[i])
        print(name_rest[i])
    rest_data=pd.DataFrame({'ID':rest_Id,\
                            'Flux':rest_flux,\
                            'Wavelength':rest_wave})
    return fpca_data,rest_data,fpca_category,rest_category



###############################################################################
'''
Step 5 Integrated.
Here, you can choose:
        which kinds of spectra will be used.
        the ratio of the spectra in two seperated datasets.
This program will yield 4 csv files.
    two categories about the filename of the chosen spectra.
    other two are for fpca-eigenfunctions and neuro-network seperately.
To notice, here I also utilized a wave-rescaller and truncated the 
wavelength into (0,1).
'''
def every(where_is_the_spectra=['II','IIL','IIP','IIb','IIn']\
          ,save_the_csv_for_R='Rdata.csv'\
          ,save_the_rest='restdata.csv'\
          ,ratios={'II':0.8,'IIP':0.9,'IIL':0.9,'IIb':0.9,'IIn':0.9}
          ,shortwave=4000,longwave=9000):
    Rdata=[]
    restdata=[]
    Rcat=[]
    restcat=[]
    i=0
    for i in where_is_the_spectra:
        specdir='/'+i+'/'
        fpca_data,rest_data,fpca_category,rest_category=read_spectra(specdir,i,ratios[i],shortwave,longwave)
        Rdata.append(fpca_data)
        restdata.append(rest_data)
        Rcat.extend(fpca_category)
        restcat.extend(rest_category)
    Rdata=pd.concat(Rdata,ignore_index=True)
    Rdata['Wavelength']=(Rdata['Wavelength']-shortwave+Shorten)/(longwave-shortwave+Shorten*2)
    restdata=pd.concat(restdata,ignore_index=True)
    restdata['Wavelength']=(restdata['Wavelength']-shortwave+Shorten)/(longwave-shortwave+Shorten*2)
    Rdata.to_csv(save_the_csv_for_R+'data.csv',encoding='utf-8',index=False)
    restdata.to_csv(save_the_rest+'data.csv',encoding='utf-8',index=False)
    Rcat=pd.DataFrame(Rcat)
    restcat=pd.DataFrame(restcat)
    Rcat.to_csv(save_the_csv_for_R+'cat.csv',encoding='utf-8',index=False)
    restcat.to_csv(save_the_rest+'cat.csv',encoding='utf-8',index=False)
    WaveRescaller={'Shortest Wavelength':[shortwave],'Longest Wavelength':[longwave],'Shorten':[Shorten]}
    WaveRescaller=pd.DataFrame(WaveRescaller)
    WaveRescaller.to_csv('WaveRescaller.csv',encoding='utf-8',index=False)
    return Rcat,restcat



'''
For simplicity, just commanding every() will return what you want.
Unfortunately I forgot the WaveRescaller here, you can just add it afterwards.
Next:
    1. Insert the Rdata.csv into Rstudio, 
       and straighten out the eigenfunctions.
    2. Insert the restdata.csv into Rstudio as well, 
       but only use fpca.score and acquire the eigenvalues
    3. Based on the eigenvalues, apply everything you can imagine, 
       SVM/CNN/K-means...you name it!
'''

'''
I have refined the code.
Now, you can type 

Rcat,restcat=every(#something you want)
j=0
for i in restcat.loc[:,0]:
    if np.size(glob.glob('IIL/'+i))!=0:
        j=j+1
for i in Rcat.loc[:,0]:
    if np.size(glob.glob('IIL/'+i))!=0:
        j=j+1
print(j)

to see how many IIL you had got.
'''
