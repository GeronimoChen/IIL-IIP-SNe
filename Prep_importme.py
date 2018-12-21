# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 15:04:05 2018

@author: Geronimo Estellar Chen
"""

'''
I am the ultimate preprocessing script, I serve for R-FPCA and Classifiers.  
I will do:  
1. Read the spectra from files.
2. De-redshift.
3. Normalize the whole spectra and keep their average flux to 1.  
4. Calculate the average flux in the selected wave-window.  
5. Shift the select spectra in the wave-window, make sure their average 
   is 0, it is extremely important for the FPCA algorithm.  
6. Choose some spectra for FPCA algorithm, because extracting the components
   may drain your RAM, it is not wise to use all of them. However, we will 
   calculate the FPCA score for all the spectra.  
7. Outputs are six csv files.  
   7.1. Rcat.csv Save the category used to solve the FPCA components.  
   7.2. restcat.csv Save the rest spectra which are not used for FPCA.  
   7.3. Rdata.csv The ID, Wavelength, Flux. the three-column shape, for the
        sake of R-FPCA input.  
   7.4. restdata.csv Also the ID, Wavelength, Flux. but they are only used 
        to evaluate the scores.  
   7.5. Rphoto.csv The photometric data in the selected band.  
   7.6. restphoto.csv The photometric data in the selected band. 

To notice, not every functions defined here are useful.  
'''
###############################################################################
'''
Step 1 Import the packages.
'''
import numpy as np
import glob
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
OBDIR=DIR+'/CAT/OB/'
SPDIR=DIR+'/CAT/SP/'
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
    1. Confirm the spectra is ordered, not a mess. Have you seen a spectra file
       set the red wave first and blue wave last? It is reversed!  
    2. Remove the redshift in the spectra.  
    3. The Savitzky-Golay filter, might be useful, but I have no idea about how
       to use it.  
    4. The spline function, I use the UnivariateSpline to remove the blackbody 
       radiation background. However, because it is not the Planck Function, it
       seems delete more than the mere black body. With this function, the 
       spectrum will be squeezed to a plane, and only some atomic data reserved. 
       Also, I didn't use it in the ultimate preprocessor.  
    5. Another spline function, and even introduced the notorious downsampling! 
       It is too dirty, I reject it!  
    6. This function will tell you what is in the folder, maybe I should use 
       another function, glob.  
    7. The function integrate all feasible rescale methods. However, I didn't 
       use any of them.  
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
def spl(a):
    f2=UnivariateSpline(a[:,0],a[:,1])
    #wavn=np.linspace(a[:,0].min(),a[:,0].max(),num=number)
    wavn=a[:,0]
    anew=np.array([wavn,a[:,1]-f2(wavn)])
    return np.transpose(anew)
def spl_downsample(a,number):
    f2=UnivariateSpline(a[:,0],a[:,1])
    wavn=np.linspace(a[:,0].min(),a[:,0].max(),num=number)
    #wavn=a[:,0]
    anew=np.array([wavn,a[:,1]-f2(wavn)])
    return np.transpose(anew)
def walker(specdir):
    i=''
    for i in os.walk(DIR+specdir):
        i
    loc_spec=i[2]
    return loc_spec
def rescaller(a,squeezer='Uni_Spline',log=False,shortwave=7000,longwave=7400):
    if log==True: a[:,1]=np.log10(a[:,1])
    if squeezer=='to_one':
        a=pd.DataFrame({'wave':a[:,0],'flux':a[:,1]})
        norm=a.copy()
        norm=norm[norm['wave']>shortwave]
        norm=norm[norm['wave']<longwave]
        a['flux']=a['flux']/np.average(norm['flux'])
        a=np.array(a)
    if squeezer=='scale': a[:,1]=preprocessing.scale(a[:,1])
    if squeezer=='Uni_Spline':a=spl(a)
    if squeezer=='Uni_DownSample':a=spl_downsample(a,5000)
    return a
def logsg(a,window=67):
    a[:,0]=np.log10(a[:,0])
    wavlog=np.arange(a[:,0].min(),a[:,0].max(),0.00001)
    f1=interp1d(a[:,0],a[:,1],fill_value='extrapolate')
    flux=savgol_filter(f1(wavlog),window,2)
    anew=np.array([10**wavlog,flux]).T
    return anew




###############################################################################
'''
Step 4 Read the spectra of a specific type.
(because I stored a certain type of SNe into one directory)
    1. Read the spectra.  
    2. If the spectrum doesn't cover the whole visible
       wavelength, discard it.  
    3. Normalize the whole spectra.  
    4. Seperate the spectra into two category, one for
       fpca eigenfunctions, the other for neuro-network,
       they are denoted as fpca_data and rest_data.
    5. specdir: the relative-path. '/'
    6. fpca_category and rest_category records the selected
       spectra, name and spectral Id.
'''
def read_spectra(specdir,obloc,percent,shortwave,longwave,log_flux=False,**kwangs):
    loc_spec=walker(specdir)
    obdata=ob[obloc]
    spectra=[]
    chosen_name=[]
    photo=[]
    amp=[]
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
        if a.shape[0]<80:continue
        a[:,1]=a[:,1]/np.average(a[:,1])
        a=pd.DataFrame({'Wavelength':a[:,0],'Flux':a[:,1]})
        a=a[a['Wavelength']>shortwave]
        a=a[a['Wavelength']<longwave]
        a=np.array(a)
        a=logsg(a)
        wave=np.arange(shortwave,longwave,1)
        f1=interp1d(a[:,0],a[:,1],fill_value='extrapolate')
        #print(a)
        a=np.array([wave,f1(wave)]).T
        PhotoColor=np.average(a[:,1])
        a[:,1]=a[:,1]-np.average(a[:,1])
        amplitude=np.std(a[:,1])
        a[:,1]=a[:,1]/amplitude
        #a=rescaller(a,squeezer='scale')
        if pd.isnull(a).sum()>0:continue
        #a[:,1]=savgol_filter(a[:,1],11,2)
        #a[:,1]=preprocessing.scale(a[:,1])
        spectra.append(a)
        chosen_name.append(i)
        photo.append(PhotoColor)
        amp.append(amplitude)
    chosen_name={'name':chosen_name,'color':photo,'amp':amp}
    chosen_name=pd.DataFrame(chosen_name)
    spectra_fpca,spectra_rest,name_fpca,name_rest\
    =train_test_split(spectra,chosen_name,test_size=percent)
    photo_fpca=list(name_fpca['color'])
    photo_rest=list(name_rest['color'])
    amp_fpca=list(name_fpca['amp'])
    amp_rest=list(name_rest['amp'])
    name_fpca=list(name_fpca['name'])
    name_rest=list(name_rest['name'])
    
    fpca_flux=np.zeros(0)
    fpca_wave=np.zeros(0)
    fpca_Id=[]
    fpca_category=[]
    fpca_photo=[]
    fpca_amp=[]
    for i in range(len(spectra_fpca)):
        fpca_flux=np.concatenate([fpca_flux,spectra_fpca[i][:,1]])
        fpca_wave=np.concatenate([fpca_wave,spectra_fpca[i][:,0]])
        fpca_Id.extend([name_fpca[i].split('_')[0] \
                        for j in range(np.shape(spectra_fpca[i])[0])])
        fpca_category.append(name_fpca[i])
        fpca_photo.append(photo_fpca[i])
        fpca_amp.append(amp_fpca[i])
        print(name_fpca[i])
        #if the program fails, sometimes the last line of a file is the troublemaker.
    fpca_data=pd.DataFrame({'ID':fpca_Id,'Flux':fpca_flux,'Wavelength':fpca_wave})
    rest_flux=np.zeros(0)
    rest_wave=np.zeros(0)
    rest_Id=[]
    rest_category=[]
    rest_photo=[]
    rest_amp=[]
    for i in range(len(spectra_rest)):
        rest_flux=np.concatenate([rest_flux,spectra_rest[i][:,1]])
        rest_wave=np.concatenate([rest_wave,spectra_rest[i][:,0]])
        rest_Id.extend([name_rest[i].split('_')[0] \
                        for j in range(np.shape(spectra_rest[i])[0])])
        rest_category.append(name_rest[i])
        rest_photo.append(photo_rest[i])
        rest_amp.append(amp_rest[i])
        print(name_rest[i])
    rest_data=pd.DataFrame({'ID':rest_Id,\
                            'Flux':rest_flux,\
                            'Wavelength':rest_wave})
    return fpca_data,rest_data,fpca_category,rest_category,fpca_photo,rest_photo,fpca_amp,rest_amp



'''
Step 5. Integrate all the parts into one function.  
Maybe I need to add some **kwangs.  
'''
def every(where_is_the_spectra=['II','IIL','IIP','IIb','IIn']\
          ,save_the_csv_for_R='R'\
          ,save_the_rest='rest'\
          ,ratios={'II':0.999,'IIP':0.001,'IIL':0.001,'IIb':0.999,'IIn':0.999}
          ,shortwave=4000,longwave=9000,**kwangs):
    Rdata=[]
    restdata=[]
    Rcat=[]
    restcat=[]
    Rphoto=[]
    restphoto=[]
    Ramp=[]
    restamp=[]
    i=0
    for i in where_is_the_spectra:
        specdir='/'+i+'/'
        fpca_data,rest_data,fpca_category,rest_category,fpca_photo,rest_photo,fpca_amp,rest_amp\
        =read_spectra(specdir,i,ratios[i],shortwave,longwave)
        Rdata.append(fpca_data)
        restdata.append(rest_data)
        Rcat.extend(fpca_category)
        restcat.extend(rest_category)
        Rphoto.extend(fpca_photo)
        restphoto.extend(rest_photo)
        Ramp.extend(fpca_amp)
        restamp.extend(rest_amp)
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
    
    Rphoto=pd.DataFrame(Rphoto)
    restphoto=pd.DataFrame(restphoto)
    Rphoto.to_csv(save_the_csv_for_R+'photo.csv',encoding='utf-8',index=False)
    restphoto.to_csv(save_the_rest+'photo.csv',encoding='utf-8',index=False)
    
    Ramp=pd.DataFrame(Ramp)
    restamp=pd.DataFrame(restamp)
    Ramp.to_csv(save_the_csv_for_R+'amp.csv',encoding='utf-8',index=False)
    restamp.to_csv(save_the_rest+'amp.csv',encoding='utf-8',index=False)
    
    WaveRescaller={'Shortest Wavelength':[shortwave],'Longest Wavelength':[longwave],'Shorten':[Shorten]}
    WaveRescaller=pd.DataFrame(WaveRescaller)
    WaveRescaller.to_csv('WaveRescaller.csv',encoding='utf-8',index=False)
    return Rcat,restcat



