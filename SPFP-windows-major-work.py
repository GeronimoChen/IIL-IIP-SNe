# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 21:22:03 2018

@author: Geronimo Estellar Chen
"""

'''
If nothing is wrong, this program shall be the ultimate one and her peers' 
abilities have been properly succeeded to her, albeit some little defects. 

There are three things she will do.
1. Read the data from R, and the raw data which is also stored in the path.
2. Plot the spectra, and fit the result from R-fpca onto the authentic one.
   Also, she can provide you informations about the principal components
   via scatter plots. Sometimes you can find some clue from these messy dots.
3. Make classification via Neuro-Network or SVM, starting from reading the 
   principal components provided from R-fpca. Both the Neuro-Network and the 
   SVM classifiers are simple enough but promises a moderate performance, 
   their F1-scores in IIL/IIP SNe can reach 0.8-0.85.

I have choosen 6 small scale wavebands and two comprehensive wavebands. 
Their feature elements, wavelength range and the number of principal 
components are shown below. The wavelengths are in Angstron, at rest-frame.
elements: ['Halpha','Hbeta','Ca'  ,'FeMg','FeOMgSi','S'   ,'Na' ,'NaMg','Gap' ]
start:    [6150    ,4600   ,8200  ,4200  ,4900     ,5250  ,5800 ,7700  ,7000  ]
end:      [6800    ,4900   ,8900  ,4600  ,5250     ,5800  ,6150 ,8200  ,7400  ]
FPCA:     [30      ,30     ,30    ,30    ,30       ,30    ,30   ,30    ,30    ]

Longbands:['Visible','Expand']
start:    [4000     ,4000    ]
end:      [7000     ,9000    ]
FPCA:     [50       ,40      ]

To notice, there are still two gaps remain unsampled, 
because they are telluric. [6800-7000,7400-7700].
Another Gap band is dangerous because H2O line within.
'''
###############################################################################
'''
Step 1
Import everything we need. 
The dash here is unused, maybe I am going to make evaluation between my model 
and the dash. 
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
from sklearn import svm,metrics,preprocessing
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import regularizers
import glob



###############################################################################
'''
Step 2
Read the csv file.
What we need are:
    Principal functions, Components, Tags, Raw data.
Where:
    EigFuns.csv, FPCS.csv, Category.csv, Rdata.csv

The obcat here stores a dataframe of all the information about type II SNe. 
Its formation is identical to the sheet on wiserep (because I grabbed it). 

Also, the formation of spcat dataframe is identical to the sheet on wiserep. 
However, I add another two lines to record whether it is properly downloaded 
and the filename (unfortunately, they are unuseful at all). 

The principal components' parameters are also concatenated after the spcat, 
which will be substantiated in step 2.2. 
'''
DIR=os.getcwd()
obdir=DIR+'/CAT/OB/'
spdir=DIR+'/CAT/SP/'
obcat=pd.concat([pd.read_csv(obdir+'OBII.csv'),\
                 pd.read_csv(obdir+'OBIIP.csv'),\
                 pd.read_csv(obdir+'OBIIL.csv'),\
                 pd.read_csv(obdir+'OBIIb.csv'),\
                 pd.read_csv(obdir+'OBIIn.csv')],ignore_index=True)
spcat=pd.concat([pd.read_csv(spdir+'SPII.csv'),\
                 pd.read_csv(spdir+'SPIIP.csv'),\
                 pd.read_csv(spdir+'SPIIL.csv'),\
                 pd.read_csv(spdir+'SPIIb.csv'),\
                 pd.read_csv(spdir+'SPIIn.csv')],ignore_index=True)
waverescaller=pd.read_csv('WaveRescaller-All.csv')
highz=obcat[(obcat['Type']=='SN IIP')&(obcat['Redshift']>0.04)]['Obj. Name']
highz=np.array(highz)
def reader(element='Ca'):
    element0=element
    element=DIR+'/FPCA-'+element+'/'
    fpcs=pd.concat([pd.read_csv(element+'FPCS.csv'),\
                    pd.read_csv(element+'FPCS2.csv')],ignore_index=True)
    filename=pd.concat([pd.read_csv(element+'Rcat.csv'),\
                        pd.read_csv(element+'restcat.csv')],ignore_index=True)
    EigFuns=np.transpose(np.array(pd.read_csv(element+'EigFuns.csv')))
    raw=pd.concat([pd.read_csv(element+'Rdata.csv'),\
                   pd.read_csv(element+'restdata.csv')])
    grid=np.array(pd.read_csv(element+'GRID.csv')).flatten()
    muest=pd.read_csv(element+'muest.csv')
    muest=np.array(muest).flatten()
    [[trash,ShortWave,LongWave,Shorten]]\
    =np.array(waverescaller[waverescaller['Elements']==element0]).tolist()
    return fpcs,filename,EigFuns,raw,grid,muest,ShortWave,LongWave,Shorten



###############################################################################
'''
Step 2.1
Reconstruct the wavelength grid
'''
def rescale_the_wave(raw,grid,ShortWave,LongWave,Shorten):
    raw['Wavelength']=raw['Wavelength']\
    *(LongWave-ShortWave+Shorten*2)+ShortWave-Shorten
    grid=grid*(LongWave-ShortWave+Shorten*2)+ShortWave-Shorten
    return raw,grid



###############################################################################
'''
Step 2.2
Link the spectra to the type etc in the filename category.
That is quite interesting that some data are overlapped, like categoryout
and fpcsout, comparing to spcat. But unfortunately, I am too lazy to fix
this defect, hope your RAM is big enough.

An interesting thing, (something)out like rawout, muestout are idling outside 
the functions, while the element or the waveband is settled, (something) raw 
and muest will take action, just in the function.
'''
def catlink(filename):
    category={'Id':[],'Name':[],'Type':[],'Redshift':[],'Spectra Date':[]}
    for i in np.array(filename):
        j=i[0]
        Id=int(j.split('_')[0])
        Name=j.split('_',1)[1].split('.ascii')[0]
        Type=np.array(obcat[obcat['Obj. Name']==Name]['Type'])[0]
        Redshift=np.array(obcat[obcat['Obj. Name']==Name]['Redshift'])[0]
        SpecDat=np.array(spcat[spcat['Id']==Id]['Obs. Date'])[0]
        category['Id'].append(Id)
        category['Name'].append(Name)
        category['Type'].append(Type)
        category['Redshift'].append(Redshift)
        category['Spectra Date'].append(SpecDat)
    category=pd.DataFrame(category)
    return category
rawout={}
muestout={}
gridout={}
eigout={}
fpcsout={}
categoryout={}
for i in waverescaller['Elements']:
    fpcs,filename,EigFuns,raw,grid,muest,ShortWave,LongWave,Shorten\
    =reader(element=i)
    category=catlink(filename)
    raw,grid=rescale_the_wave(raw,grid,ShortWave,LongWave,Shorten)
    fpcs.rename(columns=lambda x:x.replace('V',i), inplace=True)
    rawout[i]=raw
    gridout[i]=grid
    muestout[i]=muest
    eigout[i]=EigFuns
    fpcsout[i]=fpcs
    categoryout[i]=category
    fpcs['Id']=category['Id']
    spcat=pd.merge(spcat,fpcs,how='left',on='Id')
    del fpcs,filename,EigFuns,raw,grid,muest,ShortWave,LongWave,Shorten



###############################################################################
print('Thankyou for your patient waiting, all the data has been prepared now.')
#%%
'''
Up to now, Every data has been loaded.
Maybe it is because I am programming on a TF card, loading the data is 
extreamely slow. As a consequence, I used an cell-spliter (as is shown 
under the ### spliter and above this paragraf). If you are using spyder 
and you want to amend the codes, you can skip loading the data --every
spyder users should know that.
'''
###############################################################################
'''
Step 3
Extract the pictures for a preview.
We will plot:
    1. Functional Principal Components --TwoDimShowing, AllPlotIterater.
    2. How the Spectra evolve with time in the FPCA parametric space
       --TwoDimLink, LinkPlotIterater.
    3. Estimated Spectra, Choose Different FPC level --EstimatedSpectra.
    4. Real Spectra, 500 gridpoints --RealSpectra.
    5. The Really Real Spectra --Really.
The examples:
    TwoDimShowing('Halpha','Hbeta',1,1,spcat)
'''
def TwoDimShowing(element1,element2,v1,v2,category,paintother=True):
    category2=category[['Id','Type',\
                        element1+str(v1),\
                        element2+str(v2)]].copy()
    category2=category2.dropna()
    IIPdata=category2[category2['Type']=='SN IIP']
    IIndata=category2[category2['Type']=='SN IIn']
    IIbdata=category2[category2['Type']=='SN IIb']
    IILdata=category2[category2['Type']=='SN IIL']
    IIdata=category2[category2['Type']=='SN II']
    if paintother==True:
        plt.scatter(IIdata[element1+str(v1)],\
                           IIdata[element2+str(v2)],label='SN II')
        plt.scatter(IIbdata[element1+str(v1)],\
                            IIbdata[element2+str(v2)],label='SN IIb')
        plt.scatter(IIndata[element1+str(v1)],\
                            IIndata[element2+str(v2)],label='SN IIn')
    
    plt.scatter(IIPdata[element1+str(v1)],\
                IIPdata[element2+str(v2)],label='SN IIP')
    plt.scatter(IILdata[element1+str(v1)],\
                IILdata[element2+str(v2)],label='SN IIL')
    xmin=category2[element1+str(v1)].min()
    xmax=category2[element1+str(v1)].max()
    ymin=category2[element2+str(v2)].min()
    ymax=category2[element2+str(v2)].max()
    if np.size(xmin)>1:xmin=np.array(xmin)[0]
    if np.size(xmax)>1:xmax=np.array(xmax)[0]
    if np.size(ymin)>1:ymin=np.array(ymin)[0]
    if np.size(ymax)>1:ymax=np.array(ymax)[0]
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    return
def TwoDimLink(element1,element2,v1,v2,category,SNtype,color,points=5):
    category2=category[['Id','Type','Obj. Name','Obs. Date',\
                        element1+str(v1),\
                        element2+str(v2)]].copy()
    category2=category2.dropna()
    category2=category2[category2['Type']==SNtype]
    j=0
    k=[]
    for i in category2['Obj. Name'].sort_values():
        if j!=i:
            k.append(i)
            j=i
    for i in k:
        if category2[category2['Obj. Name']==i].shape[0]>points:
            pldata=category2[category2['Obj. Name']==i]
            pldata=pldata.sort_values('Obs. Date')
            plt.plot(pldata[element1+str(v1)],\
                            pldata[element2+str(v2)],\
                            c=color,alpha=0.3)
            plt.scatter(pldata[element1+str(v1)].iloc[0],\
                               pldata[element2+str(v2)].iloc[0],\
                               c=color,label=SNtype+' start')
            plt.scatter(pldata[element1+str(v1)].iloc[-1],\
                               pldata[element2+str(v2)].iloc[-1],\
                               c=color,marker='x',label=SNtype+' end')
    return
def AllPlotIterater(location='twodimplots',\
                    element=['Halpha','Ca','Hbeta',\
                             'S','FeOMgSi','FeMg',\
                             'Na','NaMg','Gap'],\
                    dimension_omitter=[5,5,5,5,5,5,5,5,5]):
    category=spcat
    for i in np.arange(np.size(element)):
        for j in np.arange(dimension_omitter[i]):
            for k in np.arange(np.size(element)):
                for l in np.arange(dimension_omitter[k]):
                    if k<i:continue
                    if k==i and l<=j:continue
                    TwoDimShowing(element[i],element[k],j+1,l+1,category)
                    plt.legend(loc='best')
                    fig=plt.gcf()
                    fig.set_size_inches(8,6)
                    plt.xlabel(element[i]+str(j+1))
                    plt.ylabel(element[k]+str(l+1))
                    plt.savefig(location+'/'\
                                +element[i]+str(j)+element[k]+str(l)+'.png')
                    #plt.savefig(location+'/'\
                    #            +element[i]+str(j)+element[k]+str(l)+'.eps')
                    plt.close('all')
    return
def LinkPlotIterater(location='twodimlinks',\
                     element=['Halpha','Ca','Hbeta','S','FeOMgSi','FeMg',\
                              'Na','NaMg','Gap'],\
                     dimension_omitter=[5,5,5,5,5,5,5,5,5]):
    category=spcat
    for i in np.arange(np.size(element)):
        for j in np.arange(dimension_omitter[i]):
            for k in np.arange(np.size(element)):
                for l in np.arange(dimension_omitter[k]):
                    if k<i:continue
                    if k==i and l<=j:continue
                    plt.plot([],[],c='b',label='SN IIP')
                    plt.plot([],[],c='r',label='SN IIL')
                    plt.scatter([],[],marker='o',c='k',label='First-Observed')
                    plt.scatter([],[],marker='x',c='k',label='Last-Observed')
                    plt.legend(loc='best')
                    TwoDimLink(element[i],element[k],j+1,l+1,\
                               category,'SN IIP','b',points=10)
                    TwoDimLink(element[i],element[k],j+1,l+1,\
                               category,'SN IIL','r',points=2)
                    plt.xlabel(element[i]+str(j+1))
                    plt.ylabel(element[k]+str(l+1))
                    fig=plt.gcf()
                    fig.set_size_inches(8,6)
                    plt.savefig(location+'/'\
                                +element[i]+str(j)+element[k]+str(l)+'.png')
                    #plt.savefig(location+'/'\
                    #            +element[i]+str(j)+element[k]+str(l)+'.eps')
                    plt.close('all')
    return
def RealSpectra(a,element='Expand',label=None):
    raw=rawout[element]
    pldata=raw[raw['ID']==a]
    plt.plot(pldata['Wavelength'],pldata['Flux'],label=label)
    return
def EstimatedSpectra(a,element='Expand',power=10,label=None):
    grid=gridout[element]
    muest=muestout[element]
    fpcs=fpcsout[element]
    category=categoryout[element]
    EigFuns=eigout[element]
    pldata=np.dot(fpcs[category['Id']==a].iloc[:,0:power]\
                  ,EigFuns.transpose()[0:power]).flatten()
    pldata=pldata+muest
    plt.plot(grid,pldata,label=label)
    return
def spl(a,number):
    f1=interp1d(a[:,0],a[:,1])
    f2=UnivariateSpline(a[:,0],a[:,1])
    wavn=np.linspace(a[:,0].min(),a[:,0].max(),num=number)
    anew=np.array([wavn,f1(wavn)-f2(wavn)])
    return np.transpose(anew)
def Really(a,deredshift=True,label=None,sgf=False,ctr=False,reg=False):
    file=glob.glob('*/'+str(a)+'_*.ascii')[0]
    obname=spcat[spcat['Id']==a]['Obj. Name'].iloc[0]
    z=obcat[obcat['Obj. Name']==obname]['Redshift'].iloc[0]
    pldata=np.genfromtxt(file)
    if deredshift==True:pldata[:,0]=pldata[:,0]/(1+z)
    if sgf==True:pldata[:,1]=savgol_filter(pldata[:,1],21,1)
    if ctr==True:pldata=spl(pldata,5000)
    if reg==True:pldata[:,1]=preprocessing.scale(pldata[:,1])
    plt.plot(pldata[:,0],pldata[:,1],label=label,c='k')
    return



###############################################################################
'''
Step 3.1.
The Function defined in this part are specially for plotting in the paper. 
Here, we will plot:
    1. The composition of data: objects and spectra. 
    2. Every step in processing the data. 
        2.1. De-redshift.
        2.2. Savitz-Golay filtering and interpolate.
        2.3. Window selection.
        2.4. Reconstruct the spectra from FPCA. 
    3. Two or Three paradigms, good one or bad one.
'''
def plot_spectra_pie(location='plotout'):
    name=['II #1656','IIP #910','IIb #773','IIn #1188','IIL #241']
    count=[1656,910,773,1188,241]
    plt.pie(count,labels=name,explode=[0.1,0.1,0.1,0.1,0.1])
    plt.axis('equal')
    plt.title('The Number of every type II (subtype) Supernovae\' Spectra')
    plt.savefig(location+'/spectracount.png')
    plt.savefig(location+'/spectracount.eps')
    plt.close('all')
    return
def plot_object_pie(location='plotout'):
    name=['II #952','IIP #221','IIb #99','IIn #386','IIL #15']
    count=[952,221,99,386,15]
    plt.pie(count,labels=name,explode=[0.1,0.1,0.1,0.1,0.1])
    plt.axis('equal')
    plt.title('The Number of every type II (subtype) Supernovae\' Object')
    plt.savefig(location+'/objectcount.png')
    plt.savefig(location+'/objectcount.eps')
    plt.close('all')
    return

def rawplot(a=45232,location='plotout'):
    SNname=spcat[spcat['Id']==a].loc[:,'Obj. Name'].iloc[0]
    date=spcat[spcat['Id']==45232].loc[:,'Obs. Date'].iloc[0]
    Really(a,label='ID'+str(a)+' '+SNname+' '+date)
    Really(a,label='Savitz-Golay Filter',sgf=True)
    plt.xlabel('Rest Frame Wavelength (A)')
    plt.ylabel('Flux')
    plt.legend(loc='best')
    plt.title('De-Redshifted Raw Data and Savitz-Golay Filtered')
    plt.savefig(location+'/sgf'+str(a)+'.png')
    plt.savefig(location+'/sgf'+str(a)+'.eps')
    plt.close('all')
    return
def continuum_remover(a=45232):
    SNname=spcat[spcat['Id']==a].loc[:,'Obj. Name'].iloc[0]
    date=spcat[spcat['Id']==45232].loc[:,'Obs. Date'].iloc[0]
    Really(a,sgf=True,ctr=True,reg=True)
    plt.xlabel('Rest Frame Wavelength (A)')
    plt.ylabel('Renormalized Flux in Arbitrary Unit')
    return
'''
Sorry, something seems not working. 
Here, I was intending to plot the masks of different wave windows. 
However, It is a little bit hard compensae the x- and y- axis scale. 
'''
def element_illustrator(a=45232):
    continuum_remover(a)
    plt.ylim(-2,6)
    for i in range(9):
        sw=waverescaller.iloc[i,1]
        lw=waverescaller.iloc[i,2]
        plt.fill([sw,sw,lw,lw],[-2,6,6,-2],alpha=0.4,
                 label=waverescaller.iloc[i,0])
    return


###############################################################################
'''
Step 4
Here, I will use some algorithms to solemnly analyse the
problem. Algorithms will be sealed into functions, 
because the dataset is rather small. Then, by inserting
different parameters inwards, the functions will return
the accuracy, recall, etc.

Firstly, let us prepare the useable stuffs
---- Precision, Recall, F1-score.
It is okay if you want to call this as Step 3.0.

Also, I prepared a data spliter function here, which will be useful 
in splitting the training and testing sets. If further requirements 
about the data comes, just fix this function.

mulitdataspliter is designed to spearate multiple-taged datasets.
To notice, the multidataspliter has not been finished, I will enaviliable it 
to yield training- and testing-sets in onehot form, which will be essential 
for the neuronetworkk.

Something should shut up! I know the amount of type IIL is small!!!
'''
def scoring(confusion_matrix):
    cm=np.array(confusion_matrix)
    tp,tn,fp,fn=cm[0],cm[1],cm[2],cm[3]
    prec1=tp/(tp+fp)
    reca1=tp/(tp+fn)
    prec2=tn/(tn+fn)
    reca2=tn/(tn+fp)
    prec=(prec1+prec2)/2
    reca=(reca1+reca2)/2
    f1sc1=2/(1/prec1+1/reca1)
    f1sc2=2/(1/prec2+1/reca2)
    f1sc=(f1sc1+f1sc2)/2
    #if tn/tp>0.5 or tp/tn>0.5:print('The testing set is biased.')
    #if np.sum(cm)<40:print('The testing set is too small.')
    return prec,reca,f1sc
def dataspliter(symmetricity,type1,type2,element,dimension_omitter,testsize):
    category=spcat[(spcat['Type']==type1)|(spcat['Type']==type2)]
    category2=category[['Id','Obj. Name','Type','Obs. Date']].copy()
    for i in range(np.size(element)):
        for j in range(dimension_omitter[i]):
            category2[element[i]+str(j+1)]=category[element[i]+str(j+1)]
    category3=category2.dropna()
    X1=category3[category3['Type']==type1]
    X2=category3[category3['Type']==type2]
    Y1=X1[['Id','Obj. Name','Type','Obs. Date']]
    Y2=X2[['Id','Obj. Name','Type','Obs. Date']]
    X1=X1.drop(['Id','Obj. Name','Type','Obs. Date'],axis=1)
    X2=X2.drop(['Id','Obj. Name','Type','Obs. Date'],axis=1)
    if symmetricity==True:
        if X1.shape[0]>X2.shape[0]:
            X1,trash1,Y1,trash2\
            =train_test_split(X1,Y1,test_size=1-X2.shape[0]/X1.shape[0])
        else:
            X2,trash1,Y2,trash2\
            =train_test_split(X2,Y2,test_size=1-X1.shape[0]/X2.shape[0])
    X=np.array(pd.concat([X1,X2],ignore_index=True))
    Y=np.array(pd.concat([Y1,Y2],ignore_index=True))
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=testsize)
    Y_test_out=Y_test
    Y_test=Y_test[:,2]
    Y_train=Y_train[:,2]
    return X_train,X_test,Y_train,Y_test,Y_test_out
'''
From now on, it is the unfinished multidataspliter.
If you are vacant, please don't forget to finish it.
'''
def multidataspliter(symmetricity,types,dimension_omitter,testsize
                     ,onehot=False):
    X,Y=[],[]
    for i in types:
        X.append(fpcs[category['Type']==i])
        Y.append(fpcs[category['Type']==i])
    if symmetricity==True:
        sizes=[]
        for i in range(np.size(X)):
            sizes.append(X[i].shape)
        small=sizes.min()
        for i in range(np.size(X)):
            trash1,X[i],trash2,Y[i]=train_test_split(X[i],Y[i]\
                    ,test_size=small/sizes[i])
    X=np.array(pd.concat(X,ignore_index=True))
    Y=np.array(pd.concat(Y,ignore_index=True))
    if onehot==True:
        Y
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=testsize)
    Y_test_out=Y_test
    Y_test=Y_test[:,2]
    return X_train,X_test,Y_train,Y_test,Y_test_out



###############################################################################
'''
Step 4.1
The old-fashioned SVM method in classification.

In the confusion matrix, I regard the type IIP as the positive
and type IIL as the negative.

In the clfout, the well-trained SVM classifier will be given here.
You can just use clfout(X) to acquire the result, the output is
like ['SN IIL', 'SN IIP'...].

An confusion_matrix will also returned, which will be used to
calculate the F1-score.

I also prepared a dimension_omitter here, because not all fpcs 
scores are important, maybe the first 5 will be enough. It is 
necessary to refine the dimension_omitter, cause we should not 
only consider cutting the tail, but also popping out the middle.

Tips in using clf.predict_proba:
    there are two columns, the first column is the probablity of type1
    and the second column is the probablity of type2, because type1 
    shows before type2 in the dataset.
    What if C=8000000?
'''
def svming(type1='SN IIP',type2='SN IIL',testsize=0.2,\
           symmetricity=True,\
           element=['Halpha' ,'Ca'      ,'Hbeta',\
                    'S'      ,'FeOMgSi' ,'FeMg' ,\
                    'Na'     ,'NaMg'    ,'Gap'  ],\
           dimension_omitter=[30,30,30,30,30,30,30,30,30],verbose=0,\
           kernel='rbf',degree=80,C=3020):
    #3.1. Divide the data into training and testing sets.
    X_train,X_test,Y_train,Y_test,Y_test_out\
    =dataspliter(symmetricity,type1,type2,element,dimension_omitter,testsize)
    #3.2. Start training.
    Y_test=np.array(Y_test.tolist())
    clf=svm.SVC(tol=10**-6,kernel=kernel,degree=degree,C=C,cache_size=1000,\
                probability=True)
    clf=clf.fit(X_train,Y_train)
    #3.3. Do the prediction specially on the training set.
    Z=clf.predict(X_test)
    Z2=clf.predict_proba(X_test)[:,1]
    Z=(Z==type1)#Z will be boolean, while Z2 will be the probablity.
    if abs(Z2[0]-Z[0])>0.5:Z2=1-Z2
    Y_test=(Y_test==type1)
    Z=np.array(Z,dtype=int)
    Y_test=np.array(Y_test,dtype=int)
    clfout=clf.predict_proba
    #3.4. The confusion matrix.
    true_positive=np.sum((Z==1)&(Y_test==1))
    true_negative=np.sum((Z==0)&(Y_test==0))
    false_positive=np.sum((Z==1)&(Y_test==0))
    false_negative=np.sum((Z==0)&(Y_test==1))
    confusion_matrix=np.array([true_positive,true_negative,\
                               false_positive,false_negative])
    #3.5. The probablity given by the machine.
    return confusion_matrix,clfout,X_test,Y_test,Z2,Y_test_out



###############################################################################
'''
Step 4.2.
Let us seal the Neuro-Network into the function.

Like the SVM function discribed before, we can also define the types, 
redshifts and the test size. 

Unfortunately, adding l1 and l2 regularization cannot increase the performance 
whatsoever, please just set it to zero.

Here, the function also return the predicted outcome and the test-set,
because that can be useful in plotting ROC curve.
'''
def neuroing(type1='SN IIP',type2='SN IIL',testsize=0.2,epoch=2000,\
             element=['Halpha' ,'Ca'      ,'Hbeta',\
                      'S'      ,'FeOMgSi' ,'FeMg' ,\
                      'Na'     ,'NaMg'    ,'Gap'  ],\
             dimension_omitter=[30,30,30,30,30,30,30,30,30],verbose=0,\
             symmetricity=True,middle_units1=40,l1=0,l2=0):
    #3.1. Prepare the dataset. When symmetricity is true, a symmetric dataset 
    #will be given, number'IIL'=number'IIP'.
    X_train,X_test,Y_train,Y_test,Y_test_out\
    =dataspliter(symmetricity,type1,type2,element,dimension_omitter,testsize)
    #because of keras, I need to change the formation of Y.
    Y_train=np.array(np.array(Y_train==type1),dtype='float32')
    Y_test=np.array(np.array(Y_test==type1),dtype='float32')
    #3.2. Construct the neuro-network.
    model=Sequential()
    model.add(Dense(units=middle_units1,\
                    activation='relu',input_dim=X_train.shape[1],
                    kernel_regularizer=regularizers.l1_l2(l1=l1,l2=l2)))
    model.add(Dense(units=1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',\
                  optimizer='adam',metrics=['accuracy'])
    model.fit(X_train,Y_train,epochs=epoch,verbose=verbose)
    #3.3. Evaluation and export the neuro-network.
    Z2=model.predict(X_test).flatten()
    Z=(Z2>0.5)-0 #As before, Z2 is the probability and Z is bollean.
    true_positive=np.sum((Z==1)&(Y_test==1))
    true_negative=np.sum((Z==0)&(Y_test==0))
    false_positive=np.sum((Z==1)&(Y_test==0))
    false_negative=np.sum((Z==0)&(Y_test==1))
    confusion_matrix=np.array([true_positive,true_negative,\
                               false_positive,false_negative])
    netout=model.predict
    return confusion_matrix,netout,X_test,Y_test,Z2,Y_test_out



###############################################################################
'''
Step 4.3.
Training and modeling with multiple tags in SVM.
Here, I will try to distinguish all IIP/IIL/IIb/IIn subclasses via SVM.
Unfortunately, I don't know how to construct a confusion matrix for mulitple
tags, it is not as simple as yes/no problem. So, for now, the function will 
only return the accuracy.

This part is unfinished, because of the work load and the importance.
Maybe, I should only focus on IIP/IIL?

To notice, the confusion matrix in multi-classifying problem can be large, 
please make it sure your scoring() function is compatible.
'''
def multisvming():
    
    return confusion_matrix,svmout,X_test,Y_test,Z2,Y_test_out



###############################################################################
'''
Step 5
Evaluate the model.

1. We will integrate the training and evaluating parts into the program.
   Then, start a loop and acquire the average F1-socre.
2. Print the ROC curve of neuro-network.
3. Count the spectra and the object that often mislead the classifier.

Please be well aware what you are doing before starting averagef1(), because
running a neuro-network on keras is not as fast as expected. If method is set 
to be 'neuro', the iteration parameter ite is better smaller than 20.
'''
def averagef1(method='svm',ite=200,**kwangs):
    if method=='svm':
        function=svming
    if method=='neuro':
        function=neuroing
    precs,recas,f1scs=[],[],[]
    for i in range(ite):
        cm0,trash1,trash2,trash3,trash4,trash5=function(**kwangs)
        prec,reca,f1sc=scoring(cm0)
        precs.append(prec)
        recas.append(reca)
        f1scs.append(f1sc)
    precs,recas,f1scs=np.array(precs),np.array(recas),np.array(f1scs)
    return np.array([[np.average(precs),np.average(recas),np.average(f1scs)],\
                     [np.std(precs),np.std(recas),np.std(f1scs)]])
def rocplot(Y_test,Z2):
    fpr,tpr,trash=sklearn.metrics.roc_curve(Y_test,Z2,drop_intermediate=False)
    plt.plot(fpr,tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    return



###############################################################################
'''
Step 6.
In this part, I would like to count two things: how many times the spectrum
is calculated, how many times the machine mistaken the spectrum.
By the way, I like the metaphor about the black sheep.

Sorry, this is a mistake, I should have arranged the whole program more
carefully. Here, cat1 and cat1 should match type1 and type2 in kwangs. 
The cat1 and cat2 are prepared for counting sheets.

I also plot the troublemakers, of which get mistaken higher than 90%

Till 2018.8.1, this part is not compatible to multi-windows scenario.
It is a copy from other former programmes.
'''
def counting_black_sheep(method='svm',ite=200,\
                         type1='SN IIP',type2='SN IIL',**kwangs):
    cat=spcat.loc[(spcat['Type']==type1)|(spcat['Type']==type2)]\
    .copy()
    cat['Test']=0
    cat['Fail']=0
    cat=cat.set_index('Id')
    if method=='svm':
        function=svming
    if method=='neuro':
        function=neuroing
    for i in range(ite):
        cm,netout,X_test,Y_test,Z2,Y_test_out=function(type1,type2,**kwangs)
        cat.loc[Y_test_out[:,0],'Test']=cat.loc[Y_test_out[:,0],'Test']+1
        Z=(Z2>0.5)
        wrong=Y_test_out[Z!=Y_test][:,0]
        cat.loc[wrong,'Fail']=cat.loc[wrong,'Fail']+1
    return cat
def PlotTroubleMaker(method='svm',ite=200,dirname='troublemaker',\
                        type1='SN IIP',type2='SN IIL',**kwangs):
    cat=counting_black_sheep(method,ite,type1,type2,**kwangs)
    catfail=cat[cat['Fail']/cat['Test']>0.9].reset_index()
    for i in catfail['Id']:
        Really(i)
        SNname=catfail[catfail['Id']==i]['Name'].iloc[0]
        SNtype=catfail[catfail['Id']==i]['Type'].iloc[0]
        test=catfail[catfail['Id']==i]['Test'].iloc[0]
        fail=catfail[catfail['Id']==i]['Fail'].iloc[0]
        plt.title('The Troublemaker Plot\n'
                  'Name: '+SNname+' ID: '+str(i)+' Type: '+SNtype
                  +' Fail/Test: '+str(fail)+'/'+str(test))
        fig=plt.gcf()
        fig.set_size_inches(18.5,9)
        plt.savefig(dirname+'/'+str(i),dpi=400)
        plt.close('all')
    return


