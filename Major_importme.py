import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=''
from keras import backend as K
import tensorflow as tf
import keras
import keras.backend.tensorflow_backend as KTF
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'cpu':0})))


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
    photo=pd.concat([pd.read_csv(element+'Rphoto.csv'),\
                     pd.read_csv(element+'restphoto.csv')])
    amp=pd.concat([pd.read_csv(element+'Ramp.csv'),\
                   pd.read_csv(element+'restamp.csv')])
    [[trash,ShortWave,LongWave,Shorten]]\
    =np.array(waverescaller[waverescaller['Elements']==element0]).tolist()
    return fpcs,filename,EigFuns,photo,amp,raw,grid,muest,ShortWave,LongWave,Shorten


def rescale_the_wave(raw,grid,ShortWave,LongWave,Shorten):
    raw['Wavelength']=raw['Wavelength']\
    *(LongWave-ShortWave+Shorten*2)+ShortWave
    grid=grid*(LongWave-ShortWave+Shorten*2)+ShortWave
    return raw,grid


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
photoout={}
ampout={}
for i in waverescaller['Elements']:
#for i in ['S']:
    fpcs,filename,EigFuns,photo,amp,raw,grid,muest,ShortWave,LongWave,Shorten\
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
    photoout[i]=photo
    ampout[i]=amp
    fpcs['Id']=category['Id']
    fpcs[i+'_photo']=np.array(photo)
    fpcs[i+'_amp']=np.array(amp)
    spcat=pd.merge(spcat,fpcs,how='left',on='Id')
    del fpcs,filename,EigFuns,raw,grid,muest,ShortWave,LongWave,Shorten
print('Thankyou for your patient waiting, all the data has been prepared now.')


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
def TwoDimShowing2(element1,element2,v1,v2,category,painttype='SN IIP',**kwangs):
    category2=category[['Id','Type',\
                        element1+str(v1),\
                        element2+str(v2)]].copy()
    category2=category2.dropna()
    paintdata=category2[category2['Type']==painttype]
    plt.scatter(paintdata[element1+str(v1)],\
                paintdata[element2+str(v2)],label=painttype,**kwangs)
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
    photo=photoout[element]
    amp=ampout[element]
    amp=np.array(amp[category['Id']==a])[0]
    photo=np.array(photo[category['Id']==a])[0]
    pldata=np.dot(fpcs[category['Id']==a].iloc[:,0:power]\
                  ,EigFuns.transpose()[0:power]).flatten()
    pldata=pldata+muest
    plt.plot(grid,pldata,label=label)
    return
def EstimatedSpectra2(a,element='Expand',power=10,label=None,**kwangs):
    grid=gridout[element]
    muest=muestout[element]
    fpcs=fpcsout[element]
    category=categoryout[element]
    EigFuns=eigout[element]
    #photo=photoout[element]
    #amp=ampout[element]
    #amp=np.array(amp[category['Id']==a])[0]
    #photo=np.array(photo[category['Id']==a])[0]
    pldata=np.dot(fpcs[category['Id']==a].iloc[:,0:power]\
                  ,EigFuns.transpose()[0:power]).flatten()
    pldata=np.array(fpcs[category['Id']==a][element+'_photo'])[0]+(muest+pldata)*np.array(fpcs[category['Id']==a][element+'_amp'])[0]
    plt.plot(grid,pldata,label=label,**kwangs)
    return
def spl(a):
    f1=interp1d(a[:,0],a[:,1])
    wavn=np.arange(a[:,0].min(),a[:,0].max(),1)
    anew=np.array([wavn,f1(wavn)])
    return np.transpose(anew)
def logsg(a,window=67):
    a[:,0]=np.log10(a[:,0])
    wavlog=np.arange(a[:,0].min(),a[:,0].max(),0.00001)
    f1=interp1d(a[:,0],a[:,1],fill_value='extrapolate')
    flux=savgol_filter(f1(wavlog),window,2)
    anew=np.array([10**wavlog,flux]).T
    return anew
def Really(a,deredshift=True,label=None,sgf=False,ctr=False,reg=False,window=67,**kwangs):
    file=glob.glob('*/'+str(a)+'_*.ascii')[0]
    obname=spcat[spcat['Id']==a]['Obj. Name'].iloc[0]
    z=obcat[obcat['Obj. Name']==obname]['Redshift'].iloc[0]
    pldata=np.genfromtxt(file)
    if deredshift==True:pldata[:,0]=pldata[:,0]/(1+z)
    if sgf==True:pldata=logsg(pldata,window)
    if ctr==True:pldata=spl(pldata)
    if reg==True:pldata[:,1]=pldata[:,1]/np.average(pldata[:,1])
    plt.plot(pldata[:,0],pldata[:,1],label=label,**kwangs)
    return


def scoring(confusion_matrix):
    cm=np.array(confusion_matrix)
    tp,tn,fp,fn=cm[0],cm[1],cm[2],cm[3]
    prec1=tp/(tp+fp)
    reca1=tp/(tp+fn)
    prec2=tn/(tn+fn)
    reca2=tn/(tn+fp)
    #prec=(prec1+prec2)/2
    #reca=(reca1+reca2)/2
    f1sc1=2/(1/prec1+1/reca1)
    f1sc2=2/(1/prec2+1/reca2)
    #f1sc=(f1sc1+f1sc2)/2
    #if tn/tp>0.5 or tp/tn>0.5:print('The testing set is biased.')
    #if np.sum(cm)<40:print('The testing set is too small.')
    return prec1,reca1,f1sc1,prec2,reca2,f1sc2
def dataspliter(symmetricity,type1,type2,element,dimension_omitter,testsize):
    category=spcat[(spcat['Type']==type1)|(spcat['Type']==type2)]
    category2=category[['Id','Obj. Name','Type','Obs. Date']].copy()
    for i in range(np.size(element)):
        category2[element[i]+'_amp']=category[element[i]+'_amp']
        category2[element[i]+'_photo']=category[element[i]+'_photo']
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


def svming(type1='SN IIP',type2='SN IIL',testsize=0.2,\
           symmetricity=False,\
           element=['Halpha' ,'Ca'      ,'Hbeta',\
                    'S'      ,'FeOMgSi' ,'FeMg' ,\
                    'Na'     ,'NaMg'    ,'Gap'  ],\
           dimension_omitter=[30,30,30,30,30,30,30,30,30],verbose=0,\
           kernel='rbf',degree=80,C=3020,gamma='auto'):
    #3.1. Divide the data into training and testing sets.
    X_train,X_test,Y_train,Y_test,Y_test_out\
    =dataspliter(symmetricity,type1,type2,element,dimension_omitter,testsize)
    #3.2. Start training.
    Y_test=np.array(Y_test.tolist())
    clf=svm.SVC(tol=10**-6,kernel=kernel,degree=degree,C=C,cache_size=1000,\
                probability=True,gamma=gamma,class_weight='balanced')
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


def neuroing(type1='SN IIP',type2='SN IIL',testsize=0.2,epoch=2000,\
             element=['Halpha' ,'Ca'      ,'Hbeta',\
                      'S'      ,'FeOMgSi' ,'FeMg' ,\
                      'Na'     ,'NaMg'    ,'Gap'  ],\
             dimension_omitter=[30,30,30,30,30,30,30,30,30],verbose=0,\
             symmetricity=False,middle_units1=40,l1=0,l2=0):
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
    model.fit(X_train,Y_train,epochs=epoch,verbose=verbose,batch_size=1000,class_weight={0:np.sum(Y_train==0),1:np.sum(Y_train==1)})
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


def averagef1(method='svm',ite=200,**kwangs):
    if method=='svm':
        function=svming
    if method=='neuro':
        function=neuroing
    precs1,recas1,f1scs1,precs2,recas2,f1scs2=[],[],[],[],[],[]
    for i in range(ite):
        cm0,trash1,trash2,trash3,trash4,trash5=function(**kwangs)
        prec1,reca1,f1sc1,prec2,reca2,f1sc2=scoring(cm0)
        precs1.append(prec1)
        recas1.append(reca1)
        f1scs1.append(f1sc1)
        precs2.append(prec2)
        recas2.append(reca2)
        f1scs2.append(f1sc2)
    precs1,recas1,f1scs1=np.array(precs1),np.array(recas1),np.array(f1scs1)
    precs2,recas2,f1scs2=np.array(precs2),np.array(recas2),np.array(f1scs2)
    print('Precision/Recall/F1Score of type IIP: ',np.average(precs1),np.average(recas1),np.average(f1scs1))
    print('Standard Derivatives of type IIP: ',np.std(precs1),np.std(recas1),np.std(f1scs1))
    print('Precision/Recall/F1Score of type IIL: ',np.average(precs2),np.average(recas2),np.average(f1scs2))
    print('Standard Derivatives of type IIL: ',np.std(precs2),np.std(recas2),np.std(f1scs2))
    return #np.array([[np.average(precs),np.average(recas),np.average(f1scs)],\
            #         [np.std(precs),np.std(recas),np.std(f1scs)]])
def rocplot(Y_test,Z2):
    fpr,tpr,trash=sklearn.metrics.roc_curve(Y_test,Z2,drop_intermediate=False)
    plt.plot(fpr,tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    return


