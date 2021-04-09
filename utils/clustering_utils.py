#!/usr/bin/env python
# coding: utf-8



### imports 

# external modules
import numpy as np

# local modules




### functionality for fitting a function to point multidimensional point clouds
# DEPRECATED AND REPLACED BY EQUIVALENT CLASSES IN SRC/CLOUDFITTERS




def dummy():
    ### general remark: the functions in this script are currently not supported anymore
    # please ignore clustering_utils for now, it will probably be removed or completely reworked
    pass

def vecdist(moments,index):
    
    # does not work well if there are outliers which dominate the distance
    relmoments = moments-np.tile([moments[index,:]],(len(moments),1))
    sumsm = np.sum(relmoments[:index,:],axis=0)
    sumgr = np.sum(relmoments[index+1:,:],axis=0)
    distofsum = np.sqrt(np.sum(np.power(sumsm-sumgr,2)))
    sumofdist = np.sum(np.sqrt(np.sum(np.power(relmoments,2),axis=1)))
    return distofsum/sumofdist

def costhetadist(moments,index):
    
    # works more or less but not all bad points have small values, 
    # allows to identify problematic regions but not individual LS
    if(index==0 or index==len(moments)-1): return 0
    inprod = np.sum(np.multiply(moments[index-1,:],moments[index+1,:]))
    norms = np.sqrt(np.sum(np.power(moments[index-1,:],2)))*np.sqrt(np.sum(np.power(moments[index+1,:],2)))
    return inprod/norms

def avgnndist(moments,index,nn):
    
    # seems to work well for the runs tested!
    rng = np.array(range(index-nn,index+nn+1))
    rng = rng[(rng>=0) & (rng<len(moments))]
    moments_sec = moments[rng,:]-np.tile(moments[index,:],(len(rng),1))
    return np.sum(np.sqrt(np.sum(np.power(moments_sec,2),axis=1)))/len(rng)




def getavgnndist(hists,nmoments,xmin,xmax,nbins,nneighbours):
    
    dists = np.zeros(len(hists))
    moments = np.zeros((len(hists),nmoments))
    binwidth = (xmax-xmin)/nbins
    bins = np.tile(np.linspace(xmin+binwidth/2,xmax-binwidth/2,num=nbins,endpoint=True),(len(hists),1))
    for i in range(1,nmoments+1):
        moments[:,i-1] = moment(bins,hists,i)
    for i in range(len(hists)):
        dists[i] = avgnndist(moments,i,nneighbours)
    return dists




def filteranomalous(df,nmoments=3,rmouterflow=True,rmlargest=0.,doplot=True,):
    
    # do preliminary filtering (no DCS-bit OFF)
    # (implicitly re-index)
    print('total number of LS: '+str(len(df)))
    df = select_golden_and_bad(df)
    print('filtered number of LS (DCS-bit ON): '+str(len(df)))
    runs = get_runs(df)
    #print('found following runs: '+str(runs))
    
    # initializations
    nlumi = 0
    dists = []
    xmin = 0.
    xmax = 1.
    
    # loop over runs and calculate distances
    for run in runs:
        dfr = select_runs(df,[run])
        (hists,_,_) = get_hist_values(dfr)
        nlumi += len(hists)
        if rmouterflow: hists = hists[:,1:-1]
        rdists = getavgnndist(hists,nmoments,xmin,xmax,len(hists[0]),2)
        for d in rdists: dists.append(d)
            
    # concatenate all runs
    dists = np.array(dists)
    ind = np.linspace(0,nlumi,num=nlumi,endpoint=False)
    if doplot: plotdistance(dists,ind,rmlargest=rmlargest)
    (gmean,gstd) = getmeanstd(dists,ind,rmlargest=rmlargest)
        
    # add a columns to the original df
    df['dist'] = dists
    df['passmomentmethod'] = np.where(dists<gmean+3*gstd,1,0)
    
    # select separate df's
    dfpass = df[df['dist']<gmean+3*gstd]
    dfpass.reset_index(drop=True,inplace=True)
    dffail = df[df['dist']>gmean+3*gstd]
    dffail.reset_index(drop=True,inplace=True)
    
    return (df,dfpass,dffail,gmean,gstd)










