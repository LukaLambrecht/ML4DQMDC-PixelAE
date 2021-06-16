#!/usr/bin/env python
# coding: utf-8

# **A collection of functions used for performing clustering tasks**  
# 
# This collection of tools is a little deprecated at this moment but kept for reference; it contains functionality for pre-filtering the histograms in the training set based on their moments (e.g. mean, rms).  
# Note that the functions here have not been used in a long time and might need some maintenance before they work properly again.



### imports 

# external modules
import os
import sys
import numpy as np

# local modules
from hist_utils import moment




def vecdist(moments, index):
    ### calculate the vectorial distance between a set of moments
    # input arguments:
    # - moments: 2D numpy array of shape (ninstances,nmoments)
    # - index: index for which instance to calculate the distance relative to the other instances
    # returns:
    # - a distance measure for the given index w.r.t. the other instances in 'moments'
    # notes:
    # - for this distance measure, the points are considered as vectors and the point at index is the origin.
    #   with respect to this origin, the average vector before index and the average vector after index are calculated.
    #   the distance is then defined as the norm of the difference of these vectors, 
    #   normalized by the norms of the individual vectors.
    relmoments = moments-np.tile([moments[index,:]],(len(moments),1))
    sumsm = np.sum(relmoments[:index,:],axis=0)
    sumgr = np.sum(relmoments[index+1:,:],axis=0)
    distofsum = np.sqrt(np.sum(np.power(sumsm-sumgr,2)))
    sumofdist = np.sum(np.sqrt(np.sum(np.power(relmoments,2),axis=1)))
    return distofsum/sumofdist

def costhetadist(moments, index):
    ### calculate the costheta distance between a set of moments
    # input arguments:
    # - moments: 2D numpy array of shape (ninstances,nmoments)
    # - index: index for which instance to calculate the distance relative to the other instances
    # returns:
    # - a distance measure for the given index w.r.t. the other instances in 'moments'
    # notes:
    # - this distance measure takes the cosine of the angle between the point at index
    #   and the one at index-1 (interpreted as vectors from the origin).
    if(index==0): return 0
    inprod = np.sum(np.multiply(moments[index-1,:],moments[index,:]))
    norms = np.sqrt(np.sum(np.power(moments[index-1,:],2)))*np.sqrt(np.sum(np.power(moments[index,:],2)))
    if norms<1e-12: norms += 1e-12
    return inprod/norms

def avgnndist(moments, index, nn):
    ### calculate average euclidean distance to neighbouring points
    # input arguments:
    # - moments: 2D numpy array of shape (ninstances,nmoments)
    # - index: index for which instance to calculate the distance relative to the other instances
    # - nn: (half-) window size
    # returns:
    # - a distance measure for the given index w.r.t. the other instances in 'moments'
    # notes:
    # - for this distance measure, the average euclidean distance is calculated between the point at 'index'
    #   and the points at index-nn and index+nn (e.g. the nn previous and next lumisections).
    rng = np.array(range(index-nn,index+nn+1))
    rng = rng[(rng>=0) & (rng<len(moments))]
    moments_sec = moments[rng,:]-np.tile(moments[index,:],(len(rng),1))
    return np.sum(np.sqrt(np.sum(np.power(moments_sec,2),axis=1)))/len(rng)




def getavgnndist(hists, nmoments, xmin, xmax, nbins, nneighbours):
    ### apply avgnndist to a set of histograms
    
    dists = np.zeros(len(hists))
    moments = np.zeros((len(hists),nmoments))
    binwidth = (xmax-xmin)/nbins
    bins = np.tile(np.linspace(xmin+binwidth/2,xmax-binwidth/2,num=nbins,endpoint=True),(len(hists),1))
    for i in range(1,nmoments+1):
        moments[:,i-1] = moment(bins,hists,i)
    for i in range(len(hists)):
        dists[i] = avgnndist(moments,i,nneighbours)
    return dists




def filteranomalous(df, nmoments=3, rmouterflow=True, rmlargest=0., doplot=True):
    ### do a pre-filtering, removing the histograms with anomalous moments
    
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





