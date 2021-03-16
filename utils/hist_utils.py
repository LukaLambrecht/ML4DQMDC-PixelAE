#!/usr/bin/env python
# coding: utf-8



### imports

# external modules
import numpy as np
from sklearn.preprocessing import normalize
import importlib

# local modules
import dataframe_utils
importlib.reload(dataframe_utils)
import csv_utils
importlib.reload(csv_utils)
import plot_utils
importlib.reload(plot_utils)




### rebinning of histograms

def rebinhists(hists,factor):
    ### perform rebinning on a set of histograms
    # hists is a numpy array of shape (nhistograms,nbins)
    # factor is the rebinning factor, which must be a divisor of nbins.
    if(not hists.shape[1]%factor==0): 
        print('WARNING in hist_utils.py / rebinhists: no rebinning performed since no suitable reduction factor was given.')
        return hists
    (len1,len2) = hists.shape
    newlen = int(len2/factor)
    rebinned = np.zeros((len1,newlen))
    for i in range(newlen):
        rebinned[:,i] = np.sum(hists[:,factor*i:factor*(i+1)],axis=1)
    return rebinned

### normalization

def normalizehists(hists):
    ### perform normalization (i.e. sum of bin contents equals one for each histogram)
    return normalize(hists, norm='l1', axis=1)

### averaging a collection of histograms (e.g. for template definition)

def averagehists(hists,nout):
    ### partition hists (of shape (nhistograms,nbins)) into nout parts and take the average histogram of each part
    avghists = np.zeros((nout,hists.shape[1]))
    nsub = int(len(hists)/nout)
    for i in range(nout):
        startindex = i*nsub
        stopindex = (i+1)*nsub
        avghists[i,:] = np.mean(hists[startindex:stopindex,:],axis=0)
    return avghists




### functions for calculating moments of a histogram

def moment(bins,counts,order):
    ### get n-th central moment of a histogram
    # - bins is a 1D or 2D np array holding the bin centers
    #   (shape (nbins) or (nhistograms,nbins))
    # - array is a 2D np array containing the bin counts
    #   (shape (nhistograms,nbins))
    # - order is the order of the moment to calculate
    #   (0 = maximum, 1 = mean value)
    if len(bins.shape)==1:
        bins = np.tile(bins,(len(counts),1))
    if not bins.shape == counts.shape:
        raise Exception('ERROR in hist_utils.py / moment: bins and counts do not have the same shape!')
    if len(bins.shape)==1:
        bins = np.array([bins])
        counts = np.array([counts])
    if order==0: # return maximum
        return np.nan_to_num(np.max(counts,axis=1))
    return np.nan_to_num(np.divide(np.sum(np.multiply(counts,np.power(bins,order)),axis=1,dtype=np.float),np.sum(counts,axis=1)))

def histmean(bins,counts):
    ### special case of moment calculation
    return moment(bins,counts,1)

def histrms(bins,counts):
    ### special case of moment calculation
    return np.power(moment(bins,counts,2)-np.power(moment(bins,counts,1),2),0.5)

def histmoments(bins,counts,orders):
    ### apply moment calculation for a list of orders
    # the return type is a numpy array of shape (nhistograms,nmoments)
    moments = np.zeros((len(counts),len(orders)))
    for i,order in enumerate(orders):
        moments[:,i] = moment(bins,counts,order)
    return moments




### higher level function for automatic preprocessing of data

def preparedatafromnpy(dataname, rebinningfactor=1, donormalize=True, doplot=False):
    # read a .npy file and output the histograms
    
    hist = np.load(dataname,allow_pickle=False)
    # preprocessing of the data: rebinning and normalizing
    hist = hist[:,1:-1]
    if rebinningfactor != 1: rhist = rebinhists(hist,rebinningfactor)
    else: rhist = hist
    if donormalize: rhist = normalizehists(rhist)
        
    if not doplot: return rhist
    
    # plot histograms
    fig,ax = plot_utils.plot_hists( rhist, colorlist='b',
                                  title = 'histograms loaded from {}'.format(dataname),
                                  xaxtitle = 'bin number', yaxtitle = 'counts' )
        
    return rhist

def preparedatafromdf(df, returnrunls=False, onlygolden=False, rebinningfactor=1, donormalize=True, doplot=False):
    # prepare the data contained in a dataframe in the form of a numpy array
    # args:
    # - returnrunls: wether to return only a histogram array or 1D arrays of run and lumisection numbers as well
    # - onlygolden: if True, only lumisections in the golden json file are kept
    # - rebinningfactor: an integer number to downsample the histograms in the dataframe
    # - donormalize: if True, data are normalized
    # - doplot: if True, some example plots are made showing the histograms
    
    if onlygolden:
        df = dataframe_utils.select_golden(df)

    # preprocessing of the data: rebinning and normalizing
    (hist,runnbs,lsnbs) = dataframe_utils.get_hist_values(df)
    hist = hist[:,1:-1]
    if rebinningfactor != 1: rhist = rebinhists(hist,rebinningfactor)
    else: rhist = hist
    if donormalize: rhist = normalizehists(rhist)
        
    if not doplot:
        if returnrunls: return (rhist,runnbs,lsnbs) 
        else: return rhist
    
    # plot some examples
    nplot = min(10,len(hist))
    flatindex = np.linspace(0,len(hist),num=len(hist),endpoint=False)
    randint = np.random.choice(flatindex,size=nplot,replace=False).astype(int)
    
    fig,ax = plot_utils.plot_hists( hist[randint], colorlist='b',
                                    title = 'examples of histograms in dataframe',
                                    xaxtitle = 'bin number', yaxtitle = 'counts' )
    fig,ax = plot_utils.plot_hists( rhist[randint], colorlist='b',
                                    title = 'same histograms, but rebinned and normalized',
                                    xaxtitle = 'bin number', yaxtitle = 'counts')
        
    if returnrunls: return (rhist,runnbs,lsnbs)
    else: return rhist

def preparedatafromcsv(dataname, returnrunls=False, onlygolden=False, rebinningfactor=1, donormalize=True, doplot=False):
    # prepare the data contained in a dataframe csv file in the form of a numpy array
    # args:
    # - returnrunls: wether to return only a histogram array or 1D arrays of run and lumisection numbers as well
    # - onlygolden: if True, only lumisections in the golden json file are kept
    # - rebinningfactor: an integer number to downsample the histograms in the dataframe
    # - doplot: if True, some example plots are made showing the histograms

    # read data
    df = csv_utils.read_csv(dataname)
    # prepare data from df
    return preparedatafromdf(df, returnrunls=returnrunls,onlygolden=onlygolden,rebinningfactor=rebinningfactor,donormalize=donormalize,doplot=doplot)





