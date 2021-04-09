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




### cropping of hisograms

def crophists(hists, slices):
    ### perform cropping on a sit of histograms
    # input arguments:
    # - hists is a numpy array of shape (nhistograms,nbins) for 1D or (nhistograms,nybins,nxbins) for 2D
    # - slices is a list of slice objects (builtin python type) of length 1 (for 1D) or 2 (for 2D)
    if len(hists.shape)==2:
        return hists[:,slices[0]]
    elif len(hists.shape)==3:
        return hists[:,slices[0],slices[1]]
    else:
        raise Excepion('ERROR in hist_utils.py / crophists: histograms have invalid input shape: {}'.format(hists.shape))
        
### rebinning of histograms

def rebinhists(hists, factor):
    ### perform rebinning on a set of histograms
    # input arguments:
    # - hists is a numpy array of shape (nhistograms,nbins) for 1D or (nhistograms,nybins,nxbins) for 2D
    # - factor is the rebinning factor, or a tuple (y axis rebinning factor, x axis rebinning factor), 
    #   which must be a divisors of the respective number of bins.
    if len(hists.shape)==2:
        if(not hists.shape[1]%factor==0): 
            print('WARNING in hist_utils.py / rebinhists: no rebinning performed since no suitable reduction factor was given.')
            return hists
        (nhists,nbins) = hists.shape
        newnbins = int(nbins/factor)
        rebinned = np.zeros((nhists,newnbins))
        for i in range(newnbins):
            rebinned[:,i] = np.sum(hists[:,factor*i:factor*(i+1)],axis=1)
        return rebinned
    elif len(hists.shape)==3:
        if not len(factor)==2:
            print('WARNING in hist_utils.py / rebinhists: no rebinning performed since no suitable reduction factor was given')
            print('(must be a tuple of length 2 for 2D histograms)')
            return hists
        if( not hists.shape[1]%factor[0]==0 or not hists.shape[2]%factor[1]==0):
            print('WARNING: in hist_utils.py / rebinhists: no rebinning performed since no suitable reduction factor was given.')
            return hists
        (nhists,nybins,nxbins) = hists.shape
        newnybins = int(nybins/factor[0])
        newnxbins = int(nxbins/factor[1])
        rebinned = np.zeros((nhists,newnybins,newnxbins))
        for i in range(newnybins):
            for j in range(newnxbins):
                rebinned[:,i,j] = np.sum(hists[:,factor[0]*i:factor[0]*(i+1),factor[1]*j:factor[1]*(j+1)],axis=(1,2))
        return rebinned
    else:
        raise Excepion('ERROR in hist_utils.py / rebinhists: histograms have invalid input shape: {}'.format(hists.shape))

### normalization

def normalizehists(hists):
    ### perform normalization 
    # for 1D histograms, the sum of bin contents is set equal one for each histogram
    # for 2D histograms, the bin contents are scaled so the maximum is 1 for each histogram
    # (maybe later make more flexible by adding normalization stragy as argument)
    if len(hists.shape)==2: return normalize(hists, norm='l1', axis=1)
    elif len(hists.shape)==3:
        normhists = []
        for i in range(len(hists)):
            normhists.append( hists[i]/hists[i].max() )
        return np.array(normhists)
    else:
        raise Excepion('ERROR in hist_utils.py / normalizehists: histograms have invalid input shape: {}'.format(hists.shape))

### averaging a collection of histograms (e.g. for template definition)

def averagehists(hists, nout):
    ### partition hists (of shape (nhistograms,nbins) or (nhistograms,nybins,nxbins)) into nout parts and take the average histogram of each part
    nsub = int(len(hists)/nout)
    if len(hists.shape)==2:
        avghists = np.zeros((nout,hists.shape[1]))   
        for i in range(nout):
            startindex = i*nsub
            stopindex = (i+1)*nsub
            avghists[i,:] = np.mean(hists[startindex:stopindex,:],axis=0)
        return avghists
    elif len(hists.shape)==3:
        avghists = np.zeros((nout,hists.shape[1],hists.shape[2]))   
        for i in range(nout):
            startindex = i*nsub
            stopindex = (i+1)*nsub
            avghists[i,:] = np.mean(hists[startindex:stopindex,:,:],axis=0)
        return avghists
    else:
        raise Excepion('ERROR in hist_utils.py / averagehists: histograms have invalid input shape: {}'.format(hists.shape))




### functions for calculating moments of a histogram

def moment(bins, counts, order):
    ### get n-th central moment of a histogram
    # - bins is a 1D or 2D np array holding the bin centers
    #   (shape (nbins) or (nhistograms,nbins))
    # - array is a 2D np array containing the bin counts
    #   (shape (nhistograms,nbins))
    # - order is the order of the moment to calculate
    #   (0 = maximum, 1 = mean value)
    # note: for now only 1D histograms are supported
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

def histmean(bins, counts):
    ### special case of moment calculation (with order=1)
    return moment(bins,counts,1)

def histrms(bins, counts):
    ### special case of moment calculation
    return np.power(moment(bins,counts,2)-np.power(moment(bins,counts,1),2),0.5)

def histmoments(bins, counts, orders):
    ### apply moment calculation for a list of orders
    # the return type is a numpy array of shape (nhistograms,nmoments)
    moments = np.zeros((len(counts),len(orders)))
    for i,order in enumerate(orders):
        moments[:,i] = moment(bins,counts,order)
    return moments




### higher level function for automatic preprocessing of data

def preparedatafromnpy(dataname, cropslices=None, rebinningfactor=None, donormalize=True, doplot=False):
    ### read a .npy file and output the histograms
    
    hist = np.load(dataname,allow_pickle=False)
    # preprocessing of the data: rebinning and normalizing
    if cropslices is not None:  hist = crophists(hist,cropslices)
    if rebinningfactor is not None: hist = rebinhists(hist,rebinningfactor)
    if donormalize: hist = normalizehists(hist)
        
    if not doplot:
        return hist
    
    # plot some examples
    nplot = min(8,len(hist))
    flatindex = np.linspace(0,len(hist),num=len(hist),endpoint=False)
    randint = np.random.choice(flatindex,size=nplot,replace=False).astype(int)
    if len(hist.shape)==2:
        _,_ = plot_utils.plot_hists( hist[randint], colorlist='b',
                                    title = 'histogram examples',
                                    xaxtitle = 'bin number' )
    if len(hist.shape)==3:
        _,_ = plot_utils.plot_hists_2d( hist[randint], ncols=4, 
                                       title = 'histogram examples' )
    return hist

def preparedatafromdf(df, returnrunls=False, cropslices=None, rebinningfactor=None, donormalize=False, doplot=False):
    # prepare the data contained in a dataframe in the form of a numpy array
    # args:
    # - returnrunls: boolean whether to return a tuple of (histograms, run numbers, lumisection numbers).
    #   (default: return only histograms)
    # - cropslices: list of slices by which to crop the historams (default: no cropping)
    # - rebinningfactor: an integer (or tuple of integers for 2D histograms) to downsample/rebin the histograms (default: no rebinning)
    # - donormalize: boolean whether to normalize the data
    # - doplot: if True, some example plots are made showing the histograms

    # preprocessing of the data: rebinning and normalizing
    (hist,runnbs,lsnbs) = dataframe_utils.get_hist_values(df)
    if cropslices is not None:  hist = crophists(hist,cropslices)
    if rebinningfactor is not None: hist = rebinhists(hist,rebinningfactor)
    if donormalize: hist = normalizehists(hist)
        
    if not doplot:
        if returnrunls: return (hist,runnbs,lsnbs) 
        else: return hist
    
    # plot some examples
    nplot = min(8,len(hist))
    flatindex = np.linspace(0,len(hist),num=len(hist),endpoint=False)
    randint = np.random.choice(flatindex,size=nplot,replace=False).astype(int)
    if len(hist.shape)==2:
        _,_ = plot_utils.plot_hists( hist[randint], colorlist='b',
                                    title = 'histogram examples',
                                    xaxtitle = 'bin number' )
    if len(hist.shape)==3:
        _,_ = plot_utils.plot_hists_2d( hist[randint], ncols=4, 
                                       title = 'histogram examples' )
        
    if returnrunls: return (hist,runnbs,lsnbs)
    else: return hist

def preparedatafromcsv(dataname, returnrunls=False, cropslices=None, rebinningfactor=None, donormalize=True, doplot=False):
    ### prepare the data contained in a dataframe csv file in the form of a numpy array
    # args:
    # - returnrunls: boolean whether to return a tuple of (histograms, run numbers, lumisection numbers).
    #   (default: return only histograms)
    # - cropslices: list of slices by which to crop the historams (default: no cropping)
    # - rebinningfactor: an integer (or tuple of integers for 2D histograms) to downsample/rebin the histograms (default: no rebinning)
    # - donormalize: boolean whether to normalize the data
    # - doplot: if True, some example plots are made showing the histograms

    # read data
    df = csv_utils.read_csv(dataname)
    # prepare data from df
    return preparedatafromdf(df, returnrunls=returnrunls, cropslices=cropslices, rebinningfactor=rebinningfactor,donormalize=donormalize,doplot=doplot)





