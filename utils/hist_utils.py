#!/usr/bin/env python
# coding: utf-8

# **A collection of useful basic functions for processing histograms.**  
# 
# Functionality includes:
# - rebinning and normalization
# - moment calculation
# - averaging
# - higher-level functions preparing data for ML training, starting from a dataframe or input csv file.



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
    ### perform cropping on a set of histograms
    # input arguments:
    # - hists: a numpy array of shape (nhistograms,nbins) for 1D or (nhistograms,nybins,nxbins) for 2D
    # - slices: a slice object (builtin python type) or a list of two slices (for 2D)
    #   notes: 
    #     - a slice can be created using the builtin python syntax 'slice(start,stop,step)', 
    #       and the syntax 'list[slice]' is equivalent to 'list[start:stop:step]'.
    #       use 'None' to ignore one of the arguments for slice creation (equivalent to ':' in direct slicing)
    #     - for 1D histograms, slices can be either a slice object or a list of length 1 containing a single slice.
    # example usage:
    # - see tutorials/plot_histograms_2d.ipynb
    # returns:
    # - a numpy array containing the same histograms as input but cropped according to the slices argument
    if len(hists.shape)==2:
        if isinstance(slices,slice): slices=[slices]
        return hists[:,slices[0]]
    elif len(hists.shape)==3:
        return hists[:,slices[0],slices[1]]
    else:
        raise Exception('ERROR in hist_utils.py / crophists: histograms have invalid input shape: {}'.format(hists.shape))
        
### rebinning of histograms

def rebinhists(hists, factor):
    ### perform rebinning on a set of histograms
    # input arguments:
    # - hists: a numpy array of shape (nhistograms,nbins) for 1D or (nhistograms,nybins,nxbins) for 2D
    # - factor: the rebinning factor (for 1D), or a tuple of (y axis rebinning factor, x axis rebinning factor) (for 2D) 
    #   note: the rebinning applied here is simple summing of bin contents,
    #         and the rebinning factors must be divisors of the respective number of bins!
    # example usage:
    # - see tutorials/plot_histograms_2d.ipynb
    # returns:
    # - a numpy array containing the same histograms as input but rebinned according to the factor argument
    if len(hists.shape)==2:
        if(not hists.shape[1]%factor==0): 
            print('WARNING in hist_utils.py / rebinhists: no rebinning performed since no suitable reduction factor was given.'
                 +' The rebinning factor ({}) is not a divisor of the number of bins ({})'.format(factor,hists.shape[1]))
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
            print('WARNING: in hist_utils.py / rebinhists: no rebinning performed since no suitable reduction factor was given.'
                 +' The rebinning factors ({}) are not divisors of the number of bins ({})'.format(factor,(hists.shape[1],hists.shape[2])))
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
        raise Exception('ERROR in hist_utils.py / rebinhists: histograms have invalid input shape: {}'.format(hists.shape))

### normalization

def normalizehists(hists):
    ### perform normalization on a set of histograms
    # note: 
    # - for 1D histograms, the sum of bin contents is set equal one for each histogram
    # - for 2D histograms, the bin contents are scaled so the maximum is 1 for each histogram
    # - maybe later make more flexible by adding normalization stragy as argument...
    # input arguments:
    # - hists: a numpy array of shape (nhistograms,nbins) for 1D or (nhistograms,nybins,nxbins) for 2D
    # returns:
    # - a numpy array containing the same histograms as input but normalized
    if len(hists.shape)==2: return normalize(hists, norm='l1', axis=1)
    elif len(hists.shape)==3:
        normhists = []
        for i in range(len(hists)):
            normhists.append( hists[i]/hists[i].max() )
        return np.array(normhists)
    else:
        raise Exception('ERROR in hist_utils.py / normalizehists: histograms have invalid input shape: {}'.format(hists.shape))

### averaging a collection of histograms (e.g. for template definition)

def averagehists(hists, nout):
    ### partition a set of histograms into equal parts and take the average histogram of each part
    # input arguments:
    # - hists: a numpy array of shape (nhistograms,nbins) for 1D or (nhistograms,nybins,nxbins) for 2D
    # - nout: number of partitions / output histograms
    #   note: nout=1 corresponds to simply taking the average of all histograms in hists.
    # returns:
    # - a numpy array of shape (nout,<input number of bins>)
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
        raise Exception('ERROR in hist_utils.py / averagehists: histograms have invalid input shape: {}'.format(hists.shape))
        
def running_average_hists(hists, window, weights=None):
    ### replace each histogram in a collection of histograms by its running average
    # input arguments:
    # - hists: a numpy array of shape (nhistograms,nbins) for 1D or (nhistograms,nybins,nxbins) for 2D
    # - nwindow: number of histograms to consider for the averaging
    #   if window is an integer, it is the number of previous histograms in hists used for averaging
    #   (so window=0 would correspond to no averaging)
    #   if window is a tuple, it corresponds to (nprevious,nnext), and the nprevious previous and nnext next histograms in hists are used for averaging
    #   (so window=(0,0) would correspond to no averaging)
    # - weights: a list or numpy array containing the relative weights of the histograms in the averaging procedure.
    #   note: the weights can be any number, but they will be normalized to have unit sum.
    #   note: weights must have length nwindow+1 or nprevious+1+nnext.
    #   note: the default behaviour is a uniform array with values 1./(window+1) (or 1./(nprevious+1+nnext))
    # returns:
    # - a numpy array with same shape as input but where each histogram is replaced by its running average
    # notes:
    # - at the edges, the weights are cropped to match the input array and renormalized
    # - this function will crash when the length of the set of histograms is smaller than the total window length,
    #   maybe extend later (although this is not normally needed)
    
    # check input arguments
    if isinstance(window,int):
        window = (window,0)
    elif len(window)!=2:
        raise Exception('ERROR in hist_utils.py / running_average_hists: nwindow argument is invalid: {}'.format(window))
    nwindow = window[0]+window[1]+1
    if weights is None:
        weights = np.ones(nwindow)/nwindow
    elif len(weights)!=nwindow:
        raise Exception('ERROR in hist_utils.py / running_average_hists: weights argument is invalid: '
                       +'found length {} while the window has length {}'.format(len(weights),nwindow))
    avghists = np.zeros(hists.shape)
    weights = normalize([weights], norm='l1')[0]
    # first low edge
    for i in range(window[0]):
        thesehists = hists[0:i+1+window[1]]
        theseweights = normalize([weights[-len(thesehists):]], norm='l1')[0]
        avghists[i] = np.average( thesehists, weights=theseweights, axis=0 )
    # then middle part
    for i in range(window[0],len(avghists)-window[1]):
        thesehists = hists[i-window[0]:i+1+window[1]]
        avghists[i] = np.average( thesehists, weights=weights, axis=0 )
    # finally high edge
    for i in range(len(avghists)-window[1],len(avghists)):
        thesehists = hists[i-window[0]:]
        theseweights = normalize([weights[:len(thesehists)]], norm='l1')[0]
        avghists[i] = np.average( thesehists, weights=theseweights, axis=0 )
    return avghists




### functions for calculating moments of a histogram

def moment(bins, counts, order):
    ### get n-th central moment of a histogram
    # input arguments:
    # - bins: a 1D or 2D np array holding the bin centers
    #   (shape (nbins) or (nhistograms,nbins))
    # - counts: a 2D np array containing the bin counts
    #   (shape (nhistograms,nbins))
    # - order: the order of the moment to calculate
    #   (0 = maximum value, 1 = mean value)
    # returns:
    # - an array of shape (nhistograms) holding the requested moment per histogram
    # notes: 
    # - for now only 1D histograms are supported!
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
    # input arguments:
    # - see function moment(bins, counts, order),
    #   the only difference being that orders is a list instead of a single number
    # returns:
    # - a numpy array of shape (nhistograms,nmoments)
    moments = np.zeros((len(counts),len(orders)))
    for i,order in enumerate(orders):
        moments[:,i] = moment(bins,counts,order)
    return moments




### higher level function for automatic preprocessing of data

def preparedatafromnpy(dataname, cropslices=None, rebinningfactor=None, donormalize=True, doplot=False):
    ### read a .npy file and output the histograms
    # input arguments: 
    # - see e.g. preparedatafromdf
    # notes: 
    # - not yet tested for 2D histograms, but is expected to work...
    
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
    ### prepare the data contained in a dataframe in the form of a numpy array
    # input arguments:
    # - returnrunls: boolean whether to return a tuple of (histograms, run numbers, lumisection numbers).
    #   (default: return only histograms)
    # - cropslices: list of slices (one per dimension) by which to crop the historams (default: no cropping)
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
    # input arguments:
    # - returnrunls: boolean whether to return a tuple of (histograms, run numbers, lumisection numbers).
    #   (default: return only histograms)
    # - cropslices: list of slices (one per dimension) by which to crop the historams (default: no cropping)
    # - rebinningfactor: an integer (or tuple of integers for 2D histograms) to downsample/rebin the histograms (default: no rebinning)
    # - donormalize: boolean whether to normalize the data
    # - doplot: if True, some example plots are made showing the histograms

    # read data
    df = csv_utils.read_csv(dataname)
    # prepare data from df
    return preparedatafromdf(df, returnrunls=returnrunls, cropslices=cropslices, rebinningfactor=rebinningfactor,donormalize=donormalize,doplot=doplot)





