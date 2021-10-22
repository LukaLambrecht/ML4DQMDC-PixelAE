#!/usr/bin/env python
# coding: utf-8

# **Histogram classification based on maximum pull between test histogram and reference histogram.**
# 
# Specifically intended for 2D histograms, but should in principle work for 1D as well.  
# Ssee static function 'pull' for definition of bin-per-bin pull and other notes.



### imports

# external modules
import sys
import numpy as np

# local modules
from HistogramClassifier import HistogramClassifier
sys.path.append('../../utils')




def pull( testhist, refhist ):
    ### calculate bin-per-bin pull between two histograms
    # bin-per-bin pull is defined here preliminarily as (testhist(bin)-refhist(bin))/sqrt(refhist(bin))
    # notes: 
    # - bins in the denominator where refhist is < 1 are set to one! This is for histograms with absolute counts, and they should not be normalized!
    # - instead another normalization is applied: the test histogram is multiplied by sum(refhist)/sum(testhist) before computing the pulls
    # input arguments:
    # - testhist, refhist: numpy arrays of the same shape
    # output:
    # numpy array of same shape as testhist and refhist
    denom = np.power(refhist,1/2)
    denom = np.where( denom<1, 1, denom )
    norm = np.sum(refhist)/np.sum(testhist)
    return (norm*testhist-refhist)/np.power(denom,1/2)

def maxabspull( testhist, refhist, n=1 ):
    ### calculate maximum of bin-per-bin pulls (in absolute value) between two histograms
    # see definition of bin-per-bin pull in function pull (above)
    # input arguments:
    # - testhist, refhist: numpy arrays of the same shape
    # - n: nubmer of largest pull values to average over (default: 1, just take single maximum)
    # output:
    # a float
    abspull = np.abs( pull(testhist,refhist) ).flatten()
    largest = np.partition( abspull, -n )[-n:]
    return np.mean(largest)

class MaxPullClassifier(HistogramClassifier):
    ### histogram classification based on maximum pull between test histogram and reference histogram.
    # specifically intended for 2D histograms, but should in principle work for 1D as well.
    # see static function pull (above) for definition of bin-per-bin pull and other notes.
    
    def __init__( self, nmaxpulls=1 ):
        ### initializer
        # input arguments:
        # - nmaxpulls: number of largest pull values to average over 
        #   (default: 1, just take single maximum)
        super( MaxPullClassifier,self ).__init__()
        self.n = nmaxpulls
        
    def set_nmaxpulls( self, nmaxpulls ):
        ### set the nmaxpulls parameter (see also initializer)
        self.n = nmaxpulls
        
    def train( self, refhist ):
        ### 'train' the classifier, i.e. set the reference histogram.
        # input arguments:
        # - refhist: a numpy array of shape (1,nbins) or (1,nybins,nxbins)
        super( MaxPullClassifier,self).train( refhist )
        if not refhist.shape[0]==1:
            raise Exception('ERROR in MaxPullClassifier/train: first dimension of training set is expected to be 1'
                           +' (only one reference histogram allowed, provided in format (1,nbins) or (1,nybins,nxbins))')
        self.refhist = refhist[0]
        
    def evaluate( self, histograms ):
        ### classify the histograms based on their max bin-per-bin pull (in absolute value) with respect to a reference histogram
        super( MaxPullClassifier,self).evaluate( histograms )
        maxpulls = np.zeros(len(histograms))
        for i,hist in enumerate(histograms):
            maxpulls[i] = maxabspull( hist, self.refhist, n=self.n )
        return maxpulls
    
    def getpull( self, histogram ):
        ### get the pull histogram for a given test histogram
        # input arguments:
        # histogram: a single histogram, i.e. numpy array of shape (nbins) for 1D or (nybins,nxbins) for 2D.
        # output:
        # numpy array of same shape as histogram containing bin-per-bin pull w.r.t. reference histogram
        return pull( histogram, self.refhist )





