#!/usr/bin/env python
# coding: utf-8

# **Histogram classification based on nonnegative matrix factorization**
# 
# Specifically intended for 2D histograms, but should in principle work for 1D as well.  
# It is basically a wrapper for a sklearn.decomposition.NMF instance.



### imports

# external modules
import sys
import numpy as np
from sklearn.decomposition import NMF

# local modules
from HistogramClassifier import HistogramClassifier
sys.path.append('../../utils')
from autoencoder_utils import mseTopNRaw




class NMFClassifier(HistogramClassifier):
    ### histogram classification based on nonnegative matrix factorization
    # specifically intended for 2D histograms, but should in principle work for 1D as well.
    # it is basically a wrapper for a sklearn.decomposition.NMF instance.
    
    def __init__( self, histograms, ncomponents, nmax ):
        ### initializer from a collection of histograms
        # input arguments:
        # - histograms: a numpy array of shape (nhists,nbins) or (nhists,nybins,nxbins) that will be used to fit a NMF model
        # - ncomponents: number of NMF components (aka clusters aka basis vectors) to use in the decomposition
        # - nmax: number of largest elements to keep in mean square error calculation
        # TODO: add keyword arguments to pass down to sklearn.decomposition.NMF
        
        super( NMFClassifier,self ).__init__()
        self.shape = list(histograms.shape)[1:]
        if len(histograms.shape)==3:
            histograms = histograms.reshape(histograms.shape[0],-1)
        self.NMF = NMF( n_components=ncomponents )
        self.NMF.fit( histograms )
        self.nmax = nmax
        
    def set_nmax( self, nmax ):
        ### set number of largest elements to keep in mean square error calculation
        # useful to quickly re-evaluate the model with different nmax without retraining
        # input arguments:
        # - nmax: number of largest elements to keep in mean square error calculation
        self.nmax = nmax
        
    def evaluate( self, histograms ):
        ### classify the given histograms based on the MSE with respect to their reconstructed version
        # input arguments:
        # - histograms: numpy array of shape (nhists,nbins) or (nhists,nybins,nxbins)
        
        super( NMFClassifier,self ).evaluate( histograms )
        if len(histograms.shape)==3:
            histograms = histograms.reshape(histograms.shape[0],-1)
        reco = self.NMF.inverse_transform(self.NMF.transform(histograms))
        return mseTopNRaw( histograms, reco, n=self.nmax )
    
    def get_components( self ):
        ### return the NMF components (aka cluster centers aka basis vectors)
        # output:
        # a numpy array of shape (ncomponents,nbins) or (ncomponents,nybins,nxbins)
        if len(self.shape)==1: return self.NMF.components_
        if len(self.shape)==2: return self.NMF.components_.reshape(self.NMF.n_components_,self.shape[0],self.shape[1])
        
    def reconstruct( self, histograms ):
        ### return the NMF reconstruction for a given set of histograms
        # input arguments:
        # - histograms: numpy array of shape (nhists,nbins) or (nhists,nybins,nxbins)
        if len(histograms.shape)==3:
            histograms = histograms.reshape(histograms.shape[0],-1)
        reco = self.NMF.inverse_transform(self.NMF.transform(histograms))
        if len(self.shape)==2: reco = reco.reshape(len(histograms),self.shape[0],self.shape[1])
        return reco





