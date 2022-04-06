#!/usr/bin/env python
# coding: utf-8

# **Histogram classification based on nonnegative matrix factorization**
# 
# Specifically intended for 2D histograms, but should in principle work for 1D as well.  
# It is basically a wrapper for a sklearn.decomposition.NMF instance.


### imports

# external modules
import os
import sys
import numpy as np
import math
import pickle
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

# local modules
from HistogramClassifier import HistogramClassifier
sys.path.append('../../utils')
from autoencoder_utils import mseTopNRaw, chiSquaredTopNRaw
import plot_utils


class NMFClassifier(HistogramClassifier):
    ### histogram classification based on nonnegative matrix factorization
    # specifically intended for 2D histograms, but should in principle work for 1D as well.
    # it is basically a wrapper for a sklearn.decomposition.NMF instance.
    
    def __init__( self, ncomponents=5, loss_type='mse', nmax=10 ):
        ### initializer
        # input arguments:
        # - ncomponents: number of NMF components (aka clusters aka basis vectors) to use in the decomposition
        # - loss_type: choose from 'mse' (mean-squared-error) or 'chi2' (chi squared error)
        # - nmax: number of largest elements to keep in error calculation
        # TODO: add keyword arguments to pass down to sklearn.decomposition.NMF
        super( NMFClassifier,self ).__init__()
        self.NMF = NMF( n_components=ncomponents )
        self.loss_type = loss_type
        self.nmax = nmax
        self.ncomponents = ncomponents
        
    def train( self, histograms, doplot=True, ncols=None, title=None ):
        ### train the NMF model on a given set of input histograms
        # input arguments:
        # - histograms: a numpy array of shape (nhists,nbins) or (nhists,nybins,nxbins) that will be used to fit a NMF model
        super( NMFClassifier,self ).train( histograms )
        self.shape = list(histograms.shape)[1:]
        if len(histograms.shape)==3:
            histograms = histograms.reshape(histograms.shape[0],-1)
        self.NMF.fit( histograms )
        if doplot: 
            components = self.get_components()
            if title is None: title = 'NMF model components'
            if len(components.shape)==2:
                plot_utils.plot_hists_multi( components, 
                                colorlist=list(range(self.ncomponents)), 
                                title=title)
            elif len(components.shape)==3:
                if ncols is None: ncols = int(math.sqrt(self.ncomponents))
                plot_utils.plot_hists_2d(components, axsize=3, 
                                ncols=ncols, 
                                title=title)
            plt.show(block=False)
        
    def set_nmax( self, nmax ):
        ### set number of largest elements to keep in mean square error calculation
        # useful to quickly re-evaluate the model with different nmax without retraining
        # input arguments:
        # - nmax: number of largest elements to keep in mean square error calculation
        self.nmax = nmax
        
    def set_loss_type( self, loss_type ):
        ### set loss type
        # useful to quickly re-evaluate the model with different loss without retraining
        # input arguments:
        # - loss_type: choose from 'mse' (mean-squared-error) or 'chi2' (chi squared error)
        self.loss_type = loss_type
        
    def evaluate( self, histograms ):
        ### classify the given histograms based on the MSE with respect to their reconstructed version
        # input arguments:
        # - histograms: numpy array of shape (nhists,nbins) or (nhists,nybins,nxbins)
        
        super( NMFClassifier,self ).evaluate( histograms )
        if len(histograms.shape)==3:
            histograms = histograms.reshape(histograms.shape[0],-1)
        reco = self.NMF.inverse_transform(self.NMF.transform(histograms))
        if self.loss_type=='mse': return mseTopNRaw( histograms, reco, n=self.nmax )
        elif self.loss_type=='chi2': return chiSquaredTopNRaw( histograms, reco, n=self.nmax )
        else: raise Exception('ERROR in NMFClassifier.evaluate: loss_type {} not recognized'.format(self.loss_type))
    
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

    def save( self, path ):
        ### save the underlying NMF model
        super( NMFClassifier, self ).save( path )
        with open( path, 'wb' ) as f:
            pickle.dump( self, f )

    @classmethod
    def load( self, path, **kwargs ):
        ### get an NMFClassifier instance from a pkl file
        super( NMFClassifier, self ).load( path )
        with open( path, 'rb' ) as f:
            obj = pickle.load( f )
        return obj
