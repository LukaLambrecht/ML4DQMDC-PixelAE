#!/usr/bin/env python
# coding: utf-8

# **Histogram classification based on principal component analysis**
#  
# It is basically a wrapper for a sklearn.decomposition.PCA instance.



### imports

# external modules
import sys
import numpy as np
from sklearn.decomposition import PCA

# local modules
from HistogramClassifier import HistogramClassifier
sys.path.append('../../utils')
from autoencoder_utils import mseTopNRaw, chiSquaredTopNRaw




class PCAClassifier(HistogramClassifier):
    ### histogram classification based on principal component analysis
    # it is basically a wrapper for a sklearn.decomposition.PCA instance.
    
    def __init__( self, ncomponents=None, svd_solver='auto', loss_type='mse', nmax=10 ):
        ### initializer
        # input arguments:
        # - ncomponents: number of PCA components (aka clusters aka basis vectors) to use in the decomposition
        # - svd_solver: solver method to extract the PCA components
        #   note: both ncomponents and svd_solver are arguments passed down to sklearn.decomposition.PCA,
        #         see https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        # - loss_type: choose from 'mse' (mean-squared-error) or 'chi2' (chi squared error)
        # - nmax: number of largest elements to keep in error calculation
        # TODO: add keyword arguments to pass down to sklearn.decomposition.PCA
        super( PCAClassifier,self ).__init__()
        self.PCA = PCA( n_components=ncomponents, svd_solver=svd_solver )
        self.loss_type = loss_type
        self.nmax = nmax
        
    def train( self, histograms ):
        ### train the PCA model on a given set of input histograms
        # input arguments:
        # - histograms: a numpy array of shape (nhists,nbins) or (nhists,nybins,nxbins) that will be used to fit a PCA model
        super( PCAClassifier,self ).train( histograms )
        self.shape = list(histograms.shape)[1:]
        if len(histograms.shape)==3:
            histograms = histograms.reshape(histograms.shape[0],-1)
        self.PCA.fit( histograms )
        
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
        
        super( PCAClassifier,self ).evaluate( histograms )
        if len(histograms.shape)==3:
            histograms = histograms.reshape(histograms.shape[0],-1)
        reco = self.PCA.inverse_transform(self.PCA.transform(histograms))
        if self.loss_type=='mse': return mseTopNRaw( histograms, reco, n=self.nmax )
        elif self.loss_type=='chi2': return chiSquaredTopNRaw( histograms, reco, n=self.nmax )
        else: raise Exception('ERROR in PCAClassifier.evaluate: loss_type {} not recognized'.format(self.loss_type))
    
    def get_components( self ):
        ### return the PCA components (aka cluster centers aka basis vectors)
        # output:
        # a numpy array of shape (ncomponents,nbins) or (ncomponents,nybins,nxbins)
        if len(self.shape)==1: return self.PCA.components_
        if len(self.shape)==2: return self.PCA.components_.reshape(self.PCA.n_components_,self.shape[0],self.shape[1])
        
    def reconstruct( self, histograms ):
        ### return the PCA reconstruction for a given set of histograms
        # input arguments:
        # - histograms: numpy array of shape (nhists,nbins) or (nhists,nybins,nxbins)
        if len(histograms.shape)==3:
            histograms = histograms.reshape(histograms.shape[0],-1)
        reco = self.PCA.inverse_transform(self.PCA.transform(histograms))
        if len(self.shape)==2: reco = reco.reshape(len(histograms),self.shape[0],self.shape[1])
        return reco

