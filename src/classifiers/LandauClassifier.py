#!/usr/bin/env python
# coding: utf-8

# documentation: to do! 


### imports

# external modules
import os
import sys
import numpy as np
from scipy.optimize import curve_fit
import pickle
import importlib

# local modules
from HistogramClassifier import HistogramClassifier
sys.path.append('../../utils')
from autoencoder_utils import mseTop10Raw, mseTop10


def landaufun(x, landaumax, landauwidth, norm):
    # see https://en.wikipedia.org/wiki/Landau_distribution
    y = np.divide((x-landaumax),landauwidth)
    prefactor = norm/np.sqrt(2.*np.pi)
    return prefactor*np.exp( -(y+np.exp(-y))/2. )

def langaufun(x, landaumax, landauwidth, norm, gausswidth):
    y = np.divide((x-landaumax),landauwidth)
    prefactor = norm/np.sqrt(2.*np.pi)
    landau = prefactor*np.exp( -(y+np.exp(-y))/2. )
    gauss = np.exp( -0.5*np.power((x-x[int(len(x)/2)])/gausswidth,2) )/(np.sqrt(2*np.pi)*gausswidth)
    return np.convolve(landau, gauss, mode='same')
    
    
class LandauClassifier(HistogramClassifier):
    
    def __init__( self, dogauss=False ):
        super( LandauClassifier,self ).__init__()
        # initializations
        self.dogauss = dogauss
        self.fitfunc = landaufun if not dogauss else langaufun
        
    def train( self ):
        pass
    
    def fit( self, histogram ):
        # find initial guess for the parameters
        landaumax = np.argmax(histogram)
        landaumax_range = [landaumax*0.8, landaumax*1.2]
        landauwidth = landaumax/4.
        landauwidth_range = [landauwidth/4., landauwidth*4.]
        norm = np.sum(histogram)/4.
        norm_range = [0., norm*2.]
        initparams = [landaumax, landauwidth, norm]
        ranges = ([landaumax_range[0], landauwidth_range[0], norm_range[0]],
                  [landaumax_range[1], landauwidth_range[1], norm_range[1]])
        if self.dogauss:
            gausswidth = landauwidth
            initparams.append(gausswidth)
        # do the fit
        xdata = np.arange(0, len(histogram))
        try: 
            popt, _ = curve_fit( self.fitfunc, xdata, histogram, p0=initparams, 
                                 #bounds=ranges, 
                                 #method='dogbox', 
                                 ftol=1e-4, xtol=1e-4, gtol=1e-4 
                               )
        except: popt = initparams
        fitted_histogram = self.fitfunc( xdata, *popt )
        return (fitted_histogram, popt)
    
    def evaluate( self, histograms ):
        super( LandauClassifier,self ).evaluate( histograms )
        predictions = self.reconstruct(histograms)
        return mseTop10Raw( histograms, predictions )
    
    def reconstruct( self, histograms ):
        predictions = np.zeros(histograms.shape)
        for i,histogram in enumerate(histograms):
            predictions[i,:] = self.fit( histogram )[0]
        return predictions
    
    def save( self, path ):
        ### save the classifier
        super( LandauClassifier,self ).save( path )
        with open( path, 'wb' ) as f:
            pickle.dump( self, f )

    @classmethod
    def load( self, path, **kwargs ):
        ### get a LandauClassifier instance from a pkl file
        super( LandauClassifier, self ).load( path )
        with open( path, 'rb' ) as f:
            obj = pickle.load( f )
        return obj