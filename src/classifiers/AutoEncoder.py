#!/usr/bin/env python
# coding: utf-8

# **Histogram classfier based on the MSE of an autoencoder reconstruction**
# 
# The AutoEncoder derives from the generic HistogramClassifier.  
# For this specific classifier, the output score of a histogram is the mean-square-error (MSE) between the original histogram and its autoencoder reconstruction.  
# In essence, it is just a wrapper for a tensorflow model.  



### imports

# external modules
import sys
import numpy as np
import tensorflow
import importlib

# local modules
from HistogramClassifier import HistogramClassifier
sys.path.append('../../utils')
from autoencoder_utils import mseTop10Raw




class AutoEncoder(HistogramClassifier):
    ### histogram classfier based on the MSE of an autoencoder reconstruction
    # the AutoEncoder derives from the generic HistogramClassifier. 
    # for this specific classifier, the output score of a histogram is the mean-square-error (MSE) 
    # between the original histogram and its autoencoder reconstruction.
    # in essence, it is just a wrapper for a tensorflow model.
    
    def __init__( self, model=None ):
        ### intializer from a tensorflow model
        # the model is assumed to be fully trained on a suitable training set and ready for use
        # TODO: perhaps the functionality for initializing and training the model can be absorbed in the AutoEncoder class,
        #       but this is not yet supported currently
        
        super( AutoEncoder,self ).__init__()
        if model is None:
            raise NotYetImplementedError('ERROR in AutoEncoder.__init__: init must take a fully trained and ready tensorflow model as input (for now)')
        if not isinstance( model, tensorflow.keras.Model ):
            raise Exception('ERROR in AutoEncoder.init: model has type {}'.format(type(model))
                           +' while a tensorflow model is expected')
        self.model = model
        
    def evaluate( self, histograms ):
        ### classification of a collection of histograms based on their autoencoder reconstruction
        
        super( AutoEncoder,self).evaluate( histograms )
        predictions = self.model.predict( histograms )
        return mseTop10Raw( histograms, predictions )
    
    def reconstruct( self, histograms ):
        ### return the autoencoder reconstruction of a set of histograms
        return self.model.predict( histograms )





