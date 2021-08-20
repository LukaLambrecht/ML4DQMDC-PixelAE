#!/usr/bin/env python
# coding: utf-8

# **Histogram classfier based on the MSE of an autoencoder reconstruction**
# 
# The AutoEncoder derives from the generic HistogramClassifier.  
# For this specific classifier, the output score of a histogram is the mean-square-error (MSE) between the original histogram and its autoencoder reconstruction.  
# In essence, it is just a wrapper for a tensorflow model.  



### imports

# external modules
import os
import sys
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
import importlib

# local modules
from HistogramClassifier import HistogramClassifier
sys.path.append('../../utils')
from autoencoder_utils import mseTop10Raw, mseTop10
import plot_utils




class AutoEncoder(HistogramClassifier):
    ### histogram classfier based on the MSE of an autoencoder reconstruction
    # the AutoEncoder derives from the generic HistogramClassifier. 
    # for this specific classifier, the output score of a histogram is the mean-square-error (MSE) 
    # between the original histogram and its autoencoder reconstruction.
    # in essence, it is just a wrapper for a tensorflow model.
    
    def __init__( self, model=None ):
        ### intializer from a tensorflow model
        # the model is assumed to be a valid tensorflow model;
        # it can be already trained before wrapping it in an AutoEncoder object,
        # but if this is not the case, the AutoEncoder.train function can be called afterwards.
        super( AutoEncoder,self ).__init__()
        if model is None:
            raise NotYetImplementedError('ERROR in AutoEncoder.__init__: init must take a fully trained and ready tensorflow model as input (for now)')
        if not isinstance( model, tensorflow.keras.Model ):
            raise Exception('ERROR in AutoEncoder.init: model has type {}'.format(type(model))
                           +' while a tensorflow model is expected')
        self.model = model
        
    def train( self, histograms, doplot=True, **kwargs ):
        ### train the model on a given set of input histograms
        super( AutoEncoder,self ).train( histograms )
        history = self.model.fit(histograms, histograms, **kwargs)
        if doplot: plot_utils.plot_loss(history)
        
    def evaluate( self, histograms ):
        ### classification of a collection of histograms based on their autoencoder reconstruction
        super( AutoEncoder,self ).evaluate( histograms )
        predictions = self.model.predict( histograms )
        return mseTop10Raw( histograms, predictions )
    
    def reconstruct( self, histograms ):
        ### return the autoencoder reconstruction of a set of histograms
        return self.model.predict( histograms )
    
    def save( self, path ):
        ### save the underlying tensorflow model to a tensorflow SavedModel or H5 format.
        # note: depending on the extension specified in path, the SavedModel or H5 format is chosen,
        #       see https://www.tensorflow.org/guide/keras/save_and_serialize
        super( AutoEncoder,self ).save( path )
        self.model.save( path )
        
    @classmethod
    def load( self, path, **kwargs ):
        ### get an AutoEncoder instance from a saved tensorflow SavedModel or H5 file
        super( AutoEncoder,self ).load( path )
        model = load_model(path, custom_objects={'mseTop10':mseTop10}, **kwargs)
        # to do: check if the above also works if mseTop10 is not in the model
        # to do: check if it is possible to add all custom objects without using the kwargs
        classifier = AutoEncoder( model=model )
        return classifier





