#!/usr/bin/env python
# coding: utf-8

# **Histogram classfier based on the MSE of an autoencoder reconstruction**
# 
# The AutoEncoder derives from the generic HistogramClassifier.  
# For this specific classifier, the output score of a histogram is the mean-square-error (MSE) between the original histogram and its autoencoder reconstruction.  
# In essence, it is just a wrapper for a Keras model.



### imports

# external modules
import os
import sys
import numpy as np
from keras import Model
from keras.models import load_model
import importlib

# local modules
from HistogramClassifier import HistogramClassifier
sys.path.append('../../utils')
from autoencoder_utils import mseTop10Raw, mseTop10
import plot_utils




class AutoEncoder(HistogramClassifier):
    ### Histogram classfier based on the MSE of an autoencoder reconstruction
    # the AutoEncoder derives from the generic HistogramClassifier. 
    # for this specific classifier, the output score of a histogram is the mean-square-error (MSE) 
    # between the original histogram and its autoencoder reconstruction.
    # in essence, it is just a wrapper for a Keras model.
    
    def __init__( self, model=None, modelpath=None ):
        ### Intializer from a Keras model
        # Input arguments:
        # - model: a valid Keras model;
        #          it does not have to be trained already,
        #          the AutoEncoder.train function will take care of this.
        # - modelpath: path to a stored Keras model,
        #              it does not have to be trained already,
        #              the AutoEncoder.train function will take care of this.
        # note: model and modelpath are alternative options, they should not both be used simultaneously.
        super( AutoEncoder,self ).__init__()
        if( model is None and modelpath is None ):
            raise Exception('ERROR in AutoEncoder.__init__: model and modelpath cannot both be None.')
        if( model is not None and modelpath is not None ):
            raise Exception('ERROR in AutoEncoder.__init__: model and modelpath cannot both be specified.')
        # case 1: model is specified directly:
        if( model is not None ):
            if not isinstance( model, Model ):
                raise Exception('ERROR in AutoEncoder.init: model has type {}'.format(type(model))
                               +' while a Keras model is expected.')
            self.model = model
        # case 2: model path is specified
        if( modelpath is not None ):
            if not os.path.exists(modelpath):
                raise Exception('ERROR in AutoEncoder.init: model path {} does not exist.'.format(modelpath))
            self.model = load_model(modelpath,custom_objects={'mseTop10':mseTop10})
        
    def train( self, histograms, doplot=True, epochs=10, batch_size=500, shuffle=False, verbose=1, validation_split=0.1, **kwargs ):
        ### Train the model on a given set of input histograms
        # Input arguments:
        # - histograms: set of training histograms, a numpy array of shape (nhistograms,nbins)
        # - doplot: boolean whether to make a plot of the loss value
        # - others: see the keras fit function
        # - kwargs: additional arguments passed down to keras fit function
        super( AutoEncoder,self ).train( histograms )
        history = self.model.fit(histograms, histograms, epochs=epochs, 
                                 batch_size=batch_size, shuffle=shuffle, 
                                 verbose=verbose, validation_split=validation_split, 
                                 **kwargs)
        if doplot: plot_utils.plot_loss(history)
        
    def evaluate( self, histograms ):
        ### Classification of a collection of histograms based on their autoencoder reconstruction
        super( AutoEncoder,self ).evaluate( histograms )
        predictions = self.model.predict( histograms )
        return mseTop10Raw( histograms, predictions )
    
    def reconstruct( self, histograms ):
        ### Return the autoencoder reconstruction of a set of histograms
        return self.model.predict( histograms )
    
    def save( self, path ):
        ### save the underlying Keras model.
        # note:
        # - The saving format is determined from the file extension specified in `path`.
        # - The only (cross-backend) supported format for Keras 3 is the "Keras v3", using the ".keras" extension.
        # - For Keras 2 and the TensorFlow backend, the "TensorFlow SavedModel" format is chosen when the extension is ".keras".
        # - When the extension is ".h5" or ".hdf5", the HDF5 format is chosen.
        #   The HDF5 format support predates the "TensorFlow SavedModel", but is considered legacy in Keras 3.
        # - See also:
        #     - https://keras.io/guides/serialization_and_saving/
        #     - https://www.tensorflow.org/guide/keras/save_and_serialize
        #     - https://github.com/keras-team/keras/blob/v3.11.3/keras/src/saving/saving_api.py#L19
        super( AutoEncoder,self ).save( path )
        self.model.save( path )
        
    @classmethod
    def load( self, path, **kwargs ):
        ### get an AutoEncoder instance from a saved Keras model file
        super( AutoEncoder,self ).load( path )
        model = load_model(path, custom_objects={'mseTop10':mseTop10}, **kwargs)
        # to do: check if the above also works if mseTop10 is not in the model
        # to do: check if it is possible to add all custom objects without using the kwargs
        classifier = AutoEncoder( model=model )
        return classifier
