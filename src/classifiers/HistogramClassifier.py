#!/usr/bin/env python
# coding: utf-8

# **Abstract base class for histogram classifying objects**  
# 
# Note that all concrete histogram classifiers must inherit from HistogramClassifier!
# A HistogramClassifier can be any object that classifies a histogram; in more detail:
# - the input is a collection of histograms (of the same type), represented by a numpy array of shape (nhists,nbins) for 1D histograms or (nhists,nybins,nxbins) for 2D histograms.
# - the output is an array of numbers of shape (nhists).
# - the processing between input and output can in principle be anything, but usually some sort of discriminating power is assumed.
# 
# How to make a concrete HistogramClassifier class:
# - define a class that inherits from HistogramClassifier
# - make sure all functions with @abstractmethod are implemented in your class
# - it is recommended to start each overriding function with a call to super(), but this is not strictly necessary
# 
# See also the existing examples!



### imports

# external modules
import os
import sys
import numpy as np
from abc import ABC,abstractmethod
import importlib

# local modules
sys.path.append('../../utils')




class HistogramClassifier(ABC):
    ### abstract base class for histogram classifying objects
    # note that all concrete histogram classifiers must inherit from HistogramClassifier!
    # a HistogramClassifier can be any object that classifies a histogram; in more detail:
    # - the input is a collection of histograms (of the same type), represented by a numpy array of shape (nhists,nbins) for 1D histograms or (nhists,nybins,nxbins) for 2D histograms.
    # - the output is an array of numbers of shape (nhists).
    # - the processing between input and output can in principle be anything, but usually some sort of discriminating power is assumed.
    # how to make a concrete HistogramClassifier class:
    # - define a class that inherits from HistogramClassifier
    # - make sure all functions with @abstractmethod are implemented in your class
    # - it is recommended to start each overriding function with a call to super(), but this is not strictly necessary
    # see also the existing examples!
    
    @abstractmethod
    def __init__( self ):
        ### empty intializer
        # this is an @abstractmethod and must be overridden in any concrete deriving class!
        pass
    
    @abstractmethod
    def train( self, histograms ):
        ### train the classifier on a set of input histograms
        # this is an @abstractmethod and must be overridden in any concrete deriving class!
        # input arguments:
        # - histograms: numpy array of shape (nhists,nbins) or (nhists,nybins,nxbins).
        # output: expected to be none.
        
        # check input args
        if not isinstance( histograms, np.ndarray ):
            raise Exception('ERROR in HistogramClassifier.evaluate: input is of type {}'.format(type(histograms))
                           +' while a numpy array is expected.')
        if( len(histograms.shape)!=2 and len(histograms.shape)!=3 ):
            raise Exception('ERROR in HistogramClassifier.evaluate: input array has shape {}'.format(histograms.shape)
                            +' while a 2D or 3D array is expected.')
    
    @abstractmethod
    def evaluate( self, histograms ):
        ### main function used to evaluate a set of histograms
        # this is an @abstractmethod and must be overridden in any concrete deriving class!
        # input arguments:
        # - histograms: numpy array of shape (nhists,nbins) or (nhists,nybins,nxbins).
        # output: expected to be a 1D numpy array of shape (nhists), one number per histogram.
        
        # check input args
        if not isinstance( histograms, np.ndarray ):
            raise Exception('ERROR in HistogramClassifier.evaluate: input is of type {}'.format(type(histograms))
                           +' while a numpy array is expected.')
        if( len(histograms.shape)!=2 and len(histograms.shape)!=3 ):
            raise Exception('ERROR in HistogramClassifier.evaluate: input array has shape {}'.format(histograms.shape)
                            +' while a 2D or 3D array is expected.')
            
    def save( self, path ):
        ### save a classifier to disk
        # specific implementation in concrete classes, here only path creation
        
        dirname = os.path.dirname( path )
        if not os.path.exists( dirname ):
            os.makedirs( dirname )
            
    @classmethod
    def load( self, path ):
        ### load a classifier object from disk
        # specific implementation in concrete classes, here only path checking
        if not os.path.exists( path ):
            raise Exception('ERROR in HistogramClassifier.load: file path {}'.format(path)
                           +' does not seem to exist.')










