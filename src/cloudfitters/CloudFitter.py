#!/usr/bin/env python
# coding: utf-8

# **Abstract base class for all point cloud fitting algorithms** 
# 
# Note that all concrete point cloud fitters must inherit from CloudFitter!  
# 
# How to make a concrete CloudFitter class:
# - define a class that inherits from CloudFitter
# - make sure all functions with @abstractmethod are implemented in your class
# - it is recommended to start each overriding function with a call to super(), but this is not strictly necessary
# 
# See also the existing examples!



### imports

# external modules
import sys
import numpy as np
from abc import ABC,abstractmethod
import importlib

# local modules
sys.path.append('../../utils')




class CloudFitter(ABC):
    ### abstract base class for all point cloud fitting algorithms
    # note that all concrete point cloud fitters must inherit from CloudFitter!
    # how to make a concrete CloudFitter class:
    # - define a class that inherits from CloudFitter
    # - make sure all functions with @abstractmethod are implemented in your class
    # - it is recommended to start each overriding function with a call to super(), but this is not strictly necessary
    # see also the existing examples!
    
    @abstractmethod
    def __init__( self ):
        ### empty intializer
        # this is an @abstractmethod and must be overridden in any concrete deriving class!
        pass
        
    @abstractmethod
    def fit( self, points ):
        # input arguments:
        # - points: 2D numpy array of shape (npoints,ndims)
        if not isinstance( points, np.ndarray ):
            raise Exception('ERROR in CloudFitter.init: points must be a numpy array but found type {}'.format(type(points)))
        if len(points.shape)!=2:
            raise Exception('ERROR in CloudFitter.init: points must be a 2D numpy array but found shape {}'.format(points.shape))
        (self.npoints,self.ndims) = points.shape
        
    @abstractmethod
    def pdf( self, points ):
        ### evaluate the pdf (probability density function) at given points
        # this is an @abstractmethod and must be overridden in any concrete deriving class!
        # input arguments:
        # - points: a 2D numpy array of shape (npoints,ndims)
        # output: a 1D array of shape (npoints)
        if not isinstance( points, np.ndarray ):
            raise Exception('ERROR in CloudFitter.pdf: points must be a numpy array but found type {}'.format(type(points)))
        if len(points.shape)!=2:
            raise Exception('ERROR in CloudFitter.pdf: points must be a 2D numpy array but found shape {}'.format(points.shape))
        if points.shape[1]!=self.ndims:
            raise Exception('ERROR in CloudFitter.pdf: points must have {} dimensions but found {}'.format(self.ndims,points.shape[1]))