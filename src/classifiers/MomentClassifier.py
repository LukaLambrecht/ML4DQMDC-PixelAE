#!/usr/bin/env python
# coding: utf-8

# documentation: to do! 


### imports

# external modules
import os
import sys
import numpy as np
import pickle
import importlib

# local modules
from HistogramClassifier import HistogramClassifier
sys.path.append('../../utils')
from autoencoder_utils import mseTop10Raw, mseTop10
from hist_utils import histmoments

        
class MomentClassifier(HistogramClassifier):
    
    def __init__( self, orders=None ):
        super( MomentClassifier,self ).__init__()
        # initializations
        if orders is None:
            orders = [1]
        self.orders = orders
        self.moment_means = [None]*len(orders)
        self.moment_stds = [None]*len(orders)
        
    def train( self, histograms ):
        super( MomentClassifier,self ).train( histograms )
        bins = np.arange(0,histograms.shape[1])
        moments = histmoments(bins, histograms, self.orders)
        self.moment_means = np.mean(moments, axis=0)
        self.moment_stds = np.std(moments, axis=0)
    
    def evaluate( self, histograms ):
        super( MomentClassifier,self ).evaluate( histograms )
        bins = np.arange(0,histograms.shape[1])
        moments = histmoments(bins, histograms, self.orders)
        res = np.divide( np.abs(moments - self.moment_means), self.moment_stds )
        res = np.sum( np.power(res, 2), axis=1 )
        return res
    
    def printout( self, histogram ):
        bins = np.arange(0,len(histogram))
        moments = histmoments(bins, np.array([histogram]), self.orders)
        print('moments for this histogram: {}'.format(moments))
        print('moments mean: {}'.format(self.moment_means))
        print('moments std: {}'.format(self.moment_stds))
        res = np.divide( moments - self.moment_means, self.moment_stds )
        print('relative deviations: {}'.format(res))
        res = np.sum( np.power(res, 2), axis=1 )
        print('total quadratic deviation: {}'.format(res))
        return res
    
    def save( self, path ):
        ### save the classifier
        super( MomentClassifier,self ).save( path )
        with open( path, 'wb' ) as f:
            pickle.dump( self, f )

    @classmethod
    def load( self, path, **kwargs ):
        ### get a MomentClassifier instance from a pkl file
        super( MomentClassifier, self ).load( path )
        with open( path, 'rb' ) as f:
            obj = pickle.load( f )
        return obj