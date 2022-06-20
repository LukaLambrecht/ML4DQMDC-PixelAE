#!/usr/bin/env python
# coding: utf-8

# **Simple fitter making a hard cut in each dimension**
# 
# 



### imports

# external modules
import sys
import numpy as np
import importlib

# local modules
from CloudFitter import CloudFitter
sys.path.append('../../utils')




def calculate_cut_values( values, quantile, side='both' ):
    ### calculate the appropriate cut values to discard a given quantile of values
    # input arguments:
    # - values: a 1D numpy array
    # - quantile: quantile of values to discard, a float between 0 and 1
    #   (or between 0 and 0.5 for side='both')
    # - side: either 'both', 'down' or 'up'
    #   for 'up', the cut will discard the quantile highest values,
    #   for 'down', cut will discard the quantile lowest values,
    #   for 'both', the cut(s) will discard the quantile values both at the high and low end.
    # returns:
    # - a tuple of shape (lower cut, upper cut), with None entries if not applicable
    ucut = None
    dcut = None
    if( side=='up' or side=='both' ):
        ucut = np.quantile( values, 1-quantile )
    if( side=='down' or side=='both' ):
        dcut = np.quantile( values, quantile )
    return (dcut,ucut)
        
class HyperRectangleFitter(CloudFitter):
    ### Simple fitter making a hard cut in each dimension
    
    def __init__(self):
        ### empty constructor
        super( HyperRectangleFitter, self ).__init__()
        self.cuts = np.zeros()
                
    def fit(self, points, quantiles=0, side='both', verbose=False):
        ### fit to a set of points
        # input arguments:
        # - points: a np array of shape (npoints,ndims)
        # - quantiles: quantiles of values to discard.
        #   can either be a float between 0 and 1 (applied in all dimensions),
        #   or a list of such floats with same length as number of dimensions in points.
        #   (note: for side='both', quantiles above 0.5 will discard everything)
        # - side: either 'both', 'down' or 'up'
        #   for 'up', the cut will discard the quantile highest values,
        #   for 'down', cut will discard the quantile lowest values,
        #   for 'both', the cut(s) will discard the quantile values both at the high and low end.
        super( HyperRectangleFitter, self ).fit(points)
        # parse arguments
        sideoptions = ['both','up','down']
        if not side in sideoptions:
            raise Exception('ERROR in HyperRectanlgeFitter.__init__:'
                           +' keyword argument "side" is {}'.format(side)
                           +' while the allowed options are {}'.format(sideoptions))
        if isinstance(quantiles,float):
            quantiles = [quantiles]*self.ndims
        if len(quantiles)!=self.ndims:
            raise Exception('ERROR in HyperRectangleFitter.__init__:'
                           +' quantiles must have same length as number of dimensions,'
                           +' found {} and {} respectively'.format(len(quantiles),ndims))
        self.cuts = []
        for dim in range(self.ndims): 
            self.cuts.append( calculate_cut_values(points[:,dim], quantiles[dim], side=side) )
        if verbose:
            print('Constructed a HyperRectangleFitter with following cut values:')
            for dim in range(self.ndims): print('dimension {}: {}'.format(dim,self.cuts[dim]))
        
    def apply_cuts(self, point):
        ### apply the cuts to a point and return whether it passes them
        # input arguments:
        # - point: a 1D numpy array of shape (ndims,)
        # returns:
        # - boolean
        for cut,value in zip(self.cuts,point):
            if( cut[0] is not None and value<cut[0] ): return False
            if( cut[1] is not None and value>cut[1] ): return False
        return True
        
    def pdf(self, points):
        ### get pdf at points
        # note that the pdf is either 0 (does not pass cuts) or 1 (passes cuts)
        super( HyperRectangleFitter, self ).pdf(points)
        pdf = np.zeros(len(points))
        for i,point in enumerate(points):
            pdf[i] = self.apply_cuts( point )
        return pdf