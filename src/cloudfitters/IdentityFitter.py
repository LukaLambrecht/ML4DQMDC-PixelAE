#!/usr/bin/env python
# coding: utf-8

# **Class for using classifier scores directly as global scores** 

# Only intended for the case where only one histogram type is being considered,
# so no real fitting needs to be done.
# Note: the scores are not used as such, but are inverted.
#       this corresponds to the convention that good histograms have low scores (e.g. MSE)
#       and high probability density, whereas bad histograms have high scores 
#       and low probability density.
# Note: the inversion (see above) implies that the scores are assumed to be > 0.
#       this is true for most classifiers defined up to now, where the score is an MSE value.

### imports

# external modules
import numpy as np
import os

# local modules
from CloudFitter import CloudFitter


class IdentityFitter(CloudFitter):
    ### class for propagating classifier output scores (e.g. MSE) to global lumisection score
    
    def __init__(self):
        ### empty constructor
        super( IdentityFitter, self ).__init__()
        self.nanthreshold = 1e-12
        
    def fit(self, points):
        ### fit to a set of points
        # input arguments:
        # - points: a numpy array of shape (npoints,ndims) 
        #           note that ndims is supposed to be 1, 
        #           else this type of classifier is not well defined.
        super( IdentityFitter, self ).fit(points)
        if( self.ndims!=1 ):
            raise Exception('ERROR in IdentityFitter.__init__:'
                    +' dimension is found to be {}'.format(self.ndims)
                    +' while 1 is expected.')
        
    def pdf(self, points):
        ### get pdf at points
        super( IdentityFitter, self ).pdf(points)
        points = np.where( points<self.nanthreshold, self.nanthreshold, points )
        scores = np.reciprocal(points[:,0])
        scores = np.where( scores>=1./self.nanthreshold, np.nan, scores )
        return scores
    
    def save(self, path):
        ### save this fitter (dummy for now since nothing to be saved)
        txtpath = os.path.splitext(path)[0]+'.txt'
        dirname = os.path.dirname(txtpath)
        if not os.path.exists(dirname): os.makedirs(dirname)
        with open(txtpath, 'w') as f:
            f.write('dummy')
        return txtpath
    
    @classmethod
    def load(self, path):
        ### load this fitter (dummy for now since nothing to be loaded)
        obj = IdentityFitter()
        dummy_array = np.array([1,2,3])
        dummy_array = np.expand_dims(dummy_array,1)
        obj.fit(dummy_array)
        return obj