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

# local modules
from CloudFitter import CloudFitter


class IdentityFitter(CloudFitter):
    ### class for propagating classifier output scores (e.g. MSE) to global lumisection score
    
    def __init__(self, points):
        ### constructor
        # input arguments:
        # - points: a numpy array of shape (npoints,ndims) 
        #           note that ndims is supposed to be 1, 
        #           else this type of classifier is not well defined.
        super( IdentityFitter, self ).__init__(points)
        if( self.ndims!=1 ):
            raise Exception('ERROR in IdentityFitter.__init__:'
                    +' dimension is found to be {}'.format(self.ndims)
                    +' while 1 is expected.')
        self.nanthreshold = 1e-12
        
    def pdf(self, points):
        ### get pdf at points
        super( IdentityFitter, self ).pdf(points)
        points = np.where( points<self.nanthreshold, self.nanthreshold, points )
        scores = np.reciprocal(points[:,0])
        scores = np.where( scores>=1./self.nanthreshold, np.nan, scores )
        return scores
