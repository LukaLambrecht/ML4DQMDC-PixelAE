#!/usr/bin/env python
# coding: utf-8



### imports

# external modules
import sys
import numpy as np
from scipy.stats import gaussian_kde
import importlib

# local modules
from CloudFitter import CloudFitter
sys.path.append('../../utils')




class GaussianKdeFitter(CloudFitter):
    ### class for fitting a gaussian kernel density to a point cloud
    # basically a wrapper for scipy.stats.gaussian_kde.
    # parameters
    # - kernel: scipy.stats.gaussian_kde object
    # - cov: covariance matrix 
    # (use np.cov for now, maybe later replace by internal kernel.covariance)
    
    def __init__(self,points,bw_method='scott'):
        ### constructor
        # input arguments:
        # - points: a np array of shape (npoints,ndims)
        # - bw_method: method to calculate the bandwidth of the gaussians,
        #   see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
        super( GaussianKdeFitter, self ).__init__(points)
        self.cov = np.cov(points,rowvar=False)
        self.kernel = gaussian_kde(np.transpose(points),bw_method=bw_method)
        
    def pdf(self,points):
        ### get pdf at points
        super( GaussianKdeFitter, self ).pdf(points)
        return self.kernel.pdf(np.transpose(points))





