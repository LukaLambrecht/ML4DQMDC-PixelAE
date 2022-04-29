#!/usr/bin/env python
# coding: utf-8

# **Class for fitting a gaussian kernel density to a point cloud**
# 
# Basically a wrapper for scipy.stats.gaussian_kde.  
# A gaussian kernel density can be thought of as a sum of little (potentially multidimensional) gaussians, each one centered at one of the points in the cloud. Hence, the resulting distribution is a sort of smoothed version of the discrete point cloud.



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
    
    def __init__(self):
        ### empty constructor
        super( GaussianKdeFitter, self ).__init__()
        self.cov = np.zeros(0)
        self.kernel = None
        
    def fit(self, points, bw_method='scott', bw_scott_factor=None):
        ### fit to a set of points
        # input arguments:
        # - points: a np array of shape (npoints,ndims)
        # - bw_method: method to calculate the bandwidth of the gaussians,
        #   see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
        # - bw_scott_factor: additional multiplication factor applied to bandwidth in case it is set to 'scott'
        super( GaussianKdeFitter, self ).fit(points)
        self.cov = np.cov(points,rowvar=False)
        if( bw_method=='scott' and bw_scott_factor is not None ):
            scott_bw = self.npoints**(-1./(self.ndims+4))
            bw_method = bw_scott_factor*scott_bw
        self.kernel = gaussian_kde(np.transpose(points),bw_method=bw_method)
        
    def pdf(self,points):
        ### get pdf at points
        super( GaussianKdeFitter, self ).pdf(points)
        return self.kernel.pdf(np.transpose(points))