#!/usr/bin/env python
# coding: utf-8



### imports

# external modules
import sys
import numpy as np
from scipy.stats import multivariate_normal
import importlib

# local modules
from CloudFitter import CloudFitter
sys.path.append('../../utils')




class SeminormalFitter(CloudFitter):
    ### class for fitting a 'seminormal' distribution to a point cloud
    # this is not strictly speaking a probability distribution,
    # only the first quadrant of the result of fitting a normal distribution
    # to the data + its mirror image wrt the origin.
    # parameters
    # - cov: multidim covariance matrix of normal distribution
    # - mvn: scipy.stats multivariate_normal object built from the cov
        
    def __init__(self,points):
        ### constructor
        # points is a np array of shape (npoints,ndims)
        # note: points can also be an array or list with length 0,
        #       in that case the object is initialized empty.
        #       use this followed by the 'load' method to load a previously saved fit!
        if len(points)==0: 
            self.ndims = 0
            self.npoints = 0
            self.cov = None
            self.mvn = None
            return
        super( SeminormalFitter, self ).__init__(points)
        points = np.vstack((points,-points))
        self.cov = np.cov(points,rowvar=False)
        self.mvn = multivariate_normal(np.zeros(self.ndims),self.cov)
        
    def pdf(self,points):
        ### get pdf at points
        super( SeminormalFitter, self ).pdf(points)
        return self.mvn.pdf(points)
    
    def save(self,path):
        ### save the covariance matrix as a .npy file specified by path
        np.save(path,self.cov)
        
    def load(self,path):
        ### load a covariance matrix from a .npy file specified by path and build the fit from it
        self.cov = np.load(path)
        self.ndims = len(self.cov)
        self.mvn = multivariate_normal(np.zeros(self.ndims),self.cov)





