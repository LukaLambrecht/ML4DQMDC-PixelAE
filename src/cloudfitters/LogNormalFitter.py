#!/usr/bin/env python
# coding: utf-8

# **Class for fitting a log-normal distribution to a point cloud**
# 
# A log-normal distribution is constructed by fitting a normal distribution to the logarithm of the point coordinates.



### imports

# external modules
import sys
import numpy as np
from scipy.stats import multivariate_normal
import importlib

# local modules
from CloudFitter import CloudFitter
sys.path.append('../../utils')




class LogNormalFitter(CloudFitter):
    ### class for fitting a log-normal distribution to a point cloud
    # parameters:
    # - mean: multidim mean of underlying normal
    # - cov: multidim covariance matrix of underlying normal
    # - mvn: scipy.stats multivariate_normal object built from the mean and cov
    
    def __init__(self):
        ### empty constructor
        super( LogNormalFitter, self ).__init__()
        self.mean = np.zeros(0)
        self.cov = np.zeros(0)
        self.mvn = None
        
    def fit(self, points):
        ### fit to a set of points
        # input arguments:
        # - points: a np array of shape (npoints,ndims)
        super( LogNormalFitter, self ).fit(points)
        # transform the data from assumed log-normal to normal
        points_log = np.log(points)
        # fit a total multivariate normal distribution
        self.mean = np.mean(points_log,axis=0)
        self.cov = np.cov(points_log,rowvar=False)
        self.mvn = multivariate_normal(self.mean,self.cov)
        
    def pdf(self,points):
        ### get pdf at points
        super( LogNormalFitter, self ).pdf(points)
        return self.mvn.pdf(np.log(points))