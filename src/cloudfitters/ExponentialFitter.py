#!/usr/bin/env python
# coding: utf-8

# **Class for fitting an exponential distribution to a point cloud**
# 
# An exponential distribution in N dimensions is fully determined by an N-dimensional vector, representing the N-dimensional decay parameter (or lambda parameter) of the distribution. 



### imports

# external modules
import sys
import numpy as np
import importlib

# local modules
from CloudFitter import CloudFitter
sys.path.append('../../utils')




class ExponentialFitter(CloudFitter):
    ### class for fitting an exponential distribution to a point cloud
    # parameters
    # - l: multidimensional lambda parameter of exponential
    
    def __init__(self):
        ### empty constructor
        # input arguments:
        # - points: a np array of shape (npoints,ndims)
        super( ExponentialFitter, self ).__init__()
        self.l = np.zeros(0)
        
    def fit(self, points):
        ### fit to a set of points
        # input arguments:
        # - points: a np array of shape (npoints,ndims)
        super( ExponentialFitter, self ).fit(points)
        # for now use mean for beta, maybe change later!
        self.l = np.reciprocal(np.mean(points,axis=0))
        
    def pdf(self, points):
        ### get pdf at points
        super( ExponentialFitter, self ).pdf(points)
        temp = np.repeat(np.transpose(np.expand_dims(self.l,axis=1)),len(points),axis=0)
        temp = np.multiply(temp,points)
        temp = np.sum(temp,axis=1)
        return np.prod(self.l)*np.exp(-temp)





