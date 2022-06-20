#!/usr/bin/env python
# coding: utf-8

# **Class for fitting a 'seminormal' distribution to a point cloud**
# 
# This is not strictly speaking a probability distribution, only the first quadrant of the result of fitting a normal distribution to the data + its mirror image wrt the origin.  



### imports

# external modules
import sys
import os
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
        
    def __init__(self):
        ### empty constructor
        super( SeminormalFitter, self ).__init__()
        self.cov = np.zeros(0)
        self.mvn = np.zeros(0)
        
    def fit(self,points):
        ### make the fit
        # input arguments:
        # - points: a np array of shape (npoints,ndims)
        super( SeminormalFitter, self ).fit(points)
        points = np.vstack((points,-points))
        self.cov = np.cov(points,rowvar=False)
        self.mvn = multivariate_normal(np.zeros(self.ndims),self.cov)
        
    def pdf(self,points):
        ### get pdf at points
        # input arguments:
        # - points: a np array of shape (npoints,ndims)
        super( SeminormalFitter, self ).pdf(points)
        return self.mvn.pdf(points)
    
    def save(self,path):
        ### save the covariance matrix as a .npy file specified by path
        npypath = os.path.splitext(path)[0]+'.npy'
        dirname = os.path.dirname(npypath)
        if not os.path.exists(dirname): os.makedirs(dirname)
        np.save(npypath,self.cov)
        return npypath
    
    @classmethod
    def load(self,path):
        ### load a covariance matrix from a .npy file specified by path and build the fit from it
        npypath = os.path.splitext(path)[0]+'.npy'
        obj = SeminormalFitter()
        obj.cov = np.load(npypath)
        obj.ndims = len(obj.cov)
        if obj.ndims>0: obj.mvn = multivariate_normal(np.zeros(obj.ndims),obj.cov)
        return obj