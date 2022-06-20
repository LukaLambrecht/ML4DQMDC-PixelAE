#!/usr/bin/env python
# coding: utf-8

# **Class for fitting a multidimensional gaussian distribution to a PCA-reduced point cloud**
# 
# Instead of fitting the full (high-dimensional) point cloud,
# a PCA-based dimensionality reduction is first applied on it.
# This has the advantage that the fit can be visualised correctly (in case of 2 reduced dimensions),
# instead of only projections of it.
# The potential disadvantage is that the PCA reduction might distort the relative separations.


### imports

# external modules
import sys
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
import importlib

# local modules
from CloudFitter import CloudFitter
sys.path.append('../../utils')


class PCAGaussianFitter(CloudFitter):
    ### class for fitting a gaussian distribution to a PCA-reduced point cloud
    # parameters
    # - pca: sklearn.decomposition.pca object
    # - mean: multidim mean of normal distribution
    # - cov: multidim covariance matrix of normal distribution
    # - mvn: scipy.stats multivariate_normal object built from mean and cov
        
    def __init__(self):
        ### empty constructor
        # input arguments:
        super( PCAGaussianFitter, self ).__init__()
        self.npcadims = 0
        self.pca = None
        self.mean = np.zeros(0)
        self.cov = np.zeros(0)
        self.mvn = None
        
    def fit(self, points, npcadims=2):
        ### fit to a set of points
        # input arguments:
        # - points: a np array of shape (npoints,ndims)
        # - npcadims: number of PCA compoments to keep
        super( PCAGaussianFitter, self ).fit(points)
        # first apply PCA
        self.npcadims = npcadims # extension to self.ndims (dimension of non-transformed points)
        self.pca = PCA(n_components=npcadims)
        pcapoints = self.pca.fit_transform(points)
        # then fit a normal distribution
        self.mean = np.mean(pcapoints, axis=0)
        self.cov = np.cov(pcapoints, rowvar=False)
        self.mvn = multivariate_normal(self.mean,self.cov)
        
    def pdf(self, points):
        ### get pdf at points
        # note: points can be both of shape (npoints,ndims) or of shape (npoints,npcadims);
        #       in the latter case it is assumed that the points are already PCA-transformed,
        #       and only the gaussian kernel density is applied on them.
        if not isinstance( points, np.ndarray ):
            raise Exception('ERROR in PCAGaussianFitter.pdf:'
                            +' points must be a numpy array but found type {}'.format(type(points)))
        if len(points.shape)!=2:
            raise Exception('ERROR in PCAGaussianFitter.pdf:'
                            +' points must be a 2D numpy array but found shape {}'.format(points.shape))
        if( points.shape[1]!=self.ndims and points.shape[1]!=self.npcadims ):
            raise Exception('ERROR in PCAGaussianFitter.pdf:'
                            +' points must have either {} or {} dimensions'.format(self.ndims, self.npcadims)
                            +' but found {}'.format(points.shape[1]))
        # first apply PCA
        if points.shape[1]==self.ndims: pcapoints = self.pca.transform(points)
        else: pcapoints = points
        # then apply gaussian distribution
        return self.mvn.pdf(pcapoints)
    
    def transform(self, points):
        ### perform PCA transformation
        return self.pca.transform(points)