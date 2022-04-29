#!/usr/bin/env python
# coding: utf-8

# **Class for fitting a gaussian kernel density to a PCA-reduced point cloud**
# 
# Extension of GaussianKdeFitter: instead of fitting the full point cloud,
# a PCA-based dimensionality reduction is first applied on it.
# This has the advantage that the fit can be visualised correctly (in case of 2 reduced dimensions),
# instead of only projections of it.
# The potential disadvantage is that the PCA reduction might distort the relative separations.


### imports

# external modules
import sys
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
import importlib

# local modules
from CloudFitter import CloudFitter
sys.path.append('../../utils')


class PCAGaussianKdeFitter(CloudFitter):
    ### class for fitting a gaussian kernel density to a PCA-reduced point cloud
    # basically a wrapper for sklean.decomposition.PCA + scipy.stats.gaussian_kde.
    # parameters
    # - pca: sklearn.decomposition.pca object
    # - kernel: scipy.stats.gaussian_kde object
    # - cov: covariance matrix 
    # (use np.cov for now, maybe later replace by internal kernel.covariance)
    
    def __init__(self):
        ### empty constructor
        super( PCAGaussianKdeFitter, self ).__init__()
        # first apply PCA
        self.npcadims = 0
        self.pca = None
        self.cov = np.zeros(0)
        self.kernel = None
        
    def fit(self, points, npcadims=2, bw_method='scott', bw_scott_factor=None):
        ### fit to a set of points
        # input arguments:
        # - points: a np array of shape (npoints,ndims)
        # - npcadims: number of PCA compoments to keep
        # - bw_method: method to calculate the bandwidth of the gaussians,
        #   see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
        # - bw_scott_factor: additional multiplication factor applied to bandwidth in case it is set to 'scott'
        super( PCAGaussianKdeFitter, self ).fit(points)
        # first apply PCA
        self.npcadims = npcadims # extension to self.ndims (dimension of non-transformed points)
        self.pca = PCA(n_components=npcadims)
        pcapoints = self.pca.fit_transform(points)
        # then apply gaussian kernel density estimation
        self.cov = np.cov(pcapoints, rowvar=False)
        if( bw_method=='scott' and bw_scott_factor is not None ):
            scott_bw = self.npoints**(-1./(self.npcadims+4))
            bw_method = bw_scott_factor*scott_bw
        self.kernel = gaussian_kde(np.transpose(pcapoints), bw_method=bw_method)
        
    def pdf(self, points):
        ### get pdf at points
        # note: points can be both of shape (npoints,ndims) or of shape (npoints,npcadims);
        #       in the latter case it is assumed that the points are already PCA-transformed,
        #       and only the gaussian kernel density is applied on them.
        if not isinstance( points, np.ndarray ):
            raise Exception('ERROR in PCAGaussianKdeFitter.pdf:'
                            +' points must be a numpy array but found type {}'.format(type(points)))
        if len(points.shape)!=2:
            raise Exception('ERROR in PCAGaussianKdeFitter.pdf:'
                            +' points must be a 2D numpy array but found shape {}'.format(points.shape))
        if( points.shape[1]!=self.ndims and points.shape[1]!=self.npcadims ):
            raise Exception('ERROR in PCAGaussianKdeFitter.pdf:'
                            +' points must have either {} or {} dimensions'.format(self.ndims, self.npcadims)
                            +' but found {}'.format(points.shape[1]))
        # first apply PCA
        if points.shape[1]==self.ndims: pcapoints = self.pca.transform(points)
        else: pcapoints = points
        # then apply gaussian kernel density
        return self.kernel.pdf(np.transpose(pcapoints))
    
    def transform(self, points):
        ### perform PCA transformation
        return self.pca.transform(points)