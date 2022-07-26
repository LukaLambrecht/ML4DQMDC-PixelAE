#!/usr/bin/env python
# coding: utf-8

# **Model: grouping classifiers for different histogram types**  
# 
# This class represents a general model for assigning a score to a lumisection.  
# It consists of two distinct parts:  
# 
# - a collection of classifiers acting on individual histogramgs (one for each type).  
# - a fitter to assign a probability density to the output scores obtained in the previous step.  
# The types of histograms, classifiers, and fitter can be freely chosen.  
#  
# This class does not contain the histograms or other data; it only contains the classifiers and fitter.  
# Use the derived class ModelInterface to make the bridge between a HistStruct (holding the data)  
# and a Model (holding the classifiers and fitter).  

### imports

# external modules
import os
import sys
import pickle
import numpy as np
import importlib

# local modules
sys.path.append('classifiers')
sys.path.append('cloudfitters')
from HistogramClassifier import HistogramClassifier
from CloudFitter import CloudFitter


class Model(object):
    
    def __init__( self, histnames ):
        ### initializer
        # input arguments:
        # - histnames: list of the histogram names for this Model.
        #   this is the only argument needed for initialization,
        #   use the functions set_classifiers and set_fitter 
        #   to set the classifiers and fitter respectively.
        self.histnames = list(histnames)
        self.ndims = len(self.histnames)
        self.classifiers = {}
        self.classifier_types = {}
        self.fitter = None
        self.fitter_type = None
        for histname in self.histnames:
            self.classifiers[histname] = None
            self.classifier_types[histname] = None
            
    def set_classifiers( self, classifiers ):
        ### set the classifiers for this Model
        # input arguments:
        # - classifiers: dict of histnames to classifiers.
        #   the histnames must match the ones used to initialize this Model,
        #   the classifiers must be a subtype of HistogramClassifier.
        checknames = sorted(list(classifiers.keys()))
        if not checknames==sorted(self.histnames):
            raise Exception('ERROR in Model.set_classifiers: provided histogram names do not match'
                           +' the ones used to initialize the Model.'
                           +' Found {} and {} respectively'.format(checknames,self.histnames))
        for histname in self.histnames:
            classifier = classifiers[histname]
            if not isinstance(classifier,HistogramClassifier):
                raise Exception('ERROR in Model.set_classifiers: classifier for histogram type {}'.format(histname)
                               +' is of type {} which is not a HistogramClassifier.'.format(type(classifier)))
            self.classifiers[histname] = classifier
            self.classifier_types[histname] = type(classifier)
            
    def set_fitter( self, fitter ):
        ### set the fitter for this Model
        # input arguments:
        # - fitter: an (untrained) object of type CloudFitter
        if not isinstance(fitter,CloudFitter):
            raise Exception('ERROR in Model.set_fitter: fitter object is of type {}'.format(type(fitter))
                           +' which is not a CloudFitter.')
        self.fitter = fitter
        self.fitter_type = type(fitter)
        
    def check_classifier( self, histname ):
        ### check if a classifier was initialized
        # input arguments:
        # - histname: type of histogram for which to check the classifier
        if histname in self.classifiers.keys(): return True
        return False
    
    def check_fitter( self ):
        ### check if a fitter was initialized
        if self.fitter is not None: return True
        return False
        
    def train_classifier( self, histname, histograms, **kwargs ):
        ### train a classifier
        # input arguments:
        # - histname: histogram name for which to train the classifier
        # - histograms: the histograms for training, np array of shape (nhistograms,nbins)
        # - kwargs: additional keyword arguments for training
        if not self.check_classifier(histname):
            raise Exception('ERROR in Model.train_classifier: requested to train classifier'
                            +' for histogram type {},'.format(histname)
                            +' but no classifier for this type was initialized.')
        self.classifiers[histname].train( histograms, **kwargs )
        
    def train_classifiers( self, histograms, **kwargs ):
        ### train classifiers for all histograms in this Model
        # input arguments:
        # - histograms: the histograms for training, dict of histnames to np arrays of shape (nhistograms,nbins)
        # - kwargs: additional keyword arguments for training
        checknames = sorted(list(histograms.keys()))
        if not checknames==sorted(self.histnames):
            raise Exception('ERROR in Model.train_classifiers: provided histogram names do not match'
                           +' the ones used to initialize the Model.'
                           +' Found {} and {} respectively'.format(checknames,self.histnames))
        for histname in self.histnames: self.train_classifier( histname, histograms[histname], **kwargs )
            
    def evaluate_classifier( self, histname, histograms, mask=None ):
        ### evaluate a classifier and return the score
        # input arguments:
        # - histname: histogram name for which to evaluate the classifier
        # - histograms: the histograms for evaluation, np array of shape (nhistograms,nbins)
        # - mask: a np boolean array masking the histograms to be evaluated
        # returns:
        # - a np array of shape (nhistograms) with the scores
        # note: masked-out indices are set to np.nan!
        if not self.check_classifier(histname):
            raise Exception('ERROR in Model.evaluate_classifier: requested to evaluate classifier'
                            +' for histogram type {},'.format(histname)
                            +' but no classifier for this type was initialized.')
        if mask is None: 
            res = self.classifiers[histname].evaluate( histograms )
        else:
            res = np.empty(len(histograms))
            res[:] = np.nan
            mask_inds = np.nonzero(mask)[0]
            histograms = histograms[mask_inds]
            scores = self.classifiers[histname].evaluate( histograms )
            res[mask_inds] = scores
        return res
        
    def evaluate_classifiers( self, histograms, mask=None ):
        ### evaluate the classifiers and return the scores
        # input arguments:
        # - histograms: dict of histnames to histogram arrays (shape (nhistograms,nbins))
        # - mask: a np boolean array masking the histograms to be evaluated
        # returns:
        # - dict of histnames to scores (shape (nhistograms))
        # note: masked-out indices are set to np.nan!
        scores = {}
        checknames = sorted(list(histograms.keys()))
        if not checknames==sorted(self.histnames):
            raise Exception('ERROR in Model.evaluate_classifiers: provided histogram names do not match'
                           +' the ones used to initialize the Model.'
                           +' Found {} and {} respectively'.format(checknames,self.histnames))
        for histname in self.histnames: 
            res = self.evaluate_classifier( histname, histograms[histname], mask=mask )
            scores[histname] = res
        return scores
    
    def get_point_array( self, points ):
        ### for internal use in train_fitter and evaluate_fitter
        # input arguments:
        # - points: dict matching histnames to scores (np array of shape (nhistograms))
        checknames = sorted(list(points.keys()))
        if not checknames==sorted(self.histnames):
            raise Exception('ERROR in Model.get_point_array: provided histogram names do not match'
                           +' the ones used to initialize the Model.'
                           +' Found {} and {} respectively'.format(checknames,self.histnames))
        nhistograms = len(points[self.histnames[0]])
        pointsnp = np.zeros((nhistograms,self.ndims))
        for i,histname in enumerate(self.histnames):
            pointsnp[:,i] = points[histname]
        return pointsnp
            
    def train_fitter( self, points, verbose=False, **kwargs ):
        ### train the fitter
        # input arguments:
        # - points: dict matching histnames to scores (np array of shape (nhistograms))
        # - kwargs: additional keyword arguments for fitting
        if not self.check_fitter():
            raise Exception('ERROR in Model.train_fitter: requested to train fitter,'
                            +' but no fitter was initialized.')
        pointsnp = self.get_point_array( points )
        self.fitter.fit(pointsnp, **kwargs)
        if verbose: print('INFO: trained fitter on a training set of shape {}'.format(pointsnp.shape))
        
    def evaluate_fitter( self, points, mask=None, verbose=False ):
        ### evaluate the fitter and return the scores
        # input arguments:
        # - points: dict matching histnames to scores (np array of shape (nhistograms))
        # - mask: a np boolean array masking the histograms to be evaluated
        # returns:
        # - a np array of shape (nhistograms) with the scores
        # note: masked-out indices are set to np.nan!
        if not self.check_fitter():
            raise Exception('ERROR in Model.evaluate_fitter: requested to train fitter,'
                            +' but no fitter was initialized.')
        pointsnp = self.get_point_array( points )
        if mask is None: res = self.fitter.pdf(pointsnp)
        else:
            res = np.empty(len(pointsnp))
            res[:] = np.nan
            mask_inds = np.nonzero(mask)[0]
            pointsnp = pointsnp[mask_inds]
            scores = self.fitter.pdf(pointsnp)
            res[mask_inds] = scores
        if verbose: print('INFO: evaluated fitter on a testing set of shape {}'.format(pointsnp.shape))
        return res
