#!/usr/bin/env python
# coding: utf-8

# **Histogram classifier based on a direct comparison with templates (i.e. reference histograms)**



### imports

# external modules
import sys
import numpy as np
import tensorflow
import importlib

# local modules
from HistogramClassifier import HistogramClassifier
sys.path.append('../../utils')




def mseTopN_templates( histograms, templates, n=-1 ):
    ### calculate the mse between each histogram in histograms and each histogram in templates
    # input arguments:
    # - histograms: 2D numpy array of shape (nhistograms, nbins)
    # - templates: 2D numpy array of shape (ntemplates,nbins)
    # - n: integer representing the number of (sorted) bin squared errors to take into account (default: all)
    # output:
    # 2D numpy array of shape (nhistograms,ntemplates) holding the mseTopN between each
    
    nhistograms,nbins = histograms.shape
    ntemplates,_ = templates.shape
    res = np.zeros( (nhistograms,ntemplates) )
    for i in range(ntemplates):
        temp = np.tile( templates[i,:], (nhistograms,1) )
        sqdiff = np.power(histograms-temp,2)
        sqdiff[:,::-1].sort()
        if n>0: sqdiff = sqdiff[:,:n]
        mean = np.mean(sqdiff,axis=-1)
        res[:,i] = mean
    return res

def mseTopN_min( histograms, templates, n=-1 ):
    ### calculate the mse betwee a histogram and each template and return the minimum
    # input arguments:
    # - histograms: 2D numpy array of shape (nhistograms, nbins)
    # - templates: 2D numpy array of shape (ntemplates,nbins)
    # - n: integer representing the number of (sorted) bin squared errors to take into account (default: all)
    # output:
    # 1D numpy array of shape (nhistograms) holding the minimum mseTopN for each histogram
    
    allmses = mseTopN_templates( histograms, templates, n=n )
    return np.amin( allmses, axis=-1 )

def mseTop10_min( histograms, templates ):
    ### special case of above with n=10
    return mseTopN_min( histograms,templates,n=10)

def mseTopN_avg( histograms, templates, n=-1 ):
    ### calculate the mse betwee a histogram and each template and return the average
    # input arguments:
    # - histograms: 2D numpy array of shape (nhistograms, nbins)
    # - templates: 2D numpy array of shape (ntemplates,nbins)
    # - n: integer representing the number of (sorted) bin squared errors to take into account (default: all)
    # output:
    # 1D numpy array of shape (nhistograms) holding the average mseTopN for each histogram
    
    allmses = mseTopN_templates( histograms, templates, n=n )
    return np.mean( allmses, axis=-1 )

def mseTop10_avg( histograms, templates ):
    ### special case of above with n=10
    return mseTopN_avg( histograms,templates,n=10)




class TemplateBasedClassifier(HistogramClassifier):
    ### histogram classifier based on a direct comparison with templates (i.e. reference histograms)
    
    def __init__( self, templates, comparemethod='minmse' ):
        ### initializer from a set of templates (reference histograms)
        # input arguments:
        # - templates: a 2D numpy array of shape (nhistograms,nbins)
        # - comparemethod: string representing the method by which to compare a histogram with a set of templates
        #   currently supported methods are:
        #   - minmse: minimum mean square error between histogram and all templates
        #   - avgmse: average mean square error between histogram and all templates
        
        if not isinstance(templates,np.ndarray):
            raise Exception('ERROR in TemplateBasedClassifier.__init__: templates must be a numpy array.')
        if not len(templates.shape)==2:
            raise Exception('ERROR in TemplateBasedClassifier.__init__: templates must have 2D shape')
        self.methods = ({'minmse':mseTopN_min,
                         'minmsetop10': mseTop10_min,
                         'avgmse':mseTopN_avg,
                         'avgmsetop10': mseTop10_avg })
        if not comparemethod in self.methods.keys():
            raise Exception('ERROR in TemplateBasedClassifier.__init__: comparemethod not recognized: {}'.format(comparemethod))
        self.templates = templates
        self.comparemethod = comparemethod
        
    def evaluate( self, histograms ):
        ### classification of a collection of histograms based on their deviation from templates
        return self.methods[self.comparemethod]( histograms, self.templates )










