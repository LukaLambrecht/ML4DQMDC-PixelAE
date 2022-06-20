#!/usr/bin/env python
# coding: utf-8

# **ModelInterface: extension of Model class interfaced by HistStruct**  
# 
# This class is the interface between a Model (holding classifiers and fitters)  
# and a HistStruct (holding histogram data).  
# It stores the classifier and model scores for the histograms in a HistStruct.  

### imports

# external modules
import os
import sys
import pickle
import zipfile
import glob
import copy
import numpy as np
import importlib

# local modules
import Model
importlib.reload(Model)
from Model import Model
sys.path.append('classifiers')
from HistogramClassifier import HistogramClassifier


class ModelInterface(Model):
    
    def __init__( self, histnames ):
        ### initializer
        # input arguments:
        # - histnames: list of the histogram names for this Model(Interface).
        super( ModelInterface, self ).__init__(histnames)
        # (setting attributes histnames, ndims, classifiers, and fitter)
        self.scores = {}
        self.globalscores = {}
        self.setnames = []
        self.default_set_name = 'default'
        self.add_setname( self.default_set_name )
        # extra properties for plotting purposes
        self.partial_fitters = {}
        self.fitscores_array = None
        
    def __str__( self ):
        ### get a printable representation of a ModelInterface
        info = '=== ModelInterface === \n'
        if len(self.histnames)==0:
            info += '  (not initialized)'
        else:
            # classifiers
            info += '- classifiers: \n'
            for histname in self.histnames:
                info += '    -- {}: '.format(histname)
                if histname in self.classifiers.keys():
                    info += '{} \n'.format(type(self.classifiers[histname]))
                else: info += '(not initialized) \n'
            # fitter
            info += '- fitter: '
            if self.fitter is None: info += '(not initialized) \n'
            else: info += '{} \n'.format(type(self.fitter))
            # scores
            for setname in self.setnames:
                info += '- set "{}" \n'.format(setname)
                for histname in self.histnames:
                    info += '  -- scores for {}: '.format(histname)
                    if self.check_scores( setnames=[setname], histnames=[histname] ): info += 'initialized \n'
                    else: info += '(not initialized) \n'
                info += '  -- global scores: '
                if self.check_globalscores( setnames=[setname] ): info += 'initialized \n'
                else: info += '(not initialized) \n'
        return info
        
    def add_setname( self, setname ):
        ### initialize empty scores for extended set
        # input arguments:
        # - setname: name of extended set
        if setname in self.setnames:
            print('WARNING in ModelInterface.add_setname: set "{}"'.format(setname)
                  +' is already present in the ModelInterface')
            return
        self.setnames.append( setname )
        self.scores[setname] = {}
        self.globalscores[setname] = []
        
    def check_setname( self, setname ):
        ### check if a setname is present
        # input arguments:
        # - setname: name of the set to check
        if setname in self.setnames: return True
        return False
    
    def check_setnames( self, setnames ):
        ### check if all names in a list of set names are present
        for setname in setnames: 
            if not self.check_setname(setname): return False
        return True
    
    def check_scores( self, histnames=None, setnames=None ):
        ### check if scores are present for a given set name
        # input arguments:
        # - histnames: list of histogram names for which to check the scores (default: all)
        # - setname: list of set names for which to check the scores (default: standard set)
        if histnames is None: histnames = self.histnames
        if setnames is None: setnames = [self.default_set_name]
        for setname in setnames: 
            if not self.check_setname( setname ): return False
            for histname in histnames:
                if not histname in self.scores[setname].keys(): return False
        return True
    
    def check_globalscores( self, setnames=None ):
        ### check if global scores are present for a given set name
        # input arguments:
        # - setname: list of set names for which to check the scores (default: standard set)
        if setnames is None: setnames = [self.default_set_name]
        for setname in setnames:
            if not self.check_setname( setname ): return False
            if len(self.globalscores[setname])==0: return False
        return True
    
    def evaluate_store_classifier( self, histname, histograms, mask=None, setname=None ):
        ### same as Model.evaluate_classifier but store the result internally
        # input arguments:
        # - histname: histogram name for which to evaluate the classifier
        # - histograms: the histograms for evaluation, np array of shape (nhistograms,nbins)
        # - mask: a np boolean array masking the histograms to be evaluated
        # - setname: name of extended set (default: standard set)
        if setname is None: setname = self.default_set_name
        if not self.check_setname( setname ):
            raise Exception('ERROR in ModelInterface.evaluate_store_classifier:'
                           +' requested to store the classifier results in set {}'.format(setname)
                           +' but it was not yet initialized.')
        scores = super( ModelInterface, self ).evaluate_classifier(histname, histograms, mask=mask)
        self.scores[setname][histname] = scores
        
    def evaluate_store_classifiers( self, histograms, mask=None, setname=None ):
        ### same as Model.evaluate_classifiers but store the result internally
        # input arguments:
        # - histograms: dict of histnames to histogram arrays (shape (nhistograms,nbins))
        # - mask: a np boolean array masking the histograms to be evaluated
        # - setname: name of extended set (default: standard set)
        if setname is None: setname = self.default_set_name
        if not self.check_setname( setname ):
            raise Exception('ERROR in ModelInterface.evaluate_store_classifiers:'
                           +' requested to store the classifier results in set {}'.format(setname)
                           +' but it was not yet initialized.')
        scores = super( ModelInterface, self ).evaluate_classifiers(histograms, mask=mask)
        self.scores[setname] = scores
        
    def evaluate_store_fitter( self, points, mask=None, setname=None, verbose=False ):
        ### same as Model.evaluate_fitter but store the result internally
        # input arguments:
        # - points: dict matching histnames to scores (np array of shape (nhistograms))
        # - mask: a np boolean array masking the histograms to be evaluated
        # - setname: name of extended set (default: standard set)
        if setname is None: setname = self.default_set_name
        if not self.check_setname( setname ):
            raise Exception('ERROR in ModelInterface.evaluate_store_fitter:'
                           +' requested to store the fitter results in set {}'.format(setname)
                           +' but it was not yet initialized.')
        scores = super( ModelInterface, self ).evaluate_fitter(points, mask=mask, verbose=verbose)
        self.globalscores[setname] = scores
        
    def get_scores( self, setnames=None, histname=None ):
        ### get the scores stored internally
        # input arguments:
        # - setnames: list of names of extended sets (default: standard set)
        # - histname: name of histogram type for which to get the scores
        #   if specified, an array of scores is returned.
        #   if not, a dict matching histnames to arrays of scores is returned.
        if setnames is None: setnames = [self.default_set_name]
        histnames = self.histnames[:]
        if histname is not None: histnames = [histname]
        if not self.check_setnames( setnames ):
            raise Exception('ERROR in ModelInterface.get_scores: requested scores for setnames {}'.format(setnames)
                           +' but not all of them are in the current ModelInterface.')
        if not self.check_scores( histnames=histnames, setnames=setnames ):
            raise Exception('ERROR in ModelInterface.get_scores: requested scores for histname {}'.format(histname)
                           +' but it is not defined for all of the requested sets.')
        res = {}
        for hname in histnames:
            scores = []
            for setname in setnames: scores.append(self.scores[setname][hname])
            res[hname] = np.concatenate( tuple(scores), axis=0 )
        # return the result in correct format
        if histname is None: return res
        return res[histname]
        
    def get_globalscores( self, setnames=None ):
        ### get the global scores stored internally
        # input arguments:
        # - setnames: list of name of extended sets (default: standard set)
        if setnames is None: setnames = [self.default_set_name]
        if not self.check_setnames( setnames ):
            raise Exception('ERROR in ModelInterface.get_globalscores: requested setnames {}'.format(setnames)
                           +' but not all of them are in the current ModelInterface.')
        globalscores = []
        for setname in setnames: globalscores.append(self.globalscores[setname])
        globalscores = np.concatenate( tuple(globalscores), axis=0 )
        return globalscores
    
    def get_globalscores_mask( self, setnames=None, score_up=None, score_down=None ):
        ### get a mask of global scores within boundaries
        # input arguments:
        # - setnames: list of name of extended sets (default: standard set)
        # - score_up and score_down are upper and lower thresholds
        #     if both are not None, the mask for global scores between the boundaries are returned
        #     if score_up is None, the mask for global score > score_down are returned
        #     if score_down is None, the mask for global score < score_up are returned
        if( score_up is None and score_down is None ):
            raise Exception('ERROR in ModelInterface.get_globalscores_mask:'
                           +' you must specify either score_up or score_down.')
        scores = self.get_globalscores( setnames=setnames )
        if score_down is None: return (scores < score_up).astype(bool)
        elif score_up is None: return (scores > score_down).astype(bool)
        else: return ((scores>score_down) & (scores<score_up)).astype(bool)
    
    def get_globalscores_indices( self, setnames=None, score_up=None, score_down=None ):
        ### get the indices of global scores within boundaries
        # input arguments:
        # - setnames: list of name of extended sets (default: standard set)
        # - score_up and score_down are upper and lower thresholds
        #     if both are not None, the indices with global scores between the boundaries are returned
        #     if score_up is None, the indices with global score > score_down are returned
        #     if score_down is None, the indices with global score < score_up are returned
        mask = self.get_globalscores_mask( setnames=setnames, score_up=score_up, score_down=score_down )
        return np.nonzero( mask )[0]
        
    def train_partial_fitters( self, dimslist, points, **kwargs ):
        ### train partial fitters on a given set of dimensions
        # input arguments:
        # - dimslist: list of tuples with integer dimension numbers
        # - points: dict matching histnames to scores (np array of shape (nhistograms))
        # - kwargs: additional keyword arguments for fitting
        self.partial_fitters = {}
        self.fitscores_array = self.get_point_array(points)
        # loop over all combinations of dimensions
        for dims in dimslist:
            # make the partial fit and store it
            scores = self.fitscores_array[:,dims]
            if len(scores.shape)==1: scores = np.expand_dims(scores, 1)
            fitter = self.fitter_type()
            fitter.fit( scores, **kwargs )
            self.partial_fitters[dims] = fitter
        
    def save( self, path, save_classifiers=True, save_fitter=True ):
        ### save a ModelInterface object to a pkl file
        # input arguments:
        # - path where to store the file
        # - save_classifiers: a boolean whether to include the classifiers (alternative: only scores)
        # - save_fitter: a boolean whether to include the fitter (alternative: only scores)
        pklpath = os.path.splitext(path)[0]+'.pkl'
        zippath = os.path.splitext(path)[0]+'.zip'
        cpath = os.path.splitext(path)[0]+'_classifiers_storage'
        fpath = os.path.splitext(path)[0]+'_fitter_storage'
        rootpath = os.path.dirname(path)
        zipcontents = {}        
        # remove the classifiers and fitter from the object
        classifiers = dict(self.classifiers)
        self.classifiers = {}
        fitter = copy.deepcopy(self.fitter)
        self.fitter = None
        partial_fitters = dict(self.partial_fitters)
        self.partial_fitters = {}
        # pickle the rest
        with open(pklpath,'wb') as f:
            pickle.dump(self,f)
        zipcontents[pklpath] = os.path.relpath(pklpath, start=rootpath)
        # restore the classifiers and fitter
        self.classifiers = classifiers
        self.fitter = fitter
        self.partial_fitters = partial_fitters
        # case where classifiers should be stored
        if( len(self.classifiers.keys())!=0 and save_classifiers ):
            # save the classifiers
            for histname,classifier in self.classifiers.items():
                classifier.save( os.path.join(cpath,histname) )
            # get all files to store in the zip file
            for root, dirs, files in os.walk(cpath):
                for name in files:
                    thispath = os.path.join(root, name)
                    zipcontents[thispath] = os.path.relpath(thispath, start=rootpath)
        # case where fitter should be stored
        if( self.fitter is not None and save_fitter ):
            # save the fitter
            self.fitter.save( os.path.join(fpath,'fitter') )
            # get all files to store in the zip file
            for root, dirs, files in os.walk(fpath):
                for name in files:
                    thispath = os.path.join(root, name)
                    zipcontents[thispath] = os.path.relpath(thispath, start=rootpath)
        # put everything in a zip file
        with zipfile.ZipFile( zippath, 'w' ) as zipf:
            for f, fname in zipcontents.items(): zipf.write(f, fname)
        # remove individual files
        for f in zipcontents: os.system('rm {}'.format(f))
        if os.path.exists(cpath): os.system('rm -r {}'.format(cpath))
        if os.path.exists(fpath): os.system('rm -r {}'.format(fpath))
            
    @classmethod
    def load( self, path, load_classifiers=True, load_fitter=True, verbose=False ):
        ### load a ModelInterface object
        # input arguments:
        # - path to a zip file containing a ModelInterface object
        # - load_classifiers: a boolean whether to load the classifiers if present
        # - load_fitter: a boolean whether to load the fitter if present
        # - verbose: boolean whether to print some information
        zippath = os.path.splitext(path)[0]+'.zip'
        unzippath = os.path.splitext(path)[0]+'_unzipped'
        basename = os.path.splitext(os.path.basename(zippath))[0]
        pklbasename = basename+'.pkl'
        cbasename = basename+'_classifiers_storage'
        fbasename = basename+'_fitter_storage'
        zipcontents = []
        # extract the zip file
        with zipfile.ZipFile( zippath, 'r' ) as zipf:
            zipcontents = zipf.namelist()
            zipf.extractall( path=unzippath )
        with open(os.path.join(unzippath,pklbasename),'rb') as f:
            obj = pickle.load(f)
        if( load_classifiers ):
            if len(zipcontents)==1:
                print('WARNING: requested to load classifiers, '
                      +'but this stored ModelInterface object does not seem to contain any.')
            else:
                for histname in obj.histnames:
                    obj.classifiers[histname] = obj.classifier_types[histname].load( 
                        os.path.join(unzippath,cbasename,histname) )
        if( load_fitter ):
            if not os.path.exists(os.path.join(unzippath,fbasename)):
                print('WARNING: requested to load fitter, '
                      +'but this stored ModelInterface object does not seem to contain any.')
            else:
                obj.fitter = obj.fitter_type.load( os.path.join(unzippath,fbasename,'fitter') )
        # remove individual files
        if os.path.exists(unzippath): os.system('rm -r {}'.format(unzippath))
        if verbose:
            print('Loaded a ModelInterface object with following properties:')
            print(obj)
        return obj