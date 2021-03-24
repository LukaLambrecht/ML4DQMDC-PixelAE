#!/usr/bin/env python
# coding: utf-8



### imports

# external modules
import os
import sys
import pickle
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib

# local modules
sys.path.append('classifiers')
from HistogramClassifier import HistogramClassifier
sys.path.append('../utils')
import dataframe_utils as dfu
import hist_utils as hu
import json_utils as jsonu
import plot_utils as pu




class HistStruct(object):
    ### main data structure used within this framework
    # a HistStruct object basically consists of a mutually consistent collection of numpy arrays,
    # where each numpy array corresponds to one histogram type, with dimensions (number of histograms, number of bins).
    # the HistStruct has functions to easily perform the following common tasks (among others):
    # - select a subset of runs and/or lumisections (e.g. using a json file formatted selector),
    # - prepare the data for machine learning training
    # - evaluate classifiers (machine learning types or other)
    
    def __init__( self ):
        ### empty initializer, setting all containers to empty defaults
        self.histnames = [] # list of histogram names
        self.histograms = {} # dict mapping histogram name to 2D numpy array of histograms (shape (nhists,nbins))
        self.nentries = {} # dict mapping histogram name to 1D numpy array of number of entries per histogram (same length as histograms)
        self.runnbs = [] # 1D numpy array of run numbers (same length as histograms)
        self.lsnbs = [] # 1D numpy array of lumisection numbers (same length as histograms)
        self.classifiers = {} # dict mapping histogram name to object of type HistogramClassifier
        self.scores = {} # dict mapping histogram name to 1D numpy array of values associated to the histograms (same length as histograms)
        self.masks = {} # dict mapping name to 1D numpy array of booleans (same length as histograms) that can be used for masking
        self.exthistograms = {} # dict similar to histograms for additional (e.g. artificially generated) histograms
        
    def save( self, path ):
        ### save a HistStruct object to a pkl file
        path = os.path.splitext(path)[0]+'.pkl'
        with open(path,'wb') as f:
            pickle.dump(self,f)
            
    @classmethod
    def load( self, path ):
        with open(path,'rb') as f:
            obj = pickle.load(f)
        return obj
        
    def add_dataframe( self, df, donormalize=True, rebinningfactor=1 ):
        ### add a dataframe to a HistStruct
        # input arguments:
        # - df is a pandas dataframe as read from the input csv files.
        # - donormalize: boolean whether to normalize the histograms
        # - rebinningfactor: factor by which to group bins together
        # notes:
        # - the new dataframe can contain one or more histogram types
        # - the new dataframe must contain the same run and lumisection numbers (for each histogram type in it)
        #   as already present in the HistStruct, except if it is the first one to be added
        
        histnames = dfu.get_histnames(df)
        # loop over all names in the dataframe
        for histname in histnames:
            if histname in self.histnames:
                raise Exception('ERROR in HistStruct.add_dataframe: dataframe contains histogram name '.format(histname)
                               +' but this is already present in the current HistStruct.')
            thisdf = dfu.select_histnames( df, [histname] )
            # determine statistics (must be done before normalizing)
            nentries = np.array(thisdf['entries'])
            # prepare the data
            (hists_all,runnbs_all,lsnbs_all) = hu.preparedatafromdf(thisdf,returnrunls=True,
                                                                    donormalize=donormalize,
                                                                    rebinningfactor=rebinningfactor)
            runnbs_all = runnbs_all.astype(int)
            lsnbs_all = lsnbs_all.astype(int)
            # check consistency in run and lumisection numbers
            if len(self.histnames)!=0:
                if( not ( (runnbs_all==self.runnbs).all() and (lsnbs_all==self.lsnbs).all() ) ):
                    raise Exception('ERROR in HistStruct.add_dataframe: dataframe run/lumi numbers are not consistent!')
            # add everything to the structure
            self.histnames.append(histname)
            self.histograms[histname] = hists_all
            self.nentries[histname] = nentries
            self.runnbs = runnbs_all
            self.lsnbs = lsnbs_all
    
    def add_mask( self, name, mask ):
        ### add a mask to a HistStruct
        # input arguments:
        # - name: a name for the mask
        # - mask: a 1D np array of booleans  with same length as number of lumisections in HistStruct
        if name in self.masks.keys():
            raise Exception('ERROR in HistStruct.add_mask: name {} already exists!'.format(name))
        if( len(mask)!=len(self.runnbs) ):
            raise Exception('ERROR in HistStruct.add_mask: mask has length {}'.format(len(mask))
                           +' while HistStruct contains {} lumisections.'.format(len(self.runnbs)))
        self.masks[name] = mask.astype(bool)
            
    def remove_mask( self, name ):
        ### inverse operation of add_mask
        if name not in self.masks.keys():
            print('WARNING in HistStruct.remove_mask: name {} is not in list of masks...'.format(name))
            return
        self.masks.pop( name )
      
    def add_json_mask( self, name, jsondict ):
        ### add a mask corresponding to a json dict
        mask = jsonu.injson( self.runnbs, self.lsnbs, jsondict=jsondict )
        self.add_mask( name, mask )
    
    def add_goldenjson_mask( self, name ):
        ### add a mask corresponding to the golden json file
        mask = jsonu.isgolden( self.runnbs, self.lsnbs )
        self.add_mask( name, mask )
        
    def add_dcsonjson_mask( self, name ):
        ### add a mask corresponding to the DCS-bit on json file
        mask = jsonu.isdcson( self.runnbs, self.lsnbs )
        self.add_mask( name, mask )
        
    def add_hightstat_mask( self, name, histnames=None, entries_to_bins_ratio=100 ):
        ### add a mask corresponding to lumisections where all histograms have sufficient statistics
        # input arguments:
        # - histnames: list of histogram names to take into account for making the mask (default: all in the HistStruct)
        # - entries_to_bins_ratio: criterion to determine if a histogram has sufficient statistics, number of entries divided by number of bins
        if histnames is None:
            histnames = self.histnames
        mask = np.ones(len(self.runnbs)).astype(bool)
        for histname in histnames:
            if histname not in self.histnames:
                raise Exception('ERROR in HistStruct.add_highstat_mask: requested to take into account {}'.format(histname)
                               +' but no such histogram type exists in the HistStruct.')
            nbins = self.histograms[histname].shape[1]
            mask = mask & (self.nentries[histname]/nbins>entries_to_bins_ratio)
        self.add_mask( name, mask )
        
    def get_combined_mask( self, names ):
        ### get a combined mask given multiple mask names
        # mostly for internal use; externally you can use get_histograms( histname, <list of mask names>) directly
        mask = np.ones(len(self.runnbs)).astype(bool)
        for name in names:
            if name not in self.masks.keys():
                raise Exception('ERROR in HistStruct.get_combined_mask: mask {} requested but not found.'.format(name))
            mask = mask & self.masks[name]
        return mask
        
    def get_runnbs( self, masknames=None ):
        ### get the array of run numbers, optionally after masking
        if masknames is None: return self.runnbs[:]
        return self.runnbs[ self.get_combined_mask(masknames) ]
    
    def get_lsnbs( self, masknames=None ):
        ### get the array of lumisection numbers, optionally after masking
        if masknames is None: return self.lsnbs[:]
        return self.lsnbs[ self.get_combined_mask(masknames) ]
    
    def get_scores( self, histname=None, masknames=None ):
        ### get the array of scores for a given histogram type, optionally after masking
        # if histname is None, return a dict matching histnames to arrays of scores
        histnames = self.histnames[:]
        if histname is not None:
            histnames = [histname]
        mask = np.ones(len(self.lsnbs)).astype(bool)
        if masknames is not None:
            mask = self.get_combined_mask(masknames)
        res = {}
        for hname in histnames:
            res[hname] = self.scores[hname][mask]
        if histname is None: return res
        return res[histname]
    
    def get_histograms( self, histname=None, masknames=None ):
        ### get the array of histograms for a given type, optionally after masking
        # if histname is None, return a dict matching histnames to arrays of histograms
        histnames = self.histnames[:]
        if histname is not None:
            histnames = [histname]
        mask = np.ones(len(self.lsnbs)).astype(bool)
        if masknames is not None:
            mask = self.get_combined_mask(masknames)
        res = {}
        for hname in histnames:
            res[hname] = self.histograms[hname][mask]
        if histname is None: return res
        return res[histname]
    
    def add_classifier( self, histname, classifier, evaluate=False ):
        ### add a histogram classifier for a given histogram name to the HistStruct
        # classifier must be an object of type HistogramClassifier (i.e. of any class that derives from it)
        # evaluate is a bool whether to evaluate the classifier (and store the result in the 'scores' attribute)
        if not histname in self.histnames:
            raise Exception('ERROR in HistStruct.add_classifier: requested to add classifier for {}'.format(histname)
                           +' but there is not entry in the HistStruct with that name.')
        if not isinstance(classifier,HistogramClassifier):
            raise Exception('ERROR in HistStruct.add_classifier: classifier is of type {}'.format(type(classifier))
                           +' while a HistogramClassifier object is expected.')
        self.classifiers[histname] = classifier
        if evaluate:
            return self.evaluate_classifier( histname )
        
    def evaluate_classifier( self, histname ):
        ### evaluate a histogram classifier for a given histogram name in the HistStruct
        # the result is both returned and stored in the 'scores' attribute
        if not histname in self.classifiers.keys():
            raise Exception('ERROR in HistStruct.evaluate_classifier: requested to evaluate classifier for {}'.format(histname)
                           +' but it is not found in the HistStruct.')
        scores = self.classifiers[histname].evaluate(self.histograms[histname])
        self.scores[histname] = scores
        return scores
    
    def plot_ls( self, run, ls, refhists, doprint=False):
        ### plot the histograms for a given run/ls number versus their references and/or their reconstruction
        nhisttypes = len(self.histnames)
        ncols = 4
        nrows = int(math.ceil(nhisttypes/ncols))
        fig,axs = plt.subplots(nrows,ncols,figsize=(24,12))
        # find index that given run and ls number correspond to
        index = (set(list(np.where(self.runnbs==run)[0])) & set(list(np.where(self.lsnbs==ls)[0])))
        if len(index)!=1: 
            raise Exception('ERROR in HistStruct.plot_lsreco: index has unexpected shape: {}.'.format(index)
                           +' Is the requested run/lumisection in the HistStruct?')
        (index,) = index
        scores = []
        # loop over all histograms belonging to this lumisection and make the plots
        for j,name in enumerate(self.histnames):
            hist = self.histograms[name][index:index+1,:]
            score = self.classifiers[name].evaluate(hist)
            scores.append(score[0])
            reco = self.classifiers[name].model.predict(hist) # need to generalize to non-autoencoder classifiers
            pu.plot_sets([refhists[name],hist,reco],
                  fig=fig,ax=axs[int(j/ncols),j%ncols],
                  title=name,
                  colorlist=['blue','black','red'],labellist=['reference hists','hist (run: '+str(int(run))+', ls: '+str(int(ls))+')','reco'],
                  transparencylist=[0.3,1.,1.])
            # additional prints
            if doprint:
                print('mse (this histogram): '+str(score[0]))
                print('mse (average reference): '+str(np.average(self.classifiers[name].evaluate( refhists[name]))))
        return {'scorepoint':scores,'figure':fig}

    def plot_run( self, run, refhists, doprint=False):
        ### call plot_ls for all lumisections in a given run
        lsnbs = self.lsnbs[np.where(histstruct.runnbs==run)]
        print('plotting {} lumisections...'.format(len(lsnbs)))
        for lsnb in lsnbs:
            _ = self.plotlsreco(run,lsnb,refhists,doprint=doprint)





