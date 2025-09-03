#!/usr/bin/env python
# coding: utf-8

# **HistStruct: consistent treatment of multiple histogram types**  
# 
# The HistStruct class is the main data structure used within this framework.  
# A HistStruct object basically consists of a mutually consistent collection of numpy arrays,  
# where each numpy array corresponds to one histogram type, with dimensions (number of histograms, number of bins).  
# The HistStruct has functions to easily perform the following common tasks (among others):  
# 
# - select a subset of runs and/or lumisections (e.g. using a custom or predefined json file formatted selector),  
# - prepare the data for machine learning training, with all kinds of preprocessing,  
# - evaluate classifiers (machine learning types or other),  
# - go from per-histogram scores to per-lumisection scores.  
#  
# When only processing a single histogram type, the HistStruct might be a bit of an overkill.  
# One could instead choose to operate on the dataframe directly.  
# However, especially when using multiple histogram types, the HistStruct is very handy to keep everything consistent.  



### imports

# external modules
import os
import sys
import pickle
import zipfile
import glob
import shutil
import copy
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib

# local modules
import ModelInterface
from ModelInterface import ModelInterface
sys.path.append('classifiers')
from HistogramClassifier import HistogramClassifier
sys.path.append('../utils')
import dataframe_utils as dfu
import hist_utils as hu
import json_utils as jsonu
import plot_utils as pu




class HistStruct(object):
    
    def __init__( self ):
        ### empty initializer, setting all containers to empty defaults
        # a HistStruct object has the following properties:
        # histnames: list of histogram names
        # histograms: dict mapping histogram name to 2D numpy array of histograms (shape (nhists,nbins))
        # nentries: dict mapping histogram name to 1D numpy array of number of entries per histogram (same length as histograms)
        # histranges: dict mapping histogram name to tuple with (xmin, xmax)
        # runnbs: 1D numpy array of run numbers (same length as histograms)
        # lsnbs: 1D numpy array of lumisection numbers (same length as histograms)
        # masks: dict mapping name to 1D numpy array of booleans (same length as histograms) that can be used for masking
        # exthistograms: dict of dicts similar to histograms for additional (e.g. artificially generated) histograms
        # setnames: list of names of extended sets
        # models: dict mapping model names to ModelInterfaces
        # modelnames: list of model names
        self.histnames = []
        self.histograms = {}
        self.nentries = {}
        self.histranges = {}
        self.runnbs = []
        self.lsnbs = []
        self.masks = {}
        self.exthistograms = {}
        self.setnames = []
        self.models = {}
        self.modelnames = []
        
    ##############################
    # functions for input/output #
    ##############################
        
    def __str__( self ):
        ### get a printable representation of a HistStruct
        info = '=== HistStruct === \n'
        # histogram names:
        info += '- histogram types ({}): \n'.format(len(self.histnames))
        for histname in self.histnames:
            info += '  -- {} (nbins: {})\n'.format(histname,self.histograms[histname].shape[1:])
        # size of histogram sets:
        info += '- number of lumisections: {}\n'.format(len(self.lsnbs))
        # masks:
        info += '- masks ({}): \n'.format(len(self.get_masknames()))
        for maskname in self.get_masknames(): info += '  -- {}\n'.format(maskname)
        # extra histograms sets
        info += '- extra histogram sets: {}\n'.format(len(self.exthistograms.keys()))
        for setname in self.exthistograms.keys(): 
            info += '  -- {}\n'.format(setname)
            info += '     --- histogram types ({})\n'.format(len(self.exthistograms[setname].keys()))
            for histname in self.exthistograms[setname].keys():
                info += '         ---- {} (shape: {})\n'.format(histname, self.exthistograms[setname][histname].shape)
        # models:
        info += '- models: \n'
        for model in self.models.values():
            info += str(model)
        return info
        
    def save( self, path, save_models=False, save_classifiers=True, save_fitter=True ):
        ### save a HistStruct object to a pkl file
        # input arguments:
        # - path where to store the file (appendix .zip is automatically appended)
        # - save_models: a boolean whether to include the models if present in the HistStruct
        # - save_classifiers: a boolean whether to include the classifiers if present in the ModelInterfaces
        # - save_fitter: a boolean whether to include the fitter if present in the ModelInterfaces
        pklpath = os.path.splitext(path)[0]+'.pkl'
        zippath = os.path.splitext(path)[0]+'.zip'
        rootpath = os.path.dirname(path)
        zipcontents = {}
        # remove the models from the object
        models = copy.deepcopy(self.models)
        self.models = {}
        # pickle the rest
        with open(pklpath,'wb') as f:
            pickle.dump(self,f)
        zipcontents[pklpath] = os.path.relpath(pklpath, start=rootpath)
        # restore the models
        self.models = models
        # case where the models should be stored
        if( len(list(self.models.keys()))>0 and save_models ):
            for modelname,model in self.models.items():
                mpath = os.path.splitext(path)[0]+'_model_{}.zip'.format(modelname)
                model.save( mpath, save_classifiers=save_classifiers, save_fitter=save_fitter )
                zipcontents[mpath] = os.path.relpath(mpath, start=rootpath)
        # put everything in a zip file
        with zipfile.ZipFile( zippath, 'w' ) as zipf:
            for f, fname in zipcontents.items(): zipf.write(f, fname)
        # remove individual files
        for f in zipcontents: os.system('rm {}'.format(f))
            
    @classmethod
    def load( self, path, load_models=True, load_classifiers=True, load_fitter=True, verbose=False ):
        ### load a HistStruct object
        # input arguments:
        # - path to a zip file containing a HistStruct object
        # - load_models: a boolean whether to load the models if present
        # - load_classifiers: a boolean whether to load the classifiers if present
        # - load_fitter: a boolean whether to load the fitter if present
        # - verbose: boolean whether to print some information
        zippath = os.path.splitext(path)[0]+'.zip'
        unzippath = os.path.splitext(path)[0]+'_unzipped'
        basename = os.path.splitext(os.path.basename(zippath))[0]
        pklbasename = basename+'.pkl'
        zipcontents = []
        # extract the zip file
        with zipfile.ZipFile( zippath, 'r' ) as zipf:
            zipcontents = zipf.namelist()
            zipf.extractall( path=unzippath )
        with open(os.path.join(unzippath,pklbasename),'rb') as f:
            obj = pickle.load(f)
        if( load_models ):
            if len(zipcontents)==1:
                print('WARNING: requested to load models, '
                      +'but this stored HistStruct object does not seem to contain any.')
            else:
                for modelname in obj.modelnames:
                    mpath = os.path.join( unzippath,
                                          os.path.splitext(os.path.basename(path))[0]+'_model_{}.zip'.format(modelname) )
                    obj.models[modelname] = ModelInterface.load( mpath, load_classifiers=load_classifiers,
                                                                 load_fitter=load_fitter )
        # remove individual files
        if os.path.exists(unzippath): os.system('rm -r {}'.format(unzippath))
        if verbose:
            print('Loaded a HistStruct object with following properties:')
            print(obj)
        return obj
    
    ###################################
    # functions for adding histograms #
    ###################################
        
    def add_dataframe( self, df, 
                        runcolumn='run', lumicolumn='lumi', menamecolumn='mename',
                        datacolumn='data', xbinscolumn='xbins', ybinscolumn='ybins',
                        entriescolumn='entries',
                        cropslices=None, rebinningfactor=None, 
                        smoothinghalfwindow=None, smoothingweights=None,
                        averagewindow=None, averageweights=None,
                        donormalize=True ):
        ### add a dataframe to a HistStruct
        # input arguments:
        # - df: a pandas dataframe as read from the input csv files
        # - cropslices: list of slices (one per dimension) by which to crop the histograms
        #                see hist_utils.py / crophists for more info.
        # - rebinningfactor: factor by which to group bins together
        #                    see hist_utils.py / rebinhists for more info.
        # - smoothinghalfwindow: half window (int for 1D, tuple for 2D) for doing smoothing of histograms
        # - smoothingweights: weight array (1D for 1D, 2D for 2D) for smoothing of histograms
        #                     see hist_utils.py / smoothhists for more info.
        # - averagewindow: window (int or tuple) for averaging each histogram with its neighbours
        # - averageweights: weights for averaging each histogram with its neighbours
        #                   see hist_utils.py / running_average_hists for more info.
        # - donormalize: boolean whether to normalize the histograms
        #                see hist_utils.py / normalizehists for more info.
        # notes:
        # - the new dataframe can contain one or multiple histogram types
        # - the new dataframe must contain the same run and lumisection numbers (for each histogram type in it)
        #   as already present in the HistStruct, except if it is the first one to be added
        # - alternative to adding the dataframe with the preprocessing options, 
        #   one can also apply the preprocessing at a later stage using the preprocess() function
        #   with the same arguments.
        
        histnames = dfu.get_menames(df, menamecolumn=menamecolumn)
        # loop over all names in the dataframe
        for histname in histnames:
            if histname in self.histnames:
                raise Exception('ERROR in HistStruct.add_dataframe: dataframe contains histogram name {}'.format(histname)
                               +' but this is already present in the current HistStruct.')
            thisdf = dfu.select_menames( df, [histname], menamecolumn=menamecolumn )
            # determine statistics (must be done before normalizing)
            nentries = np.array(thisdf[entriescolumn])
            # get physical xmin and xmax
            xmin = thisdf.at[0, 'xmin']
            xmax = thisdf.at[0, 'xmax']
            # prepare the data
            (hists_all,runnbs_all,lsnbs_all) = hu.preparedatafromdf(thisdf,
                                                runcolumn=runcolumn,
                                                lumicolumn=lumicolumn,
                                                datacolumn=datacolumn,
                                                returnrunls=True,
                                                cropslices=cropslices,
                                                rebinningfactor=rebinningfactor,
                                                smoothinghalfwindow=smoothinghalfwindow,
                                                smoothingweights=smoothingweights,
                                                averagewindow=averagewindow,
                                                averageweights=averageweights,
                                                donormalize=donormalize)
            runnbs_all = runnbs_all.astype(int)
            lsnbs_all = lsnbs_all.astype(int)
            # check consistency in run and lumisection numbers
            if len(self.histnames)!=0:
                if( not ( (runnbs_all==self.runnbs).all() and (lsnbs_all==self.lsnbs).all() ) ):
                    raise Exception('ERROR in HistStruct.add_dataframe:'
                                    +' dataframe run/lumi numbers are not consistent with current HistStruct!')
            # add everything to the structure
            self.histnames.append(histname)
            self.histograms[histname] = hists_all
            self.nentries[histname] = nentries
            self.histranges[histname] = (xmin,xmax)
            self.runnbs = runnbs_all
            self.lsnbs = lsnbs_all
            
    def add_histograms( self, histname, histograms, runnbs, lsnbs, nentries=None ):
        ### add a set of histograms to a HistStruct
        # input arguments:
        # - histname: name of the histogram type to be added
        # - histograms: a numpy array of shape (nhistograms,nbins), assumed to be of a single type
        # - runnbs: a 1D list or array of length nhistograms containing the run number per histogram
        # - lsnbs: a 1D list or array of length nhistograms containing the lumisection number per histogram
        # - nentries: a 1D list or array of length nhistograms containing the number of entries per histogram
        #   notes:
        #   - must be provided explicitly since histograms might be normalized, 
        #     in which case the number of entries cannot be determined from the sum of bin contents.
        #   - used for (de-)selecting histograms with sufficient statistics; 
        #     if you don't need that type of selection, nentries can be left at default.
        #   - default is None, meaning all entries will be set to zero.
        # notes:
        # - no preprocessing is performed, this is assumed to have been done manually (if needed) before adding the histograms
        # - runnbs and lsnbs must correspond to what is already in the current HistStruct, except if this is the first set of histogram to be added
        # - see also add_dataframe for an alternative way of adding histograms
        
        if histname in self.histnames:
            raise Exception('ERROR in HistStruct.add_histograms: histogram name is '.format(histname)
                               +' but this is already present in the current HistStruct.')
        runnbs = np.array(runnbs).astype(int)
        lsnbs = np.array(lsnbs).astype(int)
        # check consistency in run and lumisection numbers
        if len(self.histnames)!=0:
            if( not ( (runnbs==self.runnbs).all() and (lsnbs==self.lsnbs).all() ) ):
                raise Exception('ERROR in HistStruct.add_histograms: run/lumi numbers are not consistent with current HistStruct!')
        # parse nentries
        if nentries is None: nentries = np.zeros( len(runnbs) )
        if len(nentries)!=len(runnbs):
            raise Exception('ERROR in HistStruct.add_histograms: entries are not consistent with run/lumi numbers: '
                            +'found lengths {} and {} respectively.'.format(len(nentries),len(runnbs)))
        # add everything to the structure
        self.histnames.append(histname)
        self.histograms[histname] = histograms
        self.nentries[histname] = nentries
        self.runnbs = runnbs
        self.lsnbs = lsnbs

    def preprocess( self, masknames=None, cropslices=None, rebinningfactor=None,
                    smoothinghalfwindow=None, smoothingweights=None,
                    averagewindow=None, averageweights=None,
                    donormalize=False ):
        ### do preprocessing
        # input arguments:
        # - masknames: names of masks to select histograms to which to apply the preprocessing
        #               (histograms not passing the masks are simply copied)
        # the other input arguments are equivalent to those given in add_dataframe,
        # but this function allows to do preprocessing after the dataframes have already been loaded
        # note: does not work on extended histograms sets!
        #       one needs to apply preprocessing before generating extra histograms.
        for histname in self.histnames:
            # get the histograms
            hists = self.get_histograms(histname=histname, masknames=masknames)
            # do the preprocessing
            if cropslices is not None:  hists = hu.crophists(hists, cropslices)
            if rebinningfactor is not None: hists = hu.rebinhists(hists, rebinningfactor)
            if smoothinghalfwindow is not None: hists = hu.smoothhists(hists, 
                                                        halfwindow=smoothinghalfwindow,
                                                        weights=smoothingweights)
            if averagewindow is not None: hists = hu.running_average_hists(hists,
                                                     window=averagewindow,
                                                     weights=averageweights)
            if donormalize: hists = hu.normalizehists(hists)
            # put the histograms back in the histstruct
            if masknames is None:
                self.histograms[histname] = hists
            else:
                runnbs = self.get_runnbs(masknames=masknames)
                lsnbs = self.get_lsnbs(masknames=masknames)
                ids = [self.get_index(runnb,lsnb) for runnb,lsnb in zip(runnbs,lsnbs)]
                for i,idx in enumerate(ids): self.histograms[histname][idx] = hists[i]
        
    def add_globalscores( self, globalscores ):
        ### add an array of global scores (one per lumisection)
        # DEPRECATED, DO NOT USE ANYMORE
        # input arguments:
        # - globalscores: 1D numpy array of scores (must have same length as lumisection and run numbers)
        raise Exception('ERROR: HistStruct.add_globalscores is deprecated!')
        '''if( len(globalscores)!=len(self.lsnbs) ):
            raise Exception('ERROR in HistStruct.add_globalscores: length of globalscores ({})'.format(len(globalscores))
                           +' does not match length of list of lumisections ({})'.format(len(self.lsnbs)))
        if( len(self.globalscores)>0 ):
            print('WARNING in HistStruct.add_globalscores: array of global scores appears to be already initialized; '
                    +'overwriting...')
        self.globalscores = globalscores'''
        
    def add_extglobalscores( self, extname, globalscores ):
        ### add an array of global scores (one per lumisection) for a specified extra set of histograms in the HistStruct
        # DEPRECATED, DO NOT USE ANYMORE
        # input arguments:
        # - extname: name of extra histogram set
        # - globalscores: 1D numpy array of scores
        # note: this function checks if all histogram types in this set contain the same number of histograms,
        #       (and that this number corresponds to the length of globalscores)
        #       else adding globalscores is meaningless
        raise Exception('ERROR: HistStruct.add_extglobalscores is deprecated!')
        '''if not extname in self.exthistograms.keys():
            raise Exception('ERROR in HistStruct.add_extglobalscores: requested to add scores for set {}'.format(extname)
                           +' but this is not present in the current HistStruct.')
        histnames = list(self.exthistograms[extname].keys())
        nhists = self.exthistograms[extname][histnames[0]].shape[0]
        for histname in histnames[1:]:
            if( self.exthistograms[extname][histname].shape[0]!=nhists ):
                raise Exception('ERROR in HistStruct.add_extglobalscores: requested to add scores for set {}'.format(extname)
                               +' but the number of histograms for this set is not consistent between several types.')
        if( len(globalscores)!=nhists ):
            if( self.exthistograms[extname][histname].shape[0]!=nhists ):
                raise Exception('ERROR in HistStruct.add_extglobalscores: the number of histograms ({})'.format(nhists)
                                +' fdoes not match the length of the provided array ({}).'.format(len(globalscores)))
        if( extname in self.extglobalscores.keys() ):
            print('WARNING in HistStruct.add_extglobalscores: array of global scores for set {}'.format(extname)
                  +' appears to be already initialized; overwriting...')
        self.extglobalscores[extname] = globalscores'''
        
    def add_exthistograms( self, setname, histname, histograms, overwrite=False ):
        ### add a set of extra histograms to a HistStruct
        # these histograms are not assumed to correspond to physical run/lumisections numbers (e.g. resampled ones),
        # and no consistency checks are done
        # input arguments:
        # - setname: name of the extra histogram set (you can add multiple, e.g. resampled_good, resampled_bad and/or resampled_training)
        # - histname: name of the histogram type
        # - histograms: a numpy array of shape (nhistograms,nbins)
        # - overwrite: boolean whether to overwrite a set of histograms of the same name if present (default: raise exception)
        if setname in self.setnames:
            if histname in self.exthistograms[setname].keys():
                if not overwrite:
                    raise Exception('ERROR in HistStruct.add_exthistograms: histogram name is {}'.format(histname)
                                   +' but this is already present in the set of extra histogram with name {}'.format(setname))
        else: 
            self.exthistograms[setname] = {}
            for modelname in self.modelnames:
                self.models[modelname].add_setname( setname )
            self.setnames.append(setname)
        self.exthistograms[setname][histname] = histograms
        
        
    ###################################
    # functions for mask manipulation #
    ###################################
    
    def add_mask( self, name, mask ):
        ### add a mask to a HistStruct
        # input arguments:
        # - name: a name for the mask
        # - mask: a 1D np array of booleans with same length as number of lumisections in HistStruct
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
        
    def add_index_mask( self, name, indices ):
        ### add a mask corresponding to predefined indices
        # input arguments:
        # - name: a name for the mask
        # - indices: a 1D np array of integer indices, between 0 and the number of lumisections in HistStruct
        if(np.amax(indices)>=len(self.lsnbs)):
            raise Exception('ERROR in HistStruct.add_index_mask: largest index is {}'.format(np.amax(indices))
                           +' but there are only {} lumisections in the HistStruct'.format(len(self.lsnbs)))
        mask = np.zeros(len(self.lsnbs), dtype=bool)
        mask[indices] = True
        self.add_mask( name, mask)
        
    def add_run_mask( self, name, runnb ):
        ### add a mask corresponding to a given run number
        # input arguments:
        # - name: a name for the mask
        # - runnb: run number
        json = {str(runnb):[[-1]]}
        self.add_json_mask( name, json )
        
    def add_multirun_mask( self, name, runnbs ):
        ### add a mask corresponding to a given list of run numbers
        # input arguments:
        # - name: a name for the mask
        # - runnbs: a list of run numbers
        json = {}
        for runnb in runnbs: json[str(runnb)] = [[-1]]
        self.add_json_mask( name, json )
      
    def add_json_mask( self, name, jsondict, invert=False ):
        ### add a mask corresponding to a json dict
        # input arguments:
        # - name: a name for the mask
        # - jsondict: a dictionary in typical json format (see the golden json file for inspiration)
        # all lumisections present in the jsondict will be masked True, the others False.
        mask = jsonu.injson( self.runnbs, self.lsnbs, jsondict=jsondict )
        if invert: mask = ~mask
        self.add_mask( name, mask )
    
    def add_goldenjson_mask( self, name ):
        ### add a mask corresponding to the golden json file
        # input arguments:
        # - name: a name for the mask
        mask = jsonu.isgolden( self.runnbs, self.lsnbs )
        self.add_mask( name, mask )
        
    def add_dcsonjson_mask( self, name ):
        ### add a mask corresponding to the DCS-bit on json file
        # input arguments:
        # - name: a name for the mask
        mask = jsonu.isdcson( self.runnbs, self.lsnbs )
        self.add_mask( name, mask )
    
    def add_stat_mask( self, name, histnames=None, min_entries_to_bins_ratio=-1, max_entries_to_bins_ratio=-1 ):
        ### add a mask corresponding to lumisections where all histograms have statistics within given bounds
        # input arguments:
        # - histnames: list of histogram names to take into account for making the mask (default: all in the HistStruct)
        # - min_entries_to_bins_ratio: number of entries divided by number of bins, lower boundary for statistics 
        #   (default: no lower boundary)
        # - max_entries_to_bins_ratio: same but upper boundary instead of lower boundary (default: no upper boundary)
        if histnames is None:
            histnames = self.histnames
        mask = np.ones(len(self.runnbs)).astype(bool)
        for histname in histnames:
            if histname not in self.histnames:
                raise Exception('ERROR in HistStruct.add_stat_mask: requested to take into account {}'.format(histname)
                               +' but no such histogram type exists in the HistStruct.')
            # determine number of bins
            if len(self.histograms[histname].shape)==2:
                nbins = self.histograms[histname].shape[1]
            else:
                nbins = self.histograms[histname].shape[1]*self.histograms[histname].shape[2]
            # add the mask for this type of histogram
            if min_entries_to_bins_ratio > 0:
                mask = mask & (self.nentries[histname]/nbins>min_entries_to_bins_ratio)
            if max_entries_to_bins_ratio > 0:
                mask = mask & (self.nentries[histname]/nbins<max_entries_to_bins_ratio)
        self.add_mask( name, mask )
        
    def add_highstat_mask( self, name, histnames=None, entries_to_bins_ratio=100 ):
        ### shorthand call to add_stat_mask with only lower boundary and no upper boundary for statistics
        # input arguments:
        # - entries_to_bins_ratio: number of entries divided by number of bins, lower boundary for statistics
        # others: see add_stat_mask
        self.add_stat_mask( name, histnames=histnames, min_entries_to_bins_ratio=entries_to_bins_ratio )
    
    def pass_masks( self, masknames, runnbs=None, lsnbs=None ):
        ### get a list of booleans of lumisections whether they pass a given set of masks
        # input arguments:
        # - masknames: list of mask names
        # - runnbs: list of run numbers (default: all in histstruct)
        # - lsnbs: list of lumisection numbers (equally long as runnbs) (default: al in histstruct)
        res = []
        if( runnbs is None or lsnbs is None ):
            # fast method that does not require looping
            mask = self.masks[masknames[0]]
            for maskname in masknames[1:]:
                mask = mask & self.masks[maskname]
            return mask
        for runnb, lsnb in zip(runnbs,lsnbs):
            # slower method with looping
            passmasks = True
            idx = self.get_index(runnb, lsnb)
            for mask in masknames:
                if not self.masks[mask][idx]: passmasks = False
            res.append(passmasks)
        return res
    
    def get_masknames( self ):
        ### return a list of all mask names in the current HistStruct
        return list( self.masks.keys() )
    
    def get_mask( self, name ):
        ### return a mask in the current HistStruct
        return self.masks[name][:]
    
    def get_combined_mask( self, names ):
        ### get a combined (intersection) mask given multiple mask names
        # mostly for internal use; externally you can use get_histograms( histname, <list of mask names>) directly
        mask = np.ones(len(self.runnbs)).astype(bool)
        for name in names:
            if name not in self.masks.keys():
                raise Exception('ERROR in HistStruct.get_combined_mask: mask {} requested but not found.'.format(name))
            mask = mask & self.masks[name]
        return mask
    
    def get_union_mask( self, names ):
        ### get a combined (union) mask given multiple mask names
        # mostly for internal use
        mask = np.zeros(len(self.runnbs)).astype(bool)
        for name in names:
            if name not in self.masks.keys():
                raise Exception('ERROR in HistStruct.get_union_mask: mask {} requested but not found.'.format(name))
            mask = mask | self.masks[name]
        return mask
    
    ########################################################
    # functions for retrieving run and lumisection numbers #
    ########################################################
    
    def get_runnbs( self, masknames=None ):
        ### get the array of run numbers, optionally after masking
        # input arguments:
        # - masknames: list of names of masks (default: no masking, return full array)
        if masknames is None: return self.runnbs[:]
        return self.runnbs[ self.get_combined_mask(masknames) ]

    def get_runnbs_unique( self, masknames=None ):
        ### get a list of unique run numbers
        res = []
        for runnb in self.get_runnbs( masknames=masknames ):
            if runnb not in res: res.append(runnb)
        return res
    
    def get_lsnbs( self, masknames=None ):
        ### get the array of lumisection numbers, optionally after masking
        # input arguments:
        # - masknames: list of names of masks (default: no masking, return full array)
        if masknames is None: return self.lsnbs[:]
        return self.lsnbs[ self.get_combined_mask(masknames) ]
    
    def get_index( self, runnb, lsnb ):
        ### get the index in the current HistStruct of a given run and lumisection number
        # input arguments:
        # - runnb and lsnb: run and lumisection number respectively
        index = (set(list(np.where(self.runnbs==runnb)[0])) & set(list(np.where(self.lsnbs==lsnb)[0])))
        if len(index)!=1: 
            raise Exception('ERROR in HistStruct.get_index: unexpected index of requested run/lumisection: '
                            +'found {}.'.format(index)
                            +' Is the requested run/lumisection in the HistStruct?')
        (index,) = index
        return index
    
    def get_scores( self, modelname, histname=None, setnames=None, masknames=None ):
        ### get the array of scores for a given model and for a given histogram type, optionally after masking
        # input arguments:
        # - modelname: name of the model for which to retrieve the scores
        # - histname: name of the histogram type for which to retrieve the score. 
        #   if None, return a dict matching histnames to arrays of scores
        # - setnames: list of names of the histogram sets (use None for standard set)
        # - masknames: list of names of masks (default: no masking, return full array)
        # notes:
        # - do not use setnames and masknames simultaneously, this is not defined
        # - if multiple masks are given, the intersection is taken;
        #   if multiple sets are given, the union is taken
        # - the classifiers in the appropriate model must have been evaluated before calling this method!
        
        # argument checking
        if( masknames is not None and setnames is not None ):
            raise Exception('ERROR in HistStruct.get_scores: you cannot specify both masknames and setnames.')
        if modelname not in self.models.keys():
            raise Exception('ERROR in HistStruct.get_scores: requested model {}'.format(modelname)
                           +' but it does not seem to be present in the current HistStruct.')
        histnames = self.histnames[:]
        # check if histname is valid
        if histname is not None:
            if histname not in self.histnames:
                raise Exception('ERROR in HistStruct.get_scores: requested histogram name {}'.format(histname)
                               +' but this is not present in the current HistStruct.')
            histnames = [histname]
        # make the mask
        if masknames is not None: mask = self.get_combined_mask(masknames)
        # make the result
        res = {}
        for hname in histnames:
            scores = self.models[modelname].get_scores( setnames=setnames, histname=hname )
            if masknames is not None: scores = scores[mask]
            res[hname] = scores
        if histname is None: return res
        return res[histname]
    
    def get_scores_for_runsls( self, modelname, runsls, histname=None, masknames=None ):
        ### similar to get_scores, but return scores for specified lumisections
        # input arguments:
        # - modelname: name of the model for which to retrieve the global score
        # - runsls: dict in the same format as e.g. the golden json, e.g. {"380235": [[1,10], [15,20]]}
        # - histname: name of the histogram type for which to retrieve the score. 
        #   if None, return a dict matching histnames to arrays of scores
        # - masknames: list of names of masks (that come on top of the run/lumisection mask)
        #   (default: no masking, return full array)
        # note: the run and lumisection numbers are returned as well
        mask = jsonu.injson( self.runnbs, self.lsnbs, jsondict=runsls )
        if masknames is not None: mask = ((mask) & (self.get_combined_mask(masknames)))
        scores = self.get_scores(modelname, histname=histname)
        if histname is None:
            for histname in self.histnames:
                scores[histname] = scores[histname][mask]
        else: scores = scores[mask]
        return (self.runnbs[mask][:], self.lsnbs[mask][:], scores)
    
    def get_scores_array( self, modelname, setnames=None, masknames=None ):
        ### similar to get_scores, but with different return type:
        # np array of shape (nhistograms, nhistogramtypes)
        scores = self.get_scores( modelname, setnames=setnames, masknames=masknames )
        scores_array = []
        for histname in self.histnames:
            scores_array.append(scores[histname])
        scores_array = np.transpose(np.array(scores_array))
        return scores_array
    
    def get_scores_ls( self, modelname, runnb, lsnb, histnames=None ):
        ### get the scores for a given run/lumisection number and for given histogram names
        # input arguments:
        # - modelname: name of the model for which to retrieve the score
        # - runnb: run number
        # - lsnb: lumisection number
        # - histnames: names of the histogram types for which to retrieve the score. 
        # returns:
        # - a dict matching each name in histnames to a score (or None if no valid score)
        if histnames is None: histnames = self.histnames
        scores = {}
        index = self.get_index(runnb,lsnb)
        for histname in histnames:
            # check if histname is valid
            if histname not in self.histnames:
                raise Exception('ERROR in HistStruct.get_scores_ls: requested histogram name {}'.format(histname)
                               +' but this is not present in the current HistStruct.')
            scores[histname] = self.models[modelname].get_scores( histname=histname )[index]
        return scores
    
    def get_globalscores( self, modelname, setnames=None, masknames=None ):
        ### get the array of global scores, optionally after masking
        # input arguments:
        # - modelname: name of the model for which to retrieve the global score
        # - setnames: list of names of the histogram sets (use None for standard set)
        # - masknames: list of names of masks (default: no masking, return full array)
        # notes:
        # - do not use setnames and masknames simultaneously, this is not defined
        # - if multiple masks are given, the intersection is taken;
        #   if multiple sets are given, the union is taken
        # - the classifiers in the appropriate model must have been evaluated before calling this method!
        scores = self.models[modelname].get_globalscores( setnames=setnames )
        if len(scores)==0:
            msg = 'WARNING in HistStruct.get_globalscores: no global scores found for setnames {}.'.format(setnames)
            print(msg)
            return None
        if masknames is not None:
            mask = self.get_combined_mask(masknames)
            scores = scores[mask]
        return scores
    
    def get_globalscores_for_runsls( self, modelname, runsls, masknames=None ):
        ### get the array of global scores for specific lumisections
        # input arguments:
        # - modelname: name of the model for which to retrieve the global score
        # - runsls: dict in the same format as e.g. the golden json, e.g. {"380235": [[1,10], [15,20]]}
        # - masknames: list of names of masks (that come on top of the run/lumisection mask)
        #   (default: no masking, return full array)
        # note: the run and lumisection numbers are returned as well
        mask = jsonu.injson( self.runnbs, self.lsnbs, jsondict=runsls )
        if masknames is not None: mask = ((mask) & (self.get_combined_mask(masknames)))
        scores = self.get_globalscores(modelname)
        return (self.runnbs[mask][:], self.lsnbs[mask][:], scores[mask][:])
    
    def get_globalscores_jsonformat( self, modelname=None ):
        ### make a json format listing all lumisections in this histstruct
        # the output list has entries for global scores and masks
        # input arguments:
        # - modelname: name of the model for wich to retrieve the global score
        #   if None, all available models will be used
        res = []
        for (runnb,lsnb) in zip(self.runnbs,self.lsnbs):
            res.append({'run':int(runnb), 'ls':int(lsnb)})
            # (note: need explicit conversion to int since numpy data types are not understood by json serializer)
        modelnames = self.models.keys()
        if modelname is not None:
            if( modelname not in self.models.keys() ):
                raise Exception('ERROR: HistStruct.get_globalscores_json: requested model {}'.format(modelname)
                                 +' but it is not present in the current HistStruct')
            modelnames = [modelname]
        globalscores = {}
        for modelname in modelnames:
            globalscores[modelname] = self.models[modelname].get_globalscores()
        for idx in range(len(res)):
            for modelname in modelnames:
                res[idx]['score_{}'.format(modelname)] = float(globalscores[modelname][idx])
            for maskname in self.masks.keys():
                res[idx]['mask_{}'.format(maskname)] = bool(self.masks[maskname][idx])
        return res
    
    def get_globalscore_ls( self, modelname, runnb, lsnb ):
        ### get the global score for a given run/lumisection number
        # input arguments:
        # - modelname: name of the model for which to retrieve the global score
        # - runnb: run number
        # - lsnb: lumisection number
        # - histnames: names of the histogram types for which to retrieve the score. 
        # returns:
        # - a dict matching each name in histnames to a score (or None if no valid score)
        index = self.get_index(runnb,lsnb)
        globalscores = self.models[modelname].get_globalscores()
        if( len(globalscores)==0 ):
            raise Exception('ERROR in HistStruct.get_globalscore_ls: array of global scores seems to be uninitialized.')
        return globalscores[index]
    
    def get_globalscores_mask( self, modelname, masknames=None, setnames=None, score_up=None, score_down=None ):
        ### get the mask for global score between specified boundaries
        # input arguments:
        # - modelname: name of the model for which to consider the global scores
        # - masknames: list of additional masks (on top of score boundaries) to consider
        # - setnames: list of set names for which to retrieve the global scores
        # - score_up and score_down are upper and lower thresholds
        #     if both are not None, the mask for global scores between the boundaries are returned
        #     if score_up is None, the mask for global score > score_down are returned
        #     if score_down is None, the mask for global score < score_up are returned
        if( masknames is not None and setnames is not None ):
            raise Exception('ERROR in HistStruct.get_globalscore_mask:'
                           +' you cannot specify both mask names and set names.')
        scoremask = self.models[modelname].get_globalscores_mask( setnames=setnames, 
                                                                  score_up=score_up, score_down=score_down )
        if masknames is not None: scoremask = scoremask & self.get_combined_mask( masknames )
        return scoremask
    
    def get_globalscores_indices( self, modelname, masknames=None, setnames=None, score_up=None, score_down=None ):
        ### get the indices with a global score between specified boundaries
        # input arguments: see get_globalscore_mask
        mask = self.get_globalscores_mask( modelname, masknames=masknames, setnames=setnames, 
                                           score_up=score_up, score_down=score_down )
        return np.nonzero( mask )[0]
    
    def get_globalscores_runsls( self, modelname, masknames=None, setnames=None, score_up=None, score_down=None ):
        ### get the run and lumisection numbers with a global score between specified boundaries
        # input arguments: see get_globalscore_mask
        inds = self.get_globalscores_indices( modelname, masknames=masknames, setnames=setnames, 
                                              score_up=score_up, score_down=score_down )
        runnbs = self.runnbs[inds]
        lsnbs = self.lsnbs[inds]
        return (runnbs,lsnbs)
    
    def get_extglobalscores( self, extname ):
        ### get the array of global scores for one of the extra histogram sets
        # DEPRECATED, DO NOT USE ANYMORE
        # input arguments:
        # - extname: name of the extra histogram set
        # notes:
        # - this method takes the scores from the HistStruct.extglobalscores attribute;
        #   make sure to have set this attribute with add_extglobalscores,
        #   else an exception will be thrown.
        raise Exception('ERROR: HistStruct.get_extglobalscores is deprecated!')
        '''if extname not in self.extglobalscores.keys():
            raise Exception('ERROR in HistStruct.get_extglobalscores: requested to retrieve global scores for set {}'.format(extname)
                           +' but they are not present in the current HistStruct.')
        return self.extglobalscores[extname]'''
    
    def get_histograms( self, histname=None, masknames=None, setnames=None ):
        ### get the array of histograms for a given type, optionally after masking
        # input arguments:
        # - histname: name of the histogram type to retrieve 
        #   if None, return a dict matching histnames to arrays of histograms
        # - masknames: list of names of masks (default: no masking, return full array)
        # - setnames: list of names of the sets of extra histograms (see also add_exthistograms)
        #   if multiple setnames are provided, the union/concatenation is returned
        histnames = self.histnames[:]
        if histname is not None: histnames = [histname]
        if( masknames is not None and setnames is not None ):
            raise Exception('ERROR in HistStruct.get_histograms: you cannot specify both masknames and setnames')
        # case of default set:
        if setnames is None:
            # check if histnames are valid
            for hname in histnames:
                if hname not in self.histnames:
                    raise Exception('ERROR in HistStruct.get_histograms: requested histogram name {}'.format(histname)
                                   +' but this is not present in the current HistStruct.')
            # make the mask
            mask = np.ones(len(self.lsnbs)).astype(bool)
            if masknames is not None: mask = self.get_combined_mask(masknames)
            # make the result
            res = {}
            for hname in histnames: res[hname] = self.histograms[hname][mask]
        # case of extended sets:
        else:
            # argument checking
            for setname in setnames:
                if not setname in self.exthistograms.keys():
                    raise Exception('ERROR in HistStruct.get_histograms:'
                                    +' requested to retrieve histograms in set {}'.format(setname)
                                    +' but this is not present in the current HistStruct.')
                for hname in histnames:
                    if hname not in self.exthistograms[setname].keys():
                        raise Exception('ERROR in HistStruct.get_histograms: requested histogram name {}'.format(hname)
                                       +' but this is not present in the extra set with name {}.'.format(setname))
            # make the result
            res = {}
            for hname in histnames:
                hists = []
                for setname in setnames: hists.append(self.exthistograms[setname][hname])
                res[hname] = np.concatenate( tuple(hists), axis=0 )
        # return the result in correct format
        if histname is None: return res
        return res[histname]

    def get_histogramsandscores( self, modelname=None, setnames=None, masknames=None, nrandoms=-1, nfirst=-1 ):
        ### combination of get_histograms, get_scores and get_globalscores with additional options
        # - modelname: name of the model for which to retrieve the score
        #   if None, no scores will be retrieved (only histograms)
        # - setnames: list of names of histogram sets (use None for default set)
        # - masknames: list of names of masks
        # - nrandoms: if > 0, number of random instances to draw
        # - nfirst: if > 0, number of first instances to keep
        # return type:
        # dict with keys 'histograms', 'scores' and 'globalscores'
        # note that the values of scores and globalscores may be None if not initialized

        histograms = None
        scores = None
        globalscores = None
        do_scores = bool(modelname is not None)

        # get histograms and scores
        histograms = self.get_histograms(masknames=masknames, setnames=setnames)
        if do_scores: scores = self.get_scores(modelname, masknames=masknames, setnames=setnames)
        if do_scores: globalscores = self.get_globalscores(modelname, masknames=masknames, setnames=setnames)
        # further processing
        nhists = len(histograms[self.histnames[0]])
        if(nrandoms>0 and nfirst>0):
            raise Exception('ERROR in HistStruct.get_histogramsandscores: you cannot specify both "nrandoms" and "nfirst".')
        if(nrandoms>nhists):
            msg = 'WARNING: requested {} randoms'.format(nrandoms)
            msg += ' but only {} histograms are available;'.format(nhists)
            msg += ' will use all available histograms.'
            print(msg)
            nrandoms = -1
        if(nfirst>nhists):
            msg = 'WARNING: requested {} first'.format(nfirst)
            msg += ' but only {} histograms are available;'.format(nhists)
            msg += ' will use all available histograms.'
            print(msg)
            nfirst = -1
        ids = np.arange(nhists)
        if nrandoms>0: ids = np.random.choice(ids, size=nrandoms, replace=False)
        if nfirst>0: ids = ids[:nfirst]
        if( nrandoms>0 or nfirst>0 ):
            for histname in self.histnames:
                histograms[histname] = histograms[histname][ids]
                if scores is not None: 
                    scores[histname] = scores[histname][ids]
            if globalscores is not None: globalscores = globalscores[ids]
        return {'histograms': histograms, 'scores': scores, 'globalscores': globalscores}
    
    ################################################
    # functions for adding and manipulating models #
    ################################################
    
    def add_model( self, modelname, model ):
        ### add a model to the HistStruct
        # input arguments:
        # - modelname: a name for the model
        # - model: an instance of ModelInterface class with histnames corresponding to the ones for this HistStruct
        
        #if not isinstance(model, ModelInterface):
        #    raise Exception('ERROR in HistStruct.add_model: model is of type {}'.format(type(model))
        #                   +' while a ModelInterface is expected.')
        # to do: fix! does not seem to recognize ModelInterface correctly, giving errors where it shouldn't
        print('adding model "{}" to the HistStruct'.format(modelname))
        if( sorted(self.histnames)!=sorted(model.histnames) ):
            raise Exception('ERROR in HistStruct.add_model: histogran names of HistStruct and ModelInterface do not match.')
        if modelname in self.modelnames:
            print('WARNING in HistStruct.add_model: modelname "{}" is already present; overwriting...'.format(modelname))
        # add the model
        self.models[modelname] = model
        self.modelnames.append(modelname)
        # add the correct additional sets
        for setname in self.setnames:
            if setname not in model.setnames:
                model.add_setname( setname )
                
    def check_model( self, modelname ):
        ### check if a given model name is present in the HistStruct
        # input arguments:
        # - modelname: name of the model to check
        return bool( modelname in self.modelnames )
            
    def remove_model( self, modelname ):
        ### remove a model
        # input arguments:
        # - modelname: name of the model to remove
        if modelname not in self.modelnames:
            print('WARNING in HistStruct.remove_model: modelname "{}" is not present.'.format(modelname))
            return
        # remove the model
        self.models.pop(modelname)
        self.modelnames.remove(modelname)
        
    def train_classifier( self, modelname, histname, masknames=None, setnames=None, **kwargs ):
        ### train a histogram classifier
        # input arguments:
        # - modelname: name of the model for which to train the classifiers
        # - histname: a valid histogram name present in the HistStruct for which to train the classifier
        # - masknames: list of masks the classifiers should be trained on
        # - setnames: list of names of sets of extra histograms on which the classifiers should be trained
        # - kwargs: additional keyword arguments for training
        if( masknames is not None and setnames is not None ):
            raise Exception('ERROR in HistStruct.train_classifier:'
                            +' you cannot specify both masknames and setnames.')
        histograms = None
        histograms = self.get_histograms( masknames=masknames, setnames=setnames, histname=histname )
        self.models[modelname].train_classifier( histname, histograms, **kwargs )
        
    def train_classifiers( self, modelname, masknames=None, setnames=None, **kwargs ):
        ### train histogram classifiers for all histogram types
        # input arguments:
        # - modelname: name of the model for which to train the classifiers
        # - masknames: list of masks the classifiers should be trained on
        # - setnames: list of names of sets of extra histograms on which the classifiers should be trained
        # - kwargs: additional keyword arguments for training
        if( masknames is not None and setnames is not None ):
            raise Exception('ERROR in HistStruct.train_classifiers:'
                            +' you cannot specify both masknames and setnames.')
        histograms = None
        histograms = self.get_histograms( masknames=masknames, setnames=setnames )
        self.models[modelname].train_classifiers( histograms, **kwargs )
        
    def evaluate_classifier( self, modelname, histname, masknames=None, setnames=None ):
        ### evaluate a histogram classifier
        # input arguments:
        # - modelname: name of the model for wich to evaluate the classifiers
        # - histname: a valid histogram name present in the HistStruct for which to evaluate the classifier
        # - masknames: list of masks if the classifiers should be evaluated on a subset only (e.g. for speed)
        # - setnames: list of names of sets of extra histograms for which the classifiers should be evaluated
        if( masknames is not None and setnames is not None ):
            raise Exception('ERROR in HistStruct.evaluate_classifier:'
                            +' you cannot specify both masknames and setnames.')
        histograms = None
        mask = None
        # definitions in case of default set with optional masking
        if setnames is None:
            histograms = self.get_histograms( histname=histname )
            if masknames is not None: mask = self.get_combined_mask( masknames )
            self.models[modelname].evaluate_store_classifier( histname, histograms, mask=mask )
        # definitions in case of extended set
        else:
            for setname in setnames:
                histograms = self.get_histograms( histname=histname, setnames=[setname] )
                self.models[modelname].evaluate_store_classifier( histname, histograms, setname=setname )
        
    def evaluate_classifiers( self, modelname, masknames=None, setnames=None ):
        ### evaluate histogram classifiers for all histogram types
        # input arguments:
        # - modelname: name of the model for wich to evaluate the classifiers
        # - masknames: list of masks if the classifiers should be evaluated on a subset only (e.g. for speed)
        # - setnames: list of names of a set of extra histograms for which the classifiers should be evaluated
        if( masknames is not None and setnames is not None ):
            raise Exception('ERROR in HistStruct.evaluate_classifiers:'
                            +' you cannot specify both masknames and setnames.')
        histograms = None
        mask = None
        # definitions in case of default set with optional masking
        if setnames is None:
            histograms = self.get_histograms()
            if masknames is not None: mask = self.get_combined_mask( masknames )
            self.models[modelname].evaluate_store_classifiers( histograms, mask=mask )
        # definitions in case of extended set
        else:
            for setname in setnames:
                histograms = self.get_histograms( setnames=[setname] )
                self.models[modelname].evaluate_store_classifiers( histograms, setname=setname )
        
    def set_fitter( self, modelname, fitter ):
        ### set the fitter for a given model
        self.models[modelname].set_fitter(fitter)
        
    def train_fitter( self, modelname, masknames=None, setnames=None, verbose=False, **kwargs ):
        ### train the fitter for a given model
        # input arguments:
        # - modelname: name of the model to train
        # - masknames: list of mask names for training set
        # - setnames: list of set names for training set
        # - kwargs: additional keyword arguments for fitting
        # note: use either masksnames or setnames, not both!
        points = self.get_scores( modelname, masknames=masknames, setnames=setnames )
        self.models[modelname].train_fitter( points, verbose=verbose, **kwargs )
        
    def train_partial_fitters( self, modelname, dimslist, masknames=None, setnames=None, **kwargs ):
        ### train partial fitters for a given model
        # input arguments:
        # - modelname: name of the model to train
        # - dimslist: list of tuples with integer dimension numbers
        # - masknames: list of mask names for training set
        # - setnames: list of set names for training set
        # - kwargs: additional keyword arguments for fitting
        # note: use either masksnames or setnames, not both!
        # note: see also plot_partial_fit for a convenient plotting method!
        points = self.get_scores( modelname, masknames=masknames, setnames=setnames )
        self.models[modelname].train_partial_fitters( dimslist, points, **kwargs )
        
    def evaluate_fitter( self, modelname, masknames=None, setnames=None, verbose=False ):
        ### evaluate the fitter for a given model
        # input arguments:
        # - modelname: name of the model for which to evaluate the fitter
        # - masknames: list of mask names if the fitter should be evaluated on a subset only (e.g. for speed)
        # - setnames: list of set names of extra histograms for which the fitter should be evaluated
        if( masknames is not None and setnames is not None ):
            raise Exception('ERROR in HistStruct.evaluate_fitter:'
                            +' you cannot specify both masknames and setnames.')
        if setnames is None:
            points = self.get_scores( modelname )
            mask = None
            if masknames is not None: mask = self.get_combined_mask( masknames )
            self.models[modelname].evaluate_store_fitter( points, mask=mask )
        else:
            for setname in setnames:
                points = self.get_scores( modelname, setnames=[setname] )
                self.models[modelname].evaluate_store_fitter( points, setname=setname )
        
    def evaluate_fitter_on_point( self, modelname, point ):
        ### evaluate the fitter on a given points
        # input arguments:
        # - modelname: name of the model for which to evaluate the fitter
        # - points: dict matching histnames to scores (one float per histogram type)
        #   (e.g. as returned by get_scores_ls)
        # returns:
        # - the global score for the provided point (a float)
        for histname, score in point.items(): point[histname] = np.array([score])
        return self.models[modelname].evaluate_fitter( point )
        
    def evaluate_fitter_on_points( self, modelname, points ):
        ### evaluate the fitter on a given set of points
        # input arguments:
        # - modelname: name of the model for which to evaluate the fitter
        # - points: dict matching histnames to scores (np array of shape (nhistograms))
        # returns:
        # - the global scores for the provided points
        return self.models[modelname].evaluate_fitter( points )
    
    #########################################
    # functions for plotting raw histograms #
    #########################################

    def plot_histograms( self, histnames=None, masknames=None, histograms=None, ncols=4,
                            colorlist=None, labellist=None, transparencylist=None,
                            titledict=None, extratextdict=None, xaxtitledict=None, physicalxax=False, 
                            yaxtitledict=None, **kwargs ):
        ### plot the histograms in a HistStruct, optionally after masking
        # input arguments:
        # - histnames: list of names of the histogram types to plot (default: all)
        # - masknames: list of list of mask names
        #   note: each element in masknames represents a set of masks to apply; 
        #         the histograms passing different sets of masks are plotted in different colors
        # - histograms: list of dicts of histnames to 2D arrays of histograms,
        #               can be used to plot a given collection of histograms directly,
        #               and bypass the histnames and masknames arguments
        #               (note: for use in the gui, not recommended outside of it)
        # - ncols: number of columns (only relevant for 1D histograms)
        # - colorlist: list of matplotlib colors, must have same length as masknames
        # - labellist: list of labels for the legend, must have same legnth as masknames
        # - transparencylist: list of transparency values, must have same length as masknames
        # - titledict: dict mapping histogram names to titles for the subplots (default: title = histogram name)
        # - extratextdict: dict mapping histogram names to extra text on subplots (default: no extra text)
        # - xaxtitledict: dict mapping histogram names to x-axis titles for the subplots (default: no x-axis title)
        # - yaxtitledict: dict mapping histogram names to y-axis titles for the subplots (default: no y-axis title)
        # - physicalxax: bool whether to use physical x-axis range or simply use bin number (default)
        # - kwargs: keyword arguments passed down to plot_utils.plot_sets 
        
        # check validity of requested histnames
        histnames1d = []
        histnames2d = []
        if histnames is None: histnames = self.histnames
        if histograms is not None: histnames = list(histograms[0].keys())
        for histname in histnames:
            if not histname in self.histnames:
                raise Exception('ERROR in HistStruct.plot_histograms:'
                        +' requested to plot histogram type {}'.format(histname)
                        +' but it is not present in the current HistStruct.')
            if len(self.histograms[histname].shape)==2: histnames1d.append(histname)
            elif len(self.histograms[histname].shape)==3: histnames2d.append(histname)
        
        # initializations
        fig1d = None
        axs1d = None
        res2d = None

        # make a plot of the 1D histograms
        if len(histnames1d)>0:
            fig1d,axs1d = self.plot_histograms_1d( histnames=histnames1d, 
                            ncols=ncols,
                            masknames=masknames, 
                            histograms=histograms,
                            colorlist=colorlist, labellist=labellist, transparencylist=transparencylist,
                            titledict=titledict, extratextdict=extratextdict,
                            xaxtitledict=xaxtitledict, physicalxax=physicalxax, 
                            yaxtitledict=yaxtitledict,
                            **kwargs )

        # make plots of the 2D histograms
        if len(histnames2d)>0:
            allowed_kwargs = [] # fill here allowed keyword arguments
            present_kwargs = list(kwargs.keys())
            for key in present_kwargs:
                if key not in allowed_kwargs: kwargs.pop(key) 
            res2d = self.plot_histograms_2d( histnames=histnames2d, masknames=masknames,
                            histograms=histograms,
                            labellist=labellist,
                            titledict=titledict, xaxtitledict=xaxtitledict, yaxtitledict=yaxtitledict,
                            **kwargs )

        # return the figures and axes
        if len(histnames2d)==0: return (fig1d,axs1d) # for backward compatibility
        return (fig1d,axs1d,res2d)


    def plot_histograms_1d( self, histnames=None, masknames=None, histograms=None, ncols=4,
                            colorlist=None, labellist=None, transparencylist=None,
                            titledict=None, extratextdict=None,
                            xaxtitledict=None, physicalxax=False, yaxtitledict=None, 
                            **kwargs ):
        ### plot the histograms in a histstruct, optionally after masking
        # internal helper function, use only via plot_histograms
        
        # initializations
        ncols = min(ncols,len(histnames))
        nrows = int(math.ceil(len(histnames)/ncols))
        fig,axs = plt.subplots(nrows,ncols,figsize=(5*ncols,5*nrows),squeeze=False)
        # loop over all histogram types
        for j,name in enumerate(histnames):
            # get the histograms to plot
            histlist = []
            if histograms is not None:
                for k in range(len(histograms)):
                    histlist.append( histograms[k][name] )
            else:
                for maskset in masknames:
                    histlist.append( self.get_histograms(histname=name, masknames=maskset) )
            # get the title and axes
            title = pu.make_text_latex_safe(name)
            if( titledict is not None and name in titledict ): title = titledict[name]
            extratext = None
            if( extratextdict is not None and name in extratextdict ): extratext = extratextdict[name]
            xaxtitle = None
            if( xaxtitledict is not None and name in xaxtitledict ): xaxtitle = xaxtitledict[name]
            xlims = (-0.5,-1)
            if physicalxax: xlims = self.histranges[name]
            yaxtitle = None
            if( yaxtitledict is not None and name in yaxtitledict ): yaxtitle = yaxtitledict[name]
            # make the plot
            pu.plot_sets( histlist,
                        fig=fig,ax=axs[int(j/ncols),j%ncols],
                        title=title, extratext=extratext, xaxtitle=xaxtitle, xlims=xlims, yaxtitle=yaxtitle,
                        colorlist=colorlist, labellist=labellist, transparencylist=transparencylist,
                        **kwargs )
        return fig,axs


    def plot_histograms_2d( self, histnames=None, masknames=None, histograms=None,
                            labellist=None, titledict=None, xaxtitledict=None, yaxtitledict=None,
                            **kwargs ):
        ### plot the histograms in a histstruct, optionally after masking
        # internal helper function, use only via plot_histograms

        # initializations
        res = []
        ncols = min(4,len(histnames))
        nrows = int(math.ceil(len(histnames)/ncols))
        
        # get the histograms to plot
        if histograms is None:
            histograms = []
            for maskset in masknames:
                thishistograms = {}
                for histname in histnames:
                    thishistograms[histname] = self.get_histograms(histname=histname, masknames=maskset)
                histograms.append(thishistograms)

        # check total number of plots to make
        nplot = 0
        for histset in histograms:
            nplot += len(histset[list(histset.keys())[0]])
        threshold = 10
        if nplot > threshold:
            raise Exception('ERROR in HistStruct.plot_histograms_2d:'
                    +' plotting more than {} lumisections is not supported for now.'.format(threshold))

        # loop over histogram sets
        for setn, histset in enumerate(histograms):
            # loop over instances within a set
            nplots = len(histset[list(histset.keys())[0]])
            for i in range(nplots):
                histograms = []
                subtitles = []
                xaxtitles = []
                yaxtitles = []
                for j,name in enumerate(histnames):
                    # get the histogram
                    histograms.append( histset[name][i] )
                    # get the title and axes
                    title = pu.make_text_latex_safe(name)
                    if( titledict is not None and name in titledict ): title = titledict[name]
                    subtitles.append(title)
                    xaxtitle = None
                    if( xaxtitledict is not None and name in xaxtitledict ): xaxtitle = xaxtitledict[name]
                    xaxtitles.append(xaxtitle)
                    yaxtitle = None
                    if( yaxtitledict is not None and name in yaxtitledict ): yaxtitle = yaxtitledict[name]
                    yaxtitles.append(yaxtitle)
                # make the plot
                fig,axs = pu.plot_hists_2d( histograms, ncols=ncols, 
                                        subtitles=subtitles, xaxtitles=xaxtitles, 
                                        yaxtitles=yaxtitles, **kwargs)

                # add the label
                pu.add_text( fig, labellist[setn], (0.5, 0.95), fontsize=12, 
                            horizontalalignment='center',
                            background_edgecolor='black' )
                res.append( (fig,axs) )
        return res
    
    
    def plot_histograms_run( self, histnames=None, masknames=None, histograms=None, ncols=4, 
                            titledict=None, xaxtitledict=None, physicalxax=False, 
                            yaxtitledict=None, extratextdict=None, **kwargs ):
        ### plot a set of histograms in a HistStruct with a smooth color gradient.
        # typical use case: plot a single run.
        # note: only for 1D histograms!
        # input arguments:
        # - histnames: list of names of the histogram types to plot (default: all)
        # - masknames: list mask names (typically should contain a run number mask)
        # - histograms: dict of histnames to 2D arrays of histograms,
        #               can be used to plot a given collection of histograms directly,
        #               and bypass the histnames and masknames arguments
        #               (note: for use in the gui, not recommended outside of it.
        # - titledict: dict mapping histogram names to titles for the subplots (default: title = histogram name)
        # - xaxtitledict: dict mapping histogram names to x-axis titles for the subplots (default: no x-axis title)
        # - yaxtitledict: dict mapping histogram names to y-axis titles for the subplots (default: no y-axis title)
        # - physicalxax: bool whether to use physical x-axis range or simply use bin number (default)
        # - extratextdict: dict mapping histogram names to extra text on subplots (default: no extra text)
        # - kwargs: keyword arguments passed down to plot_utils.plot_hists_multi
        
        # check validity of requested histnames
        histnames1d = []
        histnames2d = []
        if histnames is None: histnames = self.histnames
        if histograms is not None: histnames = list(histograms[0].keys())
        for histname in histnames:
            if not histname in self.histnames:
                raise Exception('ERROR in HistStruct.plot_histograms_run:'
                        +' requested to plot histogram type {}'.format(histname)
                        +' but it is not present in the current HistStruct.')
            if len(self.histograms[histname].shape)==2: histnames1d.append(histname)
            elif len(self.histograms[histname].shape)==3: histnames2d.append(histname)
        
        # initializations
        fig1d = None
        axs1d = None

        # make a plot of the 1D histograms
        if len(histnames1d)>0:
            fig1d,axs1d = self.plot_histograms_run_1d( histnames=histnames1d, masknames=masknames, 
                            histograms=histograms, ncols=ncols, extratextdict=extratextdict,
                            titledict=titledict, xaxtitledict=xaxtitledict, physicalxax=physicalxax, 
                            yaxtitledict=yaxtitledict,
                            **kwargs )

        # return the figures and axes
        return (fig1d,axs1d)


    def plot_histograms_run_1d( self, histnames=None, masknames=None, histograms=None, ncols=4,
                            titledict=None, xaxtitledict=None, physicalxax=False, yaxtitledict=None,
                            extratextdict=None,
                            **kwargs ):
        ### plot the histograms in a histstruct, optionally after masking
        # internal helper function, use only via plot_histograms_run
        
        # initializations
        ncols = min(ncols,len(histnames))
        nrows = int(math.ceil(len(histnames)/ncols))
        fig,axs = plt.subplots(nrows,ncols,figsize=(7*ncols,5*nrows),squeeze=False)
        # loop over all histogram types
        for j,name in enumerate(histnames):
            # get the histograms to plot
            if histograms is not None: hists = histograms[name]
            else: hists = self.get_histograms(histname=name, masknames=masknames)
            # get the title and axes
            title = pu.make_text_latex_safe(name)
            if( titledict is not None and name in titledict ): title = titledict[name]
            xaxtitle = None
            if( xaxtitledict is not None and name in xaxtitledict ): xaxtitle = xaxtitledict[name]
            xlims = (-0.5,-1)
            if physicalxax: xlims = self.histranges[name]
            yaxtitle = None
            if( yaxtitledict is not None and name in yaxtitledict ): yaxtitle = yaxtitledict[name]
            extratext = None
            if( extratextdict is not None and name in extratextdict ): extratext = extratextdict[name]
            # set the color
            colorlist = np.arange(len(hists))
            # make the plot
            pu.plot_hists_multi( hists,
                        fig=fig,ax=axs[int(j/ncols),j%ncols],
                        title=title, xaxtitle=xaxtitle, xlims=xlims, yaxtitle=yaxtitle,
                        colorlist=colorlist, extratext=extratext,
                        **kwargs )
        return fig,axs
    
    ########################################################
    # functions for plotting lumisections + reconstruction #
    ########################################################
    
    def plot_ls( self, runnb, lsnb, histnames=None, histlabel=None, ncols=4,
                 recohist=None, recohistlabel='Reconstruction', 
                 refhists=None, refhistslabel='Reference histograms', refhiststransparency=None,
                 titledict=None, extratextdict=None,
                 xaxtitledict=None, physicalxax=False, yaxtitledict=None, **kwargs):
        ### plot the histograms in a HistStruct for a given run/ls number versus their references and/or their reconstruction
        # input arguments:
        # - runnb: run number
        # - lsnb: lumisection number
        # - histnames: names of histogram types to plot (default: all)
        # - histlabel: legend entry for the histogram (default: run and lumisection number)
        # - recohist: dict matching histogram names to reconstructed histograms
        #   notes: - 'reconstructed histograms' refers to e.g. autoencoder or NMF reconstructions;
        #            some models (e.g. simply looking at histogram moments) might not have this kind of reconstruction
        #          - in principle one histogram per key is expected, but still the the shape must be 2D (i.e. (1,nbins))
        #          - in case recohist is set to a valid model name present in the current HistStruct, 
        #            the reconstruction is calculated on the fly for the input histograms
        # - recohistlabel: legend entry for the reco histograms
        # - refhists: dict matching histogram names to reference histograms
        #   notes: - multiple histograms (i.e. a 2D array) per key are expected;
        #            in case there is only one reference histogram, it must be reshaped into (1,nbins)
        # - refhistslabel: legend entry for the reference histograms
        # - titledict: dict mapping histogram names to titles for the subplots (default: title = histogram name)
        # - extratextdict: dict mapping histogram names to extra text on subplots (default: no extra text)
        # - xaxtitledict: dict mapping histogram names to x-axis titles for the subplots (default: no x-axis title)
        # - yaxtitledict: dict mapping histogram names to y-axis titles for the subplots (default: no y-axis title)
        # - physicalxax: bool whether to use physical x-axis range or simply use bin number (default)
        # - kwargs: keyword arguments passed down to plot_utils.plot_sets 
        
        # check validity of arguments
        if histnames is None: histnames = self.histnames
        histnames1d = []
        histnames2d = []
        for histname in histnames:
            if not histname in self.histnames:
                raise Exception('ERROR in HistStruct.plot_ls: requested to plot histogram type {}'.format(histname)
                               +' but it is not present in the current HistStruct.')
            if( isinstance(recohist,str) ):
                if( recohist not in self.modelnames ):
                    raise Exception('ERROR in HistStruct.plot_ls: requested reconstruction with model "{}"'.format(recohist)
                                    +' but this model is not present in the current HistStruct.')
                if( histname not in self.models[recohist].classifiers.keys() ): 
                    raise Exception('ERROR in HistStruct.plot_ls: requested reconstruction with model "{}"'.format(recohist)
                                    +' for histogram type {},'.format(histname)
                                    +' but this model does not seem to have a classifier yet for this histogram type.')
            elif( recohist is not None ):
                if( histname not in recohist.keys() ):
                    raise Exception('ERROR in HistStruct.plot_ls: reco histograms provided, but type {}'.format(histname)
                                    +' seems to be missing.')
                if( not isinstance(recohist[histname], np.ndarray) ):
                    raise Exception('ERROR: recohist has unexpected structure, it is supposed to be a dict'
                                    +' matching strings to 2D numpy arrays')
            if( refhists is not None and histname not in refhists.keys() ):
                raise Exception('ERROR in HistStruct.plot_ls: reference histograms provided, but type {}'.format(histname)
                               +' seems to be missing.')
                if( not isinstance(refhists[histname], np.ndarray) ):
                    raise Exception('ERROR: refhists has unexpected structure, it is supposed to be a dict '
                                    +'matching strings to 2D numpy arrays')
            if len(self.histograms[histname].shape)==2: histnames1d.append(histname)
            elif len(self.histograms[histname].shape)==3: histnames2d.append(histname)

        # initializations
        fig1d = None
        axs1d = None
        fig2d = None
        axs2d = None

        # case of 1D histograms
        if len(histnames1d)>0:
            (fig1d, axs1d) = self.plot_ls_1d( runnb, lsnb, histnames=histnames, histlabel=histlabel, ncols=ncols,
                                recohist=recohist, recohistlabel=recohistlabel,
                                refhists=refhists, refhistslabel=refhistslabel, 
                                refhiststransparency=refhiststransparency,
                                titledict=titledict, extratextdict=extratextdict,
                                xaxtitledict=xaxtitledict, physicalxax=physicalxax, 
                                yaxtitledict=yaxtitledict, **kwargs )

        if len(histnames2d)>0:
            allowed_kwargs = [] # fill here allowed keyword arguments
            present_kwargs = list(kwargs.keys())
            for key in present_kwargs:
                if key not in allowed_kwargs: kwargs.pop(key)
            (fig2d, axs2d) = self.plot_ls_2d( runnb, lsnb, histnames=histnames, histlabel=histlabel,
                                recohist=recohist, recohistlabel=recohistlabel,
                                titledict=titledict, 
                                xaxtitledict=xaxtitledict,
                                yaxtitledict=yaxtitledict, **kwargs )

        # return the figures and axes
        if len(histnames2d)==0: return (fig1d,axs1d) # for backward compatibility
        return (fig1d,axs1d,fig2d,axs2d)


    def plot_run( self, runnb, masknames=None, ncols=4, 
                  recohist=None, recohistlabel='reco', 
                  refhists=None, refhistslabel='reference', doprint=False):
        ### call plot_ls for all lumisections in a given run
        runnbs = self.get_runnbs( masknames=masknames )
        lsnbs = self.get_lsnbs( masknames=masknames )
        runsel = np.where(runnbs==runnb)
        lsnbs = lsnbs[runsel]
        print('plotting {} lumisections...'.format(len(lsnbs)))
        for lsnb in lsnbs:
            _ = self.plot_ls(runnb, lsnb, ncols=ncols, 
                             recohist=recohist, recohistlabel=recohistlabel, 
                             refhists=refhists, refhistslabel=refhistslabel)


    def plot_ls_1d( self, runnb, lsnb, histnames=None, histlabel=None, ncols=4,
                 recohist=None, recohistlabel='Reconstruction',
                 refhists=None, refhistslabel='Reference histograms', refhiststransparency=None,
                 titledict=None, extratextdict=None, xaxtitledict=None, physicalxax=False, yaxtitledict=None, **kwargs):
        ### plot the histograms in a HistStruct for a given run/ls number versus their references and/or their reconstruction
        # internal helper function, use only via plot_ls

        # find index that given run and ls number correspond to
        index = self.get_index( runnb, lsnb )
        # initializations
        ncols = min(ncols,len(histnames))
        nrows = int(math.ceil(len(histnames)/ncols))
        fig,axs = plt.subplots(nrows,ncols,figsize=(5*ncols,5*nrows),squeeze=False)
        if histlabel is None: histlabel = 'Run: '+str(int(runnb))+', LS: '+str(int(lsnb))
        # loop over all histograms belonging to this lumisection and make the plots
        for j,name in enumerate(histnames):
            histlist = []
            colorlist = []
            labellist = []
            transparencylist = []
            # first plot the provided reference histograms
            if refhists is not None:
                histlist.append(refhists[name])
                colorlist.append('blue')
                labellist.append(refhistslabel)
                if refhiststransparency is None: refhiststransparency=0.3
                transparencylist.append(refhiststransparency)
            # then plot the original histogram
            hist = self.histograms[name][index:index+1,:]
            histlist.append(hist)
            colorlist.append('black')
            labellist.append(histlabel)
            transparencylist.append(1.)
            # then plot the automatically reconstructed histogram
            if isinstance(recohist,str):
                if not recohist in self.modelnames:
                    raise Exception('ERROR in HistStruct.plot_ls: no valid model provided.')
                classifier = self.models[recohist].classifiers[name]
                if not hasattr(classifier,'reconstruct'):
                    raise Exception('ERROR in HistStruct.plot_ls: automatic calculation of reco hist requires the classifiers '
                                   +'to have a method called "reconstruct", '
                                    +'but this does not seem to be the case for histogram type {}, '.format(name)
                                   +' whose classifier is of type {}'.format(type(classifier)))
                reco = classifier.reconstruct(hist)
                histlist.append(reco)
                colorlist.append('red')
                labellist.append(recohistlabel)
                transparencylist.append(1.)
            # or get the provided reconstructed histogram
            elif recohist is not None:
                reco = recohist[name]
                histlist.append(reco)
                colorlist.append('red')
                labellist.append(recohistlabel)
                transparencylist.append(1.)
            # re-plot original histogram on top but without label
            histlist.append(hist)
            colorlist.append('black')
            labellist.append(None)
            transparencylist.append(1.)
            
            # get the title and axes
            title = pu.make_text_latex_safe(name)
            if( titledict is not None and name in titledict ): title = titledict[name]
            extratext = None
            if( extratextdict is not None and name in extratextdict): extratext = extratextdict[name]
            xaxtitle = None
            if( xaxtitledict is not None and name in xaxtitledict ): xaxtitle = xaxtitledict[name]
            xlims = (-0.5,-1)
            if physicalxax: xlims = self.histranges[name]
            yaxtitle = None
            if( yaxtitledict is not None and name in yaxtitledict ): yaxtitle = yaxtitledict[name]
            # make the plot
            pu.plot_sets(histlist,
                  fig=fig,ax=axs[int(j/ncols),j%ncols],
                  title=title, extratext=extratext, xaxtitle=xaxtitle, xlims=xlims, yaxtitle=yaxtitle,
                  colorlist=colorlist, labellist=labellist, transparencylist=transparencylist,
                  **kwargs)
        return fig,axs


    def plot_ls_2d( self, runnb, lsnb, histnames=None, histlabel=None,
                 recohist=None, recohistlabel='Reconstruction',
                 titledict=None, xaxtitledict=None, yaxtitledict=None, **kwargs):
        ### plot the histograms in a HistStruct for a given run/ls number versus their reconstruction
        # internal helper function, use only via plot_ls
        
        # find index that given run and ls number correspond to
        index = self.get_index( runnb, lsnb )
        # initializations
        if recohist is None:
            ncols = min(4,len(histnames))
            nrows = int(math.ceil(len(histnames)/ncols))
        else:
            ncols = 2 
            nrows = len(histnames)
        fig,axs = plt.subplots(nrows,ncols,figsize=(5*ncols,5*nrows),squeeze=False)
        if histlabel is None: histlabel = 'Run: '+str(int(runnb))+', LS: '+str(int(lsnb))+')'
        # loop over all histograms belonging to this lumisection and make the plots
        for j,name in enumerate(histnames):
            # get the original histogram
            hist = self.histograms[name][index:index+1]
            reco = None
            # get the automatically reconstructed histogram
            if isinstance(recohist,str):
                if not recohist in self.modelnames:
                    raise Exception('ERROR in HistStruct.plot_ls: no valid model provided.')
                classifier = self.models[recohist].classifiers[name]
                if not hasattr(classifier,'reconstruct'):
                    raise Exception('ERROR in HistStruct.plot_ls: automatic calculation of reco hist requires the classifiers '
                                   +'to have a method called "reconstruct", '
                                   +'but this does not seem to be the case for histogram type {}, '.format(name)
                                   +' whose classifier is of type {}'.format(classifier))
                reco = classifier.reconstruct(hist)
            # get the provided reconstructed histogram
            elif recohist is not None:
                reco = recohist[name]
            # get the title and axes
            title = pu.make_text_latex_safe(name)
            if( titledict is not None and name in titledict ): title = titledict[name]
            xaxtitle = None
            if( xaxtitledict is not None and name in xaxtitledict ): xaxtitle = xaxtitledict[name]
            yaxtitle = None
            if( yaxtitledict is not None and name in yaxtitledict ): yaxtitle = yaxtitledict[name]
            # make the plot for original histogram
            row = j
            column = 0
            if recohist is None:
                row = int(j/ncols)
                column = j%ncols
            pu.plot_hist_2d( hist[0],
                  fig=fig,ax=axs[row,column],
                  title=title, xaxtitle=xaxtitle, yaxtitle=yaxtitle,
                  **kwargs)
            # add the label
            pu.add_text( axs[0,0], histlabel, (0.05, 1.1), fontsize=12,
                            background_edgecolor='black' )
            # make the plot for reconstructed histogram
            if recohist is not None:
                row = j
                column = 1
                pu.plot_hist_2d( reco[0],
                    fig=fig,ax=axs[row,column],
                    title=title, xaxtitle=xaxtitle, yaxtitle=yaxtitle,
                    **kwargs)
                # add the label
                pu.add_text( axs[0,1], recohistlabel, (0.05, 1.1), fontsize=12,
                            background_edgecolor='black' )
        return fig,axs
    
    def plot_ls_score( self, modelname, runnb, lsnb, ncols=4,
                       masknames=None, setnames=None, 
                       titledict=None, extratextdict=None, **kwargs ):
        ### plot the score of a given lumisection for each histogram type compared to reference scores
        # input arguments:
        # - modelname: name of the model for which to retrieve the score
        # - runnb: run number
        # - lsnb: lumisection number
        # - masknames: list of mask names for the reference scores
        # - setnames: list of set names for the reference scores
        # - titledict: dict mapping histogram names to titles for the subplots (default: title = histogram name)
        # - extratextdict: dict mapping histogram names to extra text on subplots (default: no extra text)
        # - kwargs: additional keyword arguments passed down to pu.plot_score_dist
        
        # find index that given run and ls number correspond to
        index = self.get_index( runnb, lsnb )
        # initializations
        ncols = min(ncols,len(self.histnames))
        nrows = int(math.ceil(len(self.histnames)/ncols))
        fig,axs = plt.subplots(nrows,ncols,figsize=(5*ncols,5*nrows),squeeze=False)
        lslabel = 'This LS (Run: '+str(int(runnb))+', LS: '+str(int(lsnb))+')'
        # get score for this lumisection
        scorepoint = self.get_scores_ls( modelname, runnb, lsnb )
        # loop over histogram types
        for j,histname in enumerate(self.histnames):
            # get reference scores
            scores = self.get_scores( modelname, histname=histname, 
                                      masknames=masknames, setnames=setnames )
            nscores = len(scores)
            labels = np.zeros(nscores)
            scores = np.concatenate((scores, np.ones(int(nscores/15))*scorepoint[histname]))
            labels = np.concatenate((labels, np.ones(int(nscores/15))))
            # get the title and axes
            title = pu.make_text_latex_safe(histname)
            if( titledict is not None and histname in titledict ): title = titledict[histname]
            extratext = None
            if( extratextdict is not None and histname in extratextdict): extratext = extratextdict[histname]
            pu.plot_score_dist( scores, labels,
                                fig=fig, ax=axs[int(j/ncols),j%ncols],
                                title=title, extratext=extratext,
                                **kwargs )
        return fig,axs
    
    #######################################
    # functions for plotting partial fits #
    #######################################
    
    def plot_partial_fit( self, modelname, dims, clusters, **kwargs):
        ### plot a partial fit calculated with train_partial_fitters
        # input arguments:
        # - modelname: name of the model for which to plot the partial fits
        # - dims: a tuple of length 1 or 2 with integer dimension indices
        #   note: the partial fit for this dimension must have been calculated with train_partial_fitters first
        # - clusters: a list of the different point clusters to plot
        #             each element in the list should be a dict of the form 
        #             {'masknames': [list of mask names]} or {'setnames': [list of set names]}
        # - kwargs: plot options passed down to pu.plot_fit_1d_clusters or pu.plot_fit_2d_clusters;
        #           some of them have to have the same length as clusters (e.g. colors and labels)
        
        # argument checking
        if not (isinstance(dims,list) or isinstance(dims,tuple)):
            raise Exception('ERROR in HistStruct.plot_partial_fit: dims argument must be tuple or list.'
                           +' but found {}'.format(type(dims)))
        if not self.check_model( modelname ):
            raise Exception('ERROR in HistStruct.plot_partial_fit: model "{}" is not in current histstruct.'.format(modelname))
        if dims not in self.models[modelname].partial_fitters.keys():
            raise Exception('ERROR in HistStruct.plot_partial_fit: dims "{}"'.format(dims)
                            +' not in list of dimensions for which a partial fit was trained.')
        if len(dims)==1:
            plotfunction = pu.plot_fit_1d_clusters
            plotdim = 1
        elif len(dims)==2:
            plotfunction = pu.plot_fit_2d_clusters
            plotdim = 2
        else:
            raise Exception('ERROR in HistStruct.plot_partial_fit: length of dims is {}'.format(len(dims))
                           +' while 1 or 2 is expected.')                 
        # get fitter
        fitter = self.models[modelname].partial_fitters[dims]
        # get training points
        training_points = self.models[modelname].fitscores_array[:,dims]
        if len(training_points.shape)==1: training_points = np.expand_dims(training_points,1)
        # get clusters
        cluster_points = []
        for cluster in clusters:
            points = self.get_scores_array( modelname, **cluster )[:,dims]
            cluster_points.append(points)
        # make the plot
        (fig,ax) = plotfunction( training_points, cluster_points, fitfunc=fitter, **kwargs)
        return (fig,ax)
    
    #################################################
    # functions for plotting 1D score distributions #
    #################################################
    
    def plot_score_dist( self, modelname, histname=None, 
                         masknames_sig=None, setnames_sig=None, 
                         masknames_bkg=None, setnames_bkg=None,
                         **kwargs ):
        ### plot a 1D score distribution
        # input arguments:
        # - modelname: name of the model for which to retrieve the scores
        # - histname: type of histogram for which to retrieve the scores
        #             if None, the global scores will be retrieved
        # - masknames_sig, setnames_sig: lists of mask or set names for signal distribution
        # - masknames_bkg, setnames_bkg: lists of mask or set names for background distribution
        #   note: in case of multiple masks, the intersection is taken (as usual);
        #         in case of multiple sets, the union is taken!
        # - kwargs: additional keyword arguments passed down to pu.plot_score_dist
        scores_sig_parts = []
        scores_bkg_parts = []
        if histname is not None:
            scores_sig = self.get_scores( modelname, histname=histname, 
                                          masknames=masknames_sig, setnames=setnames_sig )
            scores_bkg = self.get_scores( modelname, histname=histname, 
                                          masknames=masknames_bkg, setnames=setnames_bkg )
        else:
            scores_sig = self.get_globalscores( modelname, masknames=masknames_sig, setnames=setnames_sig )
            scores_bkg = self.get_globalscores( modelname, masknames=masknames_bkg, setnames=setnames_bkg )
        scores = np.concatenate((scores_sig, scores_bkg))
        labels = np.concatenate((np.ones(len(scores_sig)),np.zeros(len(scores_bkg))))
        fig,ax = pu.plot_score_dist(scores, labels, **kwargs)
        return (fig,ax)
