#!/usr/bin/env python
# coding: utf-8

# **HistStruct: consistent treatment of multiple histogram types**  
# 
# The HistStruct class is intended to be the main data structure used within this framework.
# A HistStruct object basically consists of a mutually consistent collection of numpy arrays, where each numpy array corresponds to one histogram type, with dimensions (number of histograms, number of bins). The HistStruct has functions to easily perform the following common tasks (among others):  
# - select a subset of runs and/or lumisections (e.g. using a custom or predefined json file formatted selector),
# - prepare the data for machine learning training, with all kinds of preprocessing,
# - evaluate classifiers (machine learning types or other).
# 
# Up to now the HistStruct is not used in many places, the main reason being that most of the tutorials for example were written (or at leasted started) before this class.  
# When only processing a single histogram type, the HistStruct might be a bit of an overkill and one could choose to operate on the dataframe directly.  
# However, especially when using multiple histogram types, the HistStruct is very handy to keep everything consistent.  
# 
# See the tutorial autoencoder_combine.ipynb for an important example!



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
        # a HistStruct object has the following properties:
        # histnames: list of histogram names
        # histograms: dict mapping histogram name to 2D numpy array of histograms (shape (nhists,nbins))
        # nentries: dict mapping histogram name to 1D numpy array of number of entries per histogram (same length as histograms)
        # histranges: dict mapping histogram name to tuple with (xmin, xmax)
        # runnbs: 1D numpy array of run numbers (same length as histograms)
        # lsnbs: 1D numpy array of lumisection numbers (same length as histograms)
        # globalscores: 1D numpy array of global score per lumisection (same length as histograms)
        # classifiers: dict mapping histogram name to object of type HistogramClassifier
        # scores: dict mapping histogram name to 1D numpy array of values associated to the histograms (same length as histograms)
        # masks: dict mapping name to 1D numpy array of booleans (same length as histograms) that can be used for masking
        # exthistograms: dict of dicts similar to histograms for additional (e.g. artificially generated) histograms
        # extscores: dict of dicts similar to scores for additional (e.g. artificially generated) histograms
        # extglobalscores: dict of lists similar to scores for additional (e.g. artificially generated) histograms
        self.histnames = []
        self.histograms = {}
        self.nentries = {}
        self.histranges = {}
        self.runnbs = []
        self.lsnbs = []
        self.globalscores = []
        self.classifiers = {}
        self.scores = {}
        self.masks = {}
        self.exthistograms = {}
        self.extscores = {}
        self.extglobalscores = {}
        
    def __str__( self ):
        ### get a printable representation of a HistStruct
        info = ''
        # histogram names:
        info += '- histogram types ({}): \n'.format(len(self.histnames))
        for histname in self.histnames:
            info += '  -- {} (nbins: {})\n'.format(histname,self.histograms[histname].shape[1:])
        # size of histogram sets:
        info += '- number of lumisections: {}\n'.format(len(self.lsnbs))
        # masks:
        info += '- masks ({}): \n'.format(len(self.get_masknames()))
        for maskname in self.get_masknames(): info += '  -- {}\n'.format(maskname)
        # classifiers:
        info += '- classifiers: \n'
        if len(self.classifiers.keys())==0: info += '    not initialized \n'
        else:
            for histname in self.histnames: info += '  -- {}: {}\n'.format(histname,type(self.classifiers[histname]))
        # scores:
        info += '- scores: \n'
        if len(self.scores.keys())==0: info += '    not initialized \n'
        else:
            for histname in self.histnames: 
                txt = 'initialized' if histname in self.scores.keys() else 'not initialized'
                info += '  -- {}: {}\n'.format(histname,txt)
        info += '- global scores: \n'
        if len(self.globalscores)==0: info += '    not initialized \n'
        else: info += '    initialized \n'
        # extra histograms sets
        info += '- extra histogram sets: {}\n'.format(len(self.exthistograms.keys()))
        for extname in self.exthistograms.keys(): 
            info += '  -- {}\n'.format(extname)
            info += '     --- histogram types ({})\n'.format(len(self.exthistograms[extname].keys()))
            for histname in self.exthistograms[extname].keys():
                info += '         ---- {} (shape: {})\n'.format(histname, self.exthistograms[extname][histname].shape)
            info += '     --- scores: \n'
            if extname not in self.extscores.keys(): info += '           not initialized \n'
            else:
                for histname in self.exthistograms[extname].keys(): 
                    txt = 'initialized' if histname in self.extscores[extname].keys() else 'not initialized'
                    info += '         ---- {}: {}\n'.format(histname,txt)
            info += '     --- global scores: \n'
            if extname not in self.extglobalscores.keys(): info += '           not initialized \n'
            else: info += '           initialized \n'
        return info
        
    def save( self, path, save_classifiers=True ):
        ### save a HistStruct object to a pkl file
        # input arguments:
        # - path where to store the file (appendix .zip is automatically appended)
        # - save_classifiers: a boolean whether to include the classifiers if present in the HistStruct
        pklpath = os.path.splitext(path)[0]+'.pkl'
        zippath = os.path.splitext(path)[0]+'.zip'
        cpath = os.path.splitext(path)[0]+'_classifiers_storage'
        rootpath = os.path.dirname(path)
        zipcontents = {}
        # case where classifiers should not be stored
        if( len(self.classifiers.keys())==0 or not save_classifiers ):
            classifiers = dict(self.classifiers)
            self.classifiers = {}
            with open(pklpath,'wb') as f:
                pickle.dump(self,f)
            self.classifiers = classifiers
            zipcontents[pklpath] = os.path.relpath(pklpath, start=rootpath)
        # case where classifiers should be stored
        else:
            # save the classifiers and store the types in the HistStruct
            self.classifier_types = {}
            self.classifier_histnames = []
            for histname,classifier in self.classifiers.items():
                classifier.save( os.path.join(cpath,histname) )
                self.classifier_types[histname] = type(classifier)
                self.classifier_histnames.append(histname)
            # get all files to store in the zip file
            for root, dirs, files in os.walk(cpath):
                for name in files:
                    thispath = os.path.join(root, name)
                    zipcontents[thispath] = os.path.relpath(thispath, start=rootpath)
            # remove the classifiers and pickle the rest
            self.classifiers = {}
            with open(pklpath,'wb') as f:
                pickle.dump(self,f)
            zipcontents[pklpath] = os.path.relpath(pklpath, start=rootpath)
            # restore the classifiers
            for histname in self.classifier_histnames:
                self.classifiers[histname] = self.classifier_types[histname].load( os.path.join(cpath,histname) )
        # put everything in a zip file
        with zipfile.ZipFile( zippath, 'w' ) as zipf:
            for f, fname in zipcontents.items(): zipf.write(f, fname)
        # remove individual files
        for f in zipcontents: os.system('rm {}'.format(f))
        if os.path.exists(cpath): os.system('rm -r {}'.format(cpath))
            
    @classmethod
    def load( self, path, load_classifiers=True, verbose=False ):
        ### load a HistStruct object from a pkl file
        # input arguments:
        # - path to a zip file containing a HistStruct object
        # - load_classifiers: a boolean whether to load the classifiers if present
        # - verbose: boolean whether to print some information
        zippath = os.path.splitext(path)[0]+'.zip'
        unzippath = os.path.splitext(path)[0]+'_unzipped'
        basename = os.path.splitext(os.path.basename(zippath))[0]
        pklbasename = basename+'.pkl'
        cbasename = basename+'_classifiers_storage'
        zipcontents = []
        # extract the zip file
        with zipfile.ZipFile( zippath, 'r' ) as zipf:
            zipcontents = zipf.namelist()
            zipf.extractall( path=unzippath )
        with open(os.path.join(unzippath,pklbasename),'rb') as f:
            obj = pickle.load(f)
        if( load_classifiers ):
            if len(zipcontents)==1:
                print('WARNING: requested to load classifiers, but this stored HistStruct object does not seem to contain any.')
            else:
                for histname in obj.classifier_histnames:
                    obj.classifiers[histname] = obj.classifier_types[histname].load( os.path.join(unzippath,cbasename,histname) )
        # remove individual files
        if os.path.exists(unzippath): os.system('rm -r {}'.format(unzippath))
        if verbose:
            print('Loaded a HistStruct object with following properties:')
            print(obj)
        return obj
        
    def add_dataframe( self, df, cropslices=None, rebinningfactor=None, 
                        smoothinghalfwindow=None, smoothingweights=None,
                        donormalize=True ):
        ### add a dataframe to a HistStruct
        # input arguments:
        # - df: a pandas dataframe as read from the input csv files
        # - cropslices: list of slices (one per dimension) by which to crop the histograms
        # - rebinningfactor: factor by which to group bins together
        # - smoothinghalfwindow: half window (int for 1D, tuple for 2D) for doing smoothing of histograms
        # - smoothingweights: weight array (1D for 1D, 2D for 2D) for smoothing of histograms
        # - donormalize: boolean whether to normalize the histograms
        # for more details on cropslices, rebinningfactor, smoothingwindow, smoothingweights
        # and donormalize: see hist_utils.py!
        # notes:
        # - the new dataframe can contain one or multiple histogram types
        # - the new dataframe must contain the same run and lumisection numbers (for each histogram type in it)
        #   as already present in the HistStruct, except if it is the first one to be added
        # - alternative to adding the dataframe with the options cropslices, donormalize and rebinningfactor
        #   (that will be passed down to preparedatafromdf), one can also call preparedatafromdf manually and add it
        #   with add_histograms, allowing for more control over complicated preprocessing.
        
        histnames = dfu.get_histnames(df)
        # loop over all names in the dataframe
        for histname in histnames:
            if histname in self.histnames:
                raise Exception('ERROR in HistStruct.add_dataframe: dataframe contains histogram name '.format(histname)
                               +' but this is already present in the current HistStruct.')
            thisdf = dfu.select_histnames( df, [histname] )
            # determine statistics (must be done before normalizing)
            nentries = np.array(thisdf['entries'])
            # get physical xmin and xmax
            xmin = thisdf.at[0, 'Xmin']
            xmax = thisdf.at[0, 'Xmax']
            # prepare the data
            (hists_all,runnbs_all,lsnbs_all) = hu.preparedatafromdf(thisdf,returnrunls=True,
                                                cropslices=cropslices,
                                                rebinningfactor=rebinningfactor,
                                                smoothinghalfwindow=smoothinghalfwindow,
                                                smoothingweights=smoothingweights,
                                                donormalize=donormalize)
            runnbs_all = runnbs_all.astype(int)
            lsnbs_all = lsnbs_all.astype(int)
            # check consistency in run and lumisection numbers
            if len(self.histnames)!=0:
                if( not ( (runnbs_all==self.runnbs).all() and (lsnbs_all==self.lsnbs).all() ) ):
                    raise Exception('ERROR in HistStruct.add_dataframe: dataframe run/lumi numbers are not consistent with current HistStruct!')
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
        # input arguments:
        # - globalscores: 1D numpy array of scores (must have same length as lumisection and run numbers)
        
        if( len(globalscores)!=len(self.lsnbs) ):
            raise Exception('ERROR in HistStruct.add_globalscores: length of globalscores ({})'.format(len(globalscores))
                           +' does not match length of list of lumisections ({})'.format(len(self.lsnbs)))
        if( len(self.globalscores)>0 ):
            print('WARNING in HistStruct.add_globalscores: array of global scores appears to be already initialized; overwriting...')
        self.globalscores = globalscores
        
    def add_extglobalscores( self, extname, globalscores ):
        ### add an array of global scores (one per lumisection) for a specified extra set of histograms in the HistStruct
        # input arguments:
        # - extname: name of extra histogram set
        # - globalscores: 1D numpy array of scores
        # note: this function checks if all histogram types in this set contain the same number of histograms,
        #       (and that this number corresponds to the length of globalscores)
        #       else adding globalscores is meaningless
        
        if not extname in self.exthistograms.keys():
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
        self.extglobalscores[extname] = globalscores

    def get_globalscores_jsonformat( self, working_point=None ):
        ### make a json format listing all lumisections in this histstruct
        # the output list has entries for global score, pass/fail given working point, and masks
        # input arguments:
        # - working_point: if present, an entry will be made for each lumisection whether it passes this working point
        res = []
        for (runnb,lsnb) in zip(self.runnbs,self.lsnbs):
            res.append({'run':int(runnb), 'ls':int(lsnb)})
            # (note: need explicit conversion to int since numpy data types are not understood by json serializer)
        if( len(self.globalscores)==0 ):
            print('WARNING in HistStruct.get_globalscores_json: array of global scores seems to be uninitialized.')
        else:
            for idx in range(len(res)): 
                if working_point is not None: res[idx]['pass'] = bool(self.globalscores[idx]>working_point)
                res[idx]['score'] = float(self.globalscores[idx])
        for maskname in self.masks.keys():
            for idx in range(len(res)):
                res[idx][maskname] = bool(self.masks[maskname][idx])
        return res
        
    def add_exthistograms( self, extname, histname, histograms, overwrite=False ):
        ### add a set of extra histograms to a HistStruct
        # these histograms are not assumed to correspond to physical run/lumisections numbers (e.g. resampled ones),
        # and no consistency checks are done
        # input arguments:
        # - extname: name of the extra histogram set (you can add multiple, e.g. resampled_good, resampled_bad and/or resampled_training)
        # - histname: name of the histogram type
        # - histograms: a numpy array of shape (nhistograms,nbins)
        # - overwrite: boolean whether to overwrite a set of histograms of the same name if present (default: raise exception)
        if extname in self.exthistograms.keys():
            if histname in self.exthistograms[extname].keys():
                if not overwrite:
                    raise Exception('ERROR in HistStruct.add_exthistograms: histogram name is {}'.format(histname)
                                   +' but this is already present in the set of extra histogram with name {}'.format(extname))
        else: 
            self.exthistograms[extname] = {}
            self.extscores[extname] = {}
        self.exthistograms[extname][histname] = histograms
    
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
        # input arguments:
        # - name: a name for the mask
        # - jsondict: a dictionary in typical json format (see the golden json file for inspiration)
        # all lumisections present in the jsondict will be masked True, the others False.
        mask = jsonu.injson( self.runnbs, self.lsnbs, jsondict=jsondict )
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
        # - min_entries_to_bins_ratio: number of entries divided by number of bins, lower boundary for statistics (default: no lower boundary)
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
        
    def get_combined_mask( self, names ):
        ### get a combined mask given multiple mask names
        # mostly for internal use; externally you can use get_histograms( histname, <list of mask names>) directly
        mask = np.ones(len(self.runnbs)).astype(bool)
        for name in names:
            if name not in self.masks.keys():
                raise Exception('ERROR in HistStruct.get_combined_mask: mask {} requested but not found.'.format(name))
            mask = mask & self.masks[name]
        return mask
    
    def get_masknames( self ):
        ### return a simple list of all mask names in the current HistStruct
        return list( self.masks.keys() )
        
    def get_runnbs( self, masknames=None ):
        ### get the array of run numbers, optionally after masking
        # input arguments:
        # - masknames: list of names of masks (default: no masking, return full array)
        if masknames is None: return self.runnbs[:]
        return self.runnbs[ self.get_combined_mask(masknames) ]

    def get_runnbs_unique( self ):
        ### get a list of unique run numbers
        res = []
        for runnb in self.runnbs:
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
            raise Exception('ERROR in HistStruct.get_index: unexpected index of requested run/lumisection: found {}.'.format(index)
                           +' Is the requested run/lumisection in the HistStruct?')
        (index,) = index
        return index
    
    def get_scores( self, histname=None, masknames=None ):
        ### get the array of scores for a given histogram type, optionally after masking
        # input arguments:
        # - histname: name of the histogram type for which to retrieve the score. 
        #   if None, return a dict matching histnames to arrays of scores
        # - masknames: list of names of masks (default: no masking, return full array)
        # notes:
        # - this method takes the scores from the HistStruct.scores attribute;
        #   make sure to have evaluated the classifiers before calling this method,
        #   else an exception will be thrown.
        histnames = self.histnames[:]
        if histname is not None:
            # check if histname is valid
            if histname not in self.histnames:
                raise Exception('ERROR in HistStruct.get_scores: requested histogram name {}'.format(histname)
                               +' but this is not present in the current HistStruct.')
            if histname not in self.scores.keys():
                raise Exception('ERROR in HistStruct.get_scores: requested histogram name {}'.format(histname)
                               +' but the scores for this histogram type were not yet initialized.')
            histnames = [histname]
        mask = np.ones(len(self.lsnbs)).astype(bool)
        if masknames is not None:
            mask = self.get_combined_mask(masknames)
        res = {}
        for hname in histnames:
            res[hname] = self.scores[hname][mask]
        if histname is None: return res
        return res[histname]
    
    def get_scores_array( self, masknames=None ):
        ### similar to get_scores, but with different return type:
        # np array of shape (nhistograms, nhistogramtypes)
        scores = self.get_scores( masknames=masknames )
        scores_array = []
        for histname in self.histnames:
            scores_array.append(scores[histname])
        scores_array = np.transpose(np.array(scores_array))
        return scores_array
    
    def get_extscores( self, extname, histname=None ):
        ### get the array of scores for a given histogram type in a given extra set.
        # input arguments:
        # - extname: name of the extra set (see also add_exthistograms)
        # - histname: name of the histogram type for which to retrieve the score. 
        #   if None, return a dict matching histnames to arrays of scores
        # notes:
        # - this method takes the scores from the HistStruct.extscores attribute;
        #   make sure to have evaluated the classifiers before calling this method,
        #   else an exception will be thrown.
        if not extname in self.exthistograms.keys():
            raise Exception('ERROR in HistStruct.get_extscores: requested to retrieve scores for set {}'.format(extname)
                           +' but this is not present in the current HistStruct.')
        histnames = self.histnames[:]
        if histname is not None:
            # check if histname is valid
            if histname not in self.exthistograms[extname].keys():
                raise Exception('ERROR in HistStruct.get_extscores: requested histogram name {}'.format(histname)
                               +' but this is not present in the extra set with name {}.'.format(extname))
            if histname not in self.extscores[extname].keys():
                raise Exception('ERROR in HistStruct.get_extscores: requested histogram name {}'.format(histname)
                               +' in extra set with name {}'.format(extname)
                               +' but the scores for this histogram type were not yet initialized.')
            histnames = [histname]
        res = {}
        for hname in histnames:
            res[hname] = self.extscores[extname][hname]
        if histname is None: return res
        return res[histname]
    
    def get_extscores_array( self, extname ):
        ### similar to get_extscores, but with different return type:
        # np array of shape (nhistograms, nhistogramtypes)
        scores = self.get_extscores( extname )
        scores_array = []
        for histname in self.histnames:
            scores_array.append(scores[histname])
        scores_array = np.transpose(np.array(scores_array))
        return scores_array
    
    def get_scores_ls( self, runnb, lsnb, histnames=None, suppresswarnings=False ):
        ### get the scores for a given run/lumisection number and for given histogram names
        # input arguments:
        # - runnb: run number
        # - lsnb: lumisection number
        # - histnames: names of the histogram types for which to retrieve the score. 
        # returns:
        # - a dict matching each name in histnames to a score (or None if no valid score)
        # notes:
        # - this method takes the scores from the HistStruct.scores attribute;
        #   make sure to have evaluated the classifiers before calling this method,
        #   else the returned scores will be None.
        if histnames is None: histnames = self.histnames
        scores = {}
        index = self.get_index(runnb,lsnb)
        for histname in histnames:
            # check if histname is valid
            if histname not in self.histnames:
                raise Exception('ERROR in HistStruct.get_scores_ls: requested histogram name {}'.format(histname)
                               +' but this is not present in the current HistStruct.')
            if histname not in self.scores.keys():
                if not suppresswarnings: 
                    print('WARNING in HistStruct.get_scores_ls: requested histogram name {}'.format(histname)
                         +' but no score for this type is present, setting score to None')
                scores[histname] = None
            # retrieve score
            else:
                scores[histname] = self.scores[histname][index]
        return scores
    
    def get_globalscores( self, masknames=None ):
        ### get the array of global scores, optionally after masking
        # input arguments:
        # - masknames: list of names of masks (default: no masking, return full array)
        # notes:
        # - this method takes the scores from the HistStruct.globalscores attribute;
        #   make sure to have set this attribute with add_globalscores,
        #   else an exception will be thrown.
        if( len(self.globalscores)==0 ):
            raise Exception('ERROR in HistStruct.get_globalscores: array of global scores seems to be uninitialized.')
        mask = np.ones(len(self.lsnbs)).astype(bool)
        if masknames is not None:
            mask = self.get_combined_mask(masknames)
        return self.globalscores[mask]
    
    def get_globalscore_ls( self, runnb, lsnb ):
        ### get the global score for a given run/lumisection number
        # input arguments:
        # - runnb: run number
        # - lsnb: lumisection number
        # - histnames: names of the histogram types for which to retrieve the score. 
        # returns:
        # - a dict matching each name in histnames to a score (or None if no valid score)
        # notes:
        # - this method takes the scores from the HistStruct.scores attribute;
        #   make sure to have evaluated the classifiers before calling this method,
        #   else the returned scores will be None.
        index = self.get_index(runnb,lsnb)
        if( len(self.globalscores)==0 ):
            raise Exception('ERROR in HistStruct.get_globalscore_ls: array of global scores seems to be uninitialized.')
        return self.globalscores[index]
    
    def get_extglobalscores( self, extname ):
        ### get the array of global scores for one of the extra histogram sets
        # input arguments:
        # - extname: name of the extra histogram set
        # notes:
        # - this method takes the scores from the HistStruct.extglobalscores attribute;
        #   make sure to have set this attribute with add_extglobalscores,
        #   else an exception will be thrown.
        if extname not in self.extglobalscores.keys():
            raise Exception('ERROR in HistStruct.get_extglobalscores: requested to retrieve global scores for set {}'.format(extname)
                           +' but they are not present in the current HistStruct.')
        return self.extglobalscores[extname]
    
    def get_histograms( self, histname=None, masknames=None ):
        ### get the array of histograms for a given type, optionally after masking
        # input arguments:
        # - histname: name of the histogram type to retrieve 
        #   if None, return a dict matching histnames to arrays of histograms
        # - masknames: list of names of masks (default: no masking, return full array)
        histnames = self.histnames[:]
        if histname is not None:
            # check if histname is valid
            if histname not in self.histnames:
                raise Exception('ERROR in HistStruct.get_histograms: requested histogram name {}'.format(histname)
                               +' but this is not present in the current HistStruct.')
            histnames = [histname]
        mask = np.ones(len(self.lsnbs)).astype(bool)
        if masknames is not None:
            mask = self.get_combined_mask(masknames)
        res = {}
        for hname in histnames:
            res[hname] = self.histograms[hname][mask]
        if histname is None: return res
        return res[histname]
    
    def get_exthistograms( self, extname, histname=None ):
        ### get the array of extra histograms for a given set name and type name
        # input arguments:
        # - extname: name of the set of extra histograms (see also add_exthistograms)
        # - histname: name of the histogram type to retrieve 
        #   if None, return a dict matching histnames to arrays of histograms
        if not extname in self.exthistograms.keys():
            raise Exception('ERROR in HistStruct.get_exthistograms: requested to retrieve histograms in set {}'.format(extname)
                           +' but this is not present in the current HistStruct.')
        histnames = self.histnames[:]
        if histname is not None:
            # check if histname is valid
            if histname not in self.exthistograms[extname].keys():
                raise Exception('ERROR in HistStruct.get_exthistograms: requested histogram name {}'.format(histname)
                               +' but this is not present in the extra set with name {}.'.format(extname))
            histnames = [histname]
        res = {}
        for hname in histnames:
            res[hname] = self.exthistograms[extname][hname]
        if histname is None: return res
        return res[histname]

    def get_histogramsandscores( self, extname=None, masknames=None, nrandoms=-1, nfirst=-1 ):
        ### combination of get_histograms, get_scores and get_globalscores with additional options
        # - extname: use an extended histogram set
        # - nrandoms: if > 0, number of random instances to draw
        # - nfirst: if > 0, number of first instances to keep
        # return type:
        # dict with keys 'histograms', 'scores' and 'globalscores'
        # note that the values of scores and globalscores may be None if not initialized

        histograms = None
        scores = None
        globalscores = None

        # case of non-extended
        if extname is None:
            histograms = self.get_histograms(masknames=masknames)
            if len(self.scores.keys())>0:
                scores = self.get_scores(masknames=masknames)
            if len(self.globalscores)>0: 
                globalscores = self.get_globalscores(masknames=masknames)
        # case of extended
        else:
            histograms = self.get_exthistograms(extname)
            if( extname in self.extscores.keys()
                and len(self.extscores[extname].keys())>0 ):
                scores = self.get_extscores(extname)
            if extname in self.extglobalscores.keys():
                globalscores = self.histstruct.get_extglobalscores(extname)
        # rest of the processing is similar
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
    
    def add_classifier( self, histname, classifier, evaluate=False ):
        ### add a histogram classifier for a given histogram name to the HistStruct
        # input arguments:
        # - histname: a valid histogram name present in the HistStruct to which this classifier applies
        # - classifier: an object of type HistogramClassifier (i.e. of any class that derives from it)
        # - evaluate: a bool whether to evaluate the classifier (and store the result in the 'scores' attribute)
        #   if set to True, the result is both returned and stored in the 'scores' attribute.
        if not histname in self.histnames:
            raise Exception('ERROR in HistStruct.add_classifier: requested to add classifier for {}'.format(histname)
                           +' but there is n entry in the HistStruct with that name.')
        if not isinstance(classifier,HistogramClassifier):
            raise Exception('ERROR in HistStruct.add_classifier: classifier is of type {}'.format(type(classifier))
                           +' while a HistogramClassifier object is expected.')
        self.classifiers[histname] = classifier
        if evaluate:
            return self.evaluate_classifier( histname )
        
    def evaluate_classifier( self, histname, extname=None ):
        ### evaluate a histogram classifier for a given histogram name in the HistStruct
        # input arguments:
        # - histname: a valid histogram name present in the HistStruct for which to evaluate the classifier
        # - extname: name of a set of extra histograms (see add_exthistograms)
        #            if None, will evaluate the classifer for the main set of histograms
        # notes:
        # - the result is both returned and stored in the 'scores' attribute
        
        if not histname in self.classifiers.keys():
                raise Exception('ERROR in HistStruct.evaluate_classifier: requested to evaluate classifier for {}'.format(histname)
                           +' but no classifier was set for this histogram type.')
        if extname is None:
            if histname not in self.histnames:
                raise Exception('ERROR in HistStruct.evaluate_classifier: requested histogram name {}'.format(histname)
                            +' but this is not present in the current HistStruct.')
            scores = self.classifiers[histname].evaluate(self.histograms[histname])
            self.scores[histname] = scores
        else:
            if not extname in self.exthistograms.keys():
                raise Exception('ERROR in HistStruct.get_evaluate_classifier: requested to retrieve histograms in set {}'.format(extname)
                           +' but this is not present in the current HistStruct.')
            if histname not in self.exthistograms[extname].keys():
                raise Exception('ERROR in HistStruct.get_evaluate_classifier: requested histogram name {}'.format(histname)
                               +' but this is not present in the extra set with name {}.'.format(extname))
            scores = self.classifiers[histname].evaluate(self.exthistograms[extname][histname])
            self.extscores[extname][histname] = scores
        return scores


    def plot_histograms( self, histnames=None, masknames=None, histograms=None, 
                            colorlist=[], labellist=[], transparencylist=[], 
                            titledict=None, xaxtitledict=None, physicalxax=False, 
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
        # - colorlist: list of matplotlib colors, must have same length as masknames
        # - labellist: list of labels for the legend, must have same legnth as masknames
        # - transparencylist: list of transparency values, must have same length as masknames
        # - titledict: dict mapping histogram names to titles for the subplots (default: title = histogram name)
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
                raise Exception('ERROR in HistStruct.plot_ls:'
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
            fig1d,axs1d = self.plot_histograms_1d( histnames=histnames1d, masknames=masknames, 
                            histograms=histograms,
                            colorlist=colorlist, labellist=labellist, transparencylist=transparencylist,
                            titledict=titledict, xaxtitledict=xaxtitledict, physicalxax=physicalxax, 
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


    def plot_histograms_1d( self, histnames=None, masknames=None, histograms=None, 
                            colorlist=[], labellist=[], transparencylist=[],
                            titledict=None, xaxtitledict=None, physicalxax=False, yaxtitledict=None, 
                            **kwargs ):
        ### plot the histograms in a histstruct, optionally after masking
        # internal helper function, use only via plot_histograms
        
        # initializations
        ncols = min(4,len(histnames))
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
            xaxtitle = None
            if( xaxtitledict is not None and name in xaxtitledict ): xaxtitle = xaxtitledict[name]
            xlims = (-0.5,-1)
            if physicalxax: xlims = self.histranges[name]
            yaxtitle = None
            if( yaxtitledict is not None and name in yaxtitledict ): yaxtitle = yaxtitledict[name]
            # make the plot
            pu.plot_sets( histlist,
                        fig=fig,ax=axs[int(j/ncols),j%ncols],
                        title=title, xaxtitle=xaxtitle, xlims=xlims, yaxtitle=yaxtitle,
                        colorlist=colorlist, labellist=labellist, transparencylist=transparencylist,
                        **kwargs )
        return fig,axs


    def plot_histograms_2d( self, histnames=None, masknames=None, histograms=None,
                            labellist=[], titledict=None, xaxtitledict=None, yaxtitledict=None,
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
    

    def plot_ls( self, runnb, lsnb, histnames=None, histlabel=None, 
                 recohist=None, recohistlabel='Reconstruction', 
                 refhists=None, refhistslabel='Reference histograms', refhiststransparency=None,
                 titledict=None, xaxtitledict=None, physicalxax=False, yaxtitledict=None, **kwargs):
        ### plot the histograms in a HistStruct for a given run/ls number versus their references and/or their reconstruction
        # note: so far only for 1D histograms.
        #       case of 2D histograms requires different plotting method since they cannot be clearly overlaid.
        #       if a HistStruct contains both 1D and 2D histograms, the 1D histograms must be selected with the histnames argument.
        # input arguments:
        # - runnb: run number
        # - lsnb: lumisection number
        # - histnames: names of histogram types to plot (default: all)
        # - histlabel: legend entry for the histogram (default: run and lumisection number)
        # - recohist: dict matching histogram names to reconstructed histograms
        #   notes: - 'reconstructed histograms' refers to e.g. autoencoder or NMF reconstructions;
        #            some models (e.g. simply looking at histogram moments) might not have this kind of reconstruction
        #          - in principle one histogram per key is expected, but still the the shape must be 2D (i.e. (1,nbins))
        #          - in case recohist is set to 'auto', the reconstruction is calculated on the fly for the input histograms
        # - recohistlabel: legend entry for the reco histograms
        # - refhists: dict matching histogram names to reference histograms
        #   notes: - multiple histograms (i.e. a 2D array) per key are expected;
        #            in case there is only one reference histogram, it must be reshaped into (1,nbins)
        # - refhistslabel: legend entry for the reference histograms
        # - titledict: dict mapping histogram names to titles for the subplots (default: title = histogram name)
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
            if( recohist=='auto' ):
                if( histname not in self.classifiers.keys() ): 
                    raise Exception('ERROR in HistStruct.plot_ls: auto reco requested, but histogram type {}'.format(histname)
                                    +' does not seem to have a classifier yet.')
            elif( recohist is not None ):
                if( histname not in recohist.keys() ):
                    raise Exception('ERROR in HistStruct.plot_ls: reco histograms provided, but type {}'.format(histname)
                                    +' seems to be missing.')
                if( not isinstance(recohist[histname], np.ndarray) ):
                    raise Exception('ERROR: recohist has unexpected structure, it is supposed to be a dict matching strings to 2D numpy arrays')
            if( refhists is not None and histname not in refhists.keys() ):
                raise Exception('ERROR in HistStruct.plot_ls: reference histograms provided, but type {}'.format(histname)
                               +' seems to be missing.')
                if( not isinstance(refhists[histname], np.ndarray) ):
                    raise Exception('ERROR: refhists has unexpected structure, it is supposed to be a dict matching strings to 2D numpy arrays')
            if len(self.histograms[histname].shape)==2: histnames1d.append(histname)
            elif len(self.histograms[histname].shape)==3: histnames2d.append(histname)

        # initializations
        fig1d = None
        axs1d = None
        fig2d = None
        axs2d = None

        # case of 1D histograms
        if len(histnames1d)>0:
            (fig1d, axs1d) = self.plot_ls_1d( runnb, lsnb, histnames=histnames, histlabel=histlabel,
                                recohist=recohist, recohistlabel=recohistlabel,
                                refhists=refhists, refhistslabel=refhistslabel, 
                                refhiststransparency=refhiststransparency,
                                titledict=titledict, xaxtitledict=xaxtitledict, physicalxax=physicalxax, 
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


    def plot_run( self, runnb, masknames=None, recohist=None, recohistlabel='reco', refhists=None, refhistslabel='reference', doprint=False):
        ### call plot_ls for all lumisections in a given run
        runnbs = self.get_runnbs( masknames=masknames )
        lsnbs = self.get_lsnbs( masknames=masknames )
        runsel = np.where(runnbs==runnb)
        lsnbs = lsnbs[runsel]
        print('plotting {} lumisections...'.format(len(lsnbs)))
        for lsnb in lsnbs:
            _ = self.plot_ls(runnb, lsnb, recohist=recohist, recohistlabel=recohistlabel, refhists=refhists, refhistslabel=refhistslabel)


    def plot_ls_1d( self, runnb, lsnb, histnames=None, histlabel=None,
                 recohist=None, recohistlabel='Reconstruction',
                 refhists=None, refhistslabel='Reference histograms', refhiststransparency=None,
                 titledict=None, xaxtitledict=None, physicalxax=False, yaxtitledict=None, **kwargs):
        ### plot the histograms in a HistStruct for a given run/ls number versus their references and/or their reconstruction
        # internal helper function, use only via plot_ls

        # find index that given run and ls number correspond to
        index = self.get_index( runnb, lsnb )
        # initializations
        ncols = min(4,len(histnames))
        nrows = int(math.ceil(len(histnames)/ncols))
        fig,axs = plt.subplots(nrows,ncols,figsize=(5*ncols,5*nrows),squeeze=False)
        if histlabel is None: histlabel = 'Run: '+str(int(runnb))+', LS: '+str(int(lsnb))+')'
        # loop over all histograms belonging to this lumisection and make the plots
        for j,name in enumerate(histnames):
            # get the original histogram
            hist = self.histograms[name][index:index+1,:]
            histlist = [hist]
            colorlist = ['black']
            labellist = [histlabel]
            transparencylist = [1.]
            # get the automatically reconstructed histogram
            if recohist=='auto':
                if not hasattr(self.classifiers[name],'reconstruct'):
                    raise Exception('ERROR in HistStruct.plot_ls: automatic calculation of reco hist requires the classifiers '
                                   +'to have a method called "reconstruct", but this does not seem to be the case for histogram type {}, '.format(name)
                                   +' whose classifier is of type {}'.format(type(self.classifiers[name])))
                reco = self.classifiers[name].reconstruct(hist)
                histlist.insert(0,reco)
                colorlist.insert(0,'red')
                labellist.insert(0,recohistlabel)
                transparencylist.insert(0,1.)
            # get the provided reconstructed histogram
            elif recohist is not None:
                reco = recohist[name]
                histlist.insert(0,reco)
                colorlist.insert(0,'red')
                labellist.insert(0,recohistlabel)
                transparencylist.insert(0,1.)
            # get the provided reference histograms
            if refhists is not None:
                histlist.insert(0,refhists[name])
                colorlist.insert(0,'blue')
                labellist.insert(0,refhistslabel)
                if refhiststransparency is None: refhiststransparency=0.3
                transparencylist.insert(0,refhiststransparency)
            # get the title and axes
            title = pu.make_text_latex_safe(name)
            if( titledict is not None and name in titledict ): title = titledict[name]
            xaxtitle = None
            if( xaxtitledict is not None and name in xaxtitledict ): xaxtitle = xaxtitledict[name]
            xlims = (-0.5,-1)
            if physicalxax: xlims = self.histranges[name]
            yaxtitle = None
            if( yaxtitledict is not None and name in yaxtitledict ): yaxtitle = yaxtitledict[name]
            # make the plot
            pu.plot_sets(histlist,
                  fig=fig,ax=axs[int(j/ncols),j%ncols],
                  title=title, xaxtitle=xaxtitle, xlims=xlims, yaxtitle=yaxtitle,
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
            if recohist=='auto':
                if not hasattr(self.classifiers[name],'reconstruct'):
                    raise Exception('ERROR in HistStruct.plot_ls: automatic calculation of reco hist requires the classifiers '
                                   +'to have a method called "reconstruct", but this does not seem to be the case for histogram type {}, '.format(name)
                                   +' whose classifier is of type {}'.format(type(self.classifiers[name])))
                reco = self.classifiers[name].reconstruct(hist)
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
