#!/usr/bin/env python
# coding: utf-8



### imports

# external modules
import numpy as np
import matplotlib.pyplot as plt
import importlib

# local modules
import csv_utils
import json_utils
import dataframe_utils
import hist_utils
import clustering_utils
import generate_data_utils
import autoencoder_utils




### loading the main datastructure combining the information from several histograms

class histstructure:
    
    def __init__(self):
        self.names = [] # list of histogram names
        self.histograms = {} # dict mapping histogram name to 2D numpy array of histograms
        self.entries = {} # dict mapping histogram name to 1D numpy array of number of entries per histogram
        self.runnbs = [] # 1D numpy array of run numbers (same length as histograms)
        self.lsnbs = [] # 1D numpy array of lumisection numbers (same length as histograms)
        self.custom = {} # this dict remains empty, so the histstruct can be extended at runtime
    
    def create(self,year,histnames,jsonselector=None,highstatonly=False,dcsononly=False):
        ### create the histstructure given the arguments provided
        # most arguments are self-explanatory
        # remarks:
        # - if jsonselector is None, no selection will be done on run/ls number, i.e. all runs and lumisections are kept
        # - if jsonselector contains a single negative run number as key, templates will be used (e.g. averaging the dataset) instead of actual ls from the data
        #   for example, if jsonselector = {"-15":[[-1]]}, the dataset will be split in 15 parts and each part will be averaged to yield a single histogram (per type)
        
        dotemplates = False
        ntemplates = 0
        if( (jsonselector is not None) and (len(jsonselector)==1) and (int(list(jsonselector.keys())[0])<0) ): 
            dotemplates=True
            ntemplates = -int(jsonselector.keys()[0])
            
        dfstruct = {} # temporary structure for dataframes
        self.names = []
        self.histograms = {}
        self.entries = {}
        runlist = []
        selectorlist = []
        for histname in histnames:
            histfile = 'data/DF'+str(year)+'_'+histname+'.csv'
            name = histfile.replace('.csv','')
            # for each type of histogram, add the full data to the structure 
            # also build additional selectors depending on all histogram types simulaneously 
            print('reading '+histfile)
            df = csv_utils.read_csv(histfile)
            # select runs/lumisections if requested
            if( (jsonselector is not None) and (not dotemplates) ):
                df = dataframe_utils.select_runsls(df,jsonselector)
            dfstruct[name] = df
            # build high statistics selector if requested
            if highstatonly: selectorlist.append( dataframe_utils.get_highstat(df) ) # select high statistics
            else: selectorlist.append( dataframe_utils.get_runsls(df) ) # do no selection

        # make combined selector now that we know selectors for each histogram type individually
        totalselector = json_utils.get_lcs(selectorlist)

        for i,name in enumerate(dfstruct.keys()):
            print('adding '+name)
            self.names.append(name)
            df_temp = dataframe_utils.select_runsls(dfstruct[name],totalselector) # apply selector
            if dcsononly: df_temp = dataframe_utils.select_dcson(df_temp) # include only DCS-on json
            # determine statistics (must be done before normalizing)
            self.entries[name] = np.array(df_temp['entries'])
            # prepare the data
            (hists_all,runnbs_all,lsnbs_all) = hist_utils.preparedatafromdf(df_temp,returnrunls=True,onlygolden=False,rebinningfactor=1)
            # calculate templates if needed
            if dotemplates:
                hists_all = hist_tools.averagehists(hists_all,ntemplates)
                runnbs_all = np.zeros(len(hists_all))
                lsnbs_all = np.arange(len(hists_all))
            # add the histograms to the structure
            self.histograms[name] = hists_all
            runnbs_all = runnbs_all.astype(int)
            lsnbs_all = lsnbs_all.astype(int)
            # if processing first histogram type, add run numbers, lsnumbers and golden indices to histstruct
            if i==0:
                self.runnbs = runnbs_all
                self.lsnbs = lsnbs_all
            # else check consistency
            else:
                if( not ( (runnbs_all==self.runnbs).all() and (lsnbs_all==self.lsnbs).all() ) ):
                    print('### WARNING ###: incompatible run and lumisection numbers')
        # delete temporary data structure
        del dfstruct
    
    def get_golden_mask(self):
        # return a boolean mask on the lumisections whether or not they belong to the golden json
        mask = np.array( json_utils.isgolden(self.runnbs,self.lsnbs) )
        return mask
    
    def get_golden_indices(self):
        # return an array of indices of lumisections that belong to the golden json
        indices = np.arange(len(self.runnbs))[self.get_golden_mask()]
        return indices
    
    def get_perrun_indices(self):
        # return a list of arrays of indices of lumisections, one element in the list represents one run
        unique_runs = np.unique(self.runnbs)
        perrun_indices = []
        for ur in unique_runs:
            perrun_indices.append( np.arange(len(self.runnbs))[self.runnbs==ur] )
        return perrun_indices




### functions for fitting a normal-like or gaussian kernel distribution to a point cloud of mse's and making plots

def get_mse_array(histstruct,valkey,dims=[]):
    if len(dims)==0:
        dims = list(range(len(histstruct.names)))
    coords = np.expand_dims( histstruct.custom[valkey][histstruct.names[dims[0]]], axis=1 )
    for dim in dims[1:]:
        coords = np.concatenate( ( coords, np.expand_dims(histstruct.custom[valkey][histstruct.names[dim]], axis=1) ), axis=1 )
    return coords

def fitseminormal(histstruct,valkey,dims=[],fitnew=True,savefit=False):
    coords = get_mse_array(histstruct,valkey,dims=dims)
    if fitnew:
        fitfunc = clustering_utils.seminormal(coords)
        if savefit:
            fitfunc.save('seminormal_fit_'+xname+'_'+yname+'.npy')
    else:
        fitfunc = clustering_utilsseminormal()
        fitfunc.load('seminormal_fit_'+xname.replace('2018','2017')+'_'+yname.replace('2018','2017')+'.npy')
    
    return fitfunc

def fitgaussiankde(histstruct,valkey,dims=[],maxnpoints=-1):
    coords = get_mse_array(histstruct,valkey,dims=dims)
    if( maxnpoints>0 and maxnpoints<len(coords) ): coords = coords[ np.random.choice(list(range(len(coords))),size=maxnpoints,replace=False) ]
    fitfunc = clustering_utils.gaussiankde(coords)
    return fitfunc

def plotfit2d(histstruct,valkey,dims,fitfunc,doinitialplot=True,onlycontour=False,rangestd=30):
    
    xname = histstruct.names[dims[0]]
    yname = histstruct.names[dims[1]]
    xvals = histstruct.custom[valkey][xname]
    yvals = histstruct.custom[valkey][yname]
    
    if doinitialplot:
        # make an initial scatter plot of the data points
        fig,ax = plt.subplots()
        ax.plot(xvals,yvals,'.',markersize=1)
        plt.xticks(rotation=90)
        ax.set_xlabel(xname+' MSE')
        ax.set_ylabel(yname+' MSE')
        
    # determine plotting range as a fixed zoom from scatter plot
    #xlim = ax.get_xlim()[1]
    #ylim = ax.get_ylim()[1]
    #zoomxlim = xlim/1.
    #zoomylim = ylim/1.
    # determine plotting range as a fixed number of stds
    zoomxlim = rangestd*np.sqrt(fitfunc.cov[0,0])
    zoomylim = rangestd*np.sqrt(fitfunc.cov[1,1])
    
    x,y = np.mgrid[0:zoomxlim:zoomxlim/100.,
                   0:zoomylim:zoomylim/100.]
    pos = np.dstack((x, y))

    # make a new plot of probability contours and overlay data points
    fig,ax = plt.subplots()
    contourplot = ax.contourf(x, y, np.log(fitfunc.pdfgrid(pos)),30)
    plt.colorbar(contourplot)
    if not onlycontour: ax.plot(xvals,yvals,'.b',markersize=2)
    ax.set_xlim((0,zoomxlim))
    ax.set_ylim((0,zoomylim))
    #plt.xticks(rotation=90)
    ax.set_xlabel(xname+' MSE')
    ax.set_ylabel(yname+' MSE')
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    
    return (fig,ax)




### normalizer class (used for 1D mse arrays)
# still experimental, not enough checks on inputs etc.

class msenormalizer:
    
    # parameters:
    # std: std of array
    
    def __init__(self):
        self.std = 1.
        
    def fit(self,array):
        self.std = np.std(array)
        return self.apply(array)
    
    def apply(self,array):
        return array/self.std





