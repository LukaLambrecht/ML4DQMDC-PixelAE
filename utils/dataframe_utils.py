#!/usr/bin/env python
# coding: utf-8

# **A collection of useful basic functions for manipulating pandas dataframes.**  
# 
# Functionality includes (among others):
# - selecting DCS-bit on data or golden json data.
# - selecting specific runs, lumisections, or types of histograms


### imports

# external modules
import os
import pandas as pd
import numpy as np
import json
import importlib

# local modules
import json_utils
importlib.reload(json_utils)


# getter and selector for histogram names 

def get_menames(df, menamecolumn='mename'):
    ### get a list of (unique) ME names present in a df
    # df is a dataframe read from an input csv file.
    menamelist = sorted(list(set(df[menamecolumn].values)))
    return menamelist
    
def select_menames(df, menames, menamecolumn='mename'):
    ### keep only a subset of MEs in a df
    # menames is a list of ME names to keep in the df.
    df = df[df[menamecolumn].isin(menames)]
    df.reset_index(drop=True, inplace=True)
    return df


# getter and selector for run numbers

def get_runs(df, runcolumn='run'):
    ### return a list of (unique) run numbers present in a df
    # df is a dataframe read from an input csv file.
    runlist = sorted(list(set(df[runcolumn].values)))
    return runlist

def select_runs(df, runnbs, runcolumn='run'):
    ### keep only a subset of runs in a df
    # runnbs is a list of run numbers to keep in the df.
    df = df[df[runcolumn].isin(runnbs)]
    df.reset_index(drop=True, inplace=True)
    return df


# getter and selector for lumisection numbers

def get_ls(df, lumicolumn='lumi'):
    ### return a list of ls numbers present in a df
    # note: the numbers are not required to be unique!
    # note: no check is done on the run number!
    lslist = sorted(list(df[lumicolumn].values))
    return lslist

def select_ls(df, lsnbs, lumicolumn='lumi'):
    ### keep only a subset of lumisection numbers in a df
    # lsnbs is a list of lumisection numbers to keep in the df.
    # note: no check is done on the run number!
    df = df[df[lumicolumn].isin(lsnbs)]
    df.reset_index(drop=True, inplace=True)
    return df


### general getter and selector in json format

def get_runsls(df, runcolumn='run', lumicolumn='lumi'):
    ### return a dictionary with runs and lumisections in a dataframe (same format as e.g. golden json)
    runslslist = get_runs(df, runcolum=runcolumn)
    for i,run in enumerate(runslslist):
        runslslist[i] = (run, get_ls( select_runs(df,[run],runcolumn=runcolumn), lumicolumn=lumicolumn))
    return json_utils.tuplelist_to_jsondict( runslslist )

def select_json(df, jsonfile, runcolumn='run', lumicolumn='lumi'):
    ### keep only lumisections that are in the given json file
    dfres = df[ json_utils.injson( df[runcolumn].values, df[runcolumn].values, jsonfile=jsonfile) ]
    dfres.reset_index(drop=True, inplace=True)
    return dfres

def select_runsls(df, jsondict, runcolumn='run', lumicolumn='lumi'):
    ### equivalent to select_json but using a pre-loaded json dict instead of a json file on disk
    dfres = df[ json_utils.injson( df[runcolumn].values, df[lumicolumn].values, jsondict=jsondict) ]
    dfres.reset_index(drop=True, inplace=True)
    return dfres


### selectors for golden json and other important json files
# note: deprecated and not recommended, it's anyway better to select the correct json file manually
#       and use the select_json or select_runsls functions above than to rely on some shady
#       automatic assignment of the 'correct' json file.

def select_golden(df, year=None, runcolumn='run', lumicolumn='lumi'):
    ### keep only golden lumisections in df
    dfres = df[ json_utils.isgolden(df[runcolumn].values, df[lumicolumn].values, year=year) ]
    dfres.reset_index(drop=True, inplace=True)
    return dfres

def select_notgolden(df, year=None, runcolumn='run', lumicolumn='lumi'):
    ### keep all but golden lumisections in df
    dfres = df[np.invert( json_utils.isgolden(df[runcolumn].values, df[lumicolumn].values, year=year) )]
    dfres.reset_index(drop=True, inplace=True)
    return dfres

def select_dcson(df, year=None, runcolumn='run', lumicolumn='lumi'):
    ### keep only lumisections in df that have DCS-bit on
    dfres = df[ json_utils.isdcson(df[runcolumn].values, df[lumicolumn].values, year=year) ]
    dfres.reset_index(drop=True, inplace=True)
    return dfres

def select_dcsoff(df, year=None, runcolumn='run', lumicolumn='lumi'):
    ### keep only lumisections in df that have DCS-bit off
    dfres = df[np.invert( json_utils.isdcson(df[runcolumn].values,df[lumicolumn].values) )]
    dfres.reset_index(drop=True,inplace=True)
    return dfres


# getter and selector for sufficient statistics

def get_highstat(df, entriescolumn='entries', xbinscolumn='xbins', entries_to_bins_ratio=100):
    ### return a select object of runs and ls of histograms with high statistics
    return get_runsls(df[df[entriescolumn]/df[xbinscolumn]>entries_to_bins_ratio])

def select_highstat(df, entriescolumn='entries', xbinscolumn='xbins', entries_to_bins_ratio=100):
    ### keep only lumisection in df with high statistics
    return df[df[entriescolumn]/df[xbinscolumn]>entries_to_bins_ratio]
    
    
# functions to obtain histograms in np array format

def get_hist_values(df, datacolumn='data', xbinscolumn='xbins', ybinscolumn='ybins',
    runcolumn='run', lumicolumn='lumi'):
    ### same as builtin "df['histo'].values" but convert strings to np arrays
    # input arguments:
    # - df: a dataframe containing histograms (assumed to be of a single type!)
    # note: this function works for both 1D and 2D histograms,
    #       the distinction is made based on whether or not 'ybins' is present as a column in the dataframe
    #       update: 'ybins' is also present for 1D histograms, but has value 1!
    # output:
    # a tuple containing the following elements:
    # - np array of shape (nhists,nbins) (for 1D) or (nhists,nybins,nxbins) (for 2D)
    # - np array of run numbers of length nhists
    # - np array of lumisection numbers of length nhists
    # warning: no check is done to assure that all histograms are of the same type!
    
    # check for corruption of data types (observed once after merging several csv files)
    if isinstance( df.at[0,xbinscolumn], str ):
        raise Exception('ERROR in dataframe_utils.py / get_hist_values:'
                +' the "{}" entry in the dataframe is of type str,'.format(xbinscolumn)
                +' while a numpy int is expected; check for file corruption.')
    
    # check data type of data field
    # (string in csv files, numpy array in parquet files)
    # case of string
    if isinstance( df.at[0,datacolumn], str ):
        # check dimension
        dim = 1
        if ybinscolumn in df.keys():
            if int(df.at[0,ybinscolumn])>1: dim=2
        # initializations
        nxbins = int(df.at[0,xbinscolumn])+2 # +2 for under- and overflow bins
        vals = np.zeros((len(df),nxbins))
        if dim==2:
            nybins = int(df.at[0,ybinscolumn])+2
            vals = np.zeros((len(df),nybins,nxbins))
        ls = np.zeros(len(df))
        runs = np.zeros(len(df))
        # loop over all entries
        for i in range(len(df)):
            try:
                # default encoding (with comma separation)
                jsonstr = json.loads(df.at[i,datacolumn])
            except:
                # alternative encoding (with space separation)
                jsonstr = json.loads(df.at[i,datacolumn].replace(' ', ','))
            hist = np.array(jsonstr)
            if dim==2: hist = hist.reshape((nybins,nxbins))
            vals[i,:] = hist
            ls[i] = int(df.at[i,lumicolumn])
            runs[i] = int(df.at[i,runcolumn])
    
    # case of numpy array
    if isinstance( df.at[0,datacolumn], np.ndarray ):
        vals = np.vstack(df[datacolumn].values)
        ls = df[lumicolumn].values
        runs = df[runcolumn].values

    # return result
    ls = ls.astype(int)
    runs = runs.astype(int)
    return (vals,runs,ls)
