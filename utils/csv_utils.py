#!/usr/bin/env python
# coding: utf-8

# **A collection of useful basic functions for reading and processing the input csv files.**  
# 
# Functionality includes:
# - reading the raw input csv files and producing more manageable csv files (grouped per histogram type).
# - reading csv files into pandas dataframes and writing pandas dataframes back to csv files.
# 
# **Note: the functionality of these utils has been absorbed into the DataLoader class, which is now the recommended way to read the data!**



### imports

# external modules
import os
import pandas as pd
import numpy as np
import importlib

# local modules
import dataframe_utils as dfu
importlib.reload(dfu)


def get_data_dirs(year='2017', eras=None, dim=1):
    ### yield all data directories
    # note that the location of the data is hard-coded;
    # this function might break for newer or later reprocessings of the data.
    # - year is a string, either '2017' or '2018'
    # - era is a list containing a selection of era names
    #   (default empty list = all eras)
    # - dim is either 1 or 2 (for 1D or 2D plots)
    if(year=='2017' and not eras): eras = ['B','C','D','E','F']
    if(year=='2018' and not eras): eras = ['A','B','C','D']
    basedir = '/eos/project/c/cmsml4dc/ML_2020/UL'+year+'_Data/'
    for era in eras:
        eradir = basedir+'DF'+year+era+'_'+str(dim)+'D_Complete'
        if not os.path.exists(eradir):
            print('ERROR in csv_utils.py / get_data_dirs: requested directory {}'.format(eradir)
                  +' does not seem to exist, skipping it and continuing...')
        else: yield eradir

def get_csv_files(inputdir):
    ### yields paths to all csv files in input directory
    # note that the output paths consist of input_dir/filename
    # this function is only meant for 1-level down searching,
    # i.e. the .csv files listed directly under input_dir.
    if not os.path.exists(inputdir):
        raise Exception('ERROR in csv_utils.py / get_csv_files: input directory {}'.format(inputdir)
                       +' does not seem to exist.')
    for el in os.listdir(inputdir):
        if el[-4:]=='.csv':
            yield os.path.join(inputdir,el)

def sort_filenames(filelist):
    ### sort filenames in numerical order (e.g. 2 before 10)
    # note that the number is supposed to be in ..._<number>.<extension> format
    nlist = []
    for f in filelist:
        temp = f.rsplit('.',1)[0]
        temp = temp[temp.rfind('_')+1:]
        nlist.append(int(temp))
    return [f for _,f in sorted(zip(nlist,filelist))]




def read_csv(csv_file):
    ### read csv file into pandas dataframe
    # csv_file is the path to the csv file to be read
    # DEPRECATED, this function might be removed in the future;
    #             use DataLoader.get_dataframe_from_file instead.
    df = pd.read_csv(csv_file)
    df.sort_values(by=['fromrun','fromlumi'],inplace=True)
    df.reset_index(drop=True,inplace=True)
    return df

def write_csv(dataframe, csvfilename):
    ### write a dataframe to a csv file
    # note: just a wrapper for builtin dataframe.to_csv
    # DEPRECATED, this function might be removed in the future;
    #             use DataLoader.write_dataframe_to_file instead.
    dataframe.to_csv(csvfilename)

def read_and_merge_csv(csv_files, histnames=None, runnbs=None):
    ### read and merge list of csv files into a single df
    # csv_files is a list of paths to files to merge into a df
    # histnames is a list of the types of histograms to keep (default: all)
    # runnbs is a list of run numbers to keep (default: all)
    # DEPRECATED, this function might be removed in the future;
    #             use DataLoader.get_dataframe_from_files instead.
    dflist = []
    print('INFO in csv_utils.py / read_and_merge_csv:'
          +' reading and merging {} csv files...'.format(len(csv_files)))
    for i,f in enumerate(csv_files):
        print('  - now processing file {} of {}...'.format(i+1,len(csv_files)))
        dffile = read_csv(f)
        if histnames is not None and len(histnames):
            dffile = dfu.select_histnames(dffile,histnames)
        if runnbs is not None and len(runnbs):
            dffile = dfu.select_runs(dffile,runnbs)
        dflist.append(dffile)
    df = pd.concat(dflist,ignore_index=True)
    df.sort_values(by=['fromrun','fromlumi'],inplace=True)
    df.reset_index(drop=True,inplace=True)
    print('INFO in csv_utils.py / read_and_merge_csv: merged {} csv files.'.format(len(csv_files)))
    return df


def write_skimmed_csv(histnames, year, eras=None, dim=1):
    ### read all available data for a given year/era and make a file per histogram type
    # DEPRECATED, this function might be removed in the future;
    #             see tutorial read_and_write_data.ipynb for equivalent functionality.
    # input arguments:
    # - histnames: list of histogram names for which to make a separate file
    # - year: data-taking year (in string format)
    # - eras: data-taking eras for which to make a separate file (in string format)
    #         use 'all' to make a file with all eras merged, i.e. a full data taking year
    # - dim: dimension of histograms (1 or 2), needed to retrieve the correct folder containing input files
    # output:
    # - one csv file per year/era and per histogram type
    # note: this function can take quite a while to run!

    if eras is None:
        eras = ['all']

    for era in eras:
        thiseras = [era]
        erasuffix = era
        if era=='all': 
            thiseras = []
            erasuffix = ''
        datadirs = list(get_data_dirs(year=year,eras=thiseras,dim=dim))
        csvfiles = []
        for datadir in datadirs:
            csvfiles += sort_filenames(list(get_csv_files(datadir)))
        # read histograms into df
        temp = read_and_merge_csv(csvfiles,histnames=histnames)
        # write df to files
        for histname in histnames:
            seldf = dfu.select_histnames(temp,[histname])
            histname = histname.replace(' ','_')
            seldf.to_csv('DF'+year+erasuffix+'_'+histname+'.csv')
