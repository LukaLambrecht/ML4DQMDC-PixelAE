#!/usr/bin/env python
# coding: utf-8



### imports

# external modules
import os
import sys
import importlib

# local modules
sys.path.append('../utils')
import csv_utils as csvu
import dataframe_utils as dfu
importlib.reload(csvu)
importlib.reload(dfu)




class DataLoader(object):
    ### class for loading histograms from disk
    # the input usually consists of:
    # - a csv file or a folder containing csv files in the correct format
    # - a set of histogram names to load
    # - a specification in terms of eras or years
    # the output typically consists of pandas dataframes containing the requested histograms.
    
    def __init__( self ):
        ### initializer
        self.validyears = ['2017', '2018']
        self.valideras = {
            '2017': ['B','C','D','E','F'],
            '2018': ['A','B','C','D']
        }
        self.validdims = [1,2]
        
    ### help functions for default data directories
    
    def check_year( self, year ):
        if( year not in self.validyears ):
            raise Exception('ERROR in DataLoader.check_year:'
                           +' year {} not recognized;'.format(year)
                           +' should be picked from {}'.format(self.validyears))
            
    def check_eras( self, eras, year ):
        if len(eras)==0: return
        self.check_year( year )
        for era in eras:
            if( era not in self.valideras[year] ):
                raise Exception('ERROR in DataLoader.check_eras:'
                               +' era {} not recognized;'.format(era)
                               +' should be picked from {}'.format(self.valideras[year]))
                
    def check_dim( self, dim ):
        if( dim not in self.validdims ):
            raise Exception('ERROR in DataLoader.check_dim:'
                           +' dimension {} not recognized;'.format(dim)
                           +' should be picked from {}'.format(self.validdims))
            
    def check_eos( self ):
        if not os.path.exists('/eos'):
            raise Exception('ERROR in DataLoader.check_eos:'
                            +' the /eos filesystem (where the input data is stored by default) cannot be found;'
                            +' make sure DataLoader is run from a place where it has access to /eos,'
                            +' or specify explicitly the input directories on your filesystem where to search for input.')
                
    def get_default_data_dirs( self, year='2017', eras=[], dim=1 ):
        ### get the default data directories for the data for this project
        # note: this returns the directories where the data is currently stored;
        #       might change in future reprocessings of the data,
        #       and should be extended for upcoming Run-III data.
        # note: default directories are on the /eos file system.
        #       this function will throw an exception if it has not access to /eos.
        # input arguments:
        # - year: data-taking year, should be '2017' or '2018' so far (default: 2017)
        # - eras: list of valid eras for the given data-taking year (default: all eras)
        # - dim: dimension of requested histograms (1 or 2)
        self.check_year( year )
        self.check_eras( eras, year )
        self.check_dim( dim )
        self.check_eos()
        return list(csvu.get_data_dirs(year=year, eras=eras, dim=dim))
    
    ### help functions to get csv files from directories
    
    def get_csv_files_in_dir( self, inputdir, sort=True ):
        ### get a (sorted) list of csv files in a given input directory
        # input arguments:
        # - inputdir: directory to scan for csv files
        # - sort: boolean whether to sort the files
        if not os.path.exists(inputdir):
            raise Exception('ERROR in DataLoader.get_csv_files:'
                           +' input directory {}'.format(inputdir)
                           +' does not seem to exist.')
        filelist = list(csvu.get_csv_files( inputdir ))
        if sort:
            try: filelist = csvu.sort_filenames(filelist)
            except:
                print('WARNING: in DataLoader.get_csv_files:'
                     +' something went wrong in numerical sorting the filenames,'
                     +' maybe the format of the filenames is not as expected?'
                     +' the returned list of files should be complete,'
                     +' but they might not be sorted correctly.')
        return filelist
    
    def get_csv_files_in_dirs( self, inputdirs, sort=True ):
        ### find the csv files in a set of input directories and return them in one list
        # input arguments:
        # - list of input directories where to look for csv files
        # - sort: see get_csv_files_in_dir
        filelist = []
        for inputdir in inputdirs:
            for f in self.get_csv_files_in_dir( inputdir, sort=sort ):
                filelist.append(f)
        return filelist
    
    def get_default_csv_files( self, year='2017', eras=[], dim=1, sort=True ):
        ### read the csv files from the default directories with input data for this project
        # note: default directories are on the /eos file system.
        #       this function will throw an exception if it has not access to /eos.
        # input arguments: 
        # - year, eras, dim: see get_default_data_dirs!
        # - sort: see get_csv_files_in_dir!
        datadirs = self.get_default_data_dirs( year=year, eras=eras, dim=dim )
        return self.get_csv_files_in_dirs( datadirs )
    
    ### functions for reading single files
    
    def get_dataframe_from_file( self, csvfile, histnames=[] ):
        ### load histograms from a given file
        if not os.path.exists( csvfile ):
            raise Exception('ERROR in DataLoader.get_histograms_from_file:'
                           +' the requested file {}'.format(csvile)
                           +' does not seem to exist.')
        df = csvu.read_csv(csvfile)
        if len(histnames)>0: df = dfu.select_histnames(df, histnames)
        return df
    
    ### functions for reading multiple files
    
    def get_dataframe_from_files( self, csvfiles, histnames=[] ):
        ### load histograms from a given set of files
        for csvfile in csvfiles:
            if not os.path.exists( csvfile ):
                raise Exception('ERROR in DataLoader.get_histograms_from_file:'
                           +' the requested file {}'.format(csvile)
                           +' does not seem to exist.')
        df = csvu.read_and_merge_csv( csvfiles, histnames=histnames )
        return df
        
    ### functions for writing a dataframe to a csv file
    
    def write_dataframe_to_file( self, df, csvfile ):
        ### write a dataframe to a csv file
        if os.path.splitext(csvfile)[1]!='.csv':
            print('WARNING in DataLoader.write_dataframe_to_file:'
                 +' the output file does not seem to have the proper extension,'
                 +' replacing by ".csv"')
            csvfile = os.path.splitext(csvfile)[0]+'.csv'
        dirname = os.path.dirname(csvfile)
        if not os.path.exists(dirname):
            print('WARNING in DataLoader.write_dataframe_to_file:'
                 +' the output directory {}'.format(dirname)
                 +' does not exist yet; will try to create it.')
            try: os.makedirs(dirname)
            except: raise Exception('ERROR in DataLoader.write_dataframe_to_file:'
                                   +' the output directory could not be created.')
        if os.path.exists(csvfile):
            print('WARNING in DataLoader.write_dataframe_to_file:'
                 +' output file {}'.format(csvfile)
                 +' already exists; overwriting...')
        csvu.write_csv( df, csvfile )










