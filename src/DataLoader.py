#!/usr/bin/env python
# coding: utf-8

# **Class for loading monitoring elements (MEs) from disk into a pandas dataframe.**  
# 
# Typically, the input consists of a single file per ME type,  
# prepared from the (nano)DQMIO format (accessed via DAS) and converted into another file format.  
# see the tools in the 'dqmio' folder for more info on preparing the input files.  
# Currently supported input file formats:  
# 
# - csv  
# - parquet  
#
# UPDATE: this class now also supports parquet files obtained with the DIALS API  
# (see https://github.com/cms-DQM/dials-py)
# 
# Example usage:  
# ```  
# from DataLoader import DataLoader  
# dl = DataLoader()  
# df = dl.get_dataframe_from_file( <path to input file> )  
# ```  
#
# Alternatively, support is available to read the legacy per-LS csv files  
# (deprecated approach for run-II data, before nanoDQMIO in run-III).  
# In this case, the needed input consists of:  
# 
# - a set of ME names to load  
# - a specification in terms of eras or years  
# 
# Example usage:  
# ```
# from DataLoader import DataLoader  
# dl = DataLoader()  
# csvfiles = dl.get_default_csv_files( year=<year>, dim=<ME dimension> )  
# df = dl.get_dataframe_from_files( csvfiles, menames=<ME names> )  
# ```
# 
# The output consists of a pandas dataframe containing the requested MEs.  


### imports

# external modules
import os
import sys
import pandas as pd
import importlib

# local modules
sys.path.append('../utils')
import csv_utils as csvu
importlib.reload(csvu)


class DataLoader(object):
    
    def __init__( self ):
        ### initializer
        # initialization of valid years and eras for which legacy csv files exist
        # (note: only relevant for legacy csv approach, else empty initializer)
        self.validyears = ['2017', '2018']
        self.valideras = {
            '2017': ['B','C','D','E','F'],
            '2018': ['A','B','C','D']
        }
        self.validdims = [1,2]
        
    ### help functions for default data directories
    
    def check_year( self, year ):
        ### check if a provided year is valid
        # (note: only relevant for legacy csv approach)
        # (note: internal helper function, no need to call)
        # input arguments:
        # - year: year in string format
        if( year not in self.validyears ):
            raise Exception('ERROR in DataLoader.check_year:'
                           +' year {} not recognized;'.format(year)
                           +' should be picked from {}'.format(self.validyears))
            
    def check_eras( self, eras, year ):
        ### check if a list of provided eras is valid
        # (note: only relevant for legacy csv approach)
        # (note: internal helper function, no need to call)
        # input arguments:
        # - eras: list of eras in string format, e.g. ['B','C']
        # - year: year in string format
        if len(eras)==0: return
        self.check_year( year )
        for era in eras:
            if( era not in self.valideras[year] ):
                raise Exception('ERROR in DataLoader.check_eras:'
                               +' era {} not recognized;'.format(era)
                               +' should be picked from {}'.format(self.valideras[year]))
                
    def check_dim( self, dim ):
        ### check if a monitoring element dimension is valid
        # (note: only 1D and 2D monitoring elements are supported for now)
        # (note: internal helper function, no need to call)
        if( dim not in self.validdims ):
            raise Exception('ERROR in DataLoader.check_dim:'
                           +' dimension {} not recognized;'.format(dim)
                           +' should be picked from {}'.format(self.validdims))
            
    def check_eos( self ):
        ### check if the /eos directory exists and is accessible
        # (note: only relevant for legacy csv approach)
        # (note: internal helper function, no need to call)
        if not os.path.exists('/eos'):
            raise Exception('ERROR in DataLoader.check_eos:'
                            +' the /eos filesystem (where the input data is stored by default) cannot be found;'
                            +' make sure DataLoader is run from a place where it has access to /eos,'
                            +' or specify explicitly the input directories on your filesystem where to search for input.')
                
    def get_default_data_dirs( self, year='2017', eras=[], dim=1 ):
        ### get the default data directories for the data for this project
        # (note: only relevant for legacy csv approach)
        # (note: internal helper function, no need to call)
        # note: this returns the directories where the data is currently stored;
        #       might change in future reprocessings of the data,
        #       and should be extended for upcoming Run-III data.
        # note: default directories are on the /eos file system.
        #       this function will throw an exception if it does not have access to /eos.
        # input arguments:
        # - year: data-taking year, should be '2017' or '2018' so far (default: 2017)
        # - eras: list of valid eras for the given data-taking year (default: all eras)
        # - dim: dimension of requested MEs (1 or 2)
        #   note: need to provide the dimension at this stage since the files for 1D and 2D MEs
        #         are stored in different directories.
        # returns:
        # a list of directories containing the legacy csv files with the requested data.
        self.check_year( year )
        self.check_eras( eras, year )
        self.check_dim( dim )
        self.check_eos()
        return list(csvu.get_data_dirs(year=year, eras=eras, dim=dim))
    
    ### help functions to get csv files from directories
    
    def get_csv_files_in_dir( self, inputdir, sort=True ):
        ### get a (optionally sorted) list of csv files in a given input directory
        # (note: only relevant for legacy csv approach)
        # (note: internal helper function, no need to call)
        # input arguments:
        # - inputdir: directory to scan for csv files
        # - sort: boolean whether to sort the files
        # returns:
        # a list of csv files in the given directory.
        if not os.path.exists(inputdir):
            raise Exception('ERROR in DataLoader.get_csv_files_in_dir:'
                           +' input directory {}'.format(inputdir)
                           +' does not seem to exist.')
        filelist = list(csvu.get_csv_files( inputdir ))
        if sort:
            try: filelist = csvu.sort_filenames(filelist)
            except:
                print('WARNING: in DataLoader.get_csv_files_in_dir:'
                     +' something went wrong in numerical sorting the filenames,'
                     +' maybe the format of the filenames is not as expected?'
                     +' the returned list of files should be complete,'
                     +' but they might not be sorted correctly.')
        return filelist
    
    def get_csv_files_in_dirs( self, inputdirs, sort=True ):
        ### find the csv files in a set of input directories and return them in one list.
        # (note: only relevant for legacy csv approach)
        # (note: internal helper function, no need to call)
        # this function simply loops over the directories given in inputdirs,
        # calls get_csv_files_in_dir for each of them, and concatenates the results.
        # input arguments:
        # - list of input directories where to look for csv files
        # - sort: see get_csv_files_in_dir
        # returns:
        # a list of csv files in the given directories.
        filelist = []
        for inputdir in inputdirs:
            for f in self.get_csv_files_in_dir( inputdir, sort=sort ):
                filelist.append(f)
        return filelist
    
    def get_default_csv_files( self, year='2017', eras=[], dim=1, sort=True ):
        ### read the csv files from the default directories with input data for this project
        # (note: only relevant for legacy csv approach)
        # note: default directories are on the /eos file system.
        #       this function will throw an exception if it has not access to /eos.
        # input arguments: 
        # - year, eras, dim: see get_default_data_dirs!
        # - sort: see get_csv_files_in_dir!
        # returns:
        # a list of csv files with the data corresponding to the provided year, eras and dimension.
        datadirs = self.get_default_data_dirs( year=year, eras=eras, dim=dim )
        return self.get_csv_files_in_dirs( datadirs )
    
    ### functions for reading single files
    
    def get_dataframe_from_file( self, dfile, menames=[], sort=True, verbose=True,
        runcolumn='fromrun', lumicolumn='fromlumi', menamecolumn='hname',
        renamecolumns=None ):
        ### load MEs from a given file into a dataframe
        # input arguments:
        # - dfile: file containing the data.
        #   currently supported formats: csv, parquet.
        # - menames: list of ME names to keep
        #   (default: keep all MEs present in the input file).
        # - sort: whether to sort the dataframe by run and lumisection number
        #   (note: requires keys 'fromrun' and 'fromlumi' to be present in the dataframe).
        # - verbose: whether to print info messages.
        # returns:
        # a pandas dataframe
        
        # check if file exists
        if not os.path.exists( dfile ):
            raise Exception('ERROR in DataLoader.get_dataframe_from_file:'
                           +' the requested file {}'.format(dfile)
                           +' does not seem to exist.')
        # do some printouts if requested
        if verbose:
            msg = 'INFO in DataLoader.get_dataframe_from_file:'
            msg += ' loading dataframe from file {}...'.format(dfile)
            print(msg)
        # get the file extension
        ext = os.path.splitext(dfile)[1]
        # load the file into a dataframe
        if( ext=='.csv' ): df = pd.read_csv(dfile)
        elif( ext=='.parquet' ): df = pd.read_parquet(dfile)
        else:
            raise Exception('ERROR in DataLoader.get_dataframe_from_file:'
                           +' the file extension is {}'.format(ext),
                           +' which is currently not supported (must be .csv or .parquet).')
        # do selection if requested
        if len(menames)>0:
            if verbose:
                msg = 'INFO in DataLoader.get_dataframe_from_file:'
                msg += ' selecting monitoring elements {}...'.format(menames)
                print(msg)
            df = df[df[menamecolumn].isin(menames)]
            df.reset_index(drop=True, inplace=True)
        # do sorting if requested
        if sort:
            if verbose:
                msg = 'INFO in DataLoader.get_dataframe_from_file:'
                msg += ' sorting the dataframe...'
                print(msg)
            df.sort_values(by=[runcolumn, lumicolumn], inplace=True)
            df.reset_index(drop=True, inplace=True)
        # rename columns if requested
        if renamecolumns is not None:
            if verbose:
                msg = 'INFO in DataLoader.get_dataframe_from_file:'
                msg += ' renaming colums...'
                print(msg)
            df.rename(mapper=renamecolumns, axis='columns', inplace=True)
        # do some more printouts if requested
        if verbose:
            msg = 'INFO in DataLoader.get_dataframe_from_file:'
            msg += ' loaded a dataframe with {} rows and {} columns.'.format(len(df), len(df.columns))
            print(msg)
        return df

    def get_dataframe_from_legacy_file( self, dfile, **kwargs ):
        renamecolumns = {
          'fromrun': 'run',
          'fromlumi': 'lumi',
          'hname': 'mename',
          'Xmin': 'xmin',
          'Xmax': 'xmax',
          'Xbins': 'xbins',
          'Ymin': 'ymin',
          'Ymax': 'ymax',
          'Ybins': 'ybins',
          'histo': 'data'
        }
        return self.get_dataframe_from_file(dfile, **kwargs, 
                 runcolumn='fromrun', lumicolumn='fromlumi', menamecolumn='hname',
                 renamecolumns=renamecolumns)

    def get_dataframe_from_dials_file( self, dfile, **kwargs ):
        renamecolumns = {
          'run_number': 'run',
          'ls_number': 'lumi',
          'me': 'mename',
          'x_min': 'xmin',
          'x_max': 'xmax',
          'x_bin': 'xbins',
          'y_min': 'ymin',
          'y_max': 'ymax',
          'y_bin': 'ybins',
        }
        return self.get_dataframe_from_file(dfile, **kwargs, 
                 runcolumn='run_number', lumicolumn='ls_number', menamecolumn='me',
                 renamecolumns=renamecolumns)
    
    ### functions for reading multiple files
    
    def get_dataframe_from_files( self, dfiles, menames=[], sort=True, verbose=True,
        runcolumn='fromrun', lumicolumn='fromlumi', menamecolumn='hname',
        renamecolumns=None ):
        ### load MEs from a given set of files into a single dataframe
        # input arguments:
        # - dfiles: list of files containing the data.
        #   currently supported formats: csv, parquet.
        # - menames: list of ME names to keep
        #   (default: keep all MEs present in the input file).
        # - sort: whether to sort the dataframe by run and lumisection number
        #   (note: requires keys 'fromrun' and 'fromlumi' to be present in the dataframe).
        # - verbose: whether to print info messages.
        # returns:
        # a pandas dataframe
        
        # quick check if all files exist
        for dfile in dfiles:
            if not os.path.exists( dfile ):
                raise Exception('ERROR in DataLoader.get_dataframe_from_files:'
                               +' the requested file {}'.format(dfile)
                               +' does not seem to exist.')
        # do some printouts if requested
        if verbose:
            msg = 'INFO in DataLoader.get_dataframe_from_files:'
            msg += ' reading and merging {} files...'.format(len(dfiles))
            print(msg)
        # loop over files
        dflist = []
        for i,dfile in enumerate(dfiles):
            if verbose:
                msg = 'INFO in DataLoader.get_dataframe_from_files:'
                msg += ' now processing file {} of {}...'.format(i+1, len(dfiles))
                print(msg)
            # read dataframe for this file
            df = self.get_dataframe_from_file( dfile, menames=menames, sort=False, verbose=verbose,
                   runcolumn=runcolumn, lumicolumn=lumicolumn, menamecolumn=menamecolumn )
            dflist.append(df)
        # concatenate
        if verbose:
            msg = 'INFO in DataLoader.get_dataframe_from_files:'
            msg += ' merging the dataframes...'
            print(msg)
        df = pd.concat(dflist,ignore_index=True)
        # do sorting if requested
        if sort:
            if verbose:
                msg = 'INFO in DataLoader.get_dataframe_from_files:'
                msg += ' sorting the dataframe...'
                print(msg)
            df.sort_values(by=[runcolumn,lumicolumn],inplace=True)
            df.reset_index(drop=True,inplace=True)
        # rename columns if requested
        if renamecolumns is not None:
            if verbose:
                msg = 'INFO in DataLoader.get_dataframe_from_file:'
                msg += ' renaming colums...'
                print(msg)
            df.rename(mapper=renamecolumns, axis='columns', inplace=True)
        # do some more printouts if requested
        if verbose:
            msg = 'INFO in DataLoader.get_dataframe_from_files:'
            msg += ' loaded a dataframe from {} files'.format(len(dfiles))
            msg += ' with {} rows and {} columns.'.format(len(df), len(df.columns))
            print(msg)
        return df

    def get_dataframe_from_legacy_files( self, dfiles, **kwargs ):
        renamecolumns = {
          'fromrun': 'run',
          'fromlumi': 'lumi',
          'hname': 'mename',
          'Xmin': 'xmin',
          'Xmax': 'xmax',
          'Xbins': 'xbins',
          'Ymin': 'ymin',
          'Ymax': 'ymax',
          'Ybins': 'ybins',
          'histo': 'data'
        }
        return self.get_dataframe_from_files(dfiles, **kwargs,
                 runcolumn='fromrun', lumicolumn='fromlumi', menamecolumn='hname',
                 renamecolumns=renamecolumns)

    def get_dataframe_from_dials_files( self, dfiles, **kwargs ):
        renamecolumns = {
          'run_number': 'run',
          'ls_number': 'lumi',
          'me': 'mename',
          'x_min': 'xmin',
          'x_max': 'xmax',
          'x_bin': 'xbins',
          'y_min': 'ymin',
          'y_max': 'ymax',
          'y_bin': 'ybins',
        }
        return self.get_dataframe_from_files(dfiles, **kwargs,
                 runcolumn='run_number', lumicolumn='ls_number', menamecolumn='me',
                 renamecolumns=renamecolumns)
        
    ### functions for writing a dataframe to a file
    
    def write_dataframe_to_file( self, df, dfile, overwrite=False, verbose=True ):
        ### write a dataframe to a file
        # input arguments:
        # - df: a pandas dataframe.
        # - dfile: file name to write.
        #   currently supported formats: csv, parquet.
        # - overwrite: whether to overwrite if a file with the given name already exists.
        # - verbose: whether to print info messages.
        
        # check if directory exists and try to create it if not
        dirname = os.path.dirname(dfile)
        if not os.path.exists(dirname):
            if verbose:
                msg = 'WARNING in DataLoader.write_dataframe_to_file:'
                msg += ' the output directory {}'.format(dirname)
                msg += ' does not exist yet; will try to create it.'
                print(msg)
            try: os.makedirs(dirname)
            except: raise Exception('ERROR in DataLoader.write_dataframe_to_file:'
                                   +' the output directory could not be created.')
        # check if the file name is duplicate
        if os.path.exists(dfile):
            msg = 'WARNING in DataLoader.write_dataframe_to_file:'
            msg += ' output file {}'.format(dfile)
            msg += ' already exists.'
            if overwrite:
                msg += ' Overwriting...'
                print(msg)
            else: raise Exception(msg)
        # get the file extension
        ext = os.path.splitext(dfile)[1]
        # write the file
        if( ext=='.csv' ): df.to_csv( dfile )
        elif( ext=='.parquet' ): df.to_parquet( dfile )
        else:
            raise Exception('ERROR in DataLoader.write_dataframe_to_file:'
                           +' the file extension is {}'.format(ext),
                           +' which is currently not supported (must be .csv or .parquet).')
        # do some printouts if requested
        if verbose:
            msg = 'INFO in DataLoader.write_dataframe_to_file:'
            msg += ' dataframe written to file {}'.format(dfile)
