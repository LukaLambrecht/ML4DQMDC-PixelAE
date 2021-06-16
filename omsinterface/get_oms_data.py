#!/usr/bin/env python
# coding: utf-8

# **Main function in omsinterface to retrieve information from OMS**  
# 
# How to use?  
# See the readme file in this directory and the notebook example.ipynb!



### imports

# external modules
import sys
import os
import json
from getpass import getpass
import importlib

# local modules
import omstools
importlib.reload(omstools)
import cert
sys.path.append(os.path.abspath('../utils/notebook_utils'))




def get_oms_data( mode, run, hltpathname='', authmode='login' ):
    ### main function for retrieving information from the OMS database
    # input arguments:
    # - mode: a string representing the type of information to retrieve.
    #   the following options are currently supported:
    #   'run' -> retrieve information per run
    #   'lumisections' -> retrieve information per lumisection
    #   'hltpathinfos' -> get information on the available HLT paths for a given run, 
    #                     in particular their names, 
    #   'hltrate' -> get the trigger rate of a specified HLT path  
    #   'hltrates' -> get the trigger rate for all available HLT paths
    # - run: a single run number (integer format)
    #   note: in case mode is 'run', the run argument can also be a tuple
    #   representing a range of runs.
    # - hltpathname: the name of a HLT path for which to retrieve the trigger rate.
    #   ignored if mode is not 'hltrate'
    # - authmode: string representing mode of authentication.
    #   choose from 'login' (you will be prompted for your cern username and password)
    #   or 'certificate' (requires you to have set up the path to a valid certificate)
    # returns:
    # - a list or dict (depending on the specifications) containing all information.
    #   simply print it to see how to access the exact values you need.

    # parse arguments
    
    if mode=='run':
        method = omstools.get_runs
        args = [run[0],run[1]] if isinstance(run,tuple) else [run,run]
    elif mode=='lumisections':
        method = omstools.get_lumisections
        args = [run]
    elif mode=='hltpathinfos':
        method = omstools.get_hltpathinfos
        args = [run]
    elif mode=='hltrate': 
        method = omstools.get_hltpathrates
        args = [run,hltpathname]
    elif mode=='hltrates':
        method = omstools.get_all_hltpathrates
        args = [run]
    else:
        raise Exception('ERROR: mode {} not recognized'.format(mode))

    kwargs = {}

    if not omstools.check_oms_connectivity():
        print('WARNING: login or certificate required for authentication.')
        if authmode=='login':
            cern_username = input('Enter CERN username:')
            cern_password = getpass('Enter CERN password: ')
            login = (cern_username,cern_password)
            kwargs['authmode'] = 'login'
            kwargs['login'] = login
        elif authmode=='certificate':
            # WARNING: does not yet work without additionally providing username and password,
            #          temporarily added for now
            cern_username = input('Enter CERN username:')
            cern_password = getpass('Enter CERN password: ')
            login = (cern_username,cern_password)
            kwargs['authmode'] = 'certificate'
            kwargs['login'] = login
            if not (os.path.exists(cert.CERT_TUPLE[0]) and os.path.exists(cert.CERT_TUPLE[1])):
                raise Exception('ERROR: path to certificate files is invalid')
            else:
                print('using certificates: {}'.format(cert.CERT_TUPLE))
            kwargs['certificate'] = cert.CERT_TUPLE
        else:
            raise Exception('ERROR: authentication mode {} not recognized'.format(authmode))
            
    # get data

    response = method(*args,**kwargs)
    
    # return the result
    return response





