#!/usr/bin/env python
# coding: utf-8



### imports

# external modules
import sys
import os
import json
from getpass import getpass

# local modules
import omstools
import cert
sys.path.append(os.path.abspath('../utils/notebook_utils'))




def get_oms_data( mode, run, authmode='login' ):

    # parse arguments
    
    if mode=='run':
        method = omstools.get_runs
        args = [run,run]
    elif mode=='lumisections':
        method = omstools.get_lumisections
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





