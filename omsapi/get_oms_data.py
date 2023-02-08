#!/usr/bin/env python
# coding: utf-8

# **Functionality to call the OMS API with the correct query based on input parameters**  
# 
# How to use?  
# Check the readme file in this directory for the required setup!  
# In particular, you will need an application ID and client secret to authenticate.  
# 
# Once this is ready, you can do the following:  
# 
# - Import this module, for example via `from get_oms_data import get_oms_api, get_oms_data, get_oms_response_attribute`  
# - Create an instance of the OMS API class using `omsapi = get_oms_api()`  
#   This instance can be re-used for all consecutive calls to OMS, no need to recreate it for every call.  
# - Make a call to `get_oms_data`, where the first argument is the instance you just created.  
#   Other arguments: see the function documentation below.  
# - The returned object is a complicated dictionary containing all information.  
#   Simply print it to find out its exact structure and how to access exactly the values you need.  
#   The function `get_oms_response_attribute` is a small helper function to retrieve a specific attribute from this dictionary.  
#   
# See the notebook example.ipynb in this directory for some examples!  


### imports

# external modules
import sys
import os
import fnmatch

# local modules
from omsapi import OMSAPI
from urls import API_URL, API_VERSION, API_AUDIENCE
from clientid import API_CLIENT_ID, API_CLIENT_SECRET
sys.path.append(os.path.abspath('../utils/notebook_utils'))


### set names of filter attributes

attribute_name = 'attribute_name'
value = 'value'
operator = 'operator'


def get_oms_api():
    ### get an OMSAPI instance
    # takes no input arguments, as the configuration parameters are unlikely to change very often
    # if needed, these parameters can be changed in the file urls.py
    
    omsapi = OMSAPI(api_url=API_URL, api_version=API_VERSION, cert_verify=False)
    omsapi.auth_oidc(API_CLIENT_ID, API_CLIENT_SECRET, audience=API_AUDIENCE)
    return omsapi


def get_oms_data( omsapi, api_endpoint, runnb=None, fillnb=None, extrafilters=[], extraargs={}, sort=None, attributes=[], limit_entries=1000):
    ### query some data from OMS
    # input arguments:
    # - omsapi: an OMSAPI instance, e.g. created by get_oms_api()
    # - api_endpoint: string, target information, e.g. 'runs' or 'lumisections'
    #   (see the readme for a link where the available endpoints are listed)
    # - runnb: run number(s) to retrieve the info for,
    #   either integer (for single run) or tuple or list of two elements (first run and last run)
    #   (can also be None to not filter on run number but this is not recommended)
    # - fillnb: runnb but for fill number instead of run number
    # - extrafilters: list of extra filters (apart from run number),
    #   each filter is supposed to be a dict of the form {'attribute_name':<name>,'value':<value>,'operator':<operator>}
    #   where <name> must be a valid field name in the OMS data, <value> its value, and <operator> chosen from "EQ", "NEQ", "LT", "GT", "LE", "GE" or "LIKE"
    # - extraargs: dict of custom key/value pairs to add to the query
    #   (still experimental, potentially usable for changing the granularity from 'run' to 'lumisection' for e.g. L1 trigger rates, see example.ipynb)
    # - sort: valid field name in the OMS data by which to sort
    # - attributes: list of valid field names in the OMS data to return (if not specified, all information is returned)
    # - limit_entries: entry limit for output json object
    
    filters = []
    
    # check omsapi argument
    if not isinstance(omsapi,OMSAPI):
        raise Exception('ERROR in get_oms_data.py/get_oms_data:'
                       +' first argument is of type '+str(type(omsapi))+' while and OMSAPI object is expected.'
                       +' You can use get_oms_api() to create this object.')
    # check runnb argument
    if runnb is None:
        pass # special case: do not apply run number filter
    elif isinstance(runnb,int):
        filters.append({attribute_name:'run_number',value:str(runnb),operator:'EQ'})
    elif isinstance(runnb,tuple) or isinstance(runnb,list):
        filters.append({attribute_name:'run_number',value:str(runnb[0]),operator:'GE'})
        filters.append({attribute_name:'run_number',value:str(runnb[1]),operator:'LE'})
    else:
        print('WARNING in get_oms_data.py/get_oms_data:'
             +' run number {} not recognized'.format(runnb)
             +' (supposed to be an int, a tuple or list of 2 elements, or None).')
    # check fillnb argument
    if fillnb is None:
        pass # do not apply fill number filter
    elif isinstance(fillnb,int):
        filters.append({attribute_name:'fill_number',value:str(fillnb),operator:'EQ'})
    elif isinstance(fillnb,tuple) or isinstance(fillnb,list):
        filters.append({attribute_name:'fill_number',value:str(fillnb[0]),operator:'GE'})
        filters.append({attribute_name:'fill_number',value:str(fillnb[1]),operator:'LE'})
    else:
        print('WARNING in get_oms_data.py/get_oms_data:'
             +' fill number {} not recognized'.format(fillnb)
             +' (supposed to be an int, a tuple or list of 2 elements, or None).')
    # check extrafilters argument
    expected_keys = sorted([attribute_name,value,operator])
    for extrafilter in extrafilters:
        keys = sorted(extrafilter.keys())
        if not keys==expected_keys:
            print('WARNING in get_oms_data.py/get_oms_data:'
                 +' filter {} contains unexpected keys'.format(extrafilter)
                 +' (expecting only {}).'.format(expected_keys)
                 +' The filter will be added but the query might fail...')
        filters.append(extrafilter)
        
    q = omsapi.query(api_endpoint)
    if len(filters)>0: q.filters(filters)
    if sort is not None: q.sort(sort)
    if len(attributes) is not None: q.attrs(attributes)
    for key,val in extraargs.items(): q.custom(key,value=val)
    q.paginate(1, limit_entries)
    #print(q.data_query())
    response = q.data()
    try: return response.json()
    except: 
        print('ERROR in get_oms_data: could not convert response to json, returning raw response instead.')
        return response


def get_oms_response_attribute( omsresponse, attribute ):
    ### small helper function to retrieve a list of values for a single attribute
    # input arguments:
    # - omsresponse: the json-like object returned by get_oms_data
    # - attribute: name of one of the attributes present in omsresponse
    return [omsresponse['data'][i]['attributes'][attribute] for i in range(len(omsresponse['data']))]

def filter_oms_response( omsresponse, key, value ):
    ### small helper function to post-filter a response object
    # input arguments:
    # - omsresponse: the json-like object returned by get_oms_data
    # - key: name of one of the attributes present in omsresponse
    # - value: string or list of strings with values to keep
    newomsdata = []
    if not isinstance(value,list): value = [value]
    for i in range(len(omsresponse['data'])):
        thisvalue = str(omsresponse['data'][i]['attributes'][key])
        for item in value:
            if( fnmatch.fnmatch(thisvalue,str(item)) ):
                newomsdata.append(omsresponse['data'][i])
                continue
    return {'data':newomsdata}
