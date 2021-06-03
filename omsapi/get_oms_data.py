#!/usr/bin/env python
# coding: utf-8



### imports

# external modules
import sys
import os

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




def get_oms_data( omsapi, api_endpoint, runnb, extrafilters=[], extraargs={}, sort=None, attributes=[]):
    ### query some data from OMS
    # input arguments:
    # - omsapi: an OMSAPI instance, e.g. created by get_oms_api()
    # - api_endpoint: string, target information, e.g. 'runs' or 'lumisections'
    #   (see the readme for a link where the available endpoints are listed)
    # - runnb: run number(s) to retrieve the info for,
    #   either integer (for single run) or tuple or list of two elements (first run and last run)
    #   (can also be None to not filter on run number but this is not recommended)
    # - extrafilters: list of extra filters (apart from run number),
    #   each filter is supposed to be a dict of the form {'attribute_name':<name>,'value':<value>,'operator':<operator>}
    #   where <name> must be a valid field name in the OMS data, <value> its value, and <operator> chosen from "EQ", "NEQ", "LT", "GT", "LE", "GE" or "LIKE"
    # - extraargs: dict of custom key/value pairs to add to the query
    #   (still experimental, potentially usable for changing the granularity from 'run' to 'lumisection' for e.g. L1 trigger rates, see example.ipynb)
    # - sort: valid field name in the OMS data by which to sort
    # - attributes: list of valid field names in the OMS data to return (if not specified, all information is returned)
    
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
    q.paginate(1,1000)
    print(q.data_query())
    response = q.data()
    return response.json()




def get_oms_response_attribute( omsresponse, attribute ):
    ### small helper function to retrieve a list of values for a single attribute
    # input arguments:
    # - omsresponse: the json-like object returned by get_oms_data
    # - attribute: name of one of the attributes present in omsresponse
    
    return [omsresponse['data'][i]['attributes'][attribute] for i in range(len(omsresponse['data']))]





