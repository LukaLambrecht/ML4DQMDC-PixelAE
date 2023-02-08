#################################################################
# Find the number of lumisections in a given data taking period #
#################################################################
# To be used for extrapolating the file size towards a dataset size.

import sys
import os
import copy
import fnmatch
sys.path.append('../../omsapi')
from get_oms_data import get_oms_api
from get_oms_data import get_oms_data
from get_oms_data import get_oms_response_attribute
from get_oms_data import filter_oms_response

def none_to_zero(runs, lumis, verbose=True):
  for i in range(len(runs)):
    if lumis[i] is None:
      if verbose:
        print('WARNING: replacing None by 0 for run {}'.format(runs[i]))
      lumis[i] = 0

if __name__=='__main__':

  # settings
  eranames = ['Run2022?']
  endpoint = 'runs'
  attributes = ([
                  'run_number',
                  'last_lumisection_number',
                  'stable_beam',
                  'fill_type_runtime',
                  'tier0_transfer',
                  'clock_type'
                ])
  query_filters = ([
    {'attribute_name':'stable_beam', 'value':True, 'operator':'EQ'},
    {'attribute_name':'fill_type_runtime','value':'PROTONS','operator':'EQ'}
                  ])
  response_filters = ([
                        ('stable_beam', True),
                        ('fill_type_runtime', 'PROTONS'),
                        ('tier0_transfer', True),
                        ('clock_type', 'LHC')
                     ])

  # get the api client instance
  omsapi = get_oms_api()

  # get all available eras
  response = get_oms_data(omsapi, 'eras')
  print('Found following available eras:')
  print(get_oms_response_attribute(response,'name'))
  response = filter_oms_response(response, 'name', eranames)
  eranames = get_oms_response_attribute(response,'name')
  print('After filter:')
  print(eranames)

  # retrieve the first and last run number for each era
  firstruns = get_oms_response_attribute(response,'start_run')
  lastruns = get_oms_response_attribute(response,'end_run')
  print('Retrieved the following run ranges:')
  for eraname, firstrun, lastrun in zip(eranames,firstruns,lastruns):
    print('  Era: {}: {} - {}'.format(eraname,firstrun,lastrun))

  # loop over eras
  nlumis = {}
  for eraname, firstrun, lastrun in zip(eranames,firstruns,lastruns):
    print('Retrieving run info for era {}'.format(eraname))
    # get run information
    response = get_oms_data( omsapi, endpoint, 
                 runnb=(firstrun,lastrun),
                 extrafilters=query_filters,
                 attributes=attributes )

    # apply filter
    print('Number of runs: {}'.format(len(response['data'])))
    for rfilter in response_filters:
      response = filter_oms_response(response, rfilter[0], rfilter[1])
      print('After filter "{}={}": {}'.format(
            rfilter[0], rfilter[1], len(response['data']) ))
  
    # get number of lumisections
    runnbs = get_oms_response_attribute(response,'run_number')
    lastlumis = get_oms_response_attribute(response,'last_lumisection_number')
    none_to_zero(runnbs,lastlumis)
    thisnlumis = sum(lastlumis)
    print('Number of lumisections for era {}: {}'.format(eraname,thisnlumis))
    nlumis[eraname] = thisnlumis
  
  # print summary
  print('Summary:')
  for eraname in eranames:
    print('Era {}: {}'.format(eraname,nlumis[eraname]))
  print('Total: {}'.format(sum([nlumis[eraname] for eraname in eranames])))
