#!/usr/bin/env python

# **Check available lumisections**  
# 
# Run with `python check_dataset_lumis.py -h` for a list of available options.  

### imports
import sys
import os
import json
import argparse
import signal
sys.path.append('../src')
from DQMIOReader import DQMIOReader
import tools
sys.path.append('../../omsapi')
from get_oms_data import get_oms_api
from get_oms_data import get_oms_data
from get_oms_data import get_oms_response_attribute
from get_oms_data import filter_oms_response


def get_lumis_das(datasetname):
  dascmd = "dasgoclient -query 'run lumi dataset={}' --limit 0".format(datasetname)
  dasstdout = os.popen(dascmd).read()
  runlumis = sorted([el.strip(' \t') for el in dasstdout.strip('\n').split('\n')])
  # format of runlumis is a list with strings of format '<run nb> [<ls nbs>]'
  if len(runlumis)==1 and runlumis[0]=='': return []
  runsls = []
  for runlumi in runlumis:
    run = int(runlumi.split(' ',1)[0])
    lumis = runlumi.split(' ',1)[1]
    lumis = lumis.strip('[] ')
    lumis = lumis.split(',')
    lumis = [int(lumi) for lumi in lumis]
    lumis = set(lumis)
    # (note: the above is to remove duplicates that are sometimes observed;
    #  not sure where it is coming from or what it means...)
    lumis = sorted(list(lumis))
    for lumi in lumis: runsls.append((run,lumi))
  return runsls


def get_lumis_oms(eraname):

  # define filters and attributes
  lumi_attributes = ([
    'lumisection_number',
  ])
  lumi_query_filters = ([
    {'attribute_name':'recorded_lumi_per_lumisection', 'value':1e-12, 'operator':'GT'},
    {'attribute_name':'physics_flag', 'value':True, 'operator':'EQ'}
  ])
  run_attributes = ([
    'run_number',
  ])
  run_query_filters = ([
    {'attribute_name':'stable_beam', 'value':True, 'operator':'EQ'},
    {'attribute_name':'fill_type_runtime', 'value':'PROTONS', 'operator':'EQ'},
    {'attribute_name':'tier0_transfer', 'value':True, 'operator':'EQ'},
    {'attribute_name':'clock_type', 'value':'LHC', 'operator':'EQ'},
    {'attribute_name':'recorded_lumi', 'value':1e-12, 'operator':'GT'}
  ])

  # get the api client instance
  omsapi = get_oms_api()

  # get first and last run of era
  response = get_oms_data(omsapi, 'eras')
  response = filter_oms_response(response, 'name', eraname)
  eranames = get_oms_response_attribute(response, 'name')
  if len(eranames)!=1:
    msg = 'ERROR: wrong number of eras matching pattern {}: {}'.format(eraname, eranames)
    print(msg)
    return
  firstrun = get_oms_response_attribute(response, 'start_run')[0]
  lastrun = get_oms_response_attribute(response, 'end_run')[0]

  # get filtered run numbers
  response = get_oms_data( omsapi, 'runs',
                 runnb=(firstrun,lastrun),
                 extrafilters=run_query_filters,
                 attributes=run_attributes
              )
  runnbs = get_oms_response_attribute(response, 'run_number')

  # get lumisection information
  runsls = []
  for runnb in runnbs:
    response = get_oms_data( omsapi, 'lumisections',
                 runnb=runnb,
                 extrafilters=lumi_query_filters,
                 attributes=lumi_attributes 
              )
    lsnbs = get_oms_response_attribute(response, 'lumisection_number')
    for lsnb in lsnbs: runsls.append((runnb,lsnb))

  return runsls


if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Check available lumisections')
  parser.add_argument('--datasetname', required=True,
                        help='Name of the DQMIO data set on DAS)')
  parser.add_argument('--rawsetname', required=True,
                        help='Name of corresponding RAW data set on DAS')
  parser.add_argument('--outputfile', required=True,
                        help='Name of json output file to write (default: no output file)')
  parser.add_argument('--proxy', default=None,
                        help='Set the location of a valid proxy created with'
                             +' "--voms-proxy-init --voms cms";'
                             +' needed for DAS client.')
  args = parser.parse_args()
  proxy = None if args.proxy is None else os.path.abspath(args.proxy)

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # export the proxy
  if proxy is not None: tools.export_proxy( proxy )

  # initialize output structure
  info = []

  # make a list of input files
  filemode = 'das'
  if os.path.exists(args.datasetname): filemode = 'local'
  filenames = tools.format_input_files(
    args.datasetname, filemode=filemode,
    privateprod=False,
    istest=False, maxfiles=None )
  filenames = sorted(filenames)

  # prepare running with timeout options
  class TimeoutError(Exception): pass
  def handler(sig,frame): raise TimeoutError
  signal.signal(signal.SIGALRM, handler)

  # find lumisections
  runsls = []
  '''print('Finding number of lumisections for DQMIO set using DQMIOReader...')
  for filename in filenames:
    signal.alarm(30)
    try:
      reader = DQMIOReader(*[filename])
      for el in reader.listLumis(): runsls.append(el)
    except:
      msg = 'ERROR: file {} could not be opened.'.format(filename)
      print(msg)
      continue
    signal.alarm(0)
  runsls = sorted(runsls)'''

  # also get lumisections via DAS client
  print('Finding number of lumisections for DQMIO set using DAS client...')
  runslsdas = get_lumis_das(args.datasetname)

  # get runs and lumisections from raw dataset with DAS client
  print('Finding number of lumisections for RAW set using DAS client...')
  rawrunslsdas = get_lumis_das(args.rawsetname)

  # get runs and lumisections from OMS
  era = args.datasetname.strip('/').split('/')[1].split('-')[0]
  print('Finding numbr of lumisections in era {} using OMS API...'.format(era))
  runslsoms = []
  try:
    runslsoms = get_lumis_oms(era)
  except:
    print('ERROR: could not find lumisections with OMS API.')

  # write output
  res = {}
  res['dqmio'] = runsls
  res['dqmiodas'] = runslsdas
  res['rawdas'] = rawrunslsdas
  res['oms'] = runslsoms
  if args.outputfile is not None:
    with open(args.outputfile, 'w') as f:
      json.dump(res,f)
