#!/usr/bin/env python

# **Find out how runs are distributed over files in a dataset**  
# 
# Run with `python find_runs_in_dataset.py -h` for a list of available options.  

### imports
import sys
import os
import json
import argparse
sys.path.append('../src')
from DQMIOReader import DQMIOReader
import tools

if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Find which files contain which runs')
  parser.add_argument('--filemode', choices=['das','local'], default='das',
                        help='Choose from "das" or "local"')
  parser.add_argument('--datasetname', required=True,
                        help='Name of the dataset on DAS (for filemode "das")'
                             +' OR path to a local directory (for filemode "local")')
  parser.add_argument('--redirector', default='root://cms-xrd-global.cern.ch/',
                        help='Redirector used to access remote files'
                             +' (ignored in filemode "local").')
  parser.add_argument('--proxy', default=None,
                        help='Set the location of a valid proxy created with'
                             +' "--voms-proxy-init --voms cms";'
                             +' needed for DAS client;'
                             +' ignored if filemode is "local".')
  args = parser.parse_args()
  filemode = args.filemode
  datasetname = args.datasetname
  redirector = args.redirector
  proxy = None if args.proxy is None else os.path.abspath(args.proxy)

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # export the proxy
  if( filemode=='das' and proxy is not None ): tools.export_proxy( proxy )

  # get the input files
  filenames = tools.format_input_files(
    datasetname, filemode=filemode,
    privateprod=False, redirector=redirector,
    istest=False, maxfiles=None )
  filenames = sorted(filenames)

  # initialize the output dicts
  filedict = {}
  rundict = {}

  # loop over files
  for i,filename in enumerate(filenames):
    print('Processing file {}/{}...'.format(i+1,len(filenames)))
 
    # make a DQMIOReader instance and initialize it with the file
    print('initializing DQMIOReader...')
    sys.stdout.flush()
    sys.stderr.flush()
    reader = DQMIOReader(*[filename])
    print('initialized DQMIOReader.')

    # get the runs
    runsls = sorted(reader.listLumis())
    runs = set([runls[0] for runls in runsls])

    # add to the output dicts
    filedict[filename] = list(runs)
    for run in runs:
      if run not in rundict.keys(): rundict[run] = [filename]
      else: rundict[run].append(filename)

  # print results
  print('Results for file to run mapping:')
  for filename in filenames:
    print('  - File: {}'.format(filename))
    for run in filedict[filename]:
      print('    - {}'.format(run))
  print('Results for run to file mapping:')
  for run in sorted(rundict.keys()):
    print('  - Run: {}'.format(run))
    for filename in rundict[run]:
      print('    - {}'.format(filename))
