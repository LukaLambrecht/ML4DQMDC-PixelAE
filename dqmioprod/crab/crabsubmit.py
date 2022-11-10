#!/usr/bin/env python

import os
import sys
import subprocess
import argparse

if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Submit DQMIO production with CRAB')
  parser.add_argument('--conffile', required=True,
                      help='CMSSW configuration file to submit.')
  parser.add_argument('--samplelist', required=True,
                      help='Path to txt file containing sample names on DAS.')
  parser.add_argument('--productionlabel', required=True,
                      help='Unique label for this production.')
  parser.add_argument('--outputsite', required=True,
                      help='Site to write output to.')
  parser.add_argument('--outputdir', required=True,
                      help='Directory to write output to.')
  parser.add_argument('--lumisperjob', type=int, default=10,
                      help='Number of lumisections per job')
  args = parser.parse_args()

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # check configuration file
  if not os.path.exists(args.conffile):
    msg = 'ERROR: configuration file {} does not exist.'.format(args.conffile)
    raise Exception(msg)

  # check if CMSSW environment was set
  if not 'CMSSW_BASE' in os.environ.keys():
    msg = 'ERROR: no CMSSW environment was set, please run cmsenv.'
    raise Exception(msg)
  else:
    print('Found CMSSW_BASE = {}'.format(os.environ['CMSSW_BASE']))

  # read datasets from sample list
  datasets = []
  with open(args.samplelist,'r') as f:
    lines = f.readlines()
  for line in lines:
    line = line.strip(' \t\n')
    if line.startswith('#'): continue
    if line.startswith('%'): continue
    if len(line)==0: continue
    datasets.append(line)

  # set crab environment
  os.environ['CRAB_PRODUCTIONLABEL'] = args.productionlabel
  os.environ['CRAB_OUTPUTSITE'] = args.outputsite
  os.environ['CRAB_OUTPUTDIR'] = args.outputdir
  os.environ['CRAB_LUMISPERJOB'] = str(args.lumisperjob)
  os.environ['CRAB_CONFFILE'] = args.conffile

  # loop over datasets
  for dataset in datasets:
    print('Submitting {} using CRAB...'.format(dataset))
    os.environ['CRAB_DATASET'] = dataset
    os.environ['CRAB_OUTPUTFILE'] = 'nanodqmio.root'
    # for testing configuration
    #os.system('python crabconf.py')
    # for actual submission
    os.system('crab submit -c crabconf.py')
