#!/usr/bin/env python

# **Submit check_dataset_size_fill.py for a collection of datasets**  
# 
# Run with `python check_dataset_size_submit.py -h` for a list of available options.  

### imports
import sys
import os
import argparse

# Make the code work for both python 2 and 3
# Use input from Python 3
from six.moves import input

sys.path.append('../../jobsubmission')
import condortools as ct
sys.path.append('../src')
import tools

if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Profile dataset size')
  parser.add_argument('--datasetname', required=True,
                        help='Name of the dataset on DAS (can contain unix-style wildcards)')
  parser.add_argument('--proxy', default=None,
                        help='Set the location of a valid proxy created with'
                             +' "--voms-proxy-init --voms cms";'
                             +' needed for DAS client;')
  parser.add_argument('--runmode', default='local', choices=['local','condor'])
  parser.add_argument('--cmssw', default=None,
                        help='Set the location of a CMSSW release;'
                             +' needed for remote file reading with xrootd.')
  parser.add_argument('--jobflavour', default='workday',
                        help='Set the job flavour in lxplus'
                             +' (see https://batchdocs.web.cern.ch/local/submit.html)')
  args = parser.parse_args()
  datasetname = args.datasetname
  proxy = None if args.proxy is None else os.path.abspath(args.proxy)

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # export the proxy
  if proxy is not None: tools.export_proxy( proxy )

  # find datasets
  print('Finding datasets corresponding to tag "{}"...'.format(datasetname))
  dascmd = "dasgoclient -query 'dataset={}' --limit 0".format(datasetname)
  dasstdout = os.popen(dascmd).read()
  datasetnames = sorted([el.strip(' \t') for el in dasstdout.strip('\n').split('\n')])
  print('DAS client ready; found following datasets ({}):'.format(len(datasetnames)))
  for d in datasetnames: print('  - {}'.format(d))
  
  # ask for confirmation to continue
  go = input('Continue? (y/n) ')
  if not go=='y': sys.exit()

  # loop over datasets
  cmds = []
  for i,datasetname in enumerate(datasetnames):
    cmd = 'python3 check_dataset_size_fill.py'
    cmd += ' --datasetname {}'.format(datasetname)
    cmd += ' --proxy {}'.format(proxy)
    outtag = datasetname.strip('/').replace('/','-')
    cmd += ' --outputfile output_{}.json'.format(outtag)
    cmds.append(cmd)

  # run the commands
  if args.runmode=='local':
    for cmd in cmds: os.system(cmd)
  elif args.runmode=='condor':
    ct.submitCommandsAsCondorCluster('cjob_check_dataset_size', cmds,
            home='auto', cmssw_version=args.cmssw,
            proxy=proxy, jobflavour=args.jobflavour)
