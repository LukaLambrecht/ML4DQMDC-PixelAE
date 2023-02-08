#!/usr/bin/env python

# **Submit check_dataset_lumis_fill.py for a collection of datasets**  
# 
# Run with `python check_dataset_lumis_submit.py -h` for a list of available options.  

### imports
import sys
import os
import argparse
sys.path.append('../../jobsubmission')
import condortools as ct

if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Check available lumis')
  parser.add_argument('--inputfile', required=True,
                        help='Txt file containing names of DQMIO and RAW datasets.')
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
  proxy = None if args.proxy is None else os.path.abspath(args.proxy)

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # read input file
  with open(args.inputfile,'r') as f:
    lines = f.readlines()
  dqmiosets = [line.split(' ')[0].strip(' \t\n') for line in lines]
  rawsets = [line.split(' ')[1].strip(' \t\n') for line in lines]

  # loop over sets
  cmds = []
  for dqmioset,rawset in zip(dqmiosets,rawsets):
    cmd = 'python3 check_dataset_lumis_fill.py'
    cmd += ' --datasetname {}'.format(dqmioset)
    cmd += ' --rawsetname {}'.format(rawset)
    cmd += ' --proxy {}'.format(proxy)
    outtag = dqmioset.strip('/').replace('/','-')
    cmd += ' --outputfile output_checklumis_{}.json'.format(outtag)
    cmds.append(cmd)

  # run the commands
  if args.runmode=='local':
    for cmd in cmds: os.system(cmd)
  elif args.runmode=='condor':
    ct.submitCommandsAsCondorCluster('cjob_check_dataset_size', cmds,
            home='auto', cmssw_version=args.cmssw,
            proxy=proxy, jobflavour=args.jobflavour)
