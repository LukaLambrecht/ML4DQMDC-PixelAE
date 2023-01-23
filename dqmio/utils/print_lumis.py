#!/usr/bin/env python

# **Print the available lumisections in a file**  
# 
# Run with `python print_lumis.py -h` for a list of available options.  

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
  parser = argparse.ArgumentParser(description='Print available lumisections')
  parser.add_argument('--filemode', choices=['das','local'], default='das',
                        help='Choose from "das" or "local"')
  parser.add_argument('--filename', required=True,
                        help='Full name of the file on DAS (for filemode "das")'
                             +' OR path to the local file (for filemode "local")')
  parser.add_argument('--redirector', default='root://cms-xrd-global.cern.ch/',
                        help='Redirector used to access remote files'
                             +' (ignored in filemode "local").')
  parser.add_argument('--proxy', default=None,
                        help='Set the location of a valid proxy created with'
                             +' "--voms-proxy-init --voms cms";'
                             +' needed for DAS client;'
                             +' ignored if filemode is "local".')
  parser.add_argument('--runonly', default=False, action='store_true',
                        help='Print run numbers only, not lumisection numbers.')
  parser.add_argument('--runnb', default=None,
                       help='Print lumisections only for the specified run number.')
  args = parser.parse_args()
  filemode = args.filemode
  filename = args.filename
  redirector = args.redirector
  proxy = None if args.proxy is None else os.path.abspath(args.proxy)
  runonly = args.runonly
  runnb = None if args.runnb is None else int(args.runnb)

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # export the proxy
  if( filemode=='das' and proxy is not None ): tools.export_proxy( proxy )

  # format input file
  if filemode=='das':
    redirector = redirector.rstrip('/')+'/'
    filename = redirector+filename
  
  # make a DQMIOReader instance and initialize it with the file
  print('initializing DQMIOReader...')
  sys.stdout.flush()
  sys.stderr.flush()
  reader = DQMIOReader(*[filename])
  print('initialized DQMIOReader.')

  runsls = sorted(reader.listLumis())
  if runonly:
    runs = set([runls[0] for runls in runsls])
    print('Available runs ({})'.format(len(runs)))
    for run in runs: print('- {}'.format(run))
  else:
    if runnb is not None:
      runsls = [(runls[0],runls[1]) for runls in runsls if runls[0]==runnb]
    print('Available lumisections: ({})'.format(len(runsls)))
    for runls in runsls: print('- Run {}, LS {}'.format(runls[0],runls[1]))
