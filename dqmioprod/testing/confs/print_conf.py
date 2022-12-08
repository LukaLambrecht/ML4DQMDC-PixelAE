#####################################################
# Print a configuration for easier copying to CMSSW #
#####################################################

import sys
import os
import argparse
import json

if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Print configuration.')
  parser.add_argument('--conffile', required=True)
  parser.add_argument('--confname', required=True)
  args = parser.parse_args()
  conffile = os.path.abspath(args.conffile)
  confname = args.confname

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # check input file
  if not os.path.exists(conffile):
    raise Exception('ERROR: input file {} does not exist'.format(conffile))

  # read configuration file
  with open(conffile,'r') as f:
    confdict = json.load(f)
  
  # check configuration name
  if not confname in confdict.keys():
    raise Exception('ERROR: configuration file {}'.format(conffile)
                   +' does not contain configuration "{}"'.format(confname))

  # do printing
  for el in confdict[confname]:
    print('    "{}",'.format(el)) 
