#!usr/bin/env python

import sys
import os
import argparse

if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Test CMSSW configuration file.')
  parser.add_argument('--conffile', required=True,
                      help='Name of the configuration file to test.')
  parser.add_argument('--inputfile', required=True,
                      help='Path to the input file (in RAW format).')
  parser.add_argument('--outputfile', required=True,
                      help='Name of the output file.')
  parser.add_argument('--nevents', type=int, default=100,
                      help='Number of events to process.')
  args = parser.parse_args()

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # check all arguments
  if not os.path.exists(args.conffile):
    raise Exception('ERROR: configuration file {} does not exist.'.format(args.conffile))
  if os.path.exists(args.inputfile):
    args.inputfile = 'file:{}'.format(args.inputfile)
  else:
    print('WARNING: input file {} does not exist;'.format(args.inputfile)
          +' will proceed assuming this is a non-local file...')

  # make the command
  cmd = 'cmsRun {}'.format(args.conffile)
  cmd += ' inputFile={}'.format(args.inputfile)
  cmd += ' outputFile={}'.format(args.outputfile)
  cmd += ' nEvents={}'.format(args.nevents)

  # run the command
  print('Running following command:')
  print(cmd)
  os.system(cmd)
