#!/usr/bin/env python

import sys
import os
import argparse

if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Make CMSSW configuration file.')
  parser.add_argument('--cmsdriver', required=True,
                      help='Path to txt file with cmsDriver command.')
  parser.add_argument('--conffile', required=True,
                      help='Name of the output configuration file to create.')
  args = parser.parse_args()
  cmsdriverfile = args.cmsdriver
  pcfile = args.conffile

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # check all arguments
  if not os.path.exists(cmsdriverfile):
    raise Exception('ERROR: cmsDriver file {} does not exist.'.format(cmsdriverfile))

  # check if CMSSW environment was set
  if not 'CMSSW_BASE' in os.environ.keys():
    msg = 'ERROR: no CMSSW environment was set, please run cmsenv.'
    raise Exception(msg)
  else:
    print('Found CMSSW_BASE = {}'.format(os.environ['CMSSW_BASE']))

  # read the cmsDriver command
  cmsdrivercmd = ''
  with open(cmsdriverfile,'r') as f:
    cmsdrivercmd = f.readline().strip(' \t\n')
  print('Round following cmsDriver command:')
  print(cmsdrivercmd)
  # check all arguments and remove if needed
  cmsdriverargs = cmsdrivercmd.split('--')
  newcmsdriverargs = []
  replaceargs = ['filein', 'fileout', 'python_filename', 'number', 'no_exec']
  for arg in cmsdriverargs:
    # check if argument is valid and add it if so
    valid = True
    for argtag in replaceargs:
      if argtag in arg:
        print('WARNING: replacing {} arg in cmsDriver command.'.format(argtag))
        valid = False
        continue
    if valid: newcmsdriverargs.append(arg)
  cmsdriverargs = newcmsdriverargs
  cmsdrivercmd = '--'.join(cmsdriverargs)

  # add unconfigurable args to cmsDriver command
  cmsdrivercmd += ' --filein file:{}'.format('placeholderFileIn')
  cmsdrivercmd += ' --fileout file:{}'.format('placeholderFileOut')
  cmsdrivercmd += ' --number {}'.format(99999)
  cmsdrivercmd += ' --python_filename {}'.format(pcfile)
  cmsdrivercmd += ' --no_exec'

  # run the command
  os.system(cmsdrivercmd)

  # open the file and read its content
  with open(pcfile,'r') as f:
    lines = f.readlines()

  # add command line configurability
  with open('internal/clopts.txt','r') as f:
    clopts = f.readlines()
  lines = clopts + ['',''] + lines

  # replace dummy values
  for idx,line in enumerate(lines):
    line = line.replace("'file:placeholderFileIn'","inputFile")
    line = line.replace("'file:placeholderFileOut'","outputFile")
    line = line.replace("(99999)","(nEvents)")
    lines[idx] = line 

  # re-write configuration file
  with open(pcfile,'w') as f:
    for line in lines:
      f.write(line)
