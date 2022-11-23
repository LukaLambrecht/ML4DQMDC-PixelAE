######################################
# Basic cmsDriver and cmsRun wrapper #
######################################

import sys
import os
import argparse
import json

if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Basic cmsDriver wrapper')
  parser.add_argument('--rawfile', required=True,
                      help='Raw file as input.')
  parser.add_argument('--outputfile', required=True,
                      help='Name of the output nanoDQMIO file to create.')
  parser.add_argument('--cmsdriver', required=True,
                      help='Path to txt file with cmsDriver command.')
  parser.add_argument('--conffile', default='nanodqmio_conf.py',
                      help='Name of cmsRun configuration file to create.')
  parser.add_argument('--docmsrun', default=False, action='store_true',
                      help='Whether or not to execute cmsRun on the created configuration file.')
  parser.add_argument('--nevents', default=None,
                      help='Set number of events to process in cmsDriver command.'
                          +' If not specified, this argument is not added to cmsDriver.')
  args = parser.parse_args()
  rawfile = os.path.abspath(args.rawfile)
  outputfile = args.outputfile
  cmsdriverfile = os.path.abspath(args.cmsdriver)
  conffile = args.conffile
  docmsrun = args.docmsrun
  nevents = args.nevents

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # check all arguments
  if not os.path.exists(rawfile):
    raise Exception('ERROR: input file {} does not exist.'.format(rawfile))
  if not os.path.exists(cmsdriverfile):
    raise Exception('ERROR: cmsDriver file {} does not exist.'.format(cmsdriverfile))

  # read the cmsDriver command
  cmsdrivercmd = ''
  with open(cmsdriverfile,'r') as f:
    cmsdrivercmd = f.readline().strip(' \t\n')
  print('Round following cmsDriver command:')
  print(cmsdrivercmd)
  # check all arguments and remove if needed
  cmsdriverargs = cmsdrivercmd.split('--')
  newcmsdriverargs = []
  replaceargs = ['filein', 'fileout', 'python_filename', 'no_exec']
  if nevents is not None: replaceargs.append('number')
  hasreco = False
  for arg in cmsdriverargs:
    # check if argument is valid and add it if so
    valid = True
    for argtag in replaceargs:
      if argtag in arg:
        print('WARNING: replacing {} arg in cmsDriver command.'.format(argtag))
        valid = False
        continue
    if valid: newcmsdriverargs.append(arg)
    # check if the cmsDriver command includes RECO data
    # (has an impact on the output file naming)
    if( ('datatier' in arg and 'RECO' in arg) 
        or ('eventcontent' in arg and 'RECO' in arg) ):
      hasreco = True
  cmsdriverargs = newcmsdriverargs
  cmsdrivercmd = '--'.join(cmsdriverargs)
  if nevents is not None: cmsdrivercmd += ' --number {}'.format(nevents)
  if not docmsrun: cmsdrivercmd += ' --no_exec'

  # add unconfigurable args to cmsDriver command
  cmsdrivercmd += ' --filein file:{}'.format(rawfile)
  cmsdrivercmd += ' --fileout file:{}'.format(outputfile)
  cmsdrivercmd += ' --python_filename {}'.format(conffile)

  # run the command
  os.system(cmsdrivercmd)
