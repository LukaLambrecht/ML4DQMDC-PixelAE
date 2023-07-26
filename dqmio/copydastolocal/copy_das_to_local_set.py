#!/usr/bin/env python

# **Copy an entire dataset from DAS to a local area**  
# 
# This script copies a dataset from DAS to a local area.  
# Note: check the size and number of files first, not suitable for large sets.  
# 
# Run with `python copy_das_to_local_set.py -h` for a list of available command line options.  

### imports
import sys
import os
import argparse
sys.path.append('../src/')
from tools import format_input_files, export_proxy
sys.path.append('../../jobsubmission')
import condortools as ct

if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Copy dataset from DAS to local')
  parser.add_argument('--datasetname',
                      help='Full name of the dataset, as displayed on DAS')
  parser.add_argument('--redirector', default='root://cms-xrd-global.cern.ch/',
                      help='Redirector to read remote files')
  parser.add_argument('--outputdir', default=os.path.abspath('.'),
                      help='Local directory where to put the copied file;'
                           ' use "auto" to make a name based on datasetname')
  parser.add_argument('--runmode', default='condor',
                      help='Choose from "condor" or "local";'
                           +' in case of "condor", will submit job to condor cluster;'
                           +' in case of "local", will run interactively in the terminal.')
  parser.add_argument('--proxy', default=os.path.abspath('x509up_u116295'),
                      help='Set the location of a valid proxy created with'
                           +' "--voms-proxy-init --voms cms";'
                           +' needed for DAS client;'
                           +' ignored if filemode is "local".')
  parser.add_argument('--privateprod', default=False, action='store_true',
                      help='Set to true if the DAS dataset is a private production'
                           +' rather than central production (impacts the dasgoclient query);'
                           +' ignored if filemode is "local".')
  parser.add_argument('--maxfiles', default=None,
                      help='Maximum number of files to copy.')
  parser.add_argument('--resubmit', default=False, action='store_true',
                      help='Only try to copy files which are not yet in the output directory.')
  args = parser.parse_args()
  datasetname = args.datasetname
  redirector = args.redirector
  outputdir = args.outputdir
  runmode = args.runmode
  proxy = args.proxy
  privateprod = args.privateprod
  maxfiles = int(args.maxfiles) if args.maxfiles is not None else None
  resubmit = args.resubmit

  # make and execute the DAS client command
  dasfiles = format_input_files( datasetname,
                                 filemode='das',
                                 privateprod=privateprod,
                                 redirector=redirector,
                                 istest=False,
                                 maxfiles=maxfiles )

  # make output directory
  if outputdir=='auto':
    outputdir = datasetname.strip('/').replace('/','_')
  if resubmit:
    if not os.path.exists(outputdir):
      raise Exception('ERROR: output directory {} does not exist'.format(outputdir)
                       +' (required for --resubmit option)')
    existingfiles = os.listdir(outputdir)
    newdasfiles = []
    for dasfile in dasfiles:
      basename = dasfile.split('/')[-1]
      if basename not in existingfiles:
        newdasfiles.append(dasfile)
    dasfiles = newdasfiles
  else:
    if os.path.exists(outputdir):
      raise Exception('ERROR: output directory {} already exists'.format(outputdir))
    os.makedirs(outputdir)

  # make the commands
  cmds = []
  for dasfile in dasfiles:
    cmd = 'xrdcp {} {}'.format(dasfile,outputdir)
    cmds.append(cmd)

  # submit the jobs
  if runmode=='local':
    for cmd in cmds: os.system(cmd)
  elif runmode=='condor':
    ct.submitCommandsAsCondorCluster('cjob_copy_das_to_local_set', cmds, proxy=proxy)
  else:
    raise Exception('ERROR: run mode not recognized: "{}"'.format(runmode))
