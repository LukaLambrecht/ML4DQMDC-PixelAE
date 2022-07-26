#!/usr/bin/env python

# **Copy a file from DAS to a local area**  
# 
# This script copies a file from DAS to a local area.   
# Note: check the file size first, not suitable for extremely large files.  
# 
# Run with `python copy_das_to_local_file.py -h` to get a list of available command line options.  

### imports
import sys
import os
import argparse
sys.path.append('../../jobsubmission')
import condortools as ct

if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Copy file from DAS to local')
  parser.add_argument('--filename',
                      help='Full path of the file to copy, as displayed on DAS')
  parser.add_argument('--redirector', default='root://cms-xrd-global.cern.ch/',
                      help='Redirector to read remote files')
  parser.add_argument('--outputdir', default=os.path.abspath('.'),
                      help='Local directory where to put the copied file.')
  parser.add_argument('--runmode', default='condor',
                      help='Choose from "condor" or "local";'
                           +' in case of "condor", will submit job to condor cluster;'
                           +' in case of "local", will run interactively in the terminal.')
  parser.add_argument('--proxy', default=os.path.abspath('x509up_u116295'),
                      help='Set the location of a valid proxy created with'
                           +' "--voms-proxy-init --voms cms";'
                           +' needed for DAS client;'
                           +' ignored if filemode is "local".')
  args = parser.parse_args()  
  filename = args.filename
  redirector = args.redirector
  outputdir = args.outputdir
  runmode = args.runmode
  proxy = args.proxy

  # format the filename
  redirector = redirector.rstrip('/')+'/'
  filename = redirector+filename

  # make output directory
  if not os.path.exists(outputdir): os.makedirs(outputdir)

  # make and submit the command
  cmd = 'xrdcp {} {}'.format(filename,outputdir)
  if runmode=='local':
    os.system(cmd)
  elif runmode=='condor':
    ct.submitCommandAsCondorJob('cjob_copy_das_to_local_file', cmd, proxy=proxy)
  else:
    raise Exception('ERROR: run mode not recognized: "{}"'.format(runmode))
