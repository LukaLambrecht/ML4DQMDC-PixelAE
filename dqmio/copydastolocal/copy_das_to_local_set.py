###################################################
# Copy an entire dataset from DAS to a local area #
###################################################
# This script copies a dataset from DAS to a local area.
# The parameters that need to be modified for your needs are explained below.
# Note: check the size and number of files first, not suitable for large sets.

### imports
import sys
import os
import argparse
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
  args = parser.parse_args()
  datasetname = args.datasetname
  redirector = args.redirector
  outputdir = args.outputdir
  runmode = args.runmode
  proxy = args.proxy

  # temp
  print(datasetname)
  print(redirector)
  print(outputdir)
  print(runmode)
  print(proxy)

  # make and execute the DAS client command
  print('running DAS client to find files in dataset {}...'.format(datasetname))
  dascmd = "dasgoclient -query 'file dataset={}' --limit 0".format(datasetname)
  dasstdout = os.popen(dascmd).read()
  dasfiles = [el.strip(' \t') for el in dasstdout.strip('\n').split('\n')]
  print('DAS client ready; found following files ({}):'.format(len(dasfiles)))
  for f in dasfiles: print('  - {}'.format(f))
  redirector = redirector.rstrip('/')+'/'
  dasfiles = [redirector+f for f in dasfiles]

  # make output directory
  if outputdir=='auto':
    outputdir = datasetname.strip('/').replace('/','_')
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
