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

# Make the code work for both python 2 and 3
# Use input from Python 3
from six.moves import input

sys.path.append('../src/')
from tools import format_input_files, export_proxy
sys.path.append('../../jobsubmission')
import condortools as ct

if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Copy dataset from DAS to local')
  parser.add_argument('-d', '--datasetname',
                      help='Full name of the dataset, as displayed on DAS')
  parser.add_argument('-r', '--redirector', default='root://cms-xrd-global.cern.ch/',
                      help='Redirector to read remote files')
  parser.add_argument('-o', '--outputdir', default=os.path.abspath('.'),
                      help='Local directory where to put the copied file;'
                           ' use "auto" to make a name based on datasetname')
  parser.add_argument('-m', '--runmode', default='condor',
                      help='Choose from "condor" or "local";'
                           +' in case of "condor", will submit job to condor cluster;'
                           +' in case of "local", will run interactively in the terminal.')
  parser.add_argument('-p', '--proxy', default=os.path.abspath('x509up_u116295'),
                      help='Set the location of a valid proxy created with'
                           +' "--voms-proxy-init --voms cms";'
                           +' needed for DAS client.')
  parser.add_argument('--run', default=None,
                      help='Run number (optional, to download files for a specific run only)')
  parser.add_argument('--privateprod', default=False, action='store_true',
                      help='Set to true if the DAS dataset is a private production'
                           +' rather than central production (impacts the dasgoclient query).')
  parser.add_argument('--maxfiles', default=None,
                      help='Maximum number of files to copy (default: all files in dataset).')
  parser.add_argument('--filesperjob', type=int, default=1,
                      help='Number of files to copy per job (default: 1)')
  parser.add_argument('--jobflavour', default='workday',
                        help='Set the job flavour in lxplus'
                             +' (see https://batchdocs.web.cern.ch/local/submit.html)')
  parser.add_argument('--resubmit', default=False, action='store_true',
                      help='Only try to copy files which are not yet in the output directory.')
  args = parser.parse_args()
  datasetname = args.datasetname
  runnb = args.run
  redirector = args.redirector
  outputdir = args.outputdir
  runmode = args.runmode
  proxy = args.proxy
  privateprod = args.privateprod
  maxfiles = int(args.maxfiles) if args.maxfiles is not None else None
  filesperjob = args.filesperjob
  jobflavour = args.jobflavour
  resubmit = args.resubmit

  # make and execute the DAS client command
  dasfiles = format_input_files( datasetname,
                                 location='das',
                                 runnb=runnb,
                                 privateprod=privateprod,
                                 redirector=redirector,
                                 istest=False,
                                 maxfiles=maxfiles )

  # make output directory
  if outputdir=='auto': outputdir = datasetname.strip('/').replace('/','_')
  if not resubmit:
    if os.path.exists(outputdir):
      raise Exception('ERROR: output directory {} already exists'.format(outputdir))
    os.makedirs(outputdir)
  else:
    if not os.path.exists(outputdir):
      raise Exception('ERROR: output directory {} does not exist'.format(outputdir)
                       +' (required for --resubmit option)')

  # handle case of resubmission
  if resubmit:
    existingfiles = os.listdir(outputdir)
    newdasfiles = []
    for dasfile in dasfiles:
      basename = dasfile.split('/')[-1]
      if basename not in existingfiles:
        newdasfiles.append(dasfile)
    print('Found {} out of {} files already present.'.format(len(existingfiles),len(dasfiles)))
    print('Will submit remaining {} files.'.format(len(newdasfiles)))
    dasfiles = newdasfiles

  # group the files
  groupeddasfiles = []
  idx = 0
  while idx<len(dasfiles):
    groupeddasfiles.append( dasfiles[idx:idx+filesperjob] )
    idx += filesperjob
  
  # ask for confirmation
  print('Found {} files to download into {}.'.format(len(dasfiles),outputdir))
  print('Will submit {} jobs.'.format(len(groupeddasfiles)))
  go = input('Proceed? (y/n) ')
  if go!='y': sys.exit()

  # make the commands
  cmds = []
  for idx,dasfilegroup in enumerate(groupeddasfiles):
    script = 'cjob_copy_das_to_local_set_temp{}.sh'.format(idx)
    with open(script, 'w') as f:
      for dasfile in dasfilegroup:
        f.write('xrdcp {} {}\n'.format(dasfile,outputdir))
    cmds.append('bash {}'.format(script))

  # submit the jobs
  if runmode=='local':
    for cmd in cmds: os.system(cmd)
  elif runmode=='condor':
    ct.submitCommandsAsCondorCluster('cjob_copy_das_to_local_set', cmds, 
      proxy=proxy, jobflavour=jobflavour)
  else:
    raise Exception('ERROR: run mode not recognized: "{}"'.format(runmode))
