########################################
# Copy a file from DAS to a local area #
########################################
# This script copies a file from DAS to a local area.
# The parameters that need to be modified for your needs are explained below.
# Note: check the file size first, not suitable for extremely large files.

### imports
import sys
import os
sys.path.append('../../jobsubmission')
import condortools as ct

if __name__=='__main__':

  # definitions
  filename = '/store/data/Commissioning2021/MinimumBias/DQMIO/900GeVmkFit-v2/80000/'
  filename += '4003A4C0-5F91-11EC-9601-B6828E80BEEF.root'
  # (name of the data set on DAS)
  redirector = 'root://cms-xrd-global.cern.ch/'
  # (redirector used to access remote files)
  outputdir = '.'
  # (path to output directory)
  runmode = 'condor'
  # (choose from 'condor' or 'local')
  proxy = os.path.abspath('x509up_u23078')
  # (set the location of a valid proxy created with --voms-proxy-init --voms cms)

  # format the filename
  redirector = redirector.rstrip('/')+'/'
  filename = redirector+filename

  # make output directory
  if not os.path.exists(outputdir): os.makedirs(outputdir)

  # make and submit the command
  cmd = 'xrdcp {} {}'.format(filename,outputdir)
  if runmode=='local':
    os.system(cmd)
  elif runmode=='condor:':
    ct.submitCommandAsCondorJob('cjob_copy_das_to_local_file', cmd, proxy=proxy)
  else:
    raise Exception('ERROR: run mode not recognized: "{}"'.format(runmode))
