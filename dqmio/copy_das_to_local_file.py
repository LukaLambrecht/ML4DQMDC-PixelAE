########################################
# copy a file from DAS to a local area #
########################################
# check the file size first, not suitable for large files.

### imports
import sys
import os
sys.path.append('../jobsubmission')
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
  proxy = os.path.abspath('x509up_u23078')
  # (set the location of a valid proxy created with --voms-proxy-init --voms cms)

  # format the filename
  redirector = redirector.rstrip('/')+'/'
  filename = redirector+filename

  # make output directory
  if not os.path.exists(outputdir): os.makedirs(outputdir)

  # make and submit the command
  cmd = 'xrdcp {} {}'.format(filename,outputdir)
  ct.submitCommandAsCondorJob('cjob_copy_das_to_local_file', cmd, proxy=proxy)
