###################################################
# copy an entire dataset from DAS to a local area #
###################################################
# check the size and number of files first, not suitable for large sets.

### imports
import sys
import os
sys.path.append('../jobsubmission')
import condortools as ct

if __name__=='__main__':

  # definitions
  datasetname = '/MinimumBias/Commissioning2021-900GeVmkFit-v2/DQMIO'
  # (name of the data set on DAS)
  redirector = 'root://cms-xrd-global.cern.ch/'
  # (redirector used to access remote files)
  outputdir = 'auto'
  # (path to output folder; can use 'auto' to make a name based on datasetname)
  proxy = os.path.abspath('x509up_u23078')
  # (set the location of a valid proxy created with --voms-proxy-init --voms cms)

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
  ct.submitCommandsAsCondorCluster('cjob_copy_das_to_local_set', cmds, proxy=proxy)
