##########################################
# submitter for DQMIO conversion scripts #
##########################################
# this script wraps conversion scripts (harvest_nanodqmio_to_*.py) in a job.
# the parameters that should be modified according to your needs are explained below.

### imports
import sys
import os
sys.path.append('../jobsubmission')
import condortools as ct

if __name__=='__main__':

  # definitions
  filemode = 'das'
  # (choose from 'das' or 'local';
  #  in case of 'das', will read all files belonging to the specified dataset from DAS;
  #  in case of 'local', will read all files in the specified folder on the local filesystem.)
  datasetname = '/MinimumBias/Commissioning2021-900GeVmkFit-v2/DQMIO'
  # (name of the data set on DAS (or filemode 'das') 
  #  OR name of the folder holding input files (for filemode 'local'))
  redirector = 'root://cms-xrd-global.cern.ch/'
  # (redirector used to access remote files (ignored in filemode 'local'))
  mename = 'PixelPhase1/Tracks/PXBarrel/chargeInner_PXLayer_1'
  # (name of the monitoring element to store)
  outputfile = 'test.csv'
  # (path to output file)
  exe = 'python harvest_nanodqmio_to_csv.py'
  # (executable to run, should be a valid conversion script 
  # similar in structure and command line args to e.g. harvest_nanodqmio_to_csv.py)
  istest = False 
  # (if set to true, only one file will be read for speed)
  runmode = 'condor'
  # (choose from 'condor' or 'local')
  proxy = os.path.abspath('x509up_u23078')
  # (set the location of a valid proxy created with --voms-proxy-init --voms cms
  #  (ignored in filemode 'local'))

  # make a list of input files
  if filemode=='das':
    # make and execute the DAS client command
    print('running DAS client to find files in dataset {}...'.format(datasetname))
    dascmd = "dasgoclient -query 'file dataset={}' --limit 0".format(datasetname)
    dasstdout = os.popen(dascmd).read()
    dasfiles = [el.strip(' \t') for el in dasstdout.strip('\n').split('\n')]
    if istest: 
      dasfiles = [dasfiles[0]] 
    print('DAS client ready; found following files ({}):'.format(len(dasfiles)))
    for f in dasfiles: print('  - {}'.format(f))
    redirector = redirector.rstrip('/')+'/'
    inputfiles = [redirector+f for f in dasfiles]
    if len(inputfiles)==0:
      raise Exception('ERROR: no files found by the DAS client'
		      +' for the queried dataset {}'.format(datasetname))
  elif filemode=='local':
    # read all root files in the given directory
    inputfiles = ([os.path.join(datasetname,f) for f in os.listdir(datasetname)
                      if f[-5:]=='.root'])
    if istest: inputfiles = [inputfiles[0]]
    proxy = None

  # make the command
  cmd = exe
  cmd += ' '+','.join(inputfiles)
  cmd += ' {}'.format(mename)
  cmd += ' {}'.format(outputfile)

  if runmode=='local':
    os.system(cmd)
  if runmode=='condor':
    ct.submitCommandAsCondorJob('cjob_harvest_nanodqmio_submit', cmd, proxy=proxy)
