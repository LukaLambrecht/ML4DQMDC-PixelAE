##########################################
# submitter for DQMIO conversion scripts #
##########################################
# this script wraps conversion scripts (harvest_nanodqmio_to_*.py) in a job.
# the difference with respect to harvest_nanodqmio_submit.py is that this script
# makes it more easy to harvest multiple monitoring elements in one go
# (instead of modifying and resubmitting harvest_nanodqmio_submit.py sequentially).
# the parameters that should be modified according to your needs are explained below.

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
  menames = ({
    'PixelPhase1/Tracks/PXBarrel/chargeInner_PXLayer_1': 'output_chargeInner_PXLayer_1.csv',
    'PixelPhase1/Tracks/PXBarrel/chargeInner_PXLayer_2': 'output_chargeInner_PXLayer_2.csv',
    'PixelPhase1/Tracks/PXBarrel/chargeInner_PXLayer_3': 'output_chargeInner_PXLayer_3.csv',
    'PixelPhase1/Tracks/PXBarrel/chargeInner_PXLayer_4': 'output_chargeInner_PXLayer_4.csv'
	    })	
  # (names of the monitoring elements to store matched to output files)
  exe = 'python harvest_nanodqmio_to_csv.py'
  # (executable to run, should be a valid conversion script 
  # similar in structure and command line args to e.g. harvest_nanodqmio_to_csv.py)
  runmode = 'condor'
  # (choose from 'condor' or 'local')
  proxy = os.path.abspath('x509up_u23078')
  # (set location of a valid proxy created with --voms-proxy-init --voms cms)

  # make and execute the DAS client command
  print('running DAS client to find files in dataset {}...'.format(datasetname))
  dascmd = "dasgoclient -query 'file dataset={}' --limit 0".format(datasetname)
  dasstdout = os.popen(dascmd).read()
  dasfiles = [el.strip(' \t') for el in dasstdout.strip('\n').split('\n')]
  print('DAS client ready; found following files ({}):'.format(len(dasfiles)))
  for f in dasfiles: print('  - {}'.format(f))
  redirector = redirector.rstrip('/')+'/'
  dasfiles = [redirector+f for f in dasfiles]
  if len(dasfiles)==0:
    raise Exception('ERROR: no files found by the DAS client'
		    +' for the queried dataset {}'.format(datasetname))

  # loop over the monitoring elements
  cmds = []
  for mename,outputfile in menames.items():
    # make the command
    cmd = exe
    cmd += ' '+','.join(dasfiles)
    cmd += ' {}'.format(mename)
    cmd += ' {}'.format(outputfile)
    cmds.append(cmd)
  
  if runmode=='local':
    for cmd in cmds: os.system(cmd)
  if runmode=='condor':
    ct.submitCommandsAsCondorCluster('cjob_harvest_nanodqmio_submitmultiple', cmds, proxy=proxy)
