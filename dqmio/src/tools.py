#################################################################
# Collection of tools that are common to the several harvesters #
#################################################################

import sys
import os


def format_input_files( datasetname,
                        filemode='das',
                        runnb=None,
                        privateprod=False,
                        redirector='root://cms-xrd-global.cern.ch/',
                        istest=False,
                        maxfiles=None ):
  ### get a list of input files from a dataset name
  # input arguments:
  # - datasetname: name of the data set on DAS (for filemode 'das')
  #   OR name of the folder holding input files (for filemode 'local')
  #   OR str containing a comma-separated list of file names (on DAS or locally according to filemode))
  #   (note: interpreted as list of file names if a comma is present, directory or dataset otherwise!)
  # - filemode: choose from 'das' or 'local';
  #   in case of 'das', will read all files belonging to the specified dataset from DAS;
  #   in case of 'local', will read all files in the specified folder on the local filesystem.)
  # - runnb: select only files for the given run number (ignored in filemode 'local')
  # - privateprod: whether the dataset on DAS was privately produced (ignored in filemode 'local')
  # - redirector: redirector used to access remote files (ignored in filemode 'local'))
  # - istest: return only first file (for testing)
  # - maxfiles: return only specified number of first files
  # note: the DAS client requires a valid proxy to run,
  #       set it before calling this function with set_proxy() (see below)

  # check if a directory is provided or a list of filenames
  runfilesearch = True
  if ',' in datasetname: runfilesearch = False

  # parse the provided redirector
  redirector = redirector.rstrip('/')+'/'

  # make a list of input files
  if runfilesearch:
    # make a list of input files based on provided directory or dataset name,
    # details depend on the chosen filemode
    if filemode=='das':
      # make and execute the DAS client command
      dasquery = 'file dataset={}'.format(datasetname)
      if privateprod: dasquery += ' instance=prod/phys03'
      if runnb is not None: dasquery += ' run={}'.format(runnb)
      print('running DAS client with following query: {}...'.format(dasquery))
      dascmd = "dasgoclient -query '{}' --limit 0".format(dasquery)
      dasstdout = os.popen(dascmd).read()
      dasfiles = sorted([el.strip(' \t') for el in dasstdout.strip('\n').split('\n')])
      print('DAS client ready; found following files ({}):'.format(len(dasfiles)))
      for f in dasfiles: print('  - {}'.format(f))
      inputfiles = [redirector+f for f in dasfiles]
    elif filemode=='local':
      # read all root files in the given directory
      inputfiles = ([os.path.join(datasetname,f) for f in os.listdir(datasetname)
                     if f[-5:]=='.root'])
  else:
    # parse the provided comma-separated list into a list
    inputfiles = [el for el in datasetname.split(',') if len(el)!=0]
    if( filemode=='das' and redirector is not None ):
      for i,inputfile in enumerate(inputfiles):
        # check if the file name has a redirector already
        if 'root://cms-xrd' in inputfile: continue
        # add the redirector
        inputfiles[i] = redirector+inputfile

  # check number of input files
  if len(inputfiles)==0:
    raise Exception('ERROR: list of input files is empty.')
  if istest:
    print('WARNING: running in test mode, only one file will be processed.')
    inputfiles = [inputfiles[0]]
  if( maxfiles is not None and maxfiles>0 and maxfiles<len(inputfiles) ):
    print('WARNING: returning only {} out of {} files.'.format(maxfiles,len(inputfiles)))
    inputfiles = inputfiles[:maxfiles]

  return inputfiles


def export_proxy( proxy ):
  ### export a provided proxy to the system variables
  print('exporting proxy to {}'.format(proxy))
  os.environ["X509_USER_PROXY"] = proxy
