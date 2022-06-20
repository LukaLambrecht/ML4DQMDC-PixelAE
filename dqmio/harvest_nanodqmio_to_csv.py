# A script for reading (nano)DQMIO files from DAS 
# and harvesting a selected monitoring element.
#
# The output is stored in a CSV file similar to the ones for the RunII legacy campaign.
# The file format is targeted to be as close as possible to the RunII legacy files,
# with the same columns, data types and naming conventions.
# The only difference is that there are no duplicate columns.
#
# While this file format may be far from optimal,
# it has the advantage that much of the existing code was developed to run on those files,
# so this is implemented to at least have the option to run on new DQMIO files 
# without any code change.
# It was tested that the output files from this script can indeed be read correctly
# by the already existing part of the framework without any code change.
# Note: need to do definitive check (both for 1D and 2D) with collision data
# in order to verify that the shapes are correct (hard to tell with cosmics...)
#
# Note: this script was not yet tested in production, for large amounts of lumisections.
# It will probably need strong speed-ups (if possible) or to be run in a job
# in order to be feasible for harvesting large datasets.

### imports
import sys
import os
import json
import numpy as np
from DQMIOReader import DQMIOReader

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
  istest = False 
  # (if set to true, only one file will be read for speed)

  # overwrite the above default arguments with command line args
  # (mainly for use in jobs submission script)
  if len(sys.argv)>1:
    runfilesearch = False
    inputfiles = sys.argv[1].split(',')
    mename = sys.argv[2]
    outputfile = sys.argv[3]
  else: runfilesearch = True

  if runfilesearch:
    # make a list of input files,
    # details depend on the chosen filemode
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
    elif filemode=='local':
      # read all root files in the given directory
      inputfiles = ([os.path.join(datasetname,f) for f in os.listdir(datasetname)
                      if f[-5:]=='.root'])
      if istest: inputfiles = [inputfiles[0]]

  # print configuration parameters
  print('running with following parameters:')
  print('input files:')
  for inputfile in inputfiles: print('  - {}'.format(inputfile))
  print('monitoring element: {}'.format(mename))
  print('outputfile: {}'.format(outputfile))

  # make a DQMIOReader instance and initialize it with the DAS files
  print('initializing DQMIOReader...')
  sys.stdout.flush()
  sys.stderr.flush()
  reader = DQMIOReader(*inputfiles)
  reader.sortIndex()
  print('initialized DQMIOReader with following properties')
  print('number of lumisections: {}'.format(len(reader.listLumis())))
  print('number of monitoring elements per lumisection: {}'.format(len(reader.listMEs())))

  # select the monitoring element
  print('selecting monitoring element {}...'.format(mename))
  mes = reader.getSingleMEs(mename)
    
  # write selected monitoring elements to output file
  print('writing output file...')
  with open(outputfile, 'w') as f:
    # write header
    header = ',fromrun,fromlumi,hname,metype,histo,entries,Xmax,Xmin,Xbins,Ymax,Ymin,Ybins'
    f.write(header+'\n')
    # extract bin edges (assume the same for all monitoring elements!)
    metype = mes[0].type
    if metype in [3,4,5]:
      nxbins = mes[0].data.GetNbinsX()
      xmin = mes[0].data.GetBinLowEdge(1)
      xmax = mes[0].data.GetBinLowEdge(nxbins)+mes[0].data.GetBinWidth(nxbins)
      nybins = 1
      ymin = 0
      ymax = 1
    elif metype in [6,7,8]:
      nxbins = mes[0].data.GetNbinsX()
      xmin = mes[0].data.GetXaxis().GetBinLowEdge(1)
      xmax = (mes[0].data.GetXaxis().GetBinLowEdge(nxbins)
		    +mes[0].data.GetXaxis().GetBinWidth(nxbins))
      nybins = mes[0].data.GetNbinsY()
      ymin = mes[0].data.GetYaxis().GetBinLowEdge(1)
      ymax = (mes[0].data.GetYaxis().GetBinLowEdge(nybins)
		    +mes[0].data.GetYaxis().GetBinWidth(nybins))
    else:
      raise Exception('ERROR: monitoring element type not recognized: {}'.format(metype))
    # loop over monitoring elements
    for idx,me in enumerate(mes):
      # extract the histogram
      if metype in [3,4,5]:
        histo = np.zeros(nxbins+2, dtype=int)
        for i in range(nxbins+2):
          histo[i] = int(me.data.GetBinContent(i))
      elif metype in [6,7,8]:
        histo = np.zeros((nxbins+2)*(nybins+2), dtype=int)
        for i in range(nybins+2):
          for j in range(nxbins+2):
            histo[i*(nxbins+2)+j] = int(me.data.GetBinContent(j,i))
      # format info
      meinfo = ''
      meinfo += '{}'.format(idx)
      meinfo += ',{}'.format(int(me.run))
      meinfo += ',{}'.format(int(me.lumi))
      meinfo += ',{}'.format(me.name.split('/')[-1])
      meinfo += ',{}'.format(me.type)
      meinfo += ',"{}"'.format(json.dumps(list(histo)))
      meinfo += ',{}'.format(int(np.sum(histo)))
      meinfo += ',{}'.format(xmax)
      meinfo += ',{}'.format(xmin)
      meinfo += ',{}'.format(nxbins)
      meinfo += ',{}'.format(ymax)
      meinfo += ',{}'.format(ymin)
      meinfo += ',{}'.format(nybins)
      f.write(meinfo+'\n')
