#!/usr/bin/env python

# **Profile dataset size: collecting information**  
# 
# Run with `python check_dataset_size_fill.py -h` for a list of available options.  

### imports
import sys
import os
import json
import argparse
import ROOT
import signal
sys.path.append('../src')
from DQMIOReader import DQMIOReader
import tools

if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Get dataset size information')
  parser.add_argument('--datasetname', required=True,
                        help='Name of the data set on DAS)')
  parser.add_argument('--outputfile', required=True,
                        help='Name of json output file to write (default: no output file)')
  parser.add_argument('--proxy', default=None,
                        help='Set the location of a valid proxy created with'
                             +' "--voms-proxy-init --voms cms";'
                             +' needed for DAS client;')
  args = parser.parse_args()
  datasetname = args.datasetname
  proxy = None if args.proxy is None else os.path.abspath(args.proxy)

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # export the proxy
  if proxy is not None: tools.export_proxy( proxy )

  # initialize output structure
  info = []

  # make a list of input files
  filemode = 'das'
  if os.path.exists(datasetname): filemode = 'local'
  filenames = tools.format_input_files(
    datasetname, filemode=filemode,
    privateprod=False,
    istest=False, maxfiles=None )
  filenames = sorted(filenames)

  # prepare running with timeout options
  class TimeoutError(Exception): pass
  def handler(sig,frame): raise TimeoutError
  signal.signal(signal.SIGALRM, handler)

  # loop over files
  for i,filename in enumerate(filenames):
    print('Processing file {}/{}...'.format(i+1,len(filenames)))
  
    # open the file to get size
    signal.alarm(30)
    try:
      f = ROOT.TFile.Open(filename,'read')
    except:
      msg = 'ERROR: file {} could not be opened.'.format(filename)
      print(msg)
      continue
    signal.alarm(0)
    size = f.GetSize()
 
    # find number of lumisections
    signal.alarm(30)
    try:
      reader = DQMIOReader(*[filename])
    except:
      msg = 'ERROR: file {} could not be opened.'.format(filename)
      print(msg)
      continue
    signal.alarm(0)
    runsls = sorted(reader.listLumis())
    nls = len(runsls)

    # add to the info
    info.append( {'file':filename, 'nls':nls, 'size_bytes':size} )

  # write output
  if args.outputfile is not None:
    with open(args.outputfile, 'w') as f:
      json.dump(info,f)
