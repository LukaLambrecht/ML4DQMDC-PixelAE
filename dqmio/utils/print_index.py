#!/usr/bin/env python

# **Print the index of a file**  
# 
# This is relatively low-level, mostly for debugging purposes.
# Run with `python print_index.py -h` for a list of available options.  

### imports
import sys
import os
import json
import argparse
from fnmatch import fnmatch
sys.path.append('../src')
from DQMIOReader import DQMIOReader
import tools

if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Print the full index')
  parser.add_argument('-d', '--datasetname', required=True,
                        help='Full name of a file on DAS, or full name of a dataset on DAS,'
                             +' or path to the local file, or path to a local directory.')
  parser.add_argument('-r', '--redirector', default='root://cms-xrd-global.cern.ch/',
                        help='Redirector used to access remote files (ignored for local files).')
  parser.add_argument('-p', '--proxy', default=None,
                        help='Set the location of a valid proxy (needed for DAS client, ignored for local files).')
  args = parser.parse_args()
  datasetname = args.datasetname
  redirector = args.redirector
  proxy = None if args.proxy is None else os.path.abspath(args.proxy)

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # export the proxy
  if( not os.path.exists(datasetname) and proxy is not None ):
    print('Exporting proxy...')
    tools.export_proxy( proxy )

  # find files
  print('Retrieving files...')
  filenames = tools.format_input_files(
                datasetname,
                redirector=redirector)

  # make a DQMIOReader instance and initialize it with the files
  print('Initializing DQMIOReader...')
  sys.stdout.flush()
  sys.stderr.flush()
  reader = DQMIOReader(*filenames)

  # print the index
  reader.printIndex()
