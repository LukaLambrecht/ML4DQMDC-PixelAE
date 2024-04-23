#!/usr/bin/env python

# **Print the available monitoring elements in a file**  
# 
# Run with `python print_mes.py -h` for a list of available options.  

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
  parser = argparse.ArgumentParser(description='Print available monitoring elements')
  parser.add_argument('-d', '--datasetname', required=True,
                        help='Full name of a file on DAS, or full name of a dataset on DAS,'
                             +' or path to the local file, or path to a local directory.')
  parser.add_argument('-r', '--redirector', default='root://cms-xrd-global.cern.ch/',
                        help='Redirector used to access remote files (ignored for local files).')
  parser.add_argument('-p', '--proxy', default=None,
                        help='Set the location of a valid proxy (needed for DAS client, ignored for local files).')
  parser.add_argument('-o', '--outputfile', default=None,
                       help='File to write output to (default: print on screen).')
  parser.add_argument('--searchkey', default=None,
                       help='Provide a search key to filter the results;'
                            +' only results matching the searchkey will be shown;'
                            +' may contain unix-style wildcards.')
  parser.add_argument('--number_only', default=False, action='store_true',
                       help='Print number of monitoring elements only;'
                            +' not a full list of their names.')
  args = parser.parse_args()
  datasetname = args.datasetname
  redirector = args.redirector
  searchkey = args.searchkey
  proxy = None if args.proxy is None else os.path.abspath(args.proxy)
  number_only = args.number_only
  outputfile = args.outputfile

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
  print('initializing DQMIOReader...')
  sys.stdout.flush()
  sys.stderr.flush()
  reader = DQMIOReader(*filenames)
  print('initialized DQMIOReader with following properties')
  print('number of lumisections: {}'.format(len(reader.listLumis())))
  menames = reader.listMEs(namepatterns=searchkey)
  sorting_threshold = 1000
  if len(menames)<sorting_threshold: 
    menames = sorted(menames)

  # write output
  header = 'Number of monitoring elements per lumisection: {}'.format(len(menames))
  if outputfile is None:
    print(header)
    if not number_only:
      for el in menames: print('  - {}'.format(el))
  else:
    with open(outputfile,'w') as f:
      f.write(header+'\n')
      if not number_only:
        for el in menames: f.write(el+'\n')
