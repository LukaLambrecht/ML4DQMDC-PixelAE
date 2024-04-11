#!/usr/bin/env python

# **Get the file names belonging to a dataset on DAS**  
# 
# Run with `python print_dataset_files.py -h` for a list of available options.  

### imports
import sys
import os
import json
import argparse
sys.path.append('../src')
import tools

if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Print files in dataset')
  parser.add_argument('-d', '--datasetname', required=True,
                        help='Name of the dataset on DAS.')
  parser.add_argument('-p', '--proxy', default=None,
                        help='Set the location of a valid proxy (needed for DAS client, ignored for local files).')
  args = parser.parse_args()
  datasetname = args.datasetname
  proxy = None if args.proxy is None else os.path.abspath(args.proxy)

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # export the proxy
  if( not os.path.exists(datasetname) and proxy is not None ):
    print('Exporting proxy...')
    tools.export_proxy( proxy )

  # make a list of input files
  print('Retrieving files...')
  filenames = tools.format_input_files(datasetname)
