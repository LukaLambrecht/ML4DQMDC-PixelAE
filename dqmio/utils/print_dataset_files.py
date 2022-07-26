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
  parser.add_argument('--datasetname', required=True,
                        help='Name of the data set on DAS)')
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

  # make a list of input files
  print('running DAS client to find files in dataset {}...'.format(datasetname))
  dascmd = "dasgoclient -query 'file dataset={}' --limit 0".format(datasetname)
  dasstdout = os.popen(dascmd).read()
  dasfiles = [el.strip(' \t') for el in dasstdout.strip('\n').split('\n')]
  print('DAS client ready; found following files ({}):'.format(len(dasfiles)))
  for f in dasfiles: print('  - {}'.format(f))
