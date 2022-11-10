###########################################
# script to rename nanoDQMIO output files #
###########################################

import sys
import os
import fnmatch
import argparse

if __name__=='__main__':

  # parse arguments
  parser = argparse.ArgumentParser(description='Rename output files')
  parser.add_argument('--folder', required=True, type=os.path.abspath)
  parser.add_argument('--renamefrom', default='step2_inDQM.root')
  parser.add_argument('--renameto', default='nanodqmio.root')
  args = parser.parse_args()

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  renamelist = {}
  for root, dirnames, filenames in os.walk(args.folder):
    for filename in fnmatch.filter(filenames, args.renamefrom):
      rnfrom = os.path.join(root, filename)
      rnto = rnfrom.replace(args.renamefrom, args.renameto)
      renamelist[rnfrom] = rnto

  print('Will perform the following renamings:')
  for key,val in renamelist.items():
    print('{} -> {}'.format(key,val))
  print('Continue? (y/n)')
  go = raw_input()
  if not go=='y': sys.exit()

  for key,val in renamelist.items():
    cmd = 'mv {} {}'.format(key,val)
    os.system(cmd)
