#########################################################
# Small conversion utility from a txt file to json file #
#########################################################

import sys
import os
import argparse

if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Conversion from txt to json.')
  parser.add_argument('--inputfile', required=True)
  parser.add_argument('--outputfile', required=True)
  args = parser.parse_args()
  inputfile = os.path.abspath(args.inputfile)
  outputfile = os.path.abspath(args.outputfile)

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # check input file
  if not os.path.exists(inputfile):
    raise Exception('ERROR: input file {} does not exist'.format(inputfile))

  # read MEs in input file
  with open(inputfile,'r') as f:
    mes = f.readlines()
  mes = ['"{}"'.format(me.strip(' \t\n')) for me in mes]

  # make output content
  lines = []
  lines.append('{')
  lines.append('  "conf" : [')
  for me in mes: lines.append(' '*8+me+',')
  lines[-1] = lines[-1].strip(',')
  lines.append('           ]')
  lines.append('}')

  # write output file
  with open(outputfile,'w') as f:
    for line in lines: f.write('{}\n'.format(line))
