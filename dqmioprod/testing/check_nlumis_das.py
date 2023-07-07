#####################################################
# Check number of lumisections using the DAS client #
#####################################################

import sys
import os
import argparse

# Make it work under both python 2 and 3
# Use input from Python 3
from six.moves import input

if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Find number of lumisections using DAS')
  parser.add_argument('--datasettag', required=True,
                      help='DAS dataset search key, can contain wildcards.')
  parser.add_argument('--print_runs', default=False, action='store_true')
  args = parser.parse_args()

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # find datasets
  print('Finding datasets matching provided input...')
  dascmd = "dasgoclient -query 'dataset={}' --limit 0".format(args.datasettag)
  dasstdout = os.popen(dascmd).read()
  datasets = sorted([el.strip(' \t') for el in dasstdout.strip('\n').split('\n')])
  print('DAS client ready; found following datasets ({}):'.format(len(datasets)))
  for dataset in datasets: print('  - {}'.format(dataset))

  # check if need to continue
  go = input('Continue to find number of lumisections for these datasets? (y/n) ')
  if not go=='y': sys.exit()

  # find number of lumisections for each dataset
  nlumis = {}
  for dataset in datasets:
    print('Finding number of lumisections for dataset {}...'.format(dataset))
    dascmd = "dasgoclient -query 'run lumi dataset={}' --limit 0".format(dataset)
    dasstdout = os.popen(dascmd).read()
    runlumis = sorted([el.strip(' \t') for el in dasstdout.strip('\n').split('\n')])
    # format of runlumis is a list with strings of format '<run nb> [<ls nbs>]'
    runs = [int(runlumi.split(' ',1)[0]) for runlumi in runlumis]
    if args.print_runs:
      print('Found following run numbers:')
      print(runs)
    thisnlumis = 0
    for runlumi in runlumis:
      lumis = runlumi.split(' ',1)[1]
      lumis = lumis.strip('[] ')
      lumis = lumis.split(',')
      thisnlumis += len(lumis)
    nlumis[dataset] = thisnlumis
    print('DAS client ready; found {} lumisections.'.format(thisnlumis))
    
  # print summary
  print('Summary:')
  for dataset in datasets:
    print('Dataset {}: {}'.format(dataset, nlumis[dataset]))
  print('Total: {}'.format(sum([nlumis[dataset] for dataset in datasets]))) 
