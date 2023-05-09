#!/usr/bin/env python

# **Check available lumisections**  
# 
# Run with `python check_dataset_plot.py -h` for a list of available options.  

### imports
import sys
import os
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
sys.path.append(os.path.abspath('../../utils'))
import plot_utils as pu


def find_duplicates(runsls):
  duplicates = []
  for i in range(len(runsls)):
    for j in range(i+1,len(runsls)):
      if runsls[i]==runsls[j]:
        #print('{} {} {}'.format(i,j,runsls[i]))
        duplicates.append(runsls[i])
  return duplicates

def divide_elements(list1, list2):
  both = []
  only1 = []
  only2 = []
  for el in list1:
    if el in list2:
      both.append(el)
    else:
      only1.append(el)
  for el in list2:
    if el in both: continue
    only2.append(el)
  return (both,only1,only2)


if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Check available lumis')
  parser.add_argument('--inputfiles', required=True, nargs='+',
                        help='Input json file(s), space separated,'
                             +' may contain shell expandable wildcards.')
  parser.add_argument('--key1', default='dqmiodas',
                        help='Key in the input json file(s) of first element of comparison')
  parser.add_argument('--label1', default='DQMIO',
                        help='Label for the first element of comparison')
  parser.add_argument('--key2', default='rawdas',
                        help='Key in the input json file(s) of second element of comparison')
  parser.add_argument('--label2', default='RAW',
                        help='Label for the second element of comparison')
  parser.add_argument('--labels', nargs='+',
                        help='Label(s) for plot legends (if specified, must be same length'
                             +' and in same order as the input files).')
  parser.add_argument('--condlabel', default=None,
                        help='Conditions label (e.g. "2022 (13.6 TeV)")'
                             +' (if it contains spaces, wrap it in single quotes).')
  args = parser.parse_args()

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # check arguments
  if args.labels is not None:
    if args.labels==['auto']:
      args.inputfiles = sorted(args.inputfiles)
      args.labels = []
      for inputfile in args.inputfiles:
        label = inputfile.split('_',2)[2]
        label = ' '.join( label.split('-')[:2] )
        if 'Run' in label: label = label.replace('Run','')
        args.labels.append(label)
    if len(args.labels)!=len(args.inputfiles):
      msg = 'ERROR: number of input files and labels must be equal.'
      raise Exception(msg)
  else:
    args.inputfiles = sorted(args.inputfiles)
    args.labels = [None]*len(args.inputfiles)

  # loop over input files
  data = {}
  for inputfile in args.inputfiles:
  
    # read input file
    print('Reading input file {}...'.format(inputfile))
    with open(inputfile,'r') as f:
      info = json.load(f)
    runsls = info[args.key1]
    rawrunsls = info[args.key2]

    # check for duplicates
    do_check_duplicates = False
    if do_check_duplicates:
      print('Checking for duplicates...')
      dup = find_duplicates(runsls)
      if len(dup)>0:
        raise Exception('ERROR: found duplicate lumisections.')
      rawdup = find_duplicates(rawrunsls)
      if len(rawdup)>0:
        raise Exception('ERROR: found duplicate lumisections.')

    # add lumisections to collection
    print('Checking lumisection overlap...')
    (both,onlyraw,onlydqmio) = divide_elements(rawrunsls, runsls)

    # also consider runs
    print('Checking run overlap...')
    runs = sorted(list(set([el[0] for el in runsls])))
    rawruns = sorted(list(set([el[0] for el in rawrunsls])))
    (runsboth,runsonlyraw,runsonlydqmio) = divide_elements(rawruns, runs)

    # check lumisection overlap for runs in both
    print('Checking lumisection overlap in common runs...')
    selectedrunsls = [runlumi for runlumi in runsls if runlumi[0] in runsboth]
    selectedrawrunsls = [runlumi for runlumi in rawrunsls if runlumi[0] in runsboth]
    (selectedboth,selectedonlyraw,selectedonlydqmio) = divide_elements(
      selectedrawrunsls, selectedrunsls)

    # add summary to data for plotting
    completion_num = len(selectedboth)+len(selectedonlydqmio)
    completion_denom = completion_num + len(selectedonlyraw)
    completion = 0.
    if completion_denom>0:
      completion = float(completion_num) / (completion_num + len(selectedonlyraw))
    data[inputfile] = ({ 'present': completion_num, 
                         'missing': len(selectedonlyraw),
                         'completion': completion })

    # print summary
    print('Summary for {}:'.format(inputfile))
    print('  Runs:')
    print('  - runs in both {} and {}: {}'.format(args.label2, args.label1, len(runsboth)))
    print('  - runs in {} only: {}'.format(args.label2, len(runsonlyraw)))
    print('  - runs in {} only: {}'.format(args.label1, len(runsonlydqmio)))
    print('  Lumis:')
    print('  - lumis in both {} and {}: {}'.format(args.label2, args.label1, len(both)))
    print('  - lumis in {} only: {}'.format(args.label2, len(onlyraw)))
    print('  - lumis in {} only: {}'.format(args.label1, len(onlydqmio)))
    print('  Lumis for runs in both:')
    print('  - lumis in both {} and {}: {}'.format(args.label2, args.label1, len(selectedboth)))
    print('  - lumis in {} only: {}'.format(args.label2, len(selectedonlyraw)))
    print('  - lumis in {} only: {}'.format(args.label1, len(selectedonlydqmio)))

    # print runs in RAW only
    #print(sorted(list(runsonlyraw)))
    
    # print lumisections in RAW only in common runs
    print(selectedonlyraw)

    # print lumisections in DQMIO only in common runs
    #print(selectedonlydqmio)

  # define colors
  norm = mpl.colors.Normalize(vmin=0,vmax=len(data))
  cobject = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.cool)
  cobject.set_array([]) # ad-hoc bug fix
  colorlist = [cobject.to_rgba(i) for i in range(len(data))]

  # make a summary plot
  fig,ax = plt.subplots(figsize=(6.4+0.4*(len(data)-1),4.8))
  # set the x-axis tick labels
  ax.set_xticks(range(len(args.labels)))
  if not None in args.labels:
    ax.set_xticklabels(args.labels, rotation=45, horizontalalignment='right')
    fig.subplots_adjust(bottom=0.3)
  # draw the lines
  for i, f in enumerate(args.inputfiles):
    completion = data[f]['completion']
    ax.plot( [i-0.5, i+0.5], [completion, completion], 
             color=colorlist[i], linewidth=3 )
    text = '{} / {}'.format(data[f]['present'], data[f]['present']+data[f]['missing'])
    ax.text(i, completion-0.01, text, fontsize=12,
            horizontalalignment='center', verticalalignment='top')
  # layout
  ax.grid(visible=True)
  ax.set_ylim((ax.get_ylim()[0]*0.8,1.2))
  pu.add_cms_label( ax, extratext='Preliminary', pos=(0.05,0.93),
                      fontsize=12, background_alpha=1. )
  ax.set_xlabel('Dataset and era', fontsize=15)
  ax.set_ylabel('Lumisection completion', fontsize=15)
  if args.condlabel is not None:
    pu.add_text(ax, args.condlabel, (1.0,1.01), horizontalalignment='right')
  fig.savefig('test.png')
