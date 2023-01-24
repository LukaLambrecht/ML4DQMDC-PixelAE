#!/usr/bin/env python

# **Profile dataset size: plotting**  
# 
# Run with `python check_dataset_size_plot.py -h` for a list of available options.  

### imports
import sys
import os
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
sys.path.append('../../utils')
import plot_utils as pu
sys.path.append('../../dqmioprod/testing')
from check_file_size_plot import plot, scatter, scatter_polyfit

if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Make dataset size plots')
  parser.add_argument('--inputfiles', required=True, nargs='+',
                        help='Name(s) of input file(s), separated by spaces,'
                             +' can contain shell expandable wildcards.')
  parser.add_argument('--labels', nargs='+',
                        help='Label(s) for plot legends (if specified, must be same length'
                             +' and in same order as the input files).')
  parser.add_argument('--condlabel', default=None,
                        help='Conditions label (e.g. "2022 (13.6 TeV)")'
                             +' (if it contains spaces, wrap it in single quotes).')
  parser.add_argument('--sumlabel', default=None,
                        help='Label for sum/total of individual input files'
                             +' (if not specified, sum will not be plotted).')
  args = parser.parse_args()

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # check arguments
  if args.labels is not None:
    if args.labels==['auto']:
      args.labels = []
      for inputfile in args.inputfiles:
        label = inputfile.split('_',1)[1]
        label = ' '.join( label.split('-')[:2] )
        if 'Run' in label: label = label.replace('Run','')
        args.labels.append(label)
    if len(args.labels)!=len(args.inputfiles):
      msg = 'ERROR: number of input files and labels must be equal.'
      raise Exception(msg)
  else:
    args.labels = [None]*len(args.inputfiles)

  # read input files
  info = []
  for inputfile, label in zip(args.inputfiles, args.labels):
    with open(inputfile,'r') as f:
      thisinfo = json.load(f)
    info.append({'file':inputfile, 'label':label, 'info':thisinfo})
  if args.sumlabel is not None:
    totinfo = sum([el['info'] for el in info],[])
    info.append({'file':None, 'label':args.sumlabel, 'info':totinfo})

  # define colors
  norm = mpl.colors.Normalize(vmin=0,vmax=len(info))
  cobject = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.cool)
  cobject.set_array([]) # ad-hoc bug fix
  colorlist = [cobject.to_rgba(i) for i in range(len(info))]

  # make a plot of size versus number of lumisections
  fig,ax = plt.subplots()
  for i in range(len(info)):
    label = info[i]['label']
    thisinfo = info[i]['info']
    color = colorlist[i]
    parampos = (0.05, 0.8-i*0.07)
    data = {}
    data['x'] = [el['nls'] for el in thisinfo]
    data['y'] = [el['size_bytes'] for el in thisinfo]
    fig,ax = scatter( data, fig=fig, ax=ax,
                      color=color,
                      xtitle='Number of processed lumisections',
                      ytitle='nanoDQMIO file size (bytes)',
                      label=label, legendloc='lower right' )
    fig,ax = scatter_polyfit( data, fig, ax,
                      color=color, paramcolor=color,
                      parampos=parampos, paramfontsize=10,
                      degree=1, showparams=True,
                      label=None, legendloc='lower right' )
  ax.set_ylim((None,ax.get_ylim()[1]*1.2))
  if args.condlabel is not None:
    pu.add_text(ax, args.condlabel, (1.0,1.01), horizontalalignment='right')
  fig.savefig('fig_nlumis_size.png')

  # make a histogram of file size distribution
  fig,ax = plt.subplots()
  for i in range(len(info)):
    # get the data
    label = info[i]['label']
    thisinfo = info[i]['info']
    color = colorlist[i]
    data = [el['size_bytes'] for el in thisinfo]
    # make the label
    label = label + ' (files: {}, size: {:.1e})'.format(len(data),sum(data))
    # make the plot
    ax.hist(data, histtype='step', color=color, label=label)
  # layout
  ax.grid(visible=True)
  pu.add_cms_label( ax, extratext='Preliminary', pos=(0.05,0.93),
                      fontsize=12, background_alpha=1. )
  ax.set_xscale('log')
  #ax.ticklabel_format( axis='x', style='sci', scilimits=(0,0) )
  #ax.set_yscale('log')
  #ax.ticklabel_format( axis='y', style='sci', scilimits=(0,0) )
  ax.set_ylim((None,ax.get_ylim()[1]*1.5))
  ax.legend()
  ax.set_xlabel('nanoDQMIO file size (bytes)', fontsize=15)
  ax.set_ylabel('Number of files', fontsize=15)
  if args.condlabel is not None:
    pu.add_text(ax, args.condlabel, (1.0,1.01), horizontalalignment='right')
  fig.savefig('fig_size_dist.png')

  # make a histogram of lumisections distribution
  fig,ax = plt.subplots()
  for i in range(len(info)):
    # get the data
    label = info[i]['label']
    thisinfo = info[i]['info']
    color = colorlist[i]
    data = [el['nls'] for el in thisinfo]
    # make the label
    label = label + ' (files: {}, lumisections: {:.1e})'.format(len(data),sum(data))
    # make the plot
    ax.hist(data, histtype='step', color=color, label=label)
  # layout
  ax.grid(visible=True)
  pu.add_cms_label( ax, extratext='Preliminary', pos=(0.05,0.93),
                      fontsize=12, background_alpha=1. )
  ax.set_xscale('log')
  #ax.ticklabel_format( axis='x', style='sci', scilimits=(0,0) )
  #ax.set_yscale('log')
  #ax.ticklabel_format( axis='y', style='sci', scilimits=(0,0) )
  ax.set_ylim((None,ax.get_ylim()[1]*1.5))
  ax.legend()
  ax.set_xlabel('Number of lumisections in file', fontsize=15)
  ax.set_ylabel('Number of files', fontsize=15)
  if args.condlabel is not None:
    pu.add_text(ax, args.condlabel, (1.0,1.01), horizontalalignment='right')
  fig.savefig('fig_nls_dist.png')
