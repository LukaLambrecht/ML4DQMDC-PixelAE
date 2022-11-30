##########################################
# Plot the results of check_file_size.py #
##########################################

import sys
import os
import argparse
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from check_file_size import get_file_size
sys.path.append('../../dqmio/src')
from DQMIOReader import DQMIOReader
sys.path.append('../../utils')
import plot_utils as pu


def plot( data, labels=None, 
          xlog=False, ylog=False, 
          xtitle=None, ytitle=None,
          fig=None, ax=None,
          legendloc='best' ):
  # argument parsing
  dolegend = True
  if labels is None: 
    labels = ['']*len(data)
    dolegend = False
  # make colormap
  ncolors = len(data)
  norm = mpl.colors.Normalize(vmin=0,vmax=ncolors-1)
  cobject = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.rainbow)
  cobject.set_array([]) # ad-hoc bug fix
  # make the figure
  if( fig is None or ax is None ): fig,ax = plt.subplots()
  for i,(thisdata,label) in enumerate(zip(data,labels)):
    ax.plot(thisdata['x'], thisdata['y'], label=label,
            linestyle='--', marker='o', linewidth=2,
            color=cobject.to_rgba(i))
  # formatting
  ax.grid()
  pu.add_cms_label( ax, extratext='Preliminary', pos=(0.05,0.93),
                    fontsize=15, background_alpha=1. )
  if xlog: ax.set_xscale('log')
  if ylog: ax.set_yscale('log')
  ax.ticklabel_format( axis='y', style='sci', scilimits=(0,0) )
  if dolegend: ax.legend(loc=legendloc)
  if xtitle is not None: ax.set_xlabel(xtitle, fontsize=15)
  if ytitle is not None: ax.set_ylabel(ytitle, fontsize=15)
  return (fig,ax)

def scatter( data, label=None,
             xlog=False, ylog=False,
             xtitle=None, ytitle=None,
             fig=None, ax=None,
             legendloc='best' ):
  # argument parsing
  dolegend = True
  if label is None:
    label = ''
    dolegend = False
  # make the figure
  if( fig is None or ax is None): fig,ax = plt.subplots()
  ax.scatter(data['x'], data['y'], label=label)
  # formatting
  ax.grid()
  pu.add_cms_label( ax, extratext='Preliminary', pos=(0.05,0.93),
                    fontsize=15, background_alpha=1. )
  if xlog: ax.set_xscale('log')
  if ylog: ax.set_yscale('log')
  ax.ticklabel_format( axis='y', style='sci', scilimits=(0,0) )
  if dolegend: ax.legend(loc=legendloc)
  if xtitle is not None: ax.set_xlabel(xtitle, fontsize=15)
  if ytitle is not None: ax.set_ylabel(ytitle, fontsize=15)
  return (fig,ax)

def scatter_polyfit( data, fig, ax, 
                     degree=1, showparams=False, 
                     label=None, legendloc='best' ):
  ### add a fit to a scatter plot
  # argument parsing
  dolegend = True
  if label is None:
    label = ''
    dolegend = False
  # make the fit
  params = np.polyfit(data['x'],data['y'], degree)[::-1]
  xax = np.linspace(np.amin(data['x']), np.amax(data['x']), num=100)
  fit = np.zeros(len(xax))
  fitstr = ''
  for degree,param in enumerate(params):
    fit += param * np.power(xax,degree)
    strfrag = '* x$^{}$'.format(degree)
    if degree==0: strfrag = ''
    if degree==1: strfrag = '* x'
    fitstr += ' + {:.1e} {}'.format(param,strfrag)
  fitstr = fitstr.strip(' +')
  fitstr = 'Fit: y = ' + fitstr
  # plot the fit
  ax.plot(xax, fit, label=label, color='red', linestyle='--')
  if showparams: pu.add_text( ax, fitstr, (0.05, 0.8), fontsize=15,
                              background_facecolor='white' )
  # formatting
  if dolegend: ax.legend(loc=legendloc)
  return (fig,ax)
  

if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Make file size trend plots')
  parser.add_argument('--inputdir', required=True, nargs='+',
                      help='Path to directory where results are stored.'
                          +' Supposed to be the directory directly under the CMSSW src,'
                          +' containing one or more configuration directories.'
                          +' Can contain unix-style wildcards for multiple input directories.'
                          +' The correct plotting depends on the naming convention!')
  parser.add_argument('--filename', default='nanodqmio.root',
                      help='Name of the nanoDQMIO file within each directory.')
  parser.add_argument('--outputdir', default='.',
                      help='Directory where to put the plots.')
  parser.add_argument('--donls', default=False, action='store_true',
                      help='Retrieve number of lumisections for each file.')
  parser.add_argument('--donmes', default=False, action='store_true',
                      help='Retrieve number of MEs for each file.')
  args = parser.parse_args()
  inputdir = [os.path.abspath(d) for d in args.inputdir]
  filename = args.filename
  outputdir = os.path.abspath(args.outputdir)
  donls = args.donls
  donmes = args.donmes

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # check all arguments
  for d in inputdir:
    if not os.path.exists(d):
      raise Exception('ERROR: input directory {} does not exist.'.format(d))
  if not os.path.exists(outputdir):
    os.makedirs(outputdir)

  # gather all information
  print('Retrieving all file information...')
  infodict = {}
  fileids = []
  nevents = []
  confs = []
  for d in inputdir:
    # get file ID and number of events from directory name
    try:
      rem,nevent = d.split('_n_')
      nevent = int(nevent)
      fileid = rem.split('_f_')[1]
    except:
      msg = 'WARNING: directory {}'.format(d)
      msg += ' does not have the expected naming convention,'
      msg += ' will skip this directory.'
      print(msg)
      continue
    if nevent not in nevents: nevents.append(nevent)
    if fileid not in fileids: fileids.append(fileid)
    # loop over configuration subdirectories
    for conf in os.listdir(d):
      if conf not in confs: confs.append(conf)
      # get the file size
      f = os.path.join(d,conf,filename)
      if not os.path.exists(f):
        msg = 'WARNING: requested file {} does not exist,'.format(f)
        msg += ' will skip this file.'
        print(msg)
        continue
      size = get_file_size(f)[0]
      # get the number of lumisections and/or number of MEs
      nls = None
      nmes = None
      if( donls or donmes ):
        reader = DQMIOReader(*[f])
        if donls: nls = len(reader.listLumis())
        if donmes: nmes = len(reader.listMEs())
      infodict[(fileid,nevent,conf)] = {'size':size, 'nls':nls, 'nmes':nmes}
  fileids.sort()
  nevents.sort()
  confs.sort()
  print('Found following raw data:')
  print(infodict)
  print(fileids)
  print(nevents)
  print(confs)

  # make a plot as a function of nevents
  data = []
  scatterdata = {'x':[], 'y':[]}
  labels = []
  for fileid in fileids:
    for conf in confs:
      x = []
      y = []
      for nevent in nevents:
        if (fileid,nevent,conf) in infodict.keys():
          size = infodict[(fileid,nevent,conf)]['size']
          x.append(nevent)
          y.append(size)
          scatterdata['x'].append(nevent)
          scatterdata['y'].append(size)
      data.append({'x':x, 'y':y})
      labelparts = []
      if len(fileids)>1: labelparts.append('Raw file: {}'.format(fileid))
      if len(confs)>1: labelparts.append('Configuration: {}'.format(conf))
      labels.append(', '.join(labelparts))
  if( len(fileids)<2 and len(confs)<2 ): labels = None
  fig,ax = plot( data, labels=labels,
                 xlog=True, xtitle='Number of processed events',
                 ytitle='nanoDQMIO file size (bytes)',
                 legendloc='lower right' )
  fig.savefig(os.path.join(outputdir,'fig_nevents_a.png'))
  fig,ax = scatter( scatterdata,
                    xlog=True, xtitle='Number of processed events',
                    ytitle='nanoDQMIO file size (bytes)' )
  fig.savefig(os.path.join(outputdir,'fig_nevents_b.png'))

  # make a plot as a function of number of lumisections
  if donls:
    data = []
    scatterdata = {'x':[], 'y':[]}
    fitdata = {'x':[], 'y':[]}
    labels = []
    for fileid in fileids:
      for conf in confs:
        x = []
        y = []
        for nevent in nevents:
          if (fileid,nevent,conf) in infodict.keys(): 
            nls = infodict[(fileid,nevent,conf)]['nls']
            size = infodict[(fileid,nevent,conf)]['size']
            x.append(nls)
            y.append(size)
            scatterdata['x'].append(nls)
            scatterdata['y'].append(size)
            if nls>1:
              fitdata['x'].append(nls)
              fitdata['y'].append(size)
        data.append({'x':x, 'y':y})
        labelparts = []
        if len(fileids)>1: labelparts.append('Raw file: {}'.format(fileid))
        if len(confs)>1: labelparts.append('Configuration: {}'.format(conf))
        labels.append(', '.join(labelparts))
    if( len(fileids)<2 and len(confs)<2 ): labels = None
    fig,ax = plot( data, labels=labels,
                   xlog=False, xtitle='Number of processed lumisections',
                   ytitle='nanoDQMIO file size (bytes)',
                   legendloc='lower right' )
    fig.savefig(os.path.join(outputdir,'fig_nlumis_a.png'))
    fig,ax = scatter( scatterdata,
                      xlog=False, xtitle='Number of processed lumisections',
                      ytitle='nanoDQMIO file size (bytes)',
                      label='Data', legendloc='lower right' )
    fig,ax = scatter_polyfit( fitdata, fig, ax,
                              degree=1, showparams=True,
                              label='Linear fit', legendloc='lower right' )
    fig.savefig(os.path.join(outputdir,'fig_nlumis_b.png'))
