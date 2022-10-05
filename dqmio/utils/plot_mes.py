#!/usr/bin/env python

# **Plot the available monitoring elements in a file**  
# 
# Run with `python plot_mes.py -h` for a list of available options.
# Note: only very rough plotting for quick checks.

### imports
import sys
import os
import json
import argparse
from fnmatch import fnmatch
import matplotlib.pyplot as plt
sys.path.append('../src')
from DQMIOReader import DQMIOReader
import tools
import metools
sys.path.append('../../utils')
import plot_utils as pu

def makeplot( me ):
  ### make a plot of a monitoring element
  # input arguments:
  # - me: instance of MonitorElement
  if me.type in [3,4,5]: return make1dplot(me)
  elif me.type in [6,7,8]: return make2dplot(me)
  elif me.type in [9]: return make3dplot(me)
  else:
    raise Exception('ERROR in makeplot: me.type {} not recognized.'.format(me.type))

def make1dplot( me ):
  ### make a plot of a 1D monitoring element
  # input arguments:
  # - me: instance of MonitorElement with type TH1F, TH1S or TH1D
  hist = metools.me_to_nparray( me, integer_values=True )
  fig,ax = pu.plot_hists([hist], title=pu.make_text_latex_safe(me.name))
  return (fig,ax)

def make2dplot( me ):
  ### make a plot of a 2D monitoring element
  # input arguments:
  # - me: instance of MonitorElement with type TH2F, TH2S or TH2D
  hist = metools.me_to_nparray( me, integer_values=True )
  fig,ax = pu.plot_hist_2d(hist, title=pu.make_text_latex_safe(me.name))
  return (fig,ax)

def make3dplot( me ):
  raise Exception('3D plotting not yet implemented.')
  
if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Plot available monitoring elements')
  parser.add_argument('--filemode', choices=['das','local'], default='das',
                        help='Choose from "das" or "local"')
  parser.add_argument('--filename', required=True,
                        help='Full name of the file on DAS (for filemode "das")'
                             +' OR path to the local file (for filemode "local")')
  parser.add_argument('--run', default=None,
                        help='Run number for which to make the plot (default: first in file).')
  parser.add_argument('--ls', default=None,
                        help='Lumisection number for which to make the plot (default: first in file).')
  parser.add_argument('--redirector', default='root://cms-xrd-global.cern.ch/',
                        help='Redirector used to access remote files'
                             +' (ignored in filemode "local").')
  parser.add_argument('--proxy', default=None,
                        help='Set the location of a valid proxy created with'
                             +' "--voms-proxy-init --voms cms";'
                             +' needed for DAS client;'
                             +' ignored if filemode is "local".')
  parser.add_argument('--searchkey', default=None,
                       help='Provide a search key to filter the results;'
                            +' only results matching the searchkey will be plotted;'
                            +' may contain unix-style wildcards.')
  parser.add_argument('--outputdir', default=None,
                       help='Directory where to store the figures'
                            +' (default: show each figure but do not save)')
  args = parser.parse_args()
  filemode = args.filemode
  filename = args.filename
  run = args.run
  ls = args.ls
  redirector = args.redirector
  searchkey = args.searchkey
  proxy = None if args.proxy is None else os.path.abspath(args.proxy)
  outputdir = None if args.outputdir is None else os.path.abspath(args.outputdir)

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # export the proxy
  if( filemode=='das' and proxy is not None ): tools.export_proxy( proxy )

  # format input file
  if filemode=='das':
    redirector = redirector.rstrip('/')+'/'
    filename = redirector+filename

  # make output dir
  if( outputdir is not None and not os.path.exists(outputdir) ): 
    os.makedirs(outputdir)
  
  # make a DQMIOReader instance and initialize it with the file
  print('initializing DQMIOReader...')
  sys.stdout.flush()
  sys.stderr.flush()
  reader = DQMIOReader(*[filename])
  print('initialized DQMIOReader with following properties')
  print('number of lumisections: {}'.format(len(reader.listLumis())))
  menames = reader.listMEs()
  if searchkey is not None:
    res = []
    for mename in menames:
      if fnmatch(mename,searchkey): res.append(mename)
    menames = res

  # check number of selected histograms
  nmes = len(menames)
  print('{} histograms were selected; continue with plotting? (y/n)'.format(nmes))
  go = raw_input()
  if go!='y': sys.exit()

  # find run and lumisection number
  if( run is None or ls is None ):
    (run,ls) = reader.listLumis()[0]
  print('making plots for run {}, LS {}'.format(run,ls))

  # make plots
  for i,mename in enumerate(menames):
    me = reader.getSingleMEForLumi((run,ls), mename)
    fig,ax = makeplot(me)
    if outputdir is None:
      block=False
      if i==len(menames)-1: block=True 
      plt.show(block=block)
    else:
      figname = mename.replace('/','_').replace(' ','_')+'.png'
      fig.savefig(os.path.join(outputdir,figname))
