#!/usr/bin/env python

# **Make a gif of a monitoring element**  
# 
# Run with `python plot_megif.py -h` for a list of available options.
# Note: only very rough plotting for quick checks.

### imports
import sys
import os
import numpy as np
import json
import argparse
from fnmatch import fnmatch
import matplotlib.pyplot as plt
import imageio
sys.path.append('../src')
from DQMIOReader import DQMIOReader
import tools
import metools
from plot_mes import makeplot
sys.path.append('../../utils')
import plot_utils as pu

if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Make a gif')
  parser.add_argument('--filemode', choices=['das','local'], default='das',
                        help='Choose from "das" or "local"')
  parser.add_argument('--filename', required=True,
                        help='Full name of the file on DAS (for filemode "das")'
                             +' OR path to the local file (for filemode "local")')
  parser.add_argument('--mename', required=True,
                        help='Name of the monitoring element for which to make a gif.')
  parser.add_argument('--run', required=True, type=int,
                        help='Run number for which to make the gif.')
  parser.add_argument('--firstls', default=1, type=int,
                        help='Lumisection number for which to start the gif (default: 1).')
  parser.add_argument('--lastls', default=-1, type=int,
                        help='Lumisection number for which to end the gif (default: end of run)')
  parser.add_argument('--duration', default=0.3, type=float,
                        help='Duration in seconds of a single frame.')
  parser.add_argument('--keeptempdir', default=False, action='store_true',
                        help='Do not remove temporary directory with pngs.')
  parser.add_argument('--redirector', default='root://cms-xrd-global.cern.ch/',
                        help='Redirector used to access remote files'
                             +' (ignored in filemode "local").')
  parser.add_argument('--proxy', default=None,
                        help='Set the location of a valid proxy created with'
                             +' "--voms-proxy-init --voms cms";'
                             +' needed for DAS client;'
                             +' ignored if filemode is "local".')
  args = parser.parse_args()
  proxy = None if args.proxy is None else os.path.abspath(args.proxy)

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # export the proxy
  if( args.filemode=='das' and proxy is not None ): tools.export_proxy( proxy )

  # format input file
  if args.filemode=='das':
    redirector = args.redirector.rstrip('/')+'/'
    filename = redirector+args.filename

  # make temporary dir
  tempdir = 'temp'
  if( os.path.exists(tempdir) ):
    msg = 'ERROR: temporary directory "temp" already exists.'
    raise Exception(msg)
  os.makedirs(tempdir)
  
  # make a DQMIOReader instance and initialize it with the file
  print('initializing DQMIOReader...')
  sys.stdout.flush()
  sys.stderr.flush()
  reader = DQMIOReader(*[filename])

  # filter run and lumisection number
  print('filtering lumisections...')
  runsls = sorted(reader.listLumis()[:])
  runsls = [el for el in runsls if el[0]==args.run]
  runsls = [el for el in runsls if el[1]>=args.firstls]
  if args.lastls>0: runsls = [el for el in runsls if el[1]<=args.lastls]
  nlumis = len(runsls)
  print('number of selected lumisections: {}'.format(nlumis))
  if nlumis==0: raise Exception('ERROR: list of lumisections to plot is empty.')

  # make plots
  fignames = []
  for i,runls in enumerate(runsls):
    print('Making plot {}/{}...'.format(i+1,len(runsls)))
    me = reader.getSingleMEForLumi(runls, args.mename)
    fig,ax = makeplot(me)
    if fig is None: continue
    figname = 'fig{}.png'.format(i)
    figname = os.path.join(tempdir,figname)
    fignames.append(figname)
    fig.savefig(figname)

  # make a gif
  with imageio.get_writer('output_gif.gif', mode='I', duration=args.duration) as writer:
    for figname in fignames:
      image = imageio.imread(figname)
      writer.append_data(image)

  # remove temporary directory
  if not args.keeptempdir: os.system('rm -r {}'.format(tempdir))
