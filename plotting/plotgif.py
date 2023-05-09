########################
# Make an animated GIF #
########################

# external modules
import sys
import os
import numpy as np
import argparse

# local modules
sys.path.append('../utils')
import dataframe_utils as dfu
import plot_utils as pu
sys.path.append('../src')
import DataLoader

if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Make a GIF')
  parser.add_argument('--inputfile', required=True, type=os.path.abspath)
  parser.add_argument('--mename', default=None)
  parser.add_argument('--outputfile', required=True, type=os.path.abspath)
  parser.add_argument('--run', default=None)
  parser.add_argument('--lsfirst', default=1)
  parser.add_argument('--lslast', default=10)
  args = parser.parse_args()

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # read the data
  print('Loading data...')
  dloader = DataLoader.DataLoader()
  df = dloader.get_dataframe_from_file( args.inputfile )
  print('Number of entries in the dataframe: {}'.format(len(df)))

  # select the monitoring element
  mename = args.mename
  if mename is None: mename = dfu.get_histnames(df)[0]
  print('Selecting ME {}'.format(mename))
  df = dfu.select_histnames(df, histnames=[mename])
  print('Number of passing entries: {}'.format(len(df)))

  # select the run and lumisections
  run = args.run
  if run is None: run = dfu.get_runs(df)[0]
  run = int(run)
  ls = np.arange(int(args.lsfirst),int(args.lslast)+1)
  print('Selecting run {} and LS {} - {}'.format(run,ls[0],ls[-1]))
  df = dfu.select_runs(df, [run])
  df = dfu.select_ls(df, ls)
  print('Number of selected lumisections: '+str(len(df)))

  # get the histograms as a np array
  (histograms,runnbs,lsnbs) = dfu.get_hist_values(df)

  # make the gif 
  titles = []
  for lsnb,runnb in zip(lsnbs,runnbs):
    titles.append('Run {}, Lumisection {}'.format(runnb,lsnb))
  pu.plot_hists_2d_gif(histograms, titles=titles, figname=args.outputfile)
