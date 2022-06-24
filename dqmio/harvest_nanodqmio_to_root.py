#################################################################################
# A script for reading (nano)DQMIO files and storing a ME in a ROOT file format #
#################################################################################
# Run "python harvest_nanodqmio_to_root.py -h" for a list of available options.
#
# The output is stored in a plain ROOT file, containing only the raw histograms.
# Run and lumisection information is written to the name of the histogram within the ROOT file.

### imports
import sys
import os
import ROOT
import argparse
sys.path.append('src')
from DQMIOReader import DQMIOReader
import tools

if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Harvest nanoDQMIO to CSV')
  parser.add_argument('--filemode', choices=['das','local'], default='das',
                        help='Choose from "das" or "local";'
                              +' in case of "das", will read all files'
                              +' belonging to the specified dataset from DAS;'
                              +' in case of "local", will read all files'
                              +' in the specified folder on the local filesystem.')
  parser.add_argument('--datasetname', required=True,
                        help='Name of the data set on DAS (or filemode "das"'
                             +' OR name of the folder holding input files (for filemode "local"'
                             +' OR comma-separated list of file names'
                             +' (on DAS or locally according to filemode)).'
                             +' Note: interpreted as list of file names if a comma is present,'
                             +' directory or dataset otherwise!')
  parser.add_argument('--redirector', default='root://cms-xrd-global.cern.ch/',
                        help='Redirector used to access remote files'
                             +' (ignored in filemode "local").')
  parser.add_argument('--mename', required=True,
                        help='Name of the monitoring element to store.')
  parser.add_argument('--outputfile', default='test.csv',
                        help='Path to output file.')
  parser.add_argument('--proxy', default=None,
                        help='Set the location of a valid proxy created with'
                             +' "--voms-proxy-init --voms cms";'
                             +' needed for DAS client;'
                             +' ignored if filemode is "local".')
  parser.add_argument('--istest', default=False, action='store_true',
                        help='If set to true, only one file will be read for speed')
  args = parser.parse_args()
  filemode = args.filemode
  datasetname = args.datasetname
  redirector = args.redirector
  mename = args.mename
  outputfile = args.outputfile
  proxy = None if args.proxy is None else os.path.abspath(args.proxy)
  istest = args.istest

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # export the proxy
  if filemode=='das': tools.export_proxy( proxy )

  # make a list of input files
  inputfiles = tools.format_input_files( datasetname,
                                         filemode=filemode,
                                         redirector=redirector,
                                         istest=istest )

  # print configuration parameters
  print('running with following parameters:')
  print('input files:')
  for inputfile in inputfiles: print('  - {}'.format(inputfile))
  print('monitoring element: {}'.format(mename))
  print('outputfile: {}'.format(outputfile))

  # make a DQMIOReader instance and initialize it with the DAS files
  print('initializing DQMIOReader...')
  sys.stdout.flush()
  sys.stderr.flush()
  reader = DQMIOReader(*inputfiles, sortindex=True)
  print('initialized DQMIOReader with following properties')
  print('number of lumisections: {}'.format(len(reader.listLumis())))
  print('number of monitoring elements per lumisection: {}'.format(len(reader.listMEs())))

  # select the monitoring element
  print('selecting monitoring element {}...'.format(mename))
  mes = reader.getSingleMEs(mename)
    
  # write selected monitoring elements to output file
  print('writing output file...')
  f = ROOT.TFile.Open(outputfile, 'recreate')
  for me in mes:
    name = 'run{}_ls{}_{}'.format(me.run, me.lumi, me.name.replace('/','_'))
    me.data.SetName(name)
    me.data.SetTitle(name)
    me.data.Write()
  f.Close()
