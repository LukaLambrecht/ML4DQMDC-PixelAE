#####################################################
# Print the available monitoring elements in a file #
#####################################################
# Run "python print_mes.py -h" for a list of available options.

### imports
import sys
import os
import json
import argparse
from fnmatch import fnmatch
sys.path.append('../src')
from DQMIOReader import DQMIOReader
import tools

if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Print available monitoring elements')
  parser.add_argument('--filemode', choices=['das','local'], default='das',
                        help='Choose from "das" or "local"')
  parser.add_argument('--filename', required=True,
                        help='Full name of the file on DAS (for filemode "das")'
                             +' OR path to the local file (for filemode "local")')
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
                            +' only results matching the searchkey will be shown;'
                            +' may contain unix-style wildcards.')
  args = parser.parse_args()
  filemode = args.filemode
  filename = args.filename
  redirector = args.redirector
  searchkey = args.searchkey
  proxy = None if args.proxy is None else os.path.abspath(args.proxy)

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
  print('number of monitoring elements per lumisection: {}'.format(len(menames)))
  for el in menames: print('  - {}'.format(el))
