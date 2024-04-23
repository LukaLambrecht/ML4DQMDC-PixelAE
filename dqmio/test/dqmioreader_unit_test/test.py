#!/usr/bin/env python

# **Small unit test script**  
# 
# Run with `python unit_test.py -h` for a list of available options.  

### imports
import sys
import os
import json
import argparse
from fnmatch import fnmatch
sys.path.append('../../src')
from DQMIOReader import DQMIOReader
import tools

if __name__=='__main__':

  # settings
  inputfile = 'CC89B224-A251-4039-89F0-848A12B6CF84.root'
  searchkey = '*charge_PXLayer_*'
  runlumi = (374797, 80)
  mename = 'PixelPhase1/Tracks/PXBarrel/charge_PXLayer_2'

  # make a DQMIOReader instance and initialize it with the input file(s)
  print('Initializing DQMIOReader...')
  sys.stdout.flush()
  sys.stderr.flush()
  reader = DQMIOReader(*[inputfile])
  print('Initialized DQMIOReader with following properties')
  print('Lumisections: ({})'.format(len(reader.listLumis())))
  for lumi in reader.listLumis(): print('  - {}'.format(lumi))
  menames = reader.listMEs(namepatterns=searchkey)
  sorting_threshold = 1000
  if len(menames)<sorting_threshold: menames = sorted(menames)
  print('Monitoring elements per lumisection: ({})'.format(len(menames)))
  for el in menames: print('  - {}'.format(el))

  # retrieve a monitoring element and check method consistency
  me1 = reader.getMEsForLumi(runlumi, namepatterns=mename)
  me2 = reader.getSingleMEForLumi(runlumi, mename)
  me3 = reader.getMEs(runlumis=runlumi, namepatterns=mename)
  me4 = reader.getSingleMEs(mename, runlumis=runlumi)
  print(me1)
  print(me2)
  print(me3)
  print(me4)

  # retrieve lists of monitoring elements
  mes = reader.getMEs(namepatterns=searchkey)
  print(len(mes))
  mes = reader.getMEs(namepatterns=mename)
  print(len(mes))
  mes = reader.getSingleMEs(mename, callback=None)
  print(len(mes))

  # convert to dataframe
  df = reader.getSingleMEsToDataFrame(mename)
  print(df)
