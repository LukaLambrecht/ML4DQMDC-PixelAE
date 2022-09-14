###################################################################
# Check the size of a nanoDQMIO file as a function of its content #
###################################################################

import sys
import os
import argparse
import json

def convert_file_size(value_in_bytes):
  ### convert a file size in bytes to more readable scales
  # returns a tuple of the form (value as float, unit as str)
  val = value_in_bytes
  units = ['bytes', 'K', 'M', 'G', 'T']
  base = float(1024)
  for i,unit in enumerate(units):
    if( val < base or i==len(units)-1 ):
      return (val, unit)
    val /= base

def get_file_size(filepath):
  ### return the size of a file at specified path
  # return type: tuple of the form (size in bytes, size in readable str format)
  if not os.path.isfile(filepath): return None
  file_size_bytes = os.stat(filepath).st_size
  file_size_str = '{:.1f}{}'.format(*convert_file_size(file_size_bytes))
  return (file_size_bytes,file_size_str)

if __name__=='__main__':

  # read arguments
  parser = argparse.ArgumentParser(description='Check file size')
  parser.add_argument('--cmssw', required=True,
                      help='Path to CMSSW release for working environment.')
  parser.add_argument('--rawfile', required=True,
                      help='Raw file as input.')
  parser.add_argument('--cmsdriver', required=True,
                      help='Path to txt file with cmsDriver command.')
  parser.add_argument('--conf', required=True,
                      help='Path to file with configurable nanoDQMIO contents.')
  parser.add_argument('--outdirname', default='output_check_file_size',
                      help='Results directory to be created in the specified CMSSW area.')
  args = parser.parse_args()
  cmssw = os.path.join(os.path.abspath(args.cmssw),'src')
  rawfile = os.path.abspath(args.rawfile)
  cmsdriverfile = os.path.abspath(args.cmsdriver)
  conffile = os.path.abspath(args.conf)
  outdirname = args.outdirname

  # print arguments
  print('Running with following configuration:')
  for arg in vars(args):
    print('  - {}: {}'.format(arg,getattr(args,arg)))

  # check all arguments
  if not os.path.exists(cmssw):
    raise Exception('ERROR: CMSSW version {} does not exist.'.format(cmssw))
  if not os.path.exists(rawfile):
    raise Exception('ERROR: input file {} does not exist.'.format(rawfile))
  if not os.path.exists(cmsdriverfile):
    raise Exception('ERROR: cmsDriver file {} does not exist.'.format(cmsdriverfile))
  if not os.path.exists(conffile):
    raise Exception('ERROR: configuration file {} does not exist.'.format(conffile))

  # check the output directory
  outdir = os.path.join(cmssw,outdirname)
  if os.path.exists(outdir):
    raise Exception('ERROR: output dir {} already exists.'.format(outdir))

  # check nanoDQMIO configuration file
  nanoconffile = os.path.join(cmssw,'DQMServices/Core/python/nanoDQMIO_perLSoutput_cff.py')
  if not os.path.exists(nanoconffile):
    raise Exception('ERROR: configuration file {} does not exist.'.format(nanoconffile))

  # read the cmsDriver command
  cmsdrivercmd = ''
  with open(cmsdriverfile,'r') as f:
    cmsdrivercmd = f.readline().strip(' \t\n')
  print('Round following cmsDriver command:')
  print(cmsdrivercmd)
  cmsdriverargs = cmsdrivercmd.split('--')
  newcmsdriverargs = []
  replaceargs = ['filein', 'fileout', 'python_filename', 'no_exec']
  for arg in cmsdriverargs:
    valid = True
    for argtag in replaceargs:
      if argtag in arg:
        print('WARNING: replacing {} arg in cmsDriver command.'.format(argtag))
        valid = False
        continue
    if valid: newcmsdriverargs.append(arg)
  cmsdriverargs = newcmsdriverargs
  cmsdrivercmd = '--'.join(cmsdriverargs)

  # add unconfigurable args to cmsDriver command
  outfile = 'step2.root'
  pcfile = 'step2_nano_cfg.py'
  outnanofile = outfile.replace('.root','_inDQM.root')
  cmsdrivercmd += ' --filein file:{}'.format(rawfile)
  cmsdrivercmd += ' --fileout file:{}'.format(outfile)
  cmsdrivercmd += ' --python_filename {}'.format(pcfile)

  # read the configurations
  confdict = {}
  with open(conffile,'r') as f:
    confdict = json.load(f)
  print('Found following configurations:')
  for key, val in confdict.items():
    print('  - {}'.format(key))
    for el in val:
      print('    - {}'.format(el))

  # loop over configurations
  print('Looping over configurations...')
  for conf, melist in confdict.items():
    scriptname = 'temp.sh'
    # note: need to work with a script instead of os.system directly,
    #       since else cmsenv has no effect outside of its scope.
    cmds = []
    # modify the nanoDQM configuration file
    with open(nanoconffile,'w') as f:
      f.write('import FWCore.ParameterSet.Config as cms\n')
      f.write('nanoDQMIO_perLSoutput = cms.PSet(\n')
      f.write('  MEsToSave = cms.untracked.vstring( *(\n')
      for me in melist: f.write('    "{}",\n'.format(me))
      f.write('  ) )\n')
      f.write(')\n')
    # set up the working directory
    cmds.append( 'cd {}'.format(cmssw) )
    cmds.append( 'cmsenv' )
    wdir = os.path.join(outdirname,conf)
    cmds.append( 'mkdir -p {}'.format(wdir) )
    cmds.append( 'cd {}'.format(wdir) )
    # add the cmsdriver command
    cmds.append(cmsdrivercmd)
    # make and run the script
    with open(scriptname,'w') as f:
      for cmd in cmds: f.write('{}\n'.format(cmd))
    os.system('bash {}'.format(scriptname))
    # remove the script
    os.system('rm {}'.format(scriptname))

  # find the file sizes and print output
  print('Checking size of output files...')
  sizedict = {}
  for conf in confdict.keys():
    fpath = os.path.join(cmssw,outdirname,conf,outnanofile)
    res = get_file_size(fpath)
    sizedict[conf] = '{} ({} bytes)'.format(res[1],res[0])
  print('Summary:')
  for key,val in sizedict.items(): print('  {}: {}'.format(key,val))
