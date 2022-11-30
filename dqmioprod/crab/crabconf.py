from WMCore.Configuration import Configuration
import os

# get parameters from submit script
productionlabel = os.environ['CRAB_PRODUCTIONLABEL']
dataset         = os.environ['CRAB_DATASET']
outputsite      = os.environ['CRAB_OUTPUTSITE']
outputdir       = os.environ['CRAB_OUTPUTDIR']
conffile        = os.environ['CRAB_CONFFILE']
outputfile      = 'nanodqmio.root'
if 'CRAB_OUTPUTFILE' in os.environ.keys(): outputfile = os.environ['CRAB_OUTPUTFILE']
lumimask        = None
if 'CRAB_LUMIMASK' in os.environ.keys(): lumimask = os.environ['CRAB_LUMIMASK']
lumisperjob     = 10
if 'CRAB_LUMISPERJOB' in os.environ.keys(): lumisperjob = int(os.environ['CRAB_LUMISPERJOB'])

# process the dataset name
datasetparts = dataset.split('/')
pd = datasetparts[1]
era = datasetparts[2]
dtier = datasetparts[3]
requestname = era + '_' + productionlabel

# get CMSSW release
cmssw_base = os.environ['CMSSW_BASE']

# set work area
workarea = os.path.join(cmssw_base, 'src/crab', productionlabel, pd)

# format pyCfgParams argument
pycfgparams = ['outputFile='+outputfile, 'inputFile='+dataset, 'nEvents=-1']

# print all configurable arguments
print('INFO from crabconf.py')
print('Found following arguments:')
print('  productionlabel = {}'.format(productionlabel))
print('  dataset = {}'.format(dataset))
print('  outputsite = {}'.format(outputsite))
print('  outputdir = {}'.format(outputdir))
print('  conffile = {}'.format(conffile))
print('  outputfile = {}'.format(outputfile))
print('  lumimask = {}'.format(lumimask))
print('  lumisperjob = {}'.format(lumisperjob))
print('  requestname = {}'.format(requestname))
print('  cmssw_base = {}'.format(cmssw_base))
print('  workarea = {}'.format(workarea))

# initialize crab configuration
config = Configuration()

# general settings
config.section_('General')
config.General.transferLogs            = True
config.General.requestName             = requestname
config.General.workArea                = workarea

# job type settings
config.section_('JobType')
config.JobType.psetName                = conffile
config.JobType.pyCfgParams             = pycfgparams
config.JobType.pluginName              = 'analysis'
config.JobType.outputFiles             = [outputfile]
config.JobType.sendExternalFolder      = True
config.JobType.allowUndistributedCMSSW = True
config.JobType.numCores                = 1
config.JobType.maxMemoryMB             = 5000

# data settings
config.section_('Data')
config.Data.inputDataset               = dataset
config.Data.unitsPerJob                = lumisperjob
config.Data.splitting                  = 'LumiBased'
config.Data.outLFNDirBase              = outputdir
config.Data.publication                = False
config.Data.lumiMask                   = lumimask
config.Data.allowNonValidInputDataset  = True

# site settings
config.section_('Site')
config.Site.storageSite                = outputsite
