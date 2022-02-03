# A script for reading (nano)DQMIO files from DAS 
# and harvesting a selected monitoring element.
#
# The output is stored in a plain ROOT file, containing only the raw histograms.
# Run and lumisection information is written to the name of the histogram within the ROOT file.

### imports
import sys
import os
import ROOT
from DQMIOReader import DQMIOReader

if __name__=='__main__':

    # definitions
    datasetname = '/MinimumBias/Commissioning2021-900GeVmkFit-v2/DQMIO'
    # (name of the data set on DAS)
    redirector = 'root://cms-xrd-global.cern.ch/'
    # (redirector used to access remote files)
    mename = 'PixelPhase1/Tracks/PXBarrel/chargeInner_PXLayer_1'
    # (name of the monitoring element to store)
    outputfile = 'test.root'
    # (path to output file)
    istest = True 
    # (if set to true, only one file will be read for speed)

    # make and execute the DAS client command
    print('running DAS client to find files in dataset {}...'.format(datasetname))
    dascmd = "dasgoclient -query 'file dataset={}' --limit 0".format(datasetname)
    dasstdout = os.popen(dascmd).read()
    dasfiles = [el.strip(' \t') for el in dasstdout.strip('\n').split('\n')]
    if istest: dasfiles = [dasfiles[0]] 
    print('DAS client ready; found following files ({}):'.format(len(dasfiles)))
    for f in dasfiles: print('  - {}'.format(f))

    # make a DQMIOReader instance and initialize it with the DAS files
    print('initializing DQMIOReader...')
    redirector = redirector.rstrip('/')+'/'
    dasfiles = [redirector+f for f in dasfiles]
    reader = DQMIOReader(*dasfiles)
    reader.sortIndex()
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
