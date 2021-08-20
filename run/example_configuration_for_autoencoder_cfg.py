#!/usr/bin/env python
# coding: utf-8

# **Configuration file for training and evaluating autoencoder models on a set of histograms**
# 
# This file contains all settings and variables needed to define the configuration of the full processing chain. See the readme file for a list of all available parameters with some explanation of what they do.  
# 
# Before using this configuration file, a valid HistStruct object that serves as an input to everything must be defined and stored.
# This HistStruct object should contain all histograms (all lumisections and all histogram types) that will be needed for training and testing.
# See create\_histstruct.ipynb for an example script to create such a HistStruct object.  
# 
# This configuration file should first be run separately, in order to configure (but not yet train) the autoencoder models and save them to the HistStruct. See below for an example of how these autoencoder models can be defined and added to the HistStruct.  
# 
# Next, this configuration file can be imported in all following steps (training, fitting and testing).  



# external modules
import os
import sys
import importlib

# local modules: utils
sys.path.append('../utils')
import autoencoder_utils as aeu
from autoencoder_utils import mseTop10
import generate_data_utils as gdu

# local modules: src
print('importing src...')
sys.path.append('../src')
sys.path.append('../src/classifiers')
sys.path.append('../src/cloudfitters')
print('  import HistStruct'); import HistStruct
print('  import AutoEncoder'); import AutoEncoder
print('  import GaussianKdeFitter'); import GaussianKdeFitter
print('refreshing src...')
importlib.reload(HistStruct)
importlib.reload(AutoEncoder)
importlib.reload(GaussianKdeFitter)

print('done')




### parameters for loading the data

HISTSTRUCT_FILE_NAME = 'test.pkl'




### parameters defining training and evaluation masks

# for training
TRAINING_MASKS = ['dcson','highstat', 'training']

# for good test set
TEST_GOOD_MASKS = [['dcson','highstat','good']]
ngoodsets = len(TEST_GOOD_MASKS)
TEST_GOOD_PARTITIONS = [-1]*ngoodsets

# for bad test set
TEST_BAD_MASKS = [['dcson','highstat','bad{}'.format(i)] for i in [0,1,2,3,4,5,6]]
nbadsets = len(TEST_BAD_MASKS)
TEST_BAD_PARTITIONS = [-1]*nbadsets




### load the histstruct and extract some information (needed for GaussianKdeFitter)

histstruct = HistStruct.HistStruct.load( HISTSTRUCT_FILE_NAME, load_classifiers=False )
npoints = histstruct.get_histograms( histname=histstruct.histnames[0], masknames=TRAINING_MASKS ).shape[0]
ndims = len(histstruct.histnames)




### parameters for plotting the input histograms

DO_INITIAL_PLOT = True

# example for local training
INITIAL_PLOT_SETTINGS = ([ {'masknames':[['dcson','highstat','training'],['dcson','highstat','good']],
                            'labellist':['training set','application run'],
                            'colorlist':['blue','green']},
                           {'masknames':[['dcson','highstat','good'],['dcson','highstat','bad']],
                            'labellist':['application run','bad test sets'],
                            'colorlist':['green','red']} ])

# example for global training (with additional label 'good' included in histstruct)
#INITIAL_PLOT_SETTINGS = ([ {'masknames':[['dcson','highstat','good'],['dcson','highstat','bad{}'.format(i)]],
#                            'labellist':['typical good histograms','bad'],
#                            'colorlist':['blue','red'],
#                            'transparencylist':[0.01,1.]} for i in range(nbadsets) ])




### parameters for extend the training set using artificial data

EXTEND_TRAINING = True
EXTEND_TRAINING_FUNCTION = gdu.upsample_hist_set
EXTEND_TRAINING_OPTIONS = {'figname':'','ntarget':5e4}
EXTEND_TRAINING_PARTITIONS = -1




### parameters for defining and training an autoencoder for each element

DO_TRAINING = True

TRAINING_OPTIONS = {'epochs':40,'batch_size':500,'shuffle':False,'verbose':1,'validation_split':0.1}
SAVE_MODELS = False
SAVE_MODELS_DIR = 'test_models'
SAVE_MODELS_BASENAME = 'autoencoder'




### setting up the classifiers

if __name__=='__main__':
    # (do not run this cell again when importing the configuration file at later stages)

    histstruct = HistStruct.HistStruct.load( HISTSTRUCT_FILE_NAME, load_classifiers=False, verbose=True )

    if DO_TRAINING:
    
        for histname in histstruct.histnames:
            input_size = histstruct.get_histograms( histname=histname ).shape[1]
            arch = [int(input_size/2.)]
            model = aeu.getautoencoder(input_size,arch)
            classifier = AutoEncoder.AutoEncoder( model=model )
            histstruct.add_classifier(histname,classifier)

    # in case of false: load models

    else:

        modelloc = '../models/autoencoders_global_training_dcson_highstat_v20210622'
        modelbasename = ''
        for histname in histstruct.histnames:
            print('loading model for {}'.format(histname))
            modelname = modelbasename+'_'+histname+'.h5'
            modelname = os.path.join(modelloc,modelname)
            classifier = AutoEncoder.AutoEncoder.load( modelname,custom_objects={'mseTop10':mseTop10} )
            histstruct.add_classifier(histname,classifier)
        
    # save modified histstruct
    savename = os.path.splitext(HISTSTRUCT_FILE_NAME)[0]+'_configured.pkl'
    histstruct.save( savename )
    
    # delete the loaded histstruct to free some memory
    del histstruct




### parameters for plotting the multidemensional mse and fitting a distribution

CLOUDFITTER_TYPE = GaussianKdeFitter.GaussianKdeFitter
CLOUDFITTER_PLOT_TRAINING = True
CLOUDFITTER_PLOT_TEST = True

# settings for GaussianKdeFitter

scott_bw = npoints**(-1./(ndims+4))
bw_method = 20*scott_bw
CLOUDFITTER_OPTIONS = {'bw_method':bw_method}

# settings for HyperRectangleFitter
quantiles = ([0.00062,0.0006,0.00015,0.00015,
             0.0003,0.0003,0.00053,0.00065])
#CLOUDFITTER_OPTIONS = {'quantiles':quantiles}




### parameters for extending the test set using artificial data

EXTEND_TEST_GOOD = True
EXTEND_TEST_GOOD_FUNCTION = gdu.upsample_hist_set
EXTEND_TEST_GOOD_OPTIONS = {'figname':'','ntarget':7*5e3,'fourierstdfactor':20.}
EXTEND_TEST_GOOD_PARTITIONS = [-1]*ngoodsets

EXTEND_TEST_BAD = True 
EXTEND_TEST_BAD_FUNCTION = gdu.upsample_hist_set
EXTEND_TEST_BAD_OPTIONS = {'figname':'','ntarget':5e3,'fourierstdfactor':20.}
EXTEND_TEST_GOOD_PARTITIONS = [-1]*nbadsets




### parameters for making roc curves and related test statistics

PLOT_SCORE_DIST = True
PLOT_SCORE_DIST_OPTIONS = {'siglabel':'anomalous', 'sigcolor':'r', 
                   'bcklabel':'good', 'bckcolor':'g', 
                   'nbins':200, 'normalize':True,
                   'xaxtitle':'negative logarithmic probability',
                   'yaxtitle':'number of lumisections (normalized)'}

PLOT_ROC_CURVE = True
PLOT_ROC_CURVE_OPTIONS = {'mode':'geom', 'doprint':False}

PLOT_CONFUSION_MATRIX = True
PLOT_CONFUSION_MATRIX_OPTIONS = {'wp':50}




### parameters for investigating particular runs and/or lumisections

INSPECT_MODE = 'run'
INSPECT_RUN = 306458
INSPECT_LS = 204 
INSPECT_MASKS = ['dcson','highstat']
INSPECT_PLOT_SCORE = True

INSPECT_RECO_MODE = 'auto'
INSPECT_REFERENCE_MASKS = ['highstat','dcson','training']
INSPECT_REFERENCE_PARTITIONS = 15




### parameters for evaluating the model on a real set

DO_EVAL = True
EVAL_MASKS = ['golden', 'highstat', 'good']
EVAL_SCORE_UP = 50
EVAL_SCORE_DOWN = None

EVAL_NMAXPLOTS = 1
EVAL_OUTFILENAME = 'autoencoder_golden_json_flags'

EVAL_RECO_MODE = 'auto'
EVAL_REFERENCE_MASKS = ['highstat','dcson','training']
EVAL_REFERENCE_PARTITIONS = 15










