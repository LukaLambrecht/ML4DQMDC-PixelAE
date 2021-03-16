#!/usr/bin/env python
# coding: utf-8



### imports

# external modules
import numpy as np
import tensorflow as tf
from keras import backend as K
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import importlib

# local modules
import plot_utils
importlib.reload(plot_utils)




### define loss functions

def mseTop10(y_true, y_pred):
    top_values, _ = tf.nn.top_k(K.square(y_pred - y_true), k=10, sorted=True)
    mean=K.mean(top_values, axis=-1)
    return mean

def mseTop10Raw(y_true, y_pred):
    # same as above but without using tf or K
    # the version including tf or K seemed to cause randomly dying kernels, no clear reason could be found,
    # but it was solved using this loss function instead.
    # verified that it gives exactly the same output as the function above on some random arrays
    # does only work for arrays with 2D shapes, not for (nbins,)
    sqdiff = np.power(y_true-y_pred,2)
    sqdiff[:,::-1].sort()
    sqdiff = sqdiff[:,:10]
    mean = np.mean(sqdiff,axis=-1)
    return mean

def mseTopNRaw(y_true, y_pred, n=10):
    # generalization of the above
    sqdiff = np.power(y_true-y_pred,2)
    sqdiff[:,::-1].sort()
    sqdiff = sqdiff[:,:n]
    mean = np.mean(sqdiff,axis=-1)
    return mean

# attempts to use chi2 instead of mse, so far no good results, but keep for reference
def chiSquared(y_true, y_pred):
    normdiffsq = np.divide(K.square(y_pred - y_true),y_true)
    chi2 = K.sum(normdiffsq,axis=-1)
    return chi2

def chiSquaredTop10(y_true, y_pred):
    normdiffsq = np.divide(K.square(y_pred - y_true),y_true)
    top_values,_ = tf.nn.top_k(normdiffsq,k=10,sorted=True)
    chi2 = K.sum(top_values,axis=-1)
    return chi2




### get roc curve and auc score in case labels are known

def get_roc(scores, labels, mode='classic', doplot=True):
    ### make a ROC curve
    # input arguments:
    # - scores is a 1D numpy array containing output scores of any algorithm
    # - labels is a 1D numpy array (equally long as scores) containing labels
    #   note that 1 for signal and 0 for background is assumed!
    #   this convention is only used to define what scores belong to signal or background;
    #   the scores itself can be anything (not limited to (0,1)), 
    #   as long as the target for signal is higher than the target for background
    # - mode: how to plot the roc curve; options are:
    #         - 'classic' = signal efficiency afo background efficiency
    # - doplot: boolean whether to make a plot or simply return the auc.
    
    nsig = np.sum(labels)
    nback = np.sum(1-labels)
    
    # set score threshold range
    scoremin = np.amin(scores)-1e-7
    # if minimum score is below zero, shift everything up (needed for geomspace)
    if scoremin < 0.: 
        scores = scores - scoremin + 1.
        scoremin = 1.
    scoremax = np.amax(scores)+1e-7
    
    scorerange = np.geomspace(scoremin,scoremax,num=100)
    sig_eff = np.zeros(len(scorerange))
    bkg_eff = np.zeros(len(scorerange))
    
    # loop over thresholds
    print('--- making ROC curve ---')
    for i,scorethreshold in enumerate(scorerange):
        sig_eff[i] = np.sum(np.where((labels==1) & (scores>scorethreshold),1,0))/nsig
        bkg_eff[i] = np.sum(np.where((labels==0) & (scores>scorethreshold),1,0))/nback
        print('  threshold: {:.4f}, signal: {:.4f}, background: {:.4f}'.format(scorethreshold,sig_eff[i],bkg_eff[i]))
    
    # note: sig_eff = signal efficiency = tp = true positive = signal flagged as signal
    # note: bkg_eff = background efficiency = fp = false positive = background flagged as signal
    fn = 1 - sig_eff # signal marked as background
    tn = 1 - bkg_eff # background marked as background
    
    # calculate auc
    if mode=='classic':
        auc = np.trapz(sig_eff[::-1],bkg_eff[::-1])
        if not doplot: return auc
    
        # make plot
        fig,ax = plt.subplots()
        ax.scatter(bkg_eff,sig_eff)
        ax.set_title('ROC curve')
        # general axis titles:
        #ax.set_xlabel('background effiency (background marked as signal)')
        #ax.set_ylabel('signal efficiency (signal marked as signal)')
        # specific axis titles:
        ax.set_xlabel('good histograms flagged as anomalous')
        ax.set_ylabel('bad histograms flagged as anomalous')
        ax.set_xscale('log')
        # set x axis limits
        ax.set_xlim((np.amin(np.where(bkg_eff>0.,bkg_eff,1.))/2.,1.))
        # set y axis limits: general case from 0 to 1.
        #ax.set_ylim(0.,1.1)
        # set y axis limits: adaptive limits based on measured signal efficiency array.
        ylowlim = np.amin(np.where((sig_eff>0.) & (bkg_eff>0.),sig_eff,1.))
        ylowlim = 2*ylowlim-1.
        ax.set_ylim((ylowlim,1+(1-ylowlim)/5))
        ax.grid()
        auctext = str(auc)
        if auc>0.99:
            auctext = '1 - '+'{:.3e}'.format(1-auc)
        ax.text(0.7,0.1,'AUC: '+auctext,transform=ax.transAxes)
        
    else:
        print('ERROR: mode not recognized: '+str(mode))
        return 0
    
    return auc

def get_roc_from_hists(hists, labels, predicted_hists, mode='classic', doplot=True):
    ### make a ROC curve without manually calculating the scores
    # the output score is the mse between the histograms and their reconstruction
    # hists and predicted_hists are 2D numpy arrays of shape (nhistograms,nbins)
    # other arguments: see get_roc

    # determine mse
    mse = mseTop10Raw(hists, predicted_hists)
    # score equals mse, since larger mse = more signal-like (signal=anomalies)
    return get_roc(mse,labels,mode,doplot)

def get_confusion_matrix(scores, labels, wp):
    ### plot a confusion matrix
    # scores and labels are defined in the same way as for get_roc
    # wp is the chosen working point 
    # (i.e. any score above wp is flagged as signal, any below is flagged as background)
    
    nsig = np.sum(labels)
    nback = np.sum(1-labels)
    
    # get confusion matrix entries
    tp = np.sum(np.where((labels==1) & (scores>wp),1,0))/nsig
    fp = np.sum(np.where((labels==0) & (scores>wp),1,0))/nback
    tn = 1-fp
    fn = 1-tp
    cmat = np.array([[tp,fn],[fp,tn]])
    # general labels:
    #df_cm = pd.DataFrame(cmat, index = ['signal','background'],
    #              columns = ['predicted signal','predicted background'])
    # specific labels:
    df_cm = pd.DataFrame(cmat, index = ['bad','good'],
                  columns = ['predicted anomalous','predicted good'])
    #plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues)
    
def get_confusion_matrix_from_hists(hists, labels, predicted_hists, msewp):
    ### plot a confusion matrix without manually calculating the scores
    # the output score is the mse between the histograms and their reconstruction
    
    # get mse
    mse = mseTop10Raw(hists, predicted_hists)
    get_confusion_matrix(mse, labels, msewp)




### getting a keras model ready for training with minimal user inputs

def getautoencoder(input_size,arch,act=[],opt='adam',loss=mseTop10):
    # get a trainable autoencoder model
    # input args:
    # - input_size: size of vector that autoencoder will operate on
    # - arch: list of number of nodes per hidden layer (excluding input and output layer)
    # - act: list of activations per layer (default: tanh)
    # - opt: optimizer to use (default: adam)
    # - loss: loss function to use (defualt: mseTop10)
    
    import math
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    from tensorflow.keras.layers import Input, Dense
    from keras.layers.advanced_activations import PReLU
    from tensorflow.keras.models import Model, Sequential, load_model
    from keras import backend as K
    
    if len(act)==0: act = ['tanh']*len(arch)
    layers = []
    # first layer manually to set input_dim
    layers.append(Dense(arch[0],activation=act[0],input_dim=input_size))
    # rest of layers in a loop
    for nnodes,activation in zip(arch[1:],act[1:]):
        layers.append(Dense(nnodes,activation=activation))
    # last layer is decoder
    layers.append(Dense(input_size,activation='tanh'))
    autoencoder = Sequential()
    for i,l in enumerate(layers):
        #l.name = 'layer_'+str(i)
        autoencoder.add(l)
    autoencoder.compile(optimizer=opt, loss=loss)
    autoencoder.summary()
    return autoencoder




def train_simple_autoencoder(hists,nepochs=-1,modelname=''):
    ### create and train a very simple keras model
    # the model consists of one hidden layer (with half as many units as there are input bins), tanh activation, adam optimizer and mseTop10 loss.
    # input args: 
    # - hists is a 2D numpy array of shape (nhistograms, nbins)
    # - nepochs is the number of epochs to use (has a default value if left unspecified)
    # - modelname is a file name to save the model in (default: model is not saved to a file)
    input_size = hists.shape[1]
    arch = [int(hists.shape[1]/2.)]
    act = ['tanh']*len(arch)
    opt = 'adam'
    loss = mseTop10
    if nepochs<0: nepochs = int(min(40,len(hists)/400))
    model = getautoencoder(input_size,arch,act,opt,loss)
    history = model.fit(hists, hists, epochs=nepochs, batch_size=500, shuffle=False, verbose=1, validation_split=0.1)
    plot_utils.plot_loss(history)
    if len(modelname)>0: model.save(modelname.split('.')[0]+'.h5')
    return model





