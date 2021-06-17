#!/usr/bin/env python
# coding: utf-8

# **Utilities related to the training and evaluation of autoencoder models with keras**
# 
# The functionality in this script includes:
# - definition of loss functions (several flavours of MSE or chi-squared)
# - calculating and plotting ROC curves and confusion matrices
# - definition of very simple ready-to-use keras model architectures



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
    ### MSE top 10 loss function for autoencoder training
    # input arguments:
    # - y_true and y_pred: two numpy arrays of equal shape,
    #   typically a histogram and its autoencoder reconstruction.
    #   if two-dimensional, the arrays are assumed to have shape (nhists,nbins)!
    # output:
    # - mean squared error between y_true and y_pred,
    #   where only the 10 bins with largest squared error are taken into account.
    #   if y_true and y_pred are 2D arrays, this function returns 1D array (mseTop10 for each histogram)
    top_values, _ = tf.nn.top_k(K.square(y_pred - y_true), k=10, sorted=True)
    mean=K.mean(top_values, axis=-1)
    return mean

def mseTop10Raw(y_true, y_pred):
    ### same as mseTop10 but without using tf or K
    # the version including tf or K seemed to cause randomly dying kernels, no clear reason could be found,
    # but it was solved using this loss function instead.
    # verified that it gives exactly the same output as the function above on some random arrays.
    # contrary to mseTop10, this function only works for arrays with 2D shapes (so shape (nhists,nbins)), not for (nbins,).
    sqdiff = np.power(y_true-y_pred,2)
    sqdiff[:,::-1].sort()
    sqdiff = sqdiff[:,:10]
    mean = np.mean(sqdiff,axis=-1)
    return mean

def mseTopNRaw(y_true, y_pred, n=10):
    ### generalization of mseTop10Raw to any number of bins to take into account
    # note: now generalized to also work for 2D histograms, i.e. arrays of shape (nhists,nybins,nxbins)!
    #       hence this is the most general method and preferred above mseTop10 and mseTop10Raw, which are only kept for reference
    # input arguments:
    # - y_true, y_pred: numpy arrays between which to calculate the mean square difference, of shape (nhists,nbins) or (nhists,nybins,nxbins)
    # - n: number of largest elements to keep for averaging
    # output:
    # numpy array of shape (nhists)
    sqdiff = np.power(y_true-y_pred,2)
    if len(sqdiff.shape)==3:
        sqdiff = sqdiff.reshape(len(sqdiff),-1)
    sqdiff = np.partition( sqdiff, -n, axis=-1 )[:,-n:]
    mean = np.mean( sqdiff, axis=-1 )
    return mean

# attempts to use chi2 instead of mse, so far no good results, but keep for reference
def chiSquared(y_true, y_pred):
    ### chi2 loss functionfor autoencoder training
    # input arguments:
    # - y_true and y_pred: two numpy arrays of equal shape,
    #   typically a histogram and its autoencoder reconstruction.
    #   if two-dimensional, the arrays are assumed to have shape (nhists,nbins)!
    # output:
    # - relative mean squared error between y_true and y_pred,
    #   if y_true and y_pred are 2D arrays, this function returns 1D array (chiSquared for each histogram)
    normdiffsq = np.divide(K.square(y_pred - y_true),y_true)
    chi2 = K.sum(normdiffsq,axis=-1)
    return chi2

def chiSquaredTop10(y_true, y_pred):
    ### same as chiSquared but take into account only 10 largest values in averaging.
    normdiffsq = np.divide(K.square(y_pred - y_true),y_true)
    top_values,_ = tf.nn.top_k(normdiffsq,k=10,sorted=True)
    chi2 = K.sum(top_values,axis=-1)
    return chi2




### get roc curve and auc score in case labels are known

def calculate_roc(scores, labels, scoreax):
    ### calculate a roc curve
    # input arguments:
    # - scores is a 1D numpy array containing output scores of any algorithm
    # - labels is a 1D numpy array (equally long as scores) containing labels
    #   note that 1 for signal and 0 for background is assumed!
    #   this convention is only used to define what scores belong to signal or background;
    #   the scores itself can be anything (not limited to (0,1)), 
    #   as long as the target for signal is higher than the target for background
    # - scoreax is an array of score thresholds for which to compute the signal and background efficiency,
    #   assumed to be sorted in increasing order (i.e. from loose to tight)
    # output:
    # - tuple of two np arrays (signal efficiency and background efficiency)
    nsig = np.sum(labels)
    nback = np.sum(1-labels)
    sig_eff = np.zeros(len(scoreax))
    bkg_eff = np.zeros(len(scoreax))
    for i,scorethreshold in enumerate(scoreax):
        sig_eff[i] = np.sum(np.where((labels==1) & (scores>scorethreshold),1,0))/nsig
        bkg_eff[i] = np.sum(np.where((labels==0) & (scores>scorethreshold),1,0))/nback
    return (sig_eff,bkg_eff)

def get_roc(scores, labels, mode='lin', npoints=100, doprint=False, doplot=True, plotmode='classic'):
    ### make a ROC curve
    # input arguments:
    # - scores is a 1D numpy array containing output scores of any algorithm
    # - labels is a 1D numpy array (equally long as scores) containing labels
    #   note that 1 for signal and 0 for background is assumed!
    #   this convention is only used to define what scores belong to signal or background;
    #   the scores itself can be anything (not limited to (0,1)), 
    #   as long as the target for signal is higher than the target for background
    # - mode: how to determine the points where to calculate signal and background efficiencies; options are:
    #         - 'lin': np.linspace between min and max score
    #         - 'geom': np. geomspace between min and max score
    #         - 'full': one point per score instance
    # - npoints: number of points where to calculate the signal and background efficiencies
    #   (ignored if mode is 'full')
    # - doprint: boolean whether to print score thresholds and corresponding signal and background efficiencies
    # - doplot: boolean whether to make a plot or simply return the auc.
    # - plotmode: how to plot the roc curve; options are:
    #         - 'classic' = signal efficiency afo background efficiency
    
    mlist = ['lin','geom','full']
    if not mode in mlist:
        raise Exception('ERROR in autoencoder_utils.py / get_roc: mode {} not recognized;'.format(mode)
                       +' options are: {}'.format(mlist))
    
    if mode=='full':
        scoreax = np.sort(scores)
    elif mode=='lin':
        scoremin = np.amin(scores)-1e-7
        scoremax = np.amax(scores)+1e-7
        scoreax = np.linspace(scoremin,scoremax,num=npoints)
    elif mode=='geom':
        scoremin = np.amin(scores)-1e-7
        # if minimum score is below zero, shift everything up (needed for geomspace)
        if scoremin < 0.: 
            scores = scores - scoremin + 1.
            scoremin = 1.
        scoremax = np.amax(scores)+1e-7
        scoreax = np.geomspace(scoremin,scoremax,num=npoints)
    
    (sig_eff,bkg_eff) = calculate_roc( scores, labels, scoreax )
    if doprint:
        print('calculating roc curve:')
        for i in range(len(scoreax)):
            print('  threshold: {:.4e}, signal: {:.4f}, background: {:.4f}'.format(scoreax[i],sig_eff[i],bkg_eff[i]))
    
    # note: sig_eff = signal efficiency = tp = true positive = signal flagged as signal
    # note: bkg_eff = background efficiency = fp = false positive = background flagged as signal
    fn = 1 - sig_eff # signal marked as background
    tn = 1 - bkg_eff # background marked as background
    
    auc = np.trapz(sig_eff[::-1],bkg_eff[::-1])
    
    if not doplot:
        return auc
    
    # calculate auc
    if plotmode=='classic':
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
        plt.show()
        
    else:
        print('ERROR: mode not recognized: '+str(mode))
        return 0
    
    return auc

def get_roc_from_hists(hists, labels, predicted_hists, mode='lin', npoints=100, doprint=False, doplot=True, plotmode='classic'):
    ### make a ROC curve without manually calculating the scores
    # the output score is the mseTop10Raw between the histograms and their reconstruction
    # - input arguments:
    # - hists and predicted_hists are 2D numpy arrays of shape (nhistograms,nbins)
    # - other arguments: see get_roc

    # determine mse
    mse = mseTop10Raw(hists, predicted_hists)
    # score equals mse, since larger mse = more signal-like (signal=anomalies)
    return get_roc(mse,labels,mode=mode,npoints=npoints,doprint=doprint,doplot=doplot,plotmode=plotmode)

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
    plt.figure()
    sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues)
    
def get_confusion_matrix_from_hists(hists, labels, predicted_hists, msewp):
    ### plot a confusion matrix without manually calculating the scores
    # the output score is the mse between the histograms and their reconstruction
    
    # get mse
    mse = mseTop10Raw(hists, predicted_hists)
    get_confusion_matrix(mse, labels, msewp)




### getting a keras model ready for training with minimal user inputs

def getautoencoder(input_size,arch,act=[],opt='adam',loss=mseTop10):
    ### get a trainable autoencoder model
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





