#!/usr/bin/env python
# coding: utf-8

# **Utilities related to the training and evaluation of autoencoder models with keras**
# 
# The functionality in this script includes:
# - definition of loss functions (several flavours of MSE or chi-squared)
# - calculating and plotting ROC curves and confusion matrices
# - definition of very simple ready-to-use keras model architectures


### imports

import math

# external modules
import numpy as np
import keras

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dense
from keras.models import Model, Sequential, load_model
from keras import backend as K

# keras math operation module
if keras.__version__.startswith("2."):
    from keras import backend as ops
else:
    # Since Keras 3, mathematical operation functions are moved under keras.ops.
    from keras import ops

# import seaborn as sn
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
    if keras.__version__.startswith("2."):
        top_k = K.tf.math.top_k
    else:
        top_k = ops.top_k
    top_values, _ = top_k(ops.square(y_pred - y_true), k=10, sorted=True)
    mean=ops.mean(top_values, axis=-1)
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
    ### chi2 loss function for autoencoder training
    # input arguments:
    # - y_true and y_pred: two numpy arrays of equal shape,
    #   typically a histogram and its autoencoder reconstruction.
    #   if two-dimensional, the arrays are assumed to have shape (nhists,nbins)!
    # output:
    # - relative mean squared error between y_true and y_pred,
    #   if y_true and y_pred are 2D arrays, this function returns 1D array (chiSquared for each histogram)
    normdiffsq = np.divide(ops.square(y_pred - y_true),y_true)
    chi2 = ops.sum(normdiffsq,axis=-1)
    return chi2

def chiSquaredTopNRaw(y_true, y_pred, n=10):
    ### generalization of chiSquared to any number of bins to take into account
    # note: should work for 2D histograms as well (i.e. arrays of shape (nhistograms,nybins,nxbins)),
    #       but not yet tested!
    # input arguments:
    # - y_true, y_pred: numpy arrays between which to calculate the mean square difference, of shape (nhists,nbins) or (nhists,nybins,nxbins)
    # - n: number of largest elements to keep for summing
    # output:
    # numpy array of shape (nhists)
    sqdiff = np.power(y_true-y_pred,2)
    y_true_safe = np.where(y_true==0, 1, y_true)
    chi2 = np.where(y_true==0, 0, sqdiff/y_true_safe)
    if len(chi2.shape)==3:
        chi2 = chi2.reshape(len(chi2),-1)
    chi2 = np.partition( chi2, -n, axis=-1 )[:,-n:]
    chi2 = np.sum( chi2, axis=-1 )
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

def get_roc(scores, labels, mode='lin', npoints=100, doprint=False, 
            doplot=True, doshow=True,
            bootstrap_samples=None, bootstrap_size=None,
            returneffs=False ):
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
    # - doplot: boolean whether to make a plot
    # - doshow: boolean whether to call plt.show
    # - bootstrap_samples: number of bootstrap samples to assess uncertainty on ROC curve
    #                      (default: no bootstrapping)
    # - bootstrap_size: size of each bootstrap sample (default: same size as scores, i.e. full sample size)
    #   note: the bootstrapping method can be used to assess the uncertainty on the ROC curve,
    #         by recalculating it several times on samples drawn from the test set with replacement;
    #         the resulting uncertainty as calculated here does not include contributions from varying the training set!
    # - returneffs: boolean whether to return the signal and background efficiencies
    # returns:
    # - if returneffs is False, only the AUC value is returned
    # - if returneffs is True, the return type is a tuple of the form (auc,sigeff,bckeff)
    
    # argument checking and parsing
    mlist = ['lin','geom','full']
    if not mode in mlist:
        raise Exception('ERROR in autoencoder_utils.py / get_roc: mode {} not recognized;'.format(mode)
                       +' options are: {}'.format(mlist))
    dobootstrap = False
    if bootstrap_samples is not None:
        dobootstrap = True
        bootstrap_samples = int(bootstrap_samples)
    
    if mode=='full':
        scoreax = np.sort(scores)
        scoreax[-1] += 0.01 # make sure the extremal scores are fully covered
        scoreax[0] -= 0.01 # make sure the extremal scores are fully covered
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
    
    # calculate signal and background efficencies
    (sig_eff,bkg_eff) = calculate_roc( scores, labels, scoreax )
    sig_eff_unc = None
    
    # do bootstrapping if requested
    if dobootstrap:
        # parse the provided bootstrap sample size
        if bootstrap_size is None: bootstrap_size = len(scores)
        print('calculating ROC curve on {} bootstrap samples of size {}'.format(bootstrap_samples,bootstrap_size))
        # set the x-axis to which the result for each sample will be interpolated
        interpolation_axis = bkg_eff
        # loop over number of bootstrap samples
        bsigeffs = np.zeros((bootstrap_samples,len(interpolation_axis)))
        for i in range(bootstrap_samples):
            # draw a sample
            inds = np.random.randint(0, len(scores), size=bootstrap_size)
            bscores = scores[inds]
            blables = labels[inds]
            # calculate the roc curve for this sample
            (bsigeff, bbkgeff) = calculate_roc( bscores, blables, scoreax )
            # interpolate to common x-axis
            # note: np.interp requires increasing arrays so need some additional slicing
            bsigeff = np.interp( interpolation_axis[::-1], bbkgeff[::-1], bsigeff[::-1] )[::-1]
            # store result
            bsigeffs[i,:] = bsigeff
        # average the results
        sig_eff = np.mean(bsigeffs, axis=0)
        sig_eff_unc = np.std(bsigeffs, axis=0)
        bkg_eff = interpolation_axis
    
    # print results if requested
    if doprint:
        print('calculated roc curve:')
        for i in range(len(scoreax)):
            #print('  threshold: {:.4e}, signal: {:.4f}, background: {:.4f}'.format(scoreax[i],sig_eff[i],bkg_eff[i]))
            print('  threshold: {}, signal: {}, background: {}'.format(scoreax[i],sig_eff[i],bkg_eff[i]))
    
    # calculate auc
    auc = np.trapz(sig_eff[::-1],bkg_eff[::-1])
  
    # make a plot if requested
    if doplot:
        plot_utils.plot_roc( sig_eff, bkg_eff, auc=auc, sig_eff_unc=sig_eff_unc,
                      xaxtitle='Fraction of good histograms flagged as anomalous', xaxtitlesize=12,
                      yaxtitle='Fraction of bad histograms flagged as anomalous', yaxtitlesize=12,
                      doshow=doshow )
        
    # return the result
    if returneffs: return (auc, sig_eff, bkg_eff, sig_eff_unc)
    else: return auc

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

def get_confusion_matrix(scores, labels, wp='maxauc', plotwp=True,
                          true_positive_label='Good', true_negative_label='Anomalous',
                          pred_positive_label='Predicted good', pred_negative_label='Predicted anomalous',
                          xaxlabelsize=None, yaxlabelsize=None, textsize=None,
                          colormap='Blues', colortitle=None):
    ### plot a confusion matrix
    # input arguments:
    # - scores and labels: defined in the same way as for get_roc
    # - wp: the chosen working point 
    #       (i.e. any score above wp is flagged as signal, any below is flagged as background)
    #       note: wp can be a integer or float, in which case that value will be used directly,
    #             or it can be a string in which case it will be used as the 'method' argument in get_wp!
    # - plotwp: only relevant if wp is a string (see above), in which case plotwp will be used as the 'doplot' argument in get_wp
    
    if isinstance(wp,str): wp = get_wp(scores, labels, method=wp, doplot=plotwp)[0]

    nsig = np.sum(labels)
    nback = np.sum(1-labels)

    # get confusion matrix entries
    tp = np.sum(np.where((labels==1) & (scores>wp),1,0))/nsig
    fp = np.sum(np.where((labels==0) & (scores>wp),1,0))/nback
    tn = 1-fp
    fn = 1-tp
    cmat = np.array([[tp,fn],[fp,tn]])
    
    # old plotting method with seaborn
    #df_cm = pd.DataFrame(cmat, index = [true_negative_label,true_positive_label],
    #              columns = [predicted_negative_label,predicted_positive_label])
    #fig,ax = plt.subplots()
    #sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues)
    
    # new plotting method with pyplot
    fig,ax = plot_utils.plot_confusion_matrix( tp, tn, fp, fn, 
                          true_positive_label=true_positive_label, true_negative_label=true_negative_label,
                          pred_positive_label=pred_positive_label, pred_negative_label=pred_negative_label,
                          xaxlabelsize=xaxlabelsize, yaxlabelsize=yaxlabelsize, textsize=textsize,
                          colormap=colormap, colortitle=colortitle )

    # printouts for testing
    #print('working point: {}'.format(wp))
    #print('nsig: {}'.format(nsig))
    #print('nback: {}'.format(nback))
    #print('true positive / nsig: {}'.format(tp))
    #print('false positive / nback: {}'.format(fp))

    # return the working point (for later use if it was automatically calculated)
    return (wp, fig, ax)
    
def get_confusion_matrix_from_hists(hists, labels, predicted_hists, msewp=None):
    ### plot a confusion matrix without manually calculating the scores
    # the output score is the mse between the histograms and their reconstruction
    
    # get mse
    mse = mseTop10Raw(hists, predicted_hists)
    get_confusion_matrix(mse, labels, wp=msewp)



### automatically calculate a suitable working point

def get_wp(scores, labels, method='maxauc', doplot=False):
    ### automatically calculate a suitable working point
    # input arguments:
    # - scores, labels: equally long 1d numpy arrays of predictions and true labels respectively
    #                   note: in all methods, the labels are assumed to be 0 (for background) or 1 (for signal)!
    # - method: method to calculate the working point
    #           currently supported: 'maxauc'
    # - doplot: make a plot (if a plotting method exists for the chosen method)
    allowedmethods = ['maxauc']
    if method not in allowedmethods:
        raise Exception('ERRR in get_wp: method {} not recognized.'.format(method)
                +' available options are {}'.format(allowedmethods))
    if method=='maxauc': return get_wp_maxauc(scores, labels, doplot=doplot)

def get_wp_maxauc(scores, labels, doplot=False):
    ### calculate the working point corresponding to maximum pseudo-AUC
    # (i.e. maximize the rectangular area enclosed by the working point)
    signal_scores = scores[labels==1]
    background_scores = scores[labels==0]
    nsig = len(signal_scores)
    nbck = len(background_scores)
    sorted_scores = sorted(scores)
    effs = np.zeros(len(scores))
    effb = np.zeros(len(scores))
    aucs = np.zeros(len(scores))
    for i,score in enumerate(sorted_scores):
        effs[i] = np.sum(signal_scores>score)/nsig
        effb[i] = np.sum(background_scores>score)/nbck
        aucs[i] = effs[i]*(1-effb[i])
    maxidx = np.argmax(aucs)
    maxscore = sorted_scores[maxidx]
    maxauc = aucs[maxidx]

    fig = None
    ax = None
    
    if doplot:
        fig,ax,ax2 = plot_utils.plot_metric(sorted_scores, aucs, label='Pseudo-AUC',
                    sig_eff=effs, sig_label='Anomaly efficiency', sig_color='r',
                    bck_eff=effb, bck_label='Background efficiency', bck_color='g',
                    legendsize=15,
                    xaxtitle='Working point', xaxtitlesize=15,
                    yaxlog=False, ymaxfactor=1.6,
                    yaxtitle='Pseudo-AUC', yaxtitlesize=15)
        ax.scatter( [maxscore], [maxauc], s=50, c='black', label='Maximum' )
        ax.legend(loc='upper left', framealpha=0.7, facecolor='white', fontsize=15)
        auctext = '{:.3f}'.format(maxauc)
        wptext = '{:.3f}'.format(maxscore)
        if maxauc>0.99:
            auctext = '1 - '+'{:.3e}'.format(1-maxauc)
        text = ax.text(0.97,0.68,'WP: {}, pseudo-AUC: {}'.format(wptext,auctext), 
                        horizontalalignment='right', transform=ax.transAxes)
        text.set_bbox(dict(facecolor='white', edgecolor='black', alpha=0.75))
    return (maxscore, maxauc, fig, ax)



### getting a keras model ready for training with minimal user inputs

def getautoencoder(input_size,arch,act=[],opt='adam',loss=mseTop10):
    ### get a trainable autoencoder model
    # input args:
    # - input_size: size of vector that autoencoder will operate on
    # - arch: list of number of nodes per hidden layer (excluding input and output layer)
    # - act: list of activations per layer (default: tanh)
    # - opt: optimizer to use (default: adam)
    # - loss: loss function to use (defualt: mseTop10)
    
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

def train_simple_autoencoder(hists, nepochs=-1, modelname='', 
                             batch_size=500, shuffle=False, 
                             verbose=1, validation_split=0.1,
                             returnhistory=False ):
    ### create and train a very simple keras model
    # the model consists of one hidden layer (with half as many units as there are input bins), 
    # tanh activation, adam optimizer and mseTop10 loss.
    # input args: 
    # - hists is a 2D numpy array of shape (nhistograms, nbins)
    # - nepochs is the number of epochs to use (has a default value if left unspecified)
    # - modelname is a file name to save the model in (default: model is not saved to a file)
    # - batch_size, shuffle, verbose, validation_split: passed to keras .fit method
    # - returnhistory: boolean whether to return the training history (e.g. for making plots)
    # returns
    # - if returnhistory is False, only the trained keras model is returned
    # - if returnhistory is True, the return type is a tuple of the form (model, history)
    input_size = hists.shape[1]
    arch = [int(hists.shape[1]/2.)]
    act = ['tanh']*len(arch)
    opt = 'adam'
    loss = mseTop10
    if nepochs<0: nepochs = int(min(40,len(hists)/400))
    model = getautoencoder(input_size,arch,act=act,opt=opt,loss=loss)
    history = model.fit(hists, hists, epochs=nepochs, batch_size=batch_size, 
                        shuffle=shuffle, verbose=verbose, 
                        validation_split=validation_split)
    if len(modelname)>0: model.save(modelname.split('.')[0]+'.h5')
    if not returnhistory: return model
    else: return (model, history)


### replacing scores of +-inf with sensible value

def clip_scores( scores, margin=1., hard_thresholds=None ):
    ### clip +-inf values in scores
    # +inf values in scores will be replaced by the maximum value (exclucing +inf) plus one
    # -inf values in scores will be replaced by the minimim value (exclucing -inf) minus one
    # input arguments:
    # - scores: 1D numpy array
    # - margin: margin between maximum value (excluding inf) and where to put inf.
    # - hard_thresholds: tuple of values for -inf, +inf (in case the min or max cannot be determined)
    # returns
    # - array with same length as scores with elements replaced as explained above
    maxnoninf = np.max(np.where(scores==np.inf,np.min(scores),scores)) + margin
    minnoninf = np.min(np.where(scores==-np.inf,np.max(scores),scores)) - margin
    if( hard_thresholds is not None and hard_thresholds[1] is not None ):
        maxnoninf = hard_thresholds[1]
    if( hard_thresholds is not None and hard_thresholds[0] is not None ):
        minnoninf = hard_thresholds[0]
    if np.max(scores)>maxnoninf: 
        scores = np.where(scores==np.inf,maxnoninf,scores)
        print('NOTE: scores of +inf were reset to {}'.format(maxnoninf))
    if np.min(scores)<minnoninf:
        scores = np.where(scores==-np.inf,minnoninf,scores)
        print('NOTE: scores of -inf were reset to {}'.format(minnoninf))
    return scores
