#!/usr/bin/env python
# coding: utf-8

# **Extension of generate\_data\_utils.py towards 2D histograms** 



### imports

# external modules
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib

# local modules
import hist_utils
importlib.reload(hist_utils)




### help functions

def goodnoise_nd(shape, fstd=None, kmaxscale=0.25, ncomponents=3):
    ### generate one sample of 'good' noise consisting of fourier components
    # generalization of goodnoise (see generate_data_utils) to arbitrary number of dimensions
    # input args:
    # - shape: a tuple, shape of the noise array to be sampled
    #   note: in case of 1D, a comma is needed e.g. shape = (30,) else it will be automatically parsed to int and raise an error
    # - fstd: an array of shape given by shape argument, 
    #   used for scaling of the amplitude of the noise bin-by-bin
    #   (default: no scaling).
    # - kmaxscale: scale factor to limit maximum frequency (lower kmaxscale means smoother noise)
    #   note: can be a tuple with same length as shape, to scale differently in different dimensions.
    # - ncomponents: number of random sines to add per dimension
    #   note: can be a tuple with same length as shape, to use a different number of components in different dimensions.
    # output: 
    # - numpy array of shape detailed by shape argument containing the noise
    
    # check fstd argument
    if fstd is not None:
        if fstd.shape!=shape:
            raise Exception('ERROR in generate_data_2d_utils.py / goodnoisend:'
                            +' argument fstd must either be None or have same shape as shape argument')
    # parse kmaxscale argument
    if( isinstance(kmaxscale,float) or isinstance(ncomponents,int) ):
        kmaxscale = tuple([kmaxscale]*len(shape))
    # parse ncomponents argument
    if( isinstance(ncomponents,float) or isinstance(ncomponents,int) ):
        ncomponents = tuple([ncomponents]*len(shape))
    # initialize noise array
    noise = np.zeros(shape)
    # loop over axes
    for i in range(len(shape)):
        ax = np.arange(0,shape[i])
        thiscomp = np.zeros(shape[i])
        ncomps = ncomponents[i]
        # get uniformly sampled wavenumbers in range (0,kmax)
        kmax = np.pi*kmaxscale[i]
        k = np.random.uniform(low=0,high=1,size=ncomps)*kmax
        # get uniformly sampled phases in range (0,2pi)
        phase = np.random.uniform(low=0,high=1,size=ncomps)*2*np.pi
        # get uniformly sampled amplitudes in range (0,2/ncomps) (i.e. mean total amplitude = 1)
        amplitude = np.random.uniform(low=0,high=1,size=ncomps)*2/ncomps
        for j in range(ncomps): thiscomp += amplitude[j]*np.sin(k[j]*ax + phase[j])
        # expand this component to all dimensions
        reps = list(shape)
        reps[i] = 1
        for j in range(0,i): thiscomp = np.expand_dims(thiscomp,0)
        for j in range(i+1,len(shape)): thiscomp = np.expand_dims(thiscomp,-1)
        # add to noise
        noise += thiscomp
    # scale noise
    if fstd is not None: noise = np.multiply(noise,fstd)
    return noise

def whitenoise_nd(shape, fstd=None):
    ### generate one sample of white noise (standard normally distributed, uncorrelated between bins)
    # generalization of whitenoise (see generate_data_utils) to arbitrary number of dimensions
    # input args:
    # - shape: a tuple, shape of the noise array to be sampled
    #   note: in case of 1D, a comma is needed e.g. shape = (30,) else it will be automatically parsed to int and raise an error
    # - fstd: an array of shape given by shape argument, 
    #   used for scaling of the amplitude of the noise bin-by-bin
    #   (default: no scaling).
    # output: 
    # - numpy array of shape detailed by shape argument containing the noise
    noise = np.random.normal(size=shape)
    if fstd is not None: noise = np.multiply(noise,fstd)
    return noise

def random_lico_nd(hists):
    ### generate one linear combination of histograms with random coefficients in (0,1) summing to 1.
    # generalization of random_lico (see generate_data_utils) to arbitrary number of dimensions.
    # input args: 
    # - numpy array of shape (nhists,<arbitrary number of additional dimensions>)
    # output:
    # - numpy array of shape (<same dimensions as input>), containing the new histogram
    nhists = hists.shape[0]
    coeffs = np.random.uniform(low=0.,high=1.,size=nhists)
    coeffs = coeffs/np.sum(coeffs)
    for i in range(len(hists.shape[1:])): coeffs = np.expand_dims(coeffs,-1)
    res = np.sum(hists*coeffs,axis=0)
    return res




def fourier_noise_nd(hists, outfilename='', figname='', nresamples=1, nonnegative=True, 
                     stdfactor=15., kmaxscale=0.25, ncomponents=3):
    ### apply fourier noise on random histograms with simple flat amplitude scaling.
    # generalization of fourier_noise (see generate_data_utils) to arbitrary number of dimensions.
    # input args: 
    # - hists: numpy array of shape (nhists,<arbitrary number of dimensions>) used for seeding
    # - outfilename: path to csv file to write results to (default: no writing)
    # - figname: path to figure plotting examples (default: no plotting)
    # - nresamples: number of samples to draw per input histogram
    # - nonnegative: boolean whether to set all bins to minimum zero after applying noise
    # - stdfactor: factor to scale magnitude of noise (larger factor = smaller noise)
    # - kmaxscale and ncomponents: see goodnoise_nd
    
    nhists = hists.shape[0]
    histshape = hists.shape[1:]
    outshape = tuple([nresamples*nhists]+list(histshape))
    reshists = np.zeros(outshape)
    
    # generate data
    for i in range(nhists):
        for j in range(nresamples):
            reshists[nresamples*i+j] = hists[i] + goodnoise_nd(histshape,fstd=hists[i]/stdfactor,
                                                               kmaxscale=kmaxscale,ncomponents=ncomponents)
    if nonnegative:
        reshists = np.where(reshists>0,reshists,0)
    np.random.shuffle(reshists)

    # plot examples of good and bad histograms
    #if len(figname)>0: 
    #    noise_examples = []
    #    for i in range(5): noise_examples.append(goodnoise(nbins,hists[-1,:]/stdfactor))
    #    plot_noise(np.array(noise_examples),hists[-1,:]/stdfactor,figname)
    #    plot_data_and_gen(50,hists,reshists,figname)
    
    # store results if requested
    if len(outfilename)>0: np.savetxt(outfilename.split('.')[0]+'.csv',reshists)

    return reshists




def white_noise_nd(hists, figname='', nresamples=1, nonnegative=True, stdfactor=15.):
    ### apply white noise to the histograms in hists.
    # generalization of white_noise (see generate_data_utils) to arbitrary number of dimensions.
    # input args:
    # - hists: np array (nhists,<arbitrary number of dimensions>) containing input histograms
    # - figname: path to figure plotting examples (default: no plotting)
    # - nresamples: number of samples to draw per input histogram
    # - nonnegative: boolean whether to set all bins to minimum zero after applying noise
    # - stdfactor: scaling factor of white noise amplitude (higher factor = smaller noise)

    nhists = hists.shape[0]
    histshape = hists.shape[1:]
    outshape = tuple([nresamples*nhists]+list(histshape))
    reshists = np.zeros(outshape)

    for i in range(nhists):
        for j in range(nresamples):
            reshists[nresamples*i+j] = hists[i] + whitenoise_nd(histshape,fstd=hists[i]/stdfactor)
        
    if nonnegative:
        reshists = np.where(reshists>0,reshists,0)
    
    # plot examples of generated histograms
    #if len(figname)>0: plot_data_and_gen(50,hists,reshists,figname)

    return reshists




def resample_lico_nd(hists, nresamples=1, nonnegative=True):
    ### take random linear combinations of input histograms
    # generalization of fourier_noise (see generate_data_utils) to arbitrary number of dimensions.
    # input args: 
    # - hists: numpy array of shape (nhists,<arbitrary number of dimensions>) used for seeding
    # - nresamples: number of samples to draw
    # - nonnegative: boolean whether to set all bins to minimum zero after applying noise
    #   note: coefficients in linear combination are always nonnegative, so this setting is superfluous is input histograms are all nonnegative
    
    nhists = hists.shape[0]
    histshape = hists.shape[1:]
    outshape = tuple([nresamples]+list(histshape))
    reshists = np.zeros(outshape)

    for i in range(nresamples):
        reshists[i] = random_lico_nd( hists )
        
    if nonnegative:
        reshists = np.where(reshists>0,reshists,0)

    return reshists










