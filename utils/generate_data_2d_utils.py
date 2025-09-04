#!/usr/bin/env python
# coding: utf-8

# **Extension of generate\_data\_utils.py towards 2D histograms** 


### imports

# external modules
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# local modules
import plot_utils as pu

### help functions

def goodnoise_nd(shape, fstd=None, kmaxscale=0.25, ncomponents=3, rng=None):
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
    # - rng: a numpy.random.Generator instance.
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
    if rng is None:
        rng = np.random.default_rng()
    # initialize noise array
    noise = np.zeros(shape)
    # loop over axes
    for i in range(len(shape)):
        ax = np.arange(0,shape[i])
        thiscomp = np.zeros(shape[i])
        ncomps = ncomponents[i]
        # get uniformly sampled wavenumbers in range (0,kmax)
        kmax = np.pi*kmaxscale[i]
        k = rng.uniform(low=0,high=1,size=ncomps)*kmax
        # get uniformly sampled phases in range (0,2pi)
        phase = rng.uniform(low=0,high=1,size=ncomps)*2*np.pi
        # get uniformly sampled amplitudes in range (0,2/ncomps) (i.e. mean total amplitude = 1)
        amplitude = rng.uniform(low=0,high=1,size=ncomps)*2/ncomps
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


def whitenoise_nd(shape, fstd=None, rng=None):
    ### generate one sample of white noise (standard normally distributed, uncorrelated between bins)
    # generalization of whitenoise (see generate_data_utils) to arbitrary number of dimensions
    # input args:
    # - shape: a tuple, shape of the noise array to be sampled
    #   note: in case of 1D, a comma is needed e.g. shape = (30,) else it will be automatically parsed to int and raise an error
    # - fstd: an array of shape given by shape argument, 
    #   used for scaling of the amplitude of the noise bin-by-bin
    #   (default: no scaling).
    # - rng: a numpy.random.Generator instance.
    # output: 
    # - numpy array of shape detailed by shape argument containing the noise
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(size=shape)
    if fstd is not None: noise = np.multiply(noise,fstd)
    return noise


def random_lico_nd(hists, rng=None):
    ### generate one linear combination of histograms with random coefficients in (0,1) summing to 1.
    # generalization of random_lico (see generate_data_utils) to arbitrary number of dimensions.
    # input args: 
    # - numpy array of shape (nhists,<arbitrary number of additional dimensions>)
    # - rng: a numpy.random.Generator instance.
    # output:
    # - numpy array of shape (<same dimensions as input>), containing the new histogram
    if rng is None:
        rng = np.random.default_rng()
    nhists = hists.shape[0]
    coeffs = rng.uniform(low=0.,high=1.,size=nhists)
    coeffs = coeffs/np.sum(coeffs)
    for i in range(len(hists.shape[1:])): coeffs = np.expand_dims(coeffs,-1)
    res = np.sum(hists*coeffs,axis=0)
    return res


### resampling functions

def fourier_noise_nd(hists, outfilename=None, doplot=False, ntarget=None, nresamples=1, nonnegative=True, 
                     stdfactor=15., kmaxscale=0.25, ncomponents=3, rng=None):
    ### apply fourier noise on random histograms with simple flat amplitude scaling.
    # generalization of fourier_noise (see generate_data_utils) to arbitrary number of dimensions.
    # input args: 
    # - hists: numpy array of shape (nhists,<arbitrary number of dimensions>) used for seeding
    # - outfilename: path to csv file to write results to (default: no writing)
    # - doplot: boolean whether to make a plot of some examples (only for 2D histograms!)
    # - ntarget: total target number of histograms (default: use nresamples instead)
    # - nresamples: number of samples to draw per input histogram
    #   (note: ignored if ntarget is not None)
    # - nonnegative: boolean whether to set all bins to minimum zero after applying noise
    # - stdfactor: factor to scale magnitude of noise (larger factor = smaller noise)
    # - kmaxscale and ncomponents: see goodnoise_nd
    # - rng: a numpy.random.Generator instance.
    
    # initializations
    if rng is None:
        rng = rng.default_rng()
    nhists = hists.shape[0]
    histshape = hists.shape[1:]
    if ntarget is not None:
        nresamples = max(1,int(float(ntarget)/nhists))
    outshape = tuple([nresamples*nhists]+list(histshape))
    reshists = np.zeros(outshape)
    
    # generate data
    for i in range(nhists):
        for j in range(nresamples):
            noise = goodnoise_nd(histshape, fstd=hists[i]/stdfactor,
                                    kmaxscale=kmaxscale, ncomponents=ncomponents,
                                    rng=rng)
            reshists[nresamples*i+j] = hists[i] + noise
    if nonnegative:
        reshists = np.where(reshists>0,reshists,0)

    # plot examples if requested
    if doplot: 
        nplot = min(3,nhists)
        randinds = rng.choice(range(nhists), size=nplot, replace=False)
        for counter,seedidx in enumerate(randinds):
            extidx = nresamples*seedidx
            seedhist = hists[seedidx]
            exthist = reshists[extidx]
            pu.plot_hists_2d([seedhist,exthist], ncols=2, 
                                title='Resampling example no. {}'.format(counter+1),
                                subtitles=['Original','Resampled'])
            plt.show(block=False)

    # shuffle the output set
    rng.shuffle(reshists)
    
    # store results if requested
    if( outfilename is not None and len(outfilename)>0 ):
        np.savetxt(outfilename.split('.')[0]+'.csv', reshists)

    # return the result
    return reshists


def white_noise_nd(hists, doplot=False, ntarget=None, nresamples=1, nonnegative=True, stdfactor=15., rng=None):
    ### apply white noise to the histograms in hists.
    # generalization of white_noise (see generate_data_utils) to arbitrary number of dimensions.
    # input args:
    # - hists: np array (nhists,<arbitrary number of dimensions>) containing input histograms
    # - doplot: boolean whether to plot some examples (only for 2D histograms!)
    # - ntarget: total target number of histograms (default: use nresamples instead)
    # - nresamples: number of samples to draw per input histogram
    #   (note: ignored if ntarget is not None)
    # - nonnegative: boolean whether to set all bins to minimum zero after applying noise
    # - stdfactor: scaling factor of white noise amplitude (higher factor = smaller noise)
    # - rng: a numpy.random.Generator instance.

    # initializations
    if rng is None:
        rng = np.random.default_rng()
    nhists = hists.shape[0]
    histshape = hists.shape[1:]
    if ntarget is not None:
        nresamples = max(1,int(float(ntarget)/nhists))
    outshape = tuple([nresamples*nhists]+list(histshape))
    reshists = np.zeros(outshape)

    # generate the data
    for i in range(nhists):
        for j in range(nresamples):
            reshists[nresamples*i+j] = hists[i] + whitenoise_nd(histshape,fstd=hists[i]/stdfactor, rng=rng)
        
    if nonnegative:
        reshists = np.where(reshists>0,reshists,0)
    
    # plot examples if requested
    if doplot:
        nplot = min(3,nhists)
        randinds = rng.choice(range(nhists), size=nplot, replace=False)
        for counter,seedidx in enumerate(randinds):
            extidx = nresamples*seedidx
            seedhist = hists[seedidx]
            exthist = reshists[extidx]
            pu.plot_hists_2d([seedhist,exthist], ncols=2,
                                title='Resampling example no. {}'.format(counter+1),
                                subtitles=['Original','Resampled'])
            plt.show(block=False)

    # shuffle the output set
    rng.shuffle(reshists)

    return reshists


def resample_lico_nd(hists, doplot=False, ntarget=None, nonnegative=True, rng=None):
    ### take random linear combinations of input histograms
    # generalization of fourier_noise (see generate_data_utils) to arbitrary number of dimensions.
    # input args: 
    # - hists: numpy array of shape (nhists,<arbitrary number of dimensions>) used for seeding
    # - doplot: boolean whether to plot some examples (only for 2D histograms!)
    # - ntarget: total target number of histograms (default: same as number of input histograms)
    # - nonnegative: boolean whether to set all bins to minimum zero after applying noise
    #   note: coefficients in linear combination are always nonnegative, 
    #         so this setting is superfluous is input histograms are all nonnegative
    # - rng: a numpy.random.Generator instance.
    
    # initializations
    if rng is None:
        rng = np.random.default_rng()
    nhists = hists.shape[0]
    histshape = hists.shape[1:]
    if ntarget is None:
        ntarget = nhists
    outshape = tuple([ntarget]+list(histshape))
    reshists = np.zeros(outshape)

    # generate the data
    for i in range(ntarget):
        reshists[i] = random_lico_nd(hists, rng=rng)
        
    if nonnegative:
        reshists = np.where(reshists>0,reshists,0)

    # plot examples if requested
    if doplot:
        nplot = min(3,nhists)
        randseedinds = rng.choice(range(nhists), size=nplot, replace=False)
        randextinds = rng.choice(range(len(reshists)), size=nplot, replace=False)
        pu.plot_hists_2d([hists[i] for i in randseedinds], ncols=nplot,
                            title='Examples of original histograms')
        pu.plot_hists_2d([reshists[i] for i in randextinds], ncols=nplot,
                            title='Examples of resampled histograms')
        plt.show(block=False)

    return reshists
