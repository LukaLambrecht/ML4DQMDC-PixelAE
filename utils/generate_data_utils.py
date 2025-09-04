#!/usr/bin/env python
# coding: utf-8

# **A collection of functions for artificially creating a labeled dataset.**  
# 
# See the function documentation below for more details on the implemented methods.  
# Also check the tutorial generate\_data.ipynb for examples!


### imports

# external modules
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import importlib

# local modules
import hist_utils
importlib.reload(hist_utils)


### help functions

def goodnoise(nbins, fstd=None, rng=None):
    ### generate one sample of 'good' noise consisting of fourier components
    # input args:
    # - nbins: number of bins, length of noise array to be sampled
    # - fstd: an array of length nbins used for scaling of the amplitude of the noise
    #         bin-by-bin.
    # - rng: a numpy.random.Generator instance.
    # output: 
    # - numpy array of length nbins containing the noise
    if rng is None:
        rng = np.random.default_rng()
    kmaxscale = 0.25 # frequency limiting factor to ensure smoothness
    ncomps = 3 # number of random sines to use
    kmax = np.pi*kmaxscale
    xax = np.arange(0,nbins)
    noise = np.zeros(nbins)
    # get uniformly sampled wavenumbers in range (0,kmax)
    k = rng.uniform(low=0,high=1,size=ncomps)*kmax
    # get uniformly sampled phases in range (0,2pi)
    phase = rng.uniform(low=0,high=1,size=ncomps)*2*np.pi
    # get uniformly sampled amplitudes in range (0,2/ncomps) (i.e. mean total amplitude = 1)
    amplitude = rng.uniform(low=0,high=1,size=ncomps)*2/ncomps
    for i in range(ncomps):
        temp = amplitude[i]*np.sin(k[i]*xax + phase[i])
        if fstd is not None: temp = np.multiply(temp,fstd)
        noise += temp
    return noise

def badnoise(nbins, fstd=None, rng=None):
    ### generate one sample of 'bad' noise consisting of fourier components
    # (higher frequency and amplitude than 'good' noise)
    # input args and output: simlar to goodnoise
    # WARNING: NOT NECESSARILY REPRESENTATIVE OF ANOMALIES TO BE EXPECTED, DO NOT USE
    if rng is None:
        rng = np.random.default_rng()
    ampscale = 10. # additional amplitude scaling
    kmaxscale = 1. # additional scaling of max frequency
    kminoffset = 0.5 # additional scaling of min frequency
    ncomps = 3 # number of fourier components
    kmax = np.pi*kmaxscale
    xax = np.arange(0,nbins)
    noise = np.zeros(nbins)
    # get uniformly sampled wavenumbers in range (kmin,kmax)
    k = rng.uniform(low=kminoffset,high=1,size=ncomps)*kmax
    # get uniformly sampled phases in range (0,2pi)
    phase = rng.uniform(low=0,high=1,size=ncomps)*2*np.pi
    # get uniformly sampled amplitudes in range (0,2*ampscale/ncomps) (i.e. mean total amplitude = ampscale)
    amplitude = ampscale*rng.uniform(low=0,high=1,size=ncomps)*2/ncomps
    for i in range(ncomps):
        temp = amplitude[i]*np.sin(k[i]*xax + phase[i])
        if fstd is not None: temp = np.multiply(temp,fstd)
        noise += temp
    return noise

def whitenoise(nbins, fstd=None, rng=None):
    ### generate one sample of white noise (uncorrelated between bins)
    # input args and output: similar to goodnoise
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(size=nbins)
    if fstd is not None: noise = np.multiply(noise,fstd)
    return noise

def random_lico(hists, rng=None):
    ### generate one linear combination of histograms with random coefficients in (0,1) summing to 1
    # input args: 
    # - numpy array of shape (nhists,nbins), the rows of which will be linearly combined
    # - rng: a numpy.random.Generator instance.
    # output:
    # - numpy array of shape (nbins), containing the new histogram
    if rng is None:
        rng = np.random.default_rng()
    nhists = hists.shape[0]
    coeffs = rng.uniform(low=0.,high=1.,size=nhists)
    coeffs = coeffs/np.sum(coeffs)
    res = np.sum(hists*coeffs[:,np.newaxis],axis=0)
    return res

def smoother(inarray, halfwidth=1):
    ### smooth the rows of a 2D array using the 2*halfwidth+1 surrounding values.
    outarray = np.zeros(inarray.shape)
    nbins = inarray.shape[1]
    for j in range(nbins):
        crange = np.arange(max(0,j-halfwidth),min(nbins,j+halfwidth+1))
        outarray[:,j] = np.sum(inarray[:,crange],axis=1)/len(crange)
    return outarray

def mse_correlation_vector(hists, index):
    ### calculate mse of a histogram at given index wrt all other histograms
    # input args:
    # - hists: numpy array of shape (nhists,nbins) containing the histograms
    # - index: the index (must be in (0,len(hists)-1)) of the histogram in question
    # output:
    # - numpy array of length nhists containing mse of the indexed histogram with respect to all other histograms
    # WARNING: can be slow if called many times on a large collection of histograms with many bins.
    corvec = np.zeros(len(hists))
    temp = hists - np.tile(hists[index:index+1],(len(hists),1))
    temp = np.power(temp,2)
    corvec = np.mean(temp,axis=1)
    return corvec

def moments_correlation_vector(moments, index):
    ### calculate moment distance of hist at index wrt all other hists
    # very similar to mse_correlation_vector but using histogram moments instead of full histograms for speed-up
    return mse_correlation_vector(moments,index)


### plot functions

def plot_data_and_gen(datahists, genhists, 
                      fig=None, axs=None,
                      datacolor='b', gencolor='b',
                      datalabel='Histograms from data', 
                      genlabel='Artificially generated histograms'):
    ### plot a couple of random examples from data and generated histograms
    # note: both are plotted in different subplots of the same figure
    # input arguments:
    # - datahists, genhists: numpy arrays of shape (nhists,nbins)
    # - fig, axs: a matplotlib figure object and a list of two axes objects
    #             (if either is None, a new figure with two subplots will be created)
    if(fig is None or axs is None): fig,axs = plt.subplots(ncols=2)
    xlims = (0,len(datahists[0]))
    xax = np.linspace(xlims[0],xlims[1],num=len(datahists[0]))
    # data
    axs[0].step(xax, datahists[0,:], color=datacolor, label=datalabel)
    for i in range(1,len(datahists)): axs[0].step(xax, datahists[int(i),:], color=datacolor)
    axs[0].legend()
    # artificial histograms
    axs[1].step(xax, genhists[0,:], color=gencolor, label=genlabel)
    for i in range(1,len(genhists)): axs[1].step(xax, genhists[int(i),:], color=gencolor)
    axs[1].legend()
    return (fig,axs)

def plot_seed_and_gen(seedhists, genhists, 
                      fig=None, ax=None,
                      seedcolor='b', gencolor='g',
                      seedlabel='Histograms from data', 
                      genlabel='Artificially generated histograms'):
    ### plot seed and generated histograms
    # note: both are plotted in the same subplot
    # input arguments:
    # - seedhists, genhists: numpy arrays of shape (nhists,nbins)
    # - fig, ax: a matplotlib figure object and an axes object
    #             (if either is None, a new figure will be created)
    if(fig is None or axs is None): fig,ax = plt.subplots()
    xlims = (0,len(datahists[0]))
    xax = np.linspace(xlims[0],xlims[1],num=len(datahists[0]))
    # data
    ax.step(xax, genhists[0,:], color=gencolor, label=genlabel)
    for i in range(1,len(genhists)): ax.plot(genhists[i,:], color=gencolor)
    # seed
    ax.step(xax, seedhists[0,:], color=seedcolor, label=seedlabel)
    for i in range(1,len(seedhists)): ax.plot(seedhists[i,:], color=seedcolor)
    ax.legend()
    return (fig,ax)
    
def plot_noise(noise, fig=None, ax=None,
               noiselabel='Examples of noise', noisecolor='b', 
               histstd=None, histstdlabel='Variation'):
    ### plot histograms in noise (numpy array of shape (nhists,nbins))
    # input arguments:
    # - noise: 2D numpy array of shape (nexamples,nbins)
    # - fig, ax: a matplotlib figure object and an axes object
    #             (if either is None, a new figure will be created)
    # - noiselabel: label for noise examples (use None to not add a legend entry for noise)
    # - noisecolor: color for noise examples on plot
    # - histstd: 1D numpy array of shape (nbins) displaying some order-of-magnitude allowed variation
    #            (typically some measure of per-bin variation in the input histogram(s))
    # - histstdlabel: label for histstd (use None to not add a legend entry for histstd)
    if(fig is None or ax is None): fig,ax = plt.subplots()
    ax.plot(noise[0,:], color=noisecolor, label=noiselabel)
    for i in range(1,len(noise)): ax.plot(noise[i,:], color=noisecolor)
    if histstd is not None:
        ax.plot(histstd, 'k--', label=histstdlabel)
        ax.plot(-histstd, 'k--')
    ax.legend()
    return (fig,ax)


### resampling functions

def fourier_noise_on_mean(hists, outfilename='', nresamples=0, nonnegative=True, doplot=True, rng=None):
    ### apply fourier noise on the bin-per-bin mean histogram, with amplitude scaling based on bin-per-bin std histogram.
    # input args:
    # - hists: numpy array of shape (nhists,nbins) used for determining mean and std
    # - outfilename: path to csv file to write results to (default: no writing)
    # - nresamples: number of samples to draw (default: number of input histograms / 10)
    # - nonnegative: boolean whether to set all bins to minimum zero after applying noise
    # - doplot: boolean whether to make a plot
    # - rng: a numpy.random.Generator instance.
    # returns:
    #   a tuple of the form (resulting histograms, maplotlib figure, matplotlib axes),
    #   figure and axes are None if doplot was set to False
    # MOSTLY SUITABLE AS HELP FUNCTION FOR RESAMPLE_SIMILAR_FOURIER_NOISE, NOT AS GENERATOR IN ITSELF
    # advantages: mean histogram is almost certainly 'good' because of averaging, eliminate bad histograms
    # disadvantages: deviations from mean are small, does not model systematic shifts by lumi.

    if rng is None:
        rng = np.random.default_rng()

    if nresamples==0: nresamples=int(len(hists)/10)

    # get mean and std histogram
    histmean = np.mean(hists,axis=0)
    histstd = np.std(hists,axis=0)
    nbins = len(histmean)
    
    # generate data
    reshists = np.zeros((nresamples,nbins))
    for i in range(nresamples):
        reshists[i,:] = histmean + goodnoise(nbins, histstd, rng=rng)
    if nonnegative:
        reshists = np.where(reshists>0,reshists,0)
    
    # make a figure
    fig = None
    axs = None
    if doplot:
        fig,axs = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
        # plot examples of histograms mean, and std
        ax = axs[0,0]
        nplot = min(200,len(hists))
        randint = rng.choice(np.arange(len(hists)), size=nplot, replace=False)
        for i in randint: ax.plot(hists[int(i),:], color='b', alpha=0.1)
        ax.plot(histmean,color='black',label='Mean histogram')
        ax.plot(histmean-histstd,color='r',label='$\pm$ 1 $\sigma$')
        ax.plot(histmean+histstd,color='r')
        ax.legend()
        # plot examples of noise
        noise_examples = []
        for i in range(5):
            noise_examples.append(goodnoise(nbins, histstd, rng=rng))
        plot_noise(np.array(noise_examples), fig=fig, ax=axs[0,1], histstd=histstd)
        # plot examples of good and bad histograms
        nplot_hists = min(20, len(hists))
        randint_hists = rng.choice(np.arange(len(hists)), size=nplot_hists, replace=False)
        nplot_reshists = min(20, len(reshists))
        randint_reshists = rng.choice(np.arange(len(reshists)), size=nplot_reshists, replace=False)
        plot_data_and_gen(hists[randint_hists], reshists[randint_reshists], fig=fig, axs=[axs[1,0],axs[1,1]])

    # store results if requested
    if len(outfilename)>0: np.savetxt(outfilename.split('.')[0]+'.csv',reshists)
    
    return (reshists, fig, axs)


def fourier_noise(hists, outfilename='', nresamples=1, nonnegative=True, stdfactor=15., doplot=True, rng=None):
    ### apply fourier noise on random histograms with simple flat amplitude scaling.
    # input args: 
    # - hists: numpy array of shape (nhists,nbins) used for seeding
    # - outfilename: path to csv file to write results to (default: no writing)
    # - nresamples: number of samples to draw per input histogram
    # - nonnegative: boolean whether to set all bins to minimum zero after applying noise
    # - stdfactor: factor to scale magnitude of noise (larger factor = smaller noise)
    # - doplot: boolean whether to make a plot
    # - rng: a numpy.random.Generator instance.
    # returns:
    #   a tuple of the form (resulting histograms, maplotlib figure, matplotlib axes),
    #   figure and axes are None if doplot was set to False
    # advantages: resampled histograms will have statistically same features as original input set
    # disadvantages: also 'bad' histograms will be resampled if included in hists
    
    if rng is None:
        rng = np.random.default_rng()

    (nhists,nbins) = hists.shape
    
    # generate data
    reshists = np.zeros((nresamples*len(hists),nbins))
    for i in range(nhists):
        for j in range(nresamples):
            reshists[nresamples*i+j,:] = hists[i,:] + goodnoise(nbins, hists[i,:]/stdfactor, rng=rng)
    if nonnegative:
        reshists = np.where(reshists>0,reshists,0)
    rng.shuffle(reshists)

    # plot examples of good and bad histograms
    fig = None
    axs = None
    if doplot: 
        fig,axs = plt.subplots(ncols=3, figsize=(15,5))
        # plot examples of noise
        noise_examples = []
        for i in range(5): noise_examples.append(goodnoise(nbins, hists[-1,:]/stdfactor, rng=rng))
        plot_noise(np.array(noise_examples), fig=fig, ax=axs[1], histstd=hists[-1,:]/stdfactor)
        # plot examples of good and bad histograms
        nplot_hists = min(20, len(hists))
        randint_hists = rng.choice(np.arange(len(hists)), size=nplot_hists, replace=False)
        nplot_reshists = min(20, len(reshists))
        randint_reshists = rng.choice(np.arange(len(reshists)), size=nplot_reshists, replace=False)
        plot_data_and_gen(hists[randint_hists], reshists[randint_reshists], fig=fig, axs=[axs[0],axs[2]])
    
    # store results if requested
    if len(outfilename)>0: np.savetxt(outfilename.split('.')[0]+'.csv',reshists)

    return (reshists, fig, axs)


def upsample_hist_set(hists, ntarget=-1, fourierstdfactor=15., doplot=True, rng=None):
    ### wrapper for fourier_noise allowing for a fixed target number of histograms instead of a fixed resampling factor.
    # useful function for quickly generating a fixed number of resampled histograms,
    # without bothering too much about what exact resampling technique or detailed settings would be most appropriate.
    # input arguments:
    # - hists: input histogram set
    # - ntarget: targetted number of resampled histograms (default: equally many as in hists)
    # - fourierstdfactor: see fourier_noise
    # - doplot: boolean whether to make a plot
    # - rng: a numpy.random.Generator instance.
    # returns:
    #   a tuple of the form (resulting histograms, maplotlib figure, matplotlib axes),
    #   figure and axes are None if doplot was set to False
    if ntarget<0: ntarget = len(hists)
    if rng is None:
        rng = np.random.default_rng()
    nresamples = max(1,int(float(ntarget)/len(hists)))    
    (hists_ext, fig, axs) = fourier_noise(hists, nresamples=nresamples, 
                                          nonnegative=True, stdfactor=fourierstdfactor, 
                                          doplot=doplot, rng=rng)
    return (hists_ext, fig, axs)


def white_noise(hists, stdfactor=15., doplot=True, rng=None):
    ### apply white noise to the histograms in hists.
    # input args:
    # - hists: np array (nhists,nbins) containing input histograms
    # - stdfactor: scaling factor of white noise amplitude (higher factor = smaller noise)
    # - doplot: boolean whether to make a plot
    # - rng: a numpy.random.Generator instance.
    # returns:
    #   a tuple of the form (resulting histograms, maplotlib figure, matplotlib axes),
    #   figure and axes are None if doplot was set to False
    
    if rng is None:
        rng = np.random.default_rng()
    (nhists,nbins) = hists.shape
    reshists = np.zeros((nhists,nbins))
    for i in range(nhists):
        reshists[i,:] = hists[i,:] + np.multiply(rng.normal(size=nbins), np.divide(hists[i,:],stdfactor) )
    
    # plot examples of generated histograms
    fig = None
    axs = None
    if doplot:
        fig,axs = plt.subplots(ncols=2, figsize=(10,5))
        nplot_hists = min(20, len(hists))
        randint_hists = rng.choice(np.arange(len(hists)), size=nplot_hists, replace=False)
        nplot_reshists = min(20, len(reshists))
        randint_reshists = rng.choice(np.arange(len(reshists)), size=nplot_reshists, replace=False)
        plot_data_and_gen(hists[randint_hists], reshists[randint_reshists], fig=fig, axs=axs)

    return (reshists, fig, axs)


def resample_bin_per_bin(hists, outfilename='', nresamples=0, nonnegative=True, smoothinghalfwidth=2, doplot=True, rng=None):
    ### do resampling from bin-per-bin probability distributions
    # input args:
    # - hists: np array (nhists,nbins) containing the histograms to draw new samples from
    # - outfilename: path to csv file to write results to (default: no writing)
    # - nresamples: number of samples to draw (default: 1/10 of number of input histograms)
    # - nonnegative: boolean whether or not to put all bins to minimum zero after applying noise
    # - smoothinghalfwidth: halfwidth of smoothing procedure to apply on the result (default: no smoothing)
    # - doplot: boolean whether to make a plot
    # - rng: a numpy.random.Generator instance.
    # returns:
    #   a tuple of the form (resulting histograms, maplotlib figure, matplotlib axes),
    #   figure and axes are None if doplot was set to False
    # advantages: no arbitrary noise modeling
    # disadvantages: bins are considered independent, shape of historams not taken into account,
    #                does not work well on small number of input histograms, 
    #                does not work well on histograms with systematic shifts
    
    if rng is None:
        rng = np.random.default_rng()
    if nresamples==0: nresamples=int(len(hists)/10)
    nbins = hists.shape[1]
    
    # generate data
    reshists = np.zeros((nresamples,nbins))
    for i in range(nbins):
        col = rng.choice(hists[:,i],size=nresamples,replace=True)
        reshists[:,i] = col
        
    # apply smoothing to compensate partially for bin independence
    if smoothinghalfwidth>0: reshists = smoother(reshists,halfwidth=smoothinghalfwidth)

    # plot examples of good and bad histograms
    fig = None
    axs = None
    if doplot:
        fig,axs = plt.subplots(ncols=2, figsize=(10,5))
        nplot_hists = min(20, len(hists))
        randint_hists = rng.choice(np.arange(len(hists)), size=nplot_hists, replace=False)
        nplot_reshists = min(20, len(reshists))
        randint_reshists = rng.choice(np.arange(len(reshists)), size=nplot_reshists, replace=False)
        plot_data_and_gen(hists[randint_hists], reshists[randint_reshists], fig=fig, axs=axs)
    
    # store results if requested
    if len(outfilename)>0: np.savetxt(outfilename.split('.')[0]+'.csv',reshists)

    return (reshists, fig, axs)


def resample_similar_bin_per_bin( allhists, selhists, outfilename='', nresamples=1, 
                                 nonnegative=True, keeppercentage=1., doplot=True, rng=None):
    ### resample from bin-per-bin probability distributions, but only from similar looking histograms.
    # input args:
    # - allhists: np array (nhists,nbins) containing all available histograms (to determine mean)
    # - selhists: np array (nhists,nbins) conataining selected histograms used as seeds (e.g. 'good' histograms)
    # - outfilename: path of csv file to write results to (default: no writing)
    # - nresamples: number of samples per input histogram in selhists
    # - nonnegative: boolean whether or not to put all bins to minimum zero after applying noise
    # - keeppercentage: percentage (between 1 and 100) of histograms in allhists to use per input histogram
    # - doplot: boolean whether to make a plot
    # - rng: a numpy.random.Generator instance.
    # returns:
    #   a tuple of the form (resulting histograms, maplotlib figure, matplotlib axes),
    #   figure and axes are None if doplot was set to False
    # advantages: no assumptions on shape of noise,
    #             can handle systematic shifts in histograms
    # disadvantages: bins are treated independently from each other

    if rng is None:
        rng = np.random.default_rng()

    # set some parameters
    (nhists,nbins) = allhists.shape
    (nsel,_) = selhists.shape
    
    # get array of moments (used to define similar histograms)
    binwidth = 1./nbins
    bincenters = np.linspace(binwidth/2,1-binwidth/2,num=nbins,endpoint=True)
    orders = [0,1,2]
    allmoments = np.zeros((nhists,len(orders)))
    for i,j in enumerate(orders): allmoments[:,i] = hist_utils.moment(bincenters,allhists,j)
    selmoments = np.zeros((nsel,len(orders)))
    for i,j in enumerate(orders): selmoments[:,i] = hist_utils.moment(bincenters,selhists,j)
    
    # make resamples
    reshists = np.zeros((nsel*nresamples,nbins))
    for i in range(nsel):
        # select similar histograms
        thisdiff = moments_correlation_vector(np.vstack((selmoments[i],allmoments)),0)[1:]
        #thisdiff = mse_correlation_vector(np.vstack((selhists[i],allhists)),0)[1:]
        threshold = np.percentile(thisdiff,keeppercentage)
        simindices = np.nonzero(np.where(thisdiff<=threshold,1,0))[0]
        for j in range(nresamples):
            reshists[nresamples*i+j,:] = resample_bin_per_bin(allhists[simindices,:],
                                           nresamples=1,nonnegative=nonnegative,smoothinghalfwidth=0,
                                           doplot=False, rng=rng)[0][0,:]
    if nonnegative: reshists = np.maximum(0,reshists)
    rng.shuffle(reshists)
    nsim = len(simindices)
    print('Note: bin-per-bin resampling performed on '+str(nsim)+' histograms.')
    print('If this number is too low, existing histograms are drawn with too small variation.')
    print('If this number is too high, systematic shifts of histograms can be averaged out.')
        
    # plot examples of good and bad histograms
    fig = None
    axs = None
    if doplot:
        fig,axs = plt.subplots(ncols=2, figsize=(10,5))
        nplot_hists = min(20, len(selhists))
        randint_hists = rng.choice(np.arange(len(selhists)), size=nplot_hists, replace=False)
        nplot_reshists = min(20, len(reshists))
        randint_reshists = rng.choice(np.arange(len(reshists)), size=nplot_reshists, replace=False)
        plot_data_and_gen(selhists[randint_hists], reshists[randint_reshists], fig=fig, axs=axs)

    # store results if requested
    if len(outfilename)>0: np.savetxt(outfilename.split('.')[0]+'.csv',reshists)
        
    return (reshists, fig, axs)


def resample_similar_fourier_noise( allhists, selhists, outfilename='', nresamples=1, 
                                   nonnegative=True, keeppercentage=1., doplot=True,
                                   rng=None):
    ### apply fourier noise on mean histogram, 
    # where the mean is determined from a set of similar-looking histograms
    # input args:
    # - allhists: np array (nhists,nbins) containing all available histograms (to determine mean)
    # - selhists: np array (nhists,nbins) conataining selected histograms used as seeds (e.g. 'good' histograms)
    # - outfilename: path of csv file to write results to (default: no writing)
    # - nresamples: number of samples per input histogram in selhists
    # - nonnegative: boolean whether or not to put all bins to minimum zero after applying noise
    # - keeppercentage: percentage (between 1 and 100) of histograms in allhists to use per input histogram
    # - doplot: boolean whether to make a plot
    # - rng: a numpy.random.Generator instance.
    # returns:
    #   a tuple of the form (resulting histograms, maplotlib figure, matplotlib axes),
    #   figure and axes are None if doplot was set to False
    # advantages: most of fourier_noise_on_mean but can additionally handle shifting histograms,
    #             apart from fourier noise, also white noise can be applied.
    # disadvantages: does not filter out odd histograms as long as enough other odd histograms look more or less similar

    if rng is None:
        rng = np.random.default_rng()

    # get some parameters
    if(len(allhists.shape)!=len(selhists.shape) or allhists.shape[1]!=selhists.shape[1]):
        print('ERROR in generate_data_utils.py / resample_similar_fourier_noise: shapes of allhists and selhists not compatible.')
        return
    (nhists,nbins) = allhists.shape
    (nsel,_) = selhists.shape

    # get array of moments (used to define similar histograms)
    binwidth = 1./nbins
    bincenters = np.linspace(binwidth/2,1-binwidth/2,num=nbins,endpoint=True)
    orders = [0,1,2]
    allmoments = np.zeros((nhists,len(orders)))
    for i,j in enumerate(orders): allmoments[:,i] = hist_utils.moment(bincenters,allhists,j)
    selmoments = np.zeros((nsel,len(orders)))
    for i,j in enumerate(orders): selmoments[:,i] = hist_utils.moment(bincenters,selhists,j)
 
    # make resampled histograms
    reshists = np.zeros((nsel*nresamples,nbins))
    for i in range(nsel):
        # select similar histograms
        thisdiff = moments_correlation_vector(np.vstack((selmoments[i],allmoments)),0)[1:]
        #thisdiff = mse_correlation_vector(np.vstack((selhists[i],allhists)),0)[1:]
        threshold = np.percentile(thisdiff,keeppercentage)
        simindices = np.nonzero(np.where(thisdiff<threshold,1,0))[0]
        for j in range(nresamples):
            reshists[nresamples*i+j,:] = fourier_noise_on_mean(allhists[simindices,:],
                                          nresamples=1, nonnegative=nonnegative, 
                                          doplot=False, rng=rng)[0][0,:]
    if nonnegative: reshists = np.maximum(0,reshists)
    rng.shuffle(reshists)
    nsim = len(simindices)
    print('Note: mean and std calculation is performed on '+str(nsim)+' histograms.')
    print('If this number is too low, histograms might be too similar for averaging to have effect.')
    print('If this number is too high, systematic shifts of histogram shapes are included into the averaging.')

    # plot examples of good and bad histograms
    # use only those histograms from real data that were used to create the resamples
    fig = None
    axs = None
    if doplot:
        fig,axs = plt.subplots(ncols=2, figsize=(10,5))
        nplot_hists = min(20, len(selhists))
        randint_hists = rng.choice(np.arange(len(selhists)), size=nplot_hists, replace=False)
        nplot_reshists = min(20, len(reshists))
        randint_reshists = rng.choice(np.arange(len(reshists)), size=nplot_reshists, replace=False)
        plot_data_and_gen(selhists[randint_hists], reshists[randint_reshists], fig=fig, axs=axs)

    # store results if requested
    if len(outfilename)>0: np.savetxt(outfilename.split('.')[0]+'.csv',reshists)

    return (reshists, fig, axs)


def resample_similar_lico( allhists, selhists, outfilename='', nresamples=1, 
                          nonnegative=True, keeppercentage=1., doplot=True,
                          rng=None):
    ### take linear combinations of similar histograms
    # input arguments:
    # - allhists: 2D np array (nhists,nbins) with all available histograms, used to take linear combinations
    # - selhists: 2D np array (nhists,nbins) with selected hists used for seeding (e.g. 'good' histograms)
    # - outfilename: path to csv file to write result to (default: no writing)
    # - nresamples: number of combinations to make per input histogram
    # - nonnegative: boolean whether to make all final histograms nonnegative
    # - keeppercentage: percentage (between 0. and 100.) of histograms in allhists to use per input histogram
    # - doplot: boolean whether to make a plot
    # - rng: a numpy.random.Generator instance.
    # returns:
    #   a tuple of the form (resulting histograms, maplotlib figure, matplotlib axes),
    #   figure and axes are None if doplot was set to False
    # advantages: no assumptions on noise
    # disadvantages: sensitive to outlying histograms (more than with averaging)

    if rng is None:
        rng = np.random.default_rng()

    # get some parameters
    if(len(allhists.shape)!=len(selhists.shape) or allhists.shape[1]!=selhists.shape[1]):
        print('### ERROR ###: shapes of allhists and selhists not compatible.')
        return
    (nhists,nbins) = allhists.shape
    (nsel,_) = selhists.shape
    
    # get array of moments (used to define similar histograms)
    binwidth = 1./nbins
    bincenters = np.linspace(binwidth/2,1-binwidth/2,num=nbins,endpoint=True)
    orders = [0,1,2]
    allmoments = np.zeros((nhists,len(orders)))
    for i,j in enumerate(orders): allmoments[:,i] = hist_utils.moment(bincenters,allhists,j)
    selmoments = np.zeros((nsel,len(orders)))
    for i,j in enumerate(orders): selmoments[:,i] = hist_utils.moment(bincenters,selhists,j)
    
    # make resampled histograms
    reshists = np.zeros((nsel*nresamples,nbins))
    for i in range(nsel):
        # select similar histograms
        thisdiff = moments_correlation_vector(np.vstack((selmoments[i],allmoments)),0)[1:]
        # printouts for testing purposes
        #for j in range(nhists):
        #    print(str(allmoments[j])+' -> '+str(thisdiff[j]))
        #thisdiff = mse_correlation_vector(np.vstack((selhists[i],allhists)),0)[1:]
        threshold = np.percentile(thisdiff,keeppercentage)
        simindices = np.nonzero(np.where(thisdiff<=threshold,1,0))[0]
        # printouts for testing purposes
        #print('---------------')
        #for j in simindices:
        #    print(str(allmoments[j])+' -> '+str(thisdiff[j]))
        for j in range(nresamples):
            reshists[nresamples*i+j,:] = random_lico(allhists[simindices,:], rng=rng)
    if nonnegative: reshists = np.maximum(0,reshists)
    rng.shuffle(reshists)
    nsim = len(simindices)
    print('Note: linear combination is taken between '+str(nsim)+' histograms.')
    print('If this number is too low, histograms might be too similar for combination to have effect.')
    print('If this number is too high, systematic shifts of histogram shapes are included into the combination')
        
    # plot examples of good and bad histograms
    # use only those histograms from real data that were used to create the resamples
    fig = None
    axs = None
    if doplot:
        fig,axs = plt.subplots(ncols=2, figsize=(10,5))
        nplot_hists = min(20, len(selhists))
        randint_hists = rng.choice(np.arange(len(selhists)), size=nplot_hists, replace=False)
        nplot_reshists = min(20, len(reshists))
        randint_reshists = rng.choice(np.arange(len(reshists)), size=nplot_reshists, replace=False)
        plot_data_and_gen(selhists[randint_hists], reshists[randint_reshists], fig=fig, axs=axs)
        
    # store results if requested
    if len(outfilename)>0: np.savetxt(outfilename.split('.')[0]+'.csv',reshists)

    return (reshists, fig, axs)


def mc_sampling(hists, nMC=10000 , nresamples=10, doplot=True, rng=None):
    ### resampling of a histogram using MC methods
    # Drawing random points from a space defined by the range of the histogram in all axes.
    # Points are "accepted" if the fall under the sampled histogram:
    # f(x) - sampled distribution
    # x_r, y_r -> randomly sampled point
    # if y_r<=f(x_r), fill the new distribution at bin corresponding to x_r with weight:
    # weight = (sum of input hist)/(#mc points accepted)
    # this is equal to 
    # weight = (MC space volume)/(# all MC points)
    
    if rng is None:
        rng = np.random.default_rng()
    (nHists,nBins) = hists.shape
    output = np.asarray( [ np.asarray([0.]*nBins) for _ in range(nHists*nresamples)])
    for i in range(nHists):
        for j in range(nresamples):
            weight = nBins*np.max(hists[i])/nMC
            for _ in range(nMC):
                x_r=rng.randint(0, nBins)
                y_r=rng.random()*np.max(hists[i])
                if( y_r <= hists[i][x_r]):
                    output[i*nresamples+j][x_r]+=weight
    output = np.array(output)
    # make a figure
    fig = None
    axs = None
    if doplot:
        fig,axs = plt.subplots(ncols=2, figsize=(10,5))
        nplot_hists = min(20, len(hists))
        randint_hists = rng.choice(np.arange(len(hists)), size=nplot_hists, replace=False)
        nplot_reshists = min(20, len(output))
        randint_reshists = rng.choice(np.arange(len(output)), size=nplot_reshists, replace=False)
        plot_data_and_gen(hists[randint_hists], output[randint_reshists], fig=fig, axs=axs)
    return (output, fig, axs)
