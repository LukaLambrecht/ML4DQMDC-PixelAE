#!/usr/bin/env python
# coding: utf-8

# **A collection of useful basic functions for plotting.**  



### imports

# external modules
import os
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from copy import copy
import imageio

# local modules




# functions for plotting 
      
def plot_hists(histlist, fig=None, ax=None, colorlist=[], labellist=[], transparency=1, xlims=(-0.5,-1),
              title=None, xaxtitle=None, yaxtitle=None, bkgcolor=None, bkgcmap='spring'):
    ### plot some histograms (in histlist) in one figure using specified colors and/or labels
    # - histlist is a list of 1D arrays containing the histograms (or a 2D array of shape (nhistograms,nbins))
    # - colorlist is a list or array containing colors (in string format), of length nhistograms
    #   note: it can also be a single string representing a color (in pyplot), then all histograms will take this color
    # - labellist is a list or array containing labels for in legend, of length nhistograms
    # - xlims is a tuple of min and max for the x-axis labels, defaults to (-0.5,nbins-0.5)
    # - title, xaxtitle, yaxtitle: strings for histogram title, x-axis title and y-axis title respectively
    # - bkgcolor: 1D array representing background color for the plot (color axis will be scaled between min and max in bkgcolor)
    #   note: if bkgcolor does not have the same length as the x-axis, it will be compressed or stretched to fit the axis,
    #         but this might be meaningless, depending on what you are trying to visualize!
    # - bkgmap: name of valid pyplot color map for plotting the background color
    # output: tuple of figure and axis objects, that can be used to further tune the look of the figure or save it
    if fig is None or ax is None: fig,ax = plt.subplots()
    dolabel = True; docolor = True
    if len(labellist)==0:
        labellist = ['']*len(histlist)
        dolabel = False
    if isinstance(colorlist,str):
        colorlist = [colorlist]*len(histlist)
    if len(colorlist)==0:
        docolor = False
    if xlims[1]<xlims[0]: xlims = (-0.5,len(histlist[0])-0.5)
    xax = np.linspace(xlims[0],xlims[1],num=len(histlist[0]))
    for i,row in enumerate(histlist):
        if docolor: ax.step(xax,row,where='mid',color=colorlist[i],label=labellist[i],alpha=transparency)
        else: ax.step(xax,row,where='mid',label=labellist[i],alpha=transparency)
    if bkgcolor is not None:
        # modify bkcolor so the automatic stretching matches the bin numbers correctly
        bkgcolor = [el for el in bkgcolor for _ in (0,1)][1:-1]
        bkgcolor = np.array(bkgcolor)
        ax.pcolorfast((xlims[0],xlims[1]), ax.get_ylim(),
              bkgcolor[np.newaxis],
              cmap=bkgcmap, alpha=0.1)
    if dolabel: ax.legend()
    if title is not None: ax.set_title(title)
    if xaxtitle is not None: ax.set_xlabel(xaxtitle)
    if yaxtitle is not None: ax.set_ylabel(yaxtitle)
    return (fig,ax)
    
def plot_hists_multi(histlist, fig=None, ax=None, colorlist=[], labellist=[], transparency=1, xlims=(-0.5,-1),
                    title=None, xaxtitle=None, yaxtitle=None):
    ### plot many histograms (in histlist) in one figure using specified colors and/or labels
    # - histlist is a list of 1D arrays containing the histograms (or a 2D array of shape (nhistograms,nbins))
    # - colorlist is a list or array containing numbers to be mapped to colors
    # - labellist is a list or array containing labels for in legend
    # output: tuple of figure and axis objects, that can be used to further tune the look of the figure or save it
    if fig is None or ax is None: fig,ax = plt.subplots()
    dolabel = True; docolor = True
    if len(labellist)==0:
        labellist = ['']*len(histlist)
        dolabel = False
    if len(colorlist)==0:
        docolor = False
    if xlims[1]<xlims[0]: xlims = (0,len(histlist[0]))
    xax = np.linspace(xlims[0],xlims[1],num=len(histlist[0]))
    if docolor:
        norm = mpl.colors.Normalize(vmin=np.min(colorlist),vmax=np.max(colorlist))
        cobject = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
        cobject.set_array([]) # ad-hoc bug fix
    for i,row in enumerate(histlist):
        if docolor: ax.step(xax,row,where='mid',color=cobject.to_rgba(colorlist[i]),label=labellist[i],alpha=transparency)
        else: ax.step(xax,row,where='mid',label=labellist[i],alpha=transparency)
    if docolor: fig.colorbar(cobject, ax=ax)
    if dolabel: ax.legend()
    if title is not None: ax.set_title(title)
    if xaxtitle is not None: ax.set_xlabel(xaxtitle)
    if yaxtitle is not None: ax.set_ylabel(yaxtitle)
    return (fig,ax)

def plot_hist_2d(hist, fig=None, ax=None, title=None, xaxtitle=None, yaxtitle=None, caxrange=None):
    ### plot a 2D histogram
    # - hist is a 2D numpy array of shape (nxbins, nybins)
    # notes:
    # - if the histogram contains only nonnegative values, values below 1e-12 will not be plotted
    #   (i.e. they will be shown as white spots in the plot) to discriminate zero from small but nonzero
    # - if the histogram contains negative values, the color axis will be symmetrized around zero
    if fig is None or ax is None: fig,ax = plt.subplots()
    histmin = np.amin(hist)
    histmax = np.amax(hist)
    hasnegative = histmin<-1e-12
    if not hasnegative: my_norm = mpl.colors.Normalize(vmin=1e-12, clip=False)
    else: 
        extremum = max(abs(histmax),abs(histmin))
        my_norm = mpl.colors.Normalize(vmin=-extremum,vmax=extremum,clip=False)
    if caxrange is not None:
        my_norm = mpl.colors.Normalize(vmin=caxrange[0],vmax=caxrange[1],clip=False)
    my_cmap = copy(mpl.cm.get_cmap('jet'))
    my_cmap.set_under('w')
    cobject = mpl.cm.ScalarMappable(norm=my_norm, cmap=my_cmap)
    cobject.set_array([]) # ad-hoc bug fix
    ax.imshow(hist, interpolation='none', norm=my_norm, cmap=my_cmap)
    fig.colorbar(cobject,ax=ax)
    if title is not None: ax.set_title(title)
    if xaxtitle is not None: ax.set_xlabel(xaxtitle)
    if yaxtitle is not None: ax.set_ylabel(yaxtitle)
    return (fig,ax)

def plot_hists_2d(hists, ncols=4, title=None, subtitles=None, xaxtitle=None, yaxtitle=None, caxrange=None):
    ### plot multiple 2D histograms next to each other
    # - hists: list of 2D numpy arrays of shape (nxbins,nybins), or an equivalent 3D numpy array
    # - ncols: number of columns to use
    nrows = int(math.ceil(len(hists)/ncols))
    fig,axs = plt.subplots(nrows,ncols,figsize=(6*ncols,4*nrows),squeeze=False)
    if( subtitles is not None and len(subtitles)!=len(hists) ):
        raise Exception('ERROR in plot_utils.py / plot_hists_2d: subtitles must have same length as hists or be None')
    # loop over all histograms belonging to this lumisection and make the plots
    for i,hist in enumerate(hists):
        subtitle = None
        if subtitles is not None: subtitle = subtitles[i]
        plot_hist_2d(hist,fig=fig,ax=axs[int(i/ncols),i%ncols],title=subtitle,xaxtitle=xaxtitle,yaxtitle=yaxtitle,caxrange=caxrange)
    if title is not None: fig.suptitle(title)
    return (fig,axs)

def plot_hists_2d_gif(hists, titles=None, xaxtitle=None, yaxtitle=None, duration=0.3, figname='temp_gif.gif'):
    nhists = len(hists)
    filenames = []
    for i in range(nhists):
        title = None
        if titles is not None: title = titles[i]
        fig,_ = plot_hist_2d(hists[i], title=title, xaxtitle=xaxtitle, yaxtitle=yaxtitle)
        filename = 'temp_gif_file_{}.png'.format(i)
        filenames.append(filename)
        fig.savefig(filename)
        plt.close()
    with imageio.get_writer(figname, mode='I', duration=duration) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    for filename in filenames:
        os.remove(filename)
        
def plot_hists_from_df(df, histtype, nhists):
    ### plot a number of histograms in a dataframe
    # - df is the dataframe from which to plot
    # - histtype is the name of the histogram type (e.g. 'chargeInner_PXLayer_1')
    # - nhists is the number of histograms to plot
    dfs = select_histnames(df,[histtype])
    nhists = min(len(dfs),nhists)
    dfs = dfs[0:nhists+1]
    val = get_hist_values(dfs)[0]
    plot_hists(val)
    
def plot_sets(setlist, fig=None, ax=None, colorlist=[], labellist=[], transparencylist=[], xlims=(-0.5,-1),
             title=None, xaxtitle=None, yaxtitle=None):
    ### plot multiple sets of histograms to compare the shapes
    # - setlist is a list of 2D numpy arrays containing histograms
    # - fig and ax: a pyplot figure and axis object (if one of both is none a new figure is created)
    # - title is a string that will be used as the title for the ax object
    # other parameters are lists of which each element applies to one list of histograms
    dolabel = True
    if len(labellist)==0:
        labellist = ['']*len(setlist)
        dolabel = False
    if len(colorlist)==0:
        colorlist = ['red','blue','green','orange']
        if len(setlist)>4:
            print('ERROR in plot_utils.py / plot_sets: please specify the colors if you plot more than four sets.')
            return
    if len(transparencylist)==0:
        transparencylist = [1.]*len(setlist)
    if xlims[1]<xlims[0]: xlims = (0,len(setlist[0][0]))
    xax = np.linspace(xlims[0],xlims[1],num=len(setlist[0][0]))
    if fig is None or ax is None: fig,ax = plt.subplots()
    for i,histlist in enumerate(setlist):
        row = histlist[0]
        ax.step(xax,row,where='mid',color=colorlist[i],label=labellist[i],alpha=transparencylist[i])
        if len(histlist)<2: continue
        for j,row in enumerate(histlist[1:,:]):
            ax.step(xax,row,where='mid',color=colorlist[i],alpha=transparencylist[i])
    if dolabel: ax.legend(loc='upper right')
    if title is not None: ax.set_title(title)
    if xaxtitle is not None: ax.set_xlabel(xaxtitle)
    if yaxtitle is not None: ax.set_ylabel(yaxtitle)
    return (fig,ax)

def plot_anomalous(histlist, ls, highlight=-1, hrange=-1):
    # histlist and ls are a list of histograms and corresponding lumisection numbers
    # lsnumber is the lumisection number of the histogram to highlight
    # hrange is the number of histograms before and after lsnumber to plot (default: whole run)
    lshist = None
    if highlight >= 0:
        if not highlight in ls:
            print('WARNING in plot_utils.py / plot_anomalous: requested lumisection number not in list of lumisections')
            return None
        index = np.where(ls==highlight)[0][0]
        lshist = histlist[index]
    if hrange > 0:
        indexmax = min(index+hrange,len(ls))
        indexmin = max(index-hrange,0)
        histlist = histlist[indexmin:indexmax]
        ls = ls[indexmin:indexmax]
    # first plot all histograms in the run
    fig,ax = plot_hists_multi(histlist,colorlist=ls,transparency=0.1)
    # now plot a single histogram on top
    if lshist is not None: 
        xlims = (0,len(lshist))
        xax = np.linspace(xlims[0],xlims[1],num=len(lshist))
        ax.step(xax,lshist,where='mid',color='black',linewidth=2)
    return (fig,ax)

def plot_moments(moments, ls, dims, fig=None, ax=None, markersize=10):
    # moments is an (nhists,nmoments) array
    # dims is a tuple of two or three values between 0 and nmoments-1
    from mpl_toolkits.mplot3d import Axes3D # specific import
    if fig==None: fig = plt.figure()
    if len(dims)==2:
        if ax==None: ax = fig.add_subplot(111)
        scpl = ax.scatter(moments[:,dims[0]],moments[:,dims[1]],s=markersize,c=ls,cmap='jet')
        plt.colorbar(scpl)
        ax.set_xlabel('moment '+str(dims[0]+1))
        ax.set_ylabel('moment '+str(dims[1]+1))
    elif len(dims)==3:
        if ax==None: ax = fig.add_subplot(111, projection='3d')
        scpl = ax.scatter(moments[:,dims[0]],moments[:,dims[1]],moments[:,dims[2]],s=markersize,c=ls,cmap='jet')
        plt.colorbar(scpl)
        ax.set_xlabel('moment '+str(dims[0]+1))
        ax.set_ylabel('moment '+str(dims[1]+1))
        ax.set_zlabel('moment '+str(dims[2]+1))
    return (fig,ax)


def plot_distance(dists, ls=None, rmlargest=0., doplot=True,
                 title=None, xaxtitle='lumisection number', yaxtitle='distance metric'):
    
    if ls is None: ls = np.arange(0,len(dists))
        
    if rmlargest>0.:
        threshold = np.quantile(dists,1-rmlargest)
        ls = ls[dists<threshold]
        dists = dists[dists<threshold]
    gmean = dists.mean()
    gstd = dists.std()
        
    if not doplot: return (gmean,gstd)
    
    fig,ax = plt.subplots()
    fig.set_size_inches(8, 6)
    
    ax.hlines(gmean,ls[0],ls[-1], color='b', label='average: {}'.format(gmean))
    ax.hlines(gmean+(1.0*gstd), ls[0],ls[-1], color='r', label='1 sigma ({})'.format(gstd))
    ax.hlines(gmean+(3.0*gstd), ls[0],ls[-1], color='r', label='3 sigma', linestyle=':')
    
    ax.set_ylim(np.min(dists)*0.9,np.max(dists)*1.1)
    ax.scatter(ls, dists, marker='+', label='data points')
    if title is not None: ax.set_title(title)
    if xaxtitle is not None: ax.set_xlabel(xaxtitle)
    if yaxtitle is not None: ax.set_ylabel(yaxtitle)
    ax.legend()
    plt.show()
    return (fig,ax)


### plot model loss as a function of training epoch
# credits to Francesco for this function
def plot_loss(data, xlims=None,
              title=None, xaxtitle='epoch', yaxtitle='loss'):
    ### plot the training and validation loss
    # data is the object returned by the .fit method when called upon a keras model
    # e.g. history = <your autoencoder>.fit(<training params>)
    #      plot_loss(history,'a title')
    fig,ax = plt.subplots()
    ax.plot(data.history['loss'], linestyle=(0,()), color="#1A237E", linewidth=3, label='training')
    ax.plot(data.history['val_loss'], linestyle=(0,(3,2)), color="#4DB6AC", linewidth=3, label='validation')
    ax.legend(loc="upper right", frameon=False)
    ax.set_yscale('log')
    if xlims is not None: ax.set_xlim(xlims)
    if title is not None: ax.set_title(title)
    if xaxtitle is not None: ax.set_xlabel(xaxtitle)
    if yaxtitle is not None: ax.set_ylabel(yaxtitle)
    plt.show()
    return (fig,ax)
    
### plot an array of mse values and get the mean and std value
# credits to Francesco for this function
def plot_mse(mse, rmlargest=0., doplot=True,
            title=None, xaxtitle='lumisection number', yaxtitle='mse'):
    ### plot the mse's and return the mean and std
    # input args:
    # - mse is a 1D numpy array of mse scores
    # - doplot: boolean whether to make a plot or simply return mean and std
    # - rmlargest: fraction of largest mse's to remove (to avoid being too sensitive to outliers)
    
    (obj1,obj2) = plot_distance(mse,rmlargest=rmlargest,doplot=doplot,title=title,xaxtitle=xaxtitle,yaxtitle=yaxtitle)
    return (obj1,obj2)


def plot_score_dist( scores, labels, nbins=20, normalize=False,
                        title='output score distributions for signal and background',
                        xaxtitle='output score', yaxtitle=None):
    ### make a plot showing the distributions of the output scores for signal and background
    minscore = np.min(scores)
    maxscore = np.max(scores)
    scorebins = np.linspace(minscore,maxscore,num=nbins+1)
    scoreax = (scorebins[1:] + scorebins[:-1])/2
    sigscores = scores[np.where(labels==1)]
    bkgscores = scores[np.where(labels==0)]
    sighist = np.histogram(sigscores,bins=scorebins)[0]
    bckhist = np.histogram(bkgscores,bins=scorebins)[0]
    if normalize:
        sighist = sighist/np.sum(sighist)
        bckhist = bckhist/np.sum(bckhist)
    (fig,ax) = plt.subplots()
    ax.step(scoreax,sighist,color='g',label='signal',where='mid')
    ax.step(scoreax,bckhist,color='r',label='background',where='mid')
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    if title is not None: ax.set_title(title)
    if xaxtitle is not None: ax.set_xlabel(xaxtitle)
    if yaxtitle is not None: ax.set_ylabel(yaxtitle)
    ax.legend()
    plt.show()
    return (fig,ax)

def plot_fit_2d( points, fitfunc=None, logprob=False, onlycontour=False, xlims=5, ylims=5, onlypositive=False,
               xaxtitle=None, yaxtitle=None ):
    ### make a scatter plot of a 2D point cloud with fitted contour
    # input arguments:
    # - points: a numpy array of shape (npoints,ndims)
    # - fitfunc: an object of type CloudFitter (see src/cloudfitters) 
    #   or any other object that implements a pdf(points) method
    # - logprob: boolean whether to plot log probability or normal probability
    # - onlycontour: a boolean whether to draw only the fit or include the data points
    # - xlims and ylims: tuples of (low,high)
    #   note: can be an integer, in which case the range will be determined automatically
    #         from the formula low = mean-xlims*std, high = mean+xlims*std,
    #         where mean and std are determined from the points array.
    # - onlypositive: overrides previous argument to set lower bound of plotting range at 0 in both dimensions.
    # - xaxtitle and yaxtitle: titles for axes.
    
    # set plotting ranges and step sizes
    if( isinstance(xlims,int) ):
        xmean = np.mean(points[:,0])
        xstd = np.std(points[:,0])
        xlims = (xmean-xlims*xstd,xmean+xlims*xstd)
    if( isinstance(ylims,int) ):
        ymean = np.mean(points[:,1])
        ystd = np.std(points[:,1])
        ylims = (ymean-ylims*ystd,ymean+ylims*ystd)
    if onlypositive:
        xlims = (0,xlims[1])
        ylims = (0,ylims[1])
    xstep = (xlims[1]-xlims[0])/100.
    ystep = (ylims[1]-ylims[0])/100.
        
    (fig,ax) = plt.subplots()
    
    if fitfunc is not None:
        
        # make a grid of points and evaluate the fitfunc
        x,y = np.mgrid[xlims[0]:xlims[1]:xstep,ylims[0]:ylims[1]:ystep]
        gridpoints = np.transpose(np.vstack((np.ravel(x),np.ravel(y))))
        evalpoints = fitfunc.pdf(gridpoints)
        if logprob: evalpoints = np.log(evalpoints)
        z = np.reshape(evalpoints,x.shape)

        # make a plot of probability contours
        contourplot = ax.contourf(x, y, z, 30)
        plt.colorbar(contourplot)
        
    if not onlycontour:
        
        # make a plot of the data points
        ax.plot(points[:,0],points[:,1],'.b',markersize=2)
    
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    if xaxtitle is not None: ax.set_xlabel(xaxtitle)
    if yaxtitle is not None: ax.set_ylabel(yaxtitle)
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    
    return (fig,ax)










