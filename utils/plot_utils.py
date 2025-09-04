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
try: import imageio
except: print('WARNING: could not import package "imageio".' 
              +' This is only used to create gif animations, so it should be safe to proceed without,'
              +' if you do not plan to do just that.')
try:
    from matplotlib import rc
    rc('text', usetex=True)
    plot_utils_latex_formatting = True
except: 
    print('WARNING: could not set LaTEX rendering for matplotlib.'
          +' Any TEX commands in figure labels might not work as expected.')
    plot_utils_latex_formatting = False
import importlib


##################
# help functions #
##################

def make_legend_opaque( leg ):
    ### set the transparency of all entries in a legend to zero
    for lh in leg.legendHandles: 
        try: lh.set_alpha(1)
        except: lh._legmarker.set_alpha(1)
            
def add_text( ax, text, pos, 
              fontsize=10,
              horizontalalignment='left',
              verticalalignment='bottom',
              background_facecolor=None, 
              background_alpha=None, 
              background_edgecolor=None,
              **kwargs ):
    ### add text to an axis at a specified position (in relative figure coordinates)
    # input arguments:
    # - ax: matplotlib axis object
    # - text: string, can contain latex syntax such as /textbf{} and /textit{}
    # - pos: tuple with relative x- and y-axis coordinates of bottom left corner
    txt = None
    if( isinstance(ax, plt.Axes) ): 
        txt = ax.text(pos[0], pos[1], text, fontsize=fontsize, 
                   horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, 
                   transform=ax.transAxes, **kwargs)
    else:
        txt = ax.text(pos[0], pos[1], text, fontsize=fontsize,
                   horizontalalignment=horizontalalignment, verticalalignment=verticalalignment,
                   **kwargs)
    if( background_facecolor is not None 
            or background_alpha is not None 
            or background_edgecolor is not None ):
        if background_facecolor is None: background_facecolor = 'white'
        if background_alpha is None: background_alpha = 1.
        if background_edgecolor is None: background_edgecolor = 'black'
        txt.set_bbox(dict(facecolor=background_facecolor, 
                            alpha=background_alpha, 
                            edgecolor=background_edgecolor))
        
def add_cms_label( ax, pos=(0.1,0.9), extratext=None, **kwargs ):
    ### add the CMS label and extra text (e.g. 'Preliminary') to a plot
    # special case of add_text, for convenience
    text = r'\textbf{CMS}'
    if extratext is not None: text += r' \textit{'+str(extratext)+r'}'
    add_text( ax, text, pos, **kwargs)
    
def add_data_label( ax, datalabel, pos=(0.9,0.9), **kwargs ):
    ### add the data label, e.g. '2023 (13.6 TeV)', to a plot
    # special case of add_text, for convenience
    add_text( ax, datalabel, pos, horizontalalignment='right', **kwargs)

def make_text_latex_safe( text ):
    ### make a string safe to process with matplotlib's latex parser in case no tex parsing is wanted
    # (e.g. escape underscores)
    # to be extended when the need arises!
    if not plot_utils_latex_formatting: return
    text = text.replace('_','\_')
    return text


########################################
# functions for plotting 1D histograms #
########################################
      
def plot_hists(histlist, fig=None, ax=None, colorlist=[], labellist=[], transparency=1, xlims=(-0.5,-1),
              title=None, titlesize=None, xaxtitle=None, xaxtitlesize=None, yaxtitle=None, yaxtitlesize=None,
              ymaxfactor=None, legendsize=None, opaque_legend=False, ticksize=None,
              bkgcolor=None, bkgcmap='spring', bkgcolorbar=False, bkgrange=None, bkgtitle=None):
    ### plot some histograms (in histlist) in one figure using specified colors and/or labels
    # - histlist is a list of 1D arrays containing the histograms (or a 2D array of shape (nhistograms,nbins))
    # - colorlist is a list or array containing colors (in string format), of length nhistograms
    #   note: it can also be a single string representing a color (in pyplot), then all histograms will take this color
    # - labellist is a list or array containing labels for in legend, of length nhistograms
    # - xlims is a tuple of min and max for the x-axis labels, defaults to (-0.5,nbins-0.5)
    # - title, xaxtitle, yaxtitle: strings for histogram title, x-axis title and y-axis title respectively
    # - bkgcolor: 1D array representing background color for the plot 
    #             (color axis will be scaled between min and max in bkgcolor)
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
        if bkgrange is None: bkgrange=(np.min(bkgcolor),np.max(bkgcolor))
        ax.pcolorfast((xlims[0],xlims[1]), ax.get_ylim(),
              bkgcolor[np.newaxis],
              cmap=bkgcmap, alpha=0.1,
              vmin=bkgrange[0], vmax=bkgrange[1])
        # add a color bar
        if bkgcolorbar:
            norm = mpl.colors.Normalize(vmin=bkgrange[0], vmax=bkgrange[1])
            cobject = mpl.cm.ScalarMappable(norm=norm, cmap=bkgcmap)
            cobject.set_array([]) # ad-hoc bug fix
            cbar = fig.colorbar(cobject, ax=ax, alpha=0.1)
            if bkgtitle is not None:
                cbar.ax.set_ylabel(bkgtitle, fontsize=yaxtitlesize,
                                   rotation=270, labelpad=20.)
    if ymaxfactor is not None:
        ymin,ymax = ax.get_ylim()
        ax.set_ylim( (ymin, ymax*ymaxfactor) )
    if dolabel: 
        leg = ax.legend(loc='upper right', fontsize=legendsize)
        if opaque_legend: make_legend_opaque(leg)
    if ticksize is not None: ax.tick_params(axis='both', labelsize=ticksize)
    if title is not None: ax.set_title(title, fontsize=titlesize)
    if xaxtitle is not None: ax.set_xlabel(xaxtitle, fontsize=xaxtitlesize)
    if yaxtitle is not None: ax.set_ylabel(yaxtitle, fontsize=yaxtitlesize)
    return (fig,ax)

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
    
def plot_hists_multi(histlist, fig=None, ax=None, figsize=None,
                     colorlist=[], labellist=[], transparency=1, xlims=(-0.5,-1),
                     title=None, titlesize=None, xaxtitle=None, xaxtitlesize=None, yaxtitle=None, yaxtitlesize=None,
                     caxtitle=None, caxtitlesize=None, caxtitleoffset=None, hidecaxis=False,
                     extratext=None, extratextsize=None, extratextloc=(0.7,0.6),
                     extratextha='left', extratextva='bottom',
                     remove_underflow=False, remove_overflow=False,
                     ylims=None, ymaxfactor=None, legendsize=None, opaque_legend=False,
                     ticksize=None ):
    ### plot many histograms (in histlist) in one figure using specified colors and/or labels
    # - histlist is a list of 1D arrays containing the histograms (or a 2D array of shape (nhistograms,nbins))
    # - colorlist is a list or array containing numbers to be mapped to colors
    # - labellist is a list or array containing labels for in legend
    # output: tuple of figure and axis objects, that can be used to further tune the look of the figure or save it
    if fig is None or ax is None: fig,ax = plt.subplots(figsize=figsize)
    dolabel = True; docolor = True
    # make label list for legend
    if len(labellist)==0:
        labellist = ['']*len(histlist)
        dolabel = False
    # make color list
    if len(colorlist)==0:
        docolor = False
    # make x-axis
    nbins = len(histlist[0])
    if remove_underflow: nbins -= 1
    if remove_overflow: nbins -= 1
    if xlims[1]<xlims[0]: xlims = (0,nbins)
    xax = np.linspace(xlims[0],xlims[1],num=nbins)
    # remove under- and overflow
    if remove_underflow: histlist = histlist[:,1:]
    if remove_overflow: histlist = histlist[:,:-1]
    # make color map
    if docolor:
        norm = mpl.colors.Normalize(vmin=np.min(colorlist),vmax=np.max(colorlist))
        cobject = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
        cobject.set_array([]) # ad-hoc bug fix
    # loop over histograms
    for i,row in enumerate(histlist):
        if docolor: ax.step(xax,row,where='mid',color=cobject.to_rgba(colorlist[i]),label=labellist[i],alpha=transparency)
        else: ax.step(xax,row,where='mid',label=labellist[i],alpha=transparency)
    if( docolor and not hidecaxis ): 
        cbar = fig.colorbar(cobject, ax=ax)
        if caxtitleoffset is not None: cbar.ax.get_yaxis().labelpad = caxtitleoffset
        if caxtitle is not None: cbar.ax.set_ylabel(caxtitle, fontsize=caxtitlesize, rotation=270)
    if ymaxfactor is not None:
        ymin,ymax = ax.get_ylim()
        ax.set_ylim( (ymin, ymax*ymaxfactor) )
    if ylims is not None:
        ax.set_ylim( ylims )
    if dolabel: 
        leg = ax.legend(loc='upper right', fontsize=legendsize)
        if opaque_legend: make_legend_opaque(leg)
    if ticksize is not None: ax.tick_params(axis='both', labelsize=ticksize)
    if title is not None: ax.set_title(title, fontsize=titlesize)
    if extratext is not None: 
        add_text( ax, extratext, extratextloc, fontsize=extratextsize, ha=extratextha, va=extratextva )
    if xaxtitle is not None: ax.set_xlabel(xaxtitle, fontsize=xaxtitlesize)
    if yaxtitle is not None: ax.set_ylabel(yaxtitle, fontsize=yaxtitlesize)      
    return (fig,ax)
    
def plot_sets(setlist, fig=None, ax=None, colorlist=[], labellist=[], transparencylist=[],
             title=None, titlesize=None,
             extratext=None, extratextsize=None, extratextloc=(0.7,0.6),
             extratextha='left', extratextva='bottom',
             xaxtitle=None, xaxtitlesize=None, xlims=(-0.5,-1), 
             remove_underflow=False, remove_overflow=False,
             yaxtitle=None, yaxtitlesize=None, ylims=None, ymaxfactor=None, 
             legendsize=None, opaque_legend=False, ticksize=None):
    ### plot multiple sets of 1D histograms to compare the shapes
    # - setlist is a list of 2D numpy arrays containing histograms
    # - fig and ax: a pyplot figure and axis object (if one of both is none a new figure is created)
    # - title is a string that will be used as the title for the ax object
    # other parameters are lists of which each element applies to one list of histograms

    # check for empty arrays
    for i,hists in enumerate(setlist):
        if hists.shape[0]==0:
            raise Exception('ERROR in plot_utils.py / plot_sets:'
                    +' the {}th histogram set is empty, '.format(i)
                    +' this is currently not supported for plotting')
    # parse arguments
    dolabel = True
    if len(labellist)==0:
        labellist = ['']*len(setlist)
        dolabel = False
    if len(colorlist)==0:
        colorlist = ['red','blue','green','orange']
        if len(setlist)>4:
            raise Exception('ERROR in plot_utils.py / plot_sets: '
                    +'please specify the colors if you plot more than four sets.')
    if len(transparencylist)==0:
        transparencylist = [1.]*len(setlist)
    # make x axis
    nbins = len(setlist[0][0])
    if remove_underflow: nbins -= 1
    if remove_overflow: nbins -= 1
    if xlims[1]<xlims[0]: xlims = (0,nbins)
    xax = np.linspace(xlims[0],xlims[1],num=nbins)
    # create the figure
    if fig is None or ax is None: fig,ax = plt.subplots()
    # loop over sets
    for i,histlist in enumerate(setlist):
        if remove_underflow: histlist = histlist[:,1:]
        if remove_overflow: histlist = histlist[:,:-1]
        row = histlist[0]
        ax.step(xax,row,where='mid',color=colorlist[i],label=labellist[i],alpha=transparencylist[i])
        if len(histlist)<2: continue
        for j,row in enumerate(histlist[1:,:]):
            ax.step(xax,row,where='mid',color=colorlist[i],alpha=transparencylist[i])
    if ymaxfactor is not None:
        ymin,ymax = ax.get_ylim()
        ax.set_ylim( (ymin, ymax*ymaxfactor) )
    if ylims is not None:
        ax.set_ylim( ylims )
    if dolabel: 
        leg = ax.legend(loc='upper right', fontsize=legendsize)
        if opaque_legend: make_legend_opaque(leg)
    if ticksize is not None: ax.tick_params(axis='both', labelsize=ticksize)
    if title is not None: ax.set_title(title, fontsize=titlesize)
    if extratext is not None: 
        add_text( ax, extratext, extratextloc, fontsize=extratextsize, ha=extratextha, va=extratextva )
    if xaxtitle is not None: ax.set_xlabel(xaxtitle, fontsize=xaxtitlesize)
    if yaxtitle is not None: ax.set_ylabel(yaxtitle, fontsize=yaxtitlesize)
    return (fig,ax)

def plot_anomalous(histlist, ls, highlight=-1, hrange=-1):
    ### plot a range of 1D histograms and highlight one of them
    # input arguments:
    # - histlist and ls: a list of histograms and corresponding lumisection numbers
    # - highlight: the lumisection number of the histogram to highlight
    # - hrange: the number of histograms before and after lsnumber to plot (default: whole run)
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


########################################
# functions for plotting 2D histograms #
########################################

def plot_hist_2d(hist, fig=None, ax=None, figsize=None, title=None, titlesize=None,
                xaxtitle=None, xaxtitlesize=None, yaxtitle=None, yaxtitlesize=None,
                ticklabelsize=None, colorticklabelsize=None, extent=None, aspect=None, caxrange=None,
                docolorbar=True, caxtitle=None, caxtitlesize=None, caxtitleoffset=None,
                origin='lower'):
    ### plot a 2D histogram
    # - hist is a 2D numpy array of shape (nxbins, nybins)
    # notes:
    # - if the histogram contains only nonnegative values, values below 1e-6 will not be plotted
    #   (i.e. they will be shown as white spots in the plot) to discriminate zero from small but nonzero
    # - if the histogram contains negative values, the color axis will be symmetrized around zero.
    # - the default behaviour of imshow() is to flip the axes w.r.t. numpy convention
    #   (i.e. the first axis is the y-axis instead of the x-axis),
    #   and to have the y-axis pointing downwards;
    #   both effects are fixed by transposing the array and using the 'lower' origin keyword.
    if fig is None or ax is None: fig,ax = plt.subplots(figsize=figsize)
    
    # settings
    histmin = np.amin(hist)
    histmax = np.amax(hist)
    hasnegative = histmin<-1e-6
    aspect_ratio = hist.shape[0]/hist.shape[1]
    if aspect is None: aspect = 'equal'
    if extent is not None: aspect = 'auto'   
        
    # make color object
    if not hasnegative:
        vmin = 1e-6
        vmax = max(vmin*2,histmax)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    else: 
        extremum = max(abs(histmax),abs(histmin))
        norm = mpl.colors.Normalize(vmin=-extremum, vmax=extremum, clip=False)
    if caxrange is not None:
        norm = mpl.colors.Normalize(vmin=caxrange[0], vmax=caxrange[1], clip=False)
    cmap = copy(mpl.cm.get_cmap('jet'))
    cmap.set_under('w')
    cobject = mpl.cm.ScalarMappable(norm=norm, cmap=cmap) # needed for colorbar
    cobject.set_array([]) # ad-hoc bug fix
    
    # make the plot
    ax.imshow(hist.T, aspect=aspect, interpolation='none', norm=norm, cmap=cmap,
              extent=extent, origin=origin)
    
    # add the colorbar
    # it is not straightforward to position it properly;
    # the 'magic values' are fraction=0.046 and pad=0.04, but have to be modified by aspect ratio;
    # for this, use the fact that imshow uses same scale for both axes, 
    # so can use array aspect ratio as proxy
    if docolorbar:
        fraction = 0.046; pad = 0.04
        fraction *= aspect_ratio
        pad *= aspect_ratio
        cbar = fig.colorbar(cobject, ax=ax, fraction=fraction, pad=pad)
        if caxtitleoffset is not None: cbar.ax.get_yaxis().labelpad = caxtitleoffset
        if caxtitle is not None: cbar.ax.set_ylabel(caxtitle, fontsize=caxtitlesize, rotation=270)
    
    # add titles
    if ticklabelsize is not None: ax.tick_params(labelsize=ticklabelsize)
    if colorticklabelsize is not None: cbar.ax.tick_params(labelsize=colorticklabelsize)
    if title is not None: ax.set_title(title, fontsize=titlesize)
    if xaxtitle is not None: ax.set_xlabel(xaxtitle, fontsize=xaxtitlesize)
    if yaxtitle is not None: ax.set_ylabel(yaxtitle, fontsize=yaxtitlesize)
    return (fig,ax)

def plot_hists_2d(hists, ncols=4, axsize=5, title=None, titlesize=None,
                    subtitles=None, subtitlesize=None, xaxtitles=None, yaxtitles=None,
                    **kwargs):
    ### plot multiple 2D histograms next to each other
    # input arguments
    # - hists: list of 2D numpy arrays of shape (nxbins,nybins), or an equivalent 3D numpy array
    # - ncols: number of columns to use
    # - figsize: approximate size of a single axis in the figure
    #            (will be modified by aspect ratio)
    # - title, titlesize: properties of the super title for the entire figure
    # - subtitles, subtitlesize: properties of the individual histogram titles
    # - xaxtitles, yaxtitles: properties of axis titles of individual histograms
    # - kwargs: passed down to plot_hist_2d

    # check for empty array
    if len(hists)==0:
        raise Exception('ERROR in plot_utils.py / plot_hists_2d:'
                    +' the histogram set is empty, '
                    +' this is currently not supported for plotting')

    # check arugments
    if( subtitles is not None and len(subtitles)!=len(hists) ):
        raise Exception('ERROR in plot_utils.py / plot_hists_2d:'
                +' subtitles must have same length as hists or be None')
    if( xaxtitles is not None and len(xaxtitles)!=len(hists) ):
        raise Exception('ERROR in plot_utils.py / plot_hists_2d:'
                +' xaxtitles must have same length as hists or be None')
    if( yaxtitles is not None and len(yaxtitles)!=len(hists) ):
        raise Exception('ERROR in plot_utils.py / plot_hists_2d:'
                +' yaxtitles must have same length as hists or be None')

    # initialize number of rows
    nrows = int(math.ceil(len(hists)/ncols))

    # calculate the aspect ratio of the plots and the size of the figure
    shapes = []
    for hist in hists: shapes.append( hist.shape )
    aspect_ratios = [el[0]/el[1] for el in shapes]
    aspect_ratio_max = max(aspect_ratios)
    aspect_ratio_min = min(aspect_ratios)
    aspect_ratio = 1
    if aspect_ratio_min > 1: aspect_ratio = aspect_ratio_min
    if aspect_ratio_max < 1: aspect_ratio = aspect_ratio_max
    figsize=None
    aspect_ratio *= 0.9 # correction for color bar
    if aspect_ratio>1: figsize = (axsize*ncols,axsize*nrows*aspect_ratio)
    if aspect_ratio<1: figsize = (axsize*ncols/aspect_ratio,axsize*nrows)

    # initalize the figure
    fig,axs = plt.subplots(nrows,ncols,figsize=figsize,squeeze=False)
    
    # loop over all histograms belonging to this lumisection and make the plots
    for i,hist in enumerate(hists):
        subtitle = None
        xaxtitle = None
        yaxtitle = None
        if subtitles is not None: subtitle = subtitles[i]
        if xaxtitles is not None: xaxtitle = xaxtitles[i]
        if yaxtitles is not None: yaxtitle = yaxtitles[i]
        # make the plot
        plot_hist_2d(hist, fig=fig,ax=axs[int(i/ncols),i%ncols],
                title=subtitle, titlesize=subtitlesize, xaxtitle=xaxtitle, yaxtitle=yaxtitle, 
                **kwargs)
    
    # add a title
    if title is not None: fig.suptitle(title, fontsize=titlesize)
    
    # return the figure and axes
    return (fig,axs)

def plot_hists_2d_gif( hists, 
                       titles=None, xaxtitle=None, yaxtitle=None,
                       duration=300, figname='temp_gif.gif',
                       mode='imageio',
                       **kwargs ):
    # manage backend
    if mode=='imageio':
        try: import imageio
        except: raise Exception('ERROR: could not import imageio')
    elif mode=='pillow':
        try: from PIL import Image
        except: raise Exception('ERROR: could not import PIL')
    else: raise Exception('ERROR: mode {} not recognized'.format(mode))
    # make individual images
    nhists = len(hists)
    filenames = []
    for i in range(nhists):
        title = None
        if titles is not None: title = titles[i]
        fig,_ = plot_hist_2d(hists[i], title=title, xaxtitle=xaxtitle, yaxtitle=yaxtitle, **kwargs)
        filename = 'temp_gif_file_{}.png'.format(i)
        filenames.append(filename)
        fig.savefig(filename, facecolor='white', transparent=False)
        plt.close()
    # convert to gif
    if mode=='imageio':
        # first approach with imageio
        with imageio.get_writer(figname, mode='I', duration=duration, loop=0) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
    elif mode=='pillow':
        # second approach with Pillow
        frames = [Image.open(filename) for filename in filenames]
        frame_one = frames[0]
        frame_one.save(figname, format="GIF", append_images=frames,
                   save_all=True, duration=duration, loop=0)
    # remove individual images
    for filename in filenames:
        os.remove(filename)
        
        
################################################################
# functions for plotting moments and distances in moment space #
################################################################

def plot_moments(moments, ls, dims=(0,1), 
                 fig=None, ax=None, markersize=10,
                 xaxtitle='auto', xaxtitlesize=12,
                 yaxtitle='auto', yaxtitlesize=12,
                 zaxtitle='auto', zaxtitlesize=12,
                 caxtitle=None, caxtitlesize=12, caxtitleoffset=15,
                 ticksize=None):
    ### plot the moments of a set of histograms
    # input arguments:
    # - moments: a numpy array of shape (nhists,nmoments)
    # - dims: a tuple of two or three values between 0 and nmoments-1
    from mpl_toolkits.mplot3d import Axes3D # specific import
    if fig==None: fig = plt.figure()
    if len(dims)==2:
        if ax==None: ax = fig.add_subplot(111)
        scpl = ax.scatter(moments[:,dims[0]],moments[:,dims[1]],s=markersize,c=ls,cmap='jet')
        cbar = plt.colorbar(scpl)
        if xaxtitle is not None:
            if xaxtitle=='auto': xaxtitle = 'Moment '+str(dims[0]+1)
            ax.set_xlabel(xaxtitle, fontsize=xaxtitlesize)
        if yaxtitle is not None:
            if yaxtitle=='auto': yaxtitle = 'Moment '+str(dims[1]+1)
            ax.set_ylabel(yaxtitle, fontsize=yaxtitlesize)
        if caxtitleoffset is not None: cbar.ax.get_yaxis().labelpad = caxtitleoffset
        if caxtitle is not None: cbar.ax.set_ylabel(caxtitle, fontsize=caxtitlesize, rotation=270)
        if ticksize is not None: ax.tick_params(axis='both', labelsize=ticksize)
    elif len(dims)==3:
        if ax==None: ax = fig.add_subplot(111, projection='3d')
        scpl = ax.scatter(moments[:,dims[0]],moments[:,dims[1]],moments[:,dims[2]],s=markersize,c=ls,cmap='jet')
        plt.colorbar(scpl)
        if xaxtitle is not None:
            if xaxtitle=='auto': xaxtitle = 'Moment '+str(dims[0]+1)
            ax.set_xlabel(xaxtitle, fontsize=xaxtitlesize)
        if yaxtitle is not None:
            if yaxtitle=='auto': yaxtitle = 'Moment '+str(dims[1]+1)
            ax.set_ylabel(yaxtitle, fontsize=yaxtitlesize)
        if zaxtitle is not None:
            if zaxtitle=='auto': zaxtitle = 'Moment '+str(dims[2]+1)
            ax.set_zlabel(zaxtitle, fontsize=zaxtitlesize)
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


###################################################################
# functions for plotting model training/validation losses and MSE #
###################################################################

### plot model loss as a function of training epoch
# credits to Francesco for this function
def plot_loss(data,
              loss_key='loss', val_loss_key='val_loss',
              title=None, titlesize=None, 
              xaxtitle='Epoch', xaxtitlesize=None, 
              yaxtitle='Loss', yaxtitlesize=None,
              xlims=None, yaxlog=True,
              legendsize=None, legendloc='best',
              doshow=True):
    ### plot the training and validation loss of a keras/tensorflow model
    # input arguments:
    # - data: the object returned by the .fit method when called upon a keras model
    # - other: plot layout options
    fig,ax = plt.subplots()
    if loss_key is not None: 
        ax.plot(data.history[loss_key], linestyle=(0,()), color="#1A237E", linewidth=3, label='Training')
    if val_loss_key is not None:
        ax.plot(data.history[val_loss_key], linestyle=(0,(3,2)), color="#4DB6AC", linewidth=3, label='Validation')
    ax.legend(loc=legendloc, fontsize=legendsize)
    if yaxlog: ax.set_yscale('log')
    if xlims is not None: ax.set_xlim(xlims)
    if title is not None: ax.set_title(title, fontsize=legendsize)
    if xaxtitle is not None: ax.set_xlabel(xaxtitle, fontsize=xaxtitlesize)
    if yaxtitle is not None: ax.set_ylabel(yaxtitle, fontsize=xaxtitlesize)
    if doshow: plt.show(block=False)
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


###########################################################
# functions for plotting model output score distributions #
###########################################################

def plot_score_dist( scores, labels, fig=None, ax=None,
                        nbins=20, normalize=False, normalizesignal=False,
                        siglabel='Signal', sigcolor='g',
                        bcklabel='Background', bckcolor='r',
                        title=None, titlesize=12,
                        extratext=None, extratextsize=12,
                        xaxtitle=None, xaxtitlesize=12, 
                        yaxtitle=None, yaxtitlesize=12,
                        legendsize=None, legendloc='best',
                        ticksize=None,
                        doshow=True):
    ### make a plot showing the distributions of the output scores for signal and background
    
    # define binning between min and max
    minscore = np.min(scores)
    maxscore = np.max(scores)
    scorebins = np.linspace(minscore,maxscore,num=nbins+1)
    scoreax = (scorebins[1:] + scorebins[:-1])/2
    # split in signal and background
    sigscores = scores[np.where(labels==1)]
    bkgscores = scores[np.where(labels==0)]
    # make histograms
    sighist = np.histogram(sigscores,bins=scorebins)[0]
    bckhist = np.histogram(bkgscores,bins=scorebins)[0]
    if normalize:
        if np.sum(sighist)!=0: sighist = sighist/np.sum(sighist)
        if np.sum(bckhist)!=0: bckhist = bckhist/np.sum(bckhist)
    if normalizesignal:
        if np.amax(sighist)!=0: sighist *= np.amax(bckhist)/np.amax(sighist)
    if( fig is None or ax is None): (fig,ax) = plt.subplots()
    ax.step(scoreax,bckhist,color=bckcolor,label=bcklabel,where='mid')
    ax.step(scoreax,sighist,color=sigcolor,label=siglabel,where='mid')
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    ax.ticklabel_format(axis='x', style='scientific', scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(ticksize)
    ax.yaxis.get_offset_text().set_fontsize(ticksize)
    if title is not None: ax.set_title(title, fontsize=titlesize)
    if extratext is not None: 
        add_text( ax, extratext, (0.95,0.6), fontsize=extratextsize, horizontalalignment='right' )
    if xaxtitle is not None: ax.set_xlabel(xaxtitle, fontsize=xaxtitlesize)
    if yaxtitle is not None: ax.set_ylabel(yaxtitle, fontsize=yaxtitlesize)
    ax.legend( loc=legendloc, fontsize=legendsize )
    if doshow: plt.show(block=False)
    return (fig,ax)

def plot_score_dist_multi( scores, labels=None, colors=None, fig=None, ax=None,
                           nbins=20, normalize=False,
                           linestyle=None, linewidth=1,
                           title=None, titlesize=12,
                           xaxtitle=None, xaxtitlesize=12, 
                           yaxtitle=None, yaxtitlesize=12,
                           legendsize=None, legendloc='best',
                           ticksize=None, dolegend=True ):
    ### plot the distribution of output scores for arbitrarily many sets (not limited to 'signal' and 'background')
    # input arguments:
    # - scores: list of numpy arrays of scores
    # - labels: list of legend entries for the scores, must have same length as scores or be None (no legend)
    # - colors: list of colors for the different score arrays, must have same length as scores or be None (default colors)
    
    # default settings
    if labels is None: 
        labels = ['']*len(scores)
        dolegend = False
    if colors is None:
        colors = ['b']*len(scores)
    # make the x-axis
    minscore = min( np.amin(score_array) for score_array in scores )
    maxscore = max( np.amax(score_array) for score_array in scores )
    scorebins = np.linspace(minscore,maxscore,num=nbins+1)
    scoreax = (scorebins[1:] + scorebins[:-1])/2
    # make the histograms
    hists = []
    for score_array in scores:
        hist = np.histogram(score_array,bins=scorebins)[0]
        if normalize: hist = hist/np.sum(hist)
        hists.append(hist)
    # make the figure and do initial plot
    if( fig is None or ax is None): (fig,ax) = plt.subplots()
    for i,hist in enumerate(hists):
        ax.step(scoreax, hist, 
                color=colors[i],label=labels[i], where='mid',
                linestyle=linestyle, linewidth=linewidth)
    # plot aesthetics
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    ax.ticklabel_format(axis='x', style='scientific', scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(ticksize)
    ax.yaxis.get_offset_text().set_fontsize(ticksize)
    if title is not None: ax.set_title(title, fontsize=titlesize)
    if xaxtitle is not None: ax.set_xlabel(xaxtitle, fontsize=xaxtitlesize)
    if yaxtitle is not None: ax.set_ylabel(yaxtitle, fontsize=yaxtitlesize)
    if dolegend: ax.legend( loc=legendloc, fontsize=legendsize )
    return (fig,ax)

def plot_score_ls( thisscore, refscores, fig=None, ax=None, 
                    thislabel='This LS', thiscolor='black',
                    reflabel='Reference LS', refcolor='dodgerblue', **kwargs ):
    ### make a plot of the score for a single lumisection comparing to some reference distribution
    
    # make a dummy plot of the score distributions with thisscore unnormalized
    scores = np.concatenate((refscores,np.array([thisscore])))
    labels = np.concatenate((np.zeros(len(refscores)),np.ones(1)))
    extraargs = {}
    if 'nbins' in kwargs.keys(): extraargs['nbins'] = kwargs['nbins']
    fig,ax = plot_score_dist( scores, labels, fig=fig, ax=ax, 
                                normalize=False, doshow=False, **extraargs )

    # make the peak score to superimpose
    ymax = ax.get_ylim()[1]
    sigscores = [thisscore]*int(ymax)
    scores = np.concatenate((refscores,sigscores))
    labels = np.concatenate((np.zeros(len(refscores)),np.ones(len(sigscores))))

    # clear the current plot
    ax.clear()

    # make the plot
    # note: for now, normalized distribution is not supported since thisscores 
    #       will always reach to 1, while refscores might have a much lower maximum
    if 'normalize' in kwargs.keys(): kwargs.pop('normalize')
    fig,ax = plot_score_dist( scores, labels, fig=fig, ax=ax,
                            siglabel=thislabel, sigcolor=thiscolor,
                            bcklabel=reflabel, bckcolor=refcolor,
                            normalize=False,
                            **kwargs )
    return (fig,ax)


##############################################
# function for metric and ROC curve plotting #
##############################################

def plot_metric( wprange, metric, label=None, color=None,
                    sig_eff=None, sig_label=None, sig_color=None,
                    bck_eff=None, bck_label=None, bck_color=None,
                    legendsize=None,
                    title=None,
                    xaxtitle='working point', xaxtitlesize=None,
                    yaxlog=False, ymaxfactor=1.3, 
                    yaxtitle='metric', yaxtitlesize=None ):
    ### plot a metric based on signal and background efficiencies.
    # along with the metric, the actual signal and background efficiencies can be plotted as well.
    # input arguments:
    # - wprange, metric: equally long 1D-numpy arrays, x- and y-data respectively
    # - label: label for the metric to put in the legend
    # - color: color for the metric (default: blue)
    # - sig_eff: 1D-numpy array of signal efficiencies corresponding to wprange
    # - sig_label: label for sig_eff in the legend
    # - color: color for sig_eff (default: green)
    # - bck_eff, bck_label, bck_color: same as for signal
    # - title, xaxtitle and yaxtitle: titles for the plot and axes
    # - yaxlog: boolean whether to put y axis in log scale
    # - ymaxfactor: factor to add extra space on the top of the plot (for the legend)
    fig,ax = plt.subplots()
    ax2 = ax.twinx()
    # parse arguments
    if label is None: label = ''
    if sig_label is None: sig_label = ''
    if bck_label is None: bck_label = ''
    if color is None: color = 'blue'
    if sig_color is None: sig_color = 'forestgreen'
    if bck_color is None: bck_color = 'firebrick'
    # make the plots
    ax.plot( wprange, metric, label=label, color=color, linewidth=3 )
    if sig_eff is not None: ax2.plot( wprange, sig_eff, label=sig_label,
                                        color=sig_color, linewidth=2 )
    if bck_eff is not None: ax2.plot( wprange, bck_eff, label=bck_label,
                                        color=bck_color, linewidth=2 )
    # draw a dashed line at unit efficiency
    ax.grid()
    ax2.plot( [wprange[0],wprange[1]], [1.,1.], color='black', linestyle='dashed')
    # set the legends
    ax.legend(loc='upper left', fontsize=legendsize)
    ax2.legend(loc='upper right', fontsize=legendsize)
    # axis properties for first axes
    if yaxlog: ax.set_yscale('log')
    ymin,ymax = ax.get_ylim()
    ax.set_ylim( (ymin, ymax*ymaxfactor) )
    if title is not None: ax.set_title(title)
    if xaxtitle is not None: ax.set_xlabel(xaxtitle, fontsize=xaxtitlesize)
    if yaxtitle is not None: ax.set_ylabel(yaxtitle, color=color, fontsize=yaxtitlesize)
    # axis properties for second axes
    ax2.set_ylabel('Efficiency', fontsize=yaxtitlesize)
    ymin,ymax = ax2.get_ylim()
    ax2.set_ylim( (ymin, ymax*ymaxfactor) )
    return (fig,ax,ax2)

def plot_roc( sig_eff, bkg_eff, auc=None, sig_eff_unc=None,
              color='b',
              title=None, titlesize=None,
              xaxtitle='Background efficiency', xaxtitlesize=None,
              yaxtitle='Signal efficiency', yaxtitlesize=None,
              xaxlog=True, yaxlog=False, xlims='auto', ylims='auto', dogrid=True,
              ticksize=None,
              doshow=True ):
    # note: automatic determination of xlims and ylims assumes log scale for x-axis and lin scale for y-axis;
    #       might not work properly in other cases and ranges should be provided manually.
    fig,ax = plt.subplots()
    # in case of log scale plots, remove problematic points
    if xaxlog:
        inds = np.nonzero(bkg_eff)
        bkg_eff = bkg_eff[inds]
        sig_eff = sig_eff[inds]
        if sig_eff_unc is not None: sig_eff_unc = sig_eff_unc[inds]
    if yaxlog:
        inds = np.nonzero(sig_eff)
        bkg_eff = bkg_eff[inds]
        sig_eff = sig_eff[inds]
        if sig_eff_unc is not None: sig_eff_unc = sig_eff_unc[inds]
    # plot the actual curve
    ax.scatter(bkg_eff, sig_eff, color=color)
    # plot uncertainty bands
    if sig_eff_unc is not None:
        ax.fill_between(bkg_eff, sig_eff - sig_eff_unc, sig_eff + sig_eff_unc,
                        color=color, alpha=0.25)
    # axis settings
    if title is not None: ax.set_title(title, fontsize=titlesize)
    if xaxtitle is not None: ax.set_xlabel(xaxtitle, fontsize=xaxtitlesize)
    if yaxtitle is not None: ax.set_ylabel(yaxtitle, fontsize=yaxtitlesize)   
    if xaxlog: ax.set_xscale('log')
    if yaxlog: ax.set_yscale('log')
    # set x axis limits
    if xlims=='auto':
        xlims = (np.amin(np.where(bkg_eff>0.,bkg_eff,1.))/2.,1.)
    ax.set_xlim(xlims)
    # set y axis limits
    if ylims=='auto':
        ylowlim = np.amin(np.where((sig_eff>0.) & (bkg_eff>0.),sig_eff,1.))
        ylowlim = 2*ylowlim-1.
        ylims = (ylowlim,1+(1-ylowlim)/5)
        if ylowlim==1: ylims = (0.95,1.05)
    ax.set_ylim(ylims)
    # tick formatting
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    #ax.ticklabel_format(axis='both', style='scientific', scilimits=(0,0))
    #ax.xaxis.get_offset_text().set_fontsize(ticksize)
    #ax.yaxis.get_offset_text().set_fontsize(ticksize)
    # enable grid for easier reading
    if dogrid: ax.grid()
    # write AUC value
    if auc is not None:
        auctext = '{:.3f}'.format(auc)
        if auc>0.99:
            auctext = '1 - '+'{:.3e}'.format(1-auc)
        ax.text(0.7,0.1,'AUC: '+auctext,transform=ax.transAxes)
    if doshow: plt.show(block=False)
    return (fig,ax)

def plot_confusion_matrix( tp, tn, fp, fn, 
                          true_positive_label='Good', true_negative_label='Anomalous',
                          pred_positive_label='Predicted good', pred_negative_label='Predicted anomalous',
                          xaxlabelsize=None, yaxlabelsize=None, textsize=None,
                          colormap='Blues', colortitle=None ):
    cmat = np.array([[tp,fn],[fp,tn]])
    fig,ax = plt.subplots()
    norm = mpl.colors.Normalize(0,1)
    alpha = 1
    # plot the matrix
    ax.imshow(cmat, cmap=colormap, norm=norm, alpha=alpha)
    # add axis labels
    ax.set_xticks((0,1))
    ax.set_xticklabels((pred_positive_label,pred_negative_label), fontsize=xaxlabelsize)
    ax.set_yticks((0,1))
    ax.set_yticklabels((true_positive_label,true_negative_label), fontsize=yaxlabelsize)
    # add a color bar
    cobject = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
    cobject.set_array([]) # ad-hoc bug fix
    cbar = fig.colorbar(cobject, ax=ax, alpha=alpha)
    if colortitle is not None: cbar.ax.set_ylabel(colortitle, rotation=270, labelpad=20.)
    # add text
    def color(value):
        if value>0.5: return 'w'
        return 'k'
    def valstr(value):
        frmt = '{:.3f}'
        if value<0.01: frmt = '{:.3e}'
        return frmt.format(value)
    ax.text(0, 0, valstr(tp), fontsize=textsize, 
                   horizontalalignment='center', verticalalignment='center',
                   color=color(tp))
    ax.text(0, 1, valstr(fn), fontsize=textsize, 
                   horizontalalignment='center', verticalalignment='center',
                   color=color(fn))
    ax.text(1, 0, valstr(fp), fontsize=textsize, 
                   horizontalalignment='center', verticalalignment='center',
                   color=color(fp))
    ax.text(1, 1, valstr(tn), fontsize=textsize, 
                   horizontalalignment='center', verticalalignment='center',
                   color=color(tn))
    
    return (fig,ax)


##################################################
# functions for plotting fits and point clusters #
##################################################

def clip_scores( scores ):
    ### clip +-inf values in scores
    # local copy of the same functions in autoencoder_utils.py
    # (need to copy here locally to use in plot_fit_2d and plot_fit_1d without circular import...)
    maxnoninf = np.max(np.where(scores==np.inf,np.min(scores),scores)) + 1
    minnoninf = np.min(np.where(scores==-np.inf,np.max(scores),scores)) -1
    if np.max(scores)>maxnoninf: 
        scores = np.where(scores==np.inf,maxnoninf,scores)
        print('NOTE: scores of +inf were reset to {}'.format(maxnoninf))
    if np.min(scores)<minnoninf:
        scores = np.where(scores==-np.inf,minnoninf,scores)
        print('NOTE: scores of -inf were reset to {}'.format(minnoninf))
    return scores

def plot_fit_2d( points, fitfunc=None, figsize=(10,8), logprob=False, clipprob=False, 
                onlycontour=False, xlims=5, ylims=5, onlypositive=False,
                xaxtitle=None, xaxtitlesize=None, yaxtitle=None, yaxtitlesize=None, 
                title=None, titlesize=None,
                caxtitle=None, caxtitlesize=None, caxrange=None,
                transparency=1, ticksize=None ):
    ### make a 2D scatter plot of a point cloud with fitted contour
    # input arguments:
    # - points: a numpy array of shape (npoints,ndims), where ndims is supposed to be 2
    # - fitfunc: an object of type CloudFitter (see src/cloudfitters) 
    #   or any other object that implements a pdf(points) method
    # - logprob: boolean whether to plot log probability or normal probability
    # - clipprob: boolean whether to replace +- inf values by (non-inf) max and min
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
    xstep = (xlims[1]-xlims[0])/300.
    ystep = (ylims[1]-ylims[0])/300.
        
    (fig,ax) = plt.subplots(figsize=figsize)
    
    if fitfunc is not None:
        
        # make a grid of points and evaluate the fitfunc
        x,y = np.mgrid[xlims[0]:xlims[1]:xstep,ylims[0]:ylims[1]:ystep]
        gridpoints = np.transpose(np.vstack((np.ravel(x),np.ravel(y))))
        evalpoints = fitfunc.pdf(gridpoints)
        if logprob: evalpoints = np.log(evalpoints)
        if clipprob: evalpoints = clip_scores(evalpoints)
        z = np.reshape(evalpoints,x.shape)

        # make a plot of probability contours
        contourplot = ax.contourf(x, y, z, 30, alpha=transparency)
        colorbar = plt.colorbar(contourplot)
        if caxtitle is not None: colorbar.set_label(caxtitle, fontsize=caxtitlesize)
        if caxrange is not None: contourplot.set_clim(caxrange[0], caxrange[1])
        colorbar.ax.tick_params(labelsize=ticksize)
        
    if not onlycontour:
        
        # make a plot of the data points
        ax.plot(points[:,0],points[:,1],'.b',markersize=2)
    
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    if title is not None: ax.set_title(title, fontsize=titlesize)
    if xaxtitle is not None: ax.set_xlabel(xaxtitle, fontsize=xaxtitlesize)
    if yaxtitle is not None: ax.set_ylabel(yaxtitle, fontsize=yaxtitlesize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    ax.ticklabel_format(axis='both', style='scientific', scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(ticksize)
    ax.yaxis.get_offset_text().set_fontsize(ticksize)
    
    return (fig,ax)

def plot_fit_2d_clusters( points, clusters, figsize=(12,8), labels=None, colors=None, 
                          legendmarkerscale=1., legendsize=10, legendloc='best', legendbbox=None, **kwargs ):
    ### make a 2D scatter plot of a fitted contour with point clouds superimposed
    # input arguments: 
    # - points: numpy arrays of shape (npoints,ndims), where ndims is supposed to be 2,
    #           usually the points to which the fit was done
    #           note: only used to determine plotting range, these points are not plotted!
    # - clusters: list of numpy arrays of shape (npoints,ndims), where ndims is supposed to be 2,
    #             clouds of points to plot
    # - labels: list with legend entries (must be same length as clusters)
    # - colors: list with colors (must be same length as clusters)
    # - kwargs: passed down to plot_fit_2d 
    #           note: onlycontour is set automatically and should not be in kwargs
    
    # first make contour plot
    fig,ax = plot_fit_2d(points, figsize=figsize, onlycontour=True, **kwargs )

    # other initializations
    dolegend = True
    if labels is None:
        dolegend = False
        labels = ['']*len(clusters)
    if colors is None:
        colors = ['b']*len(clusters)

    # make scatter plots
    for j in range(len(clusters)):
        cluster = clusters[j]
        label = labels[j]
        color = colors[j]
        ax.plot( cluster[:,0], cluster[:,1], '.', color=color, markersize=4,label=label )
    if dolegend:
        legend = ax.legend(markerscale=legendmarkerscale, fontsize=legendsize, loc=legendloc, bbox_to_anchor=legendbbox)
    return (fig,ax)

def plot_fit_1d( points, fitfunc=None, logprob=False, clipprob=False,
                onlycontour=False, xlims=5, onlypositive=False,
                xaxtitle=None, xaxtitlesize=None, yaxtitle=None, yaxtitlesize=None,
                title=None, titlesize=None ):
    ### make a 1D scatter plot of a point cloud with fitted contour
    # input arguments:
    # - points: a numpy array of shape (npoints,ndims), where ndims is supposed to be 1
    # - fitfunc: an object of type CloudFitter (see src/cloudfitters) 
    #   or any other object that implements a pdf(points) method
    # - logprob: boolean whether to plot log probability or normal probability
    # - clipprob: boolean whether to replace +- inf values by (non-inf) max and min
    # - onlycontour: a boolean whether to draw only the fit or include the data points
    # - xlims: tuple of the form (low,high)
    #   note: can be an integer, in which case the range will be determined automatically
    #         from the formula low = mean-xlims*std, high = mean+xlims*std,
    #         where mean and std are determined from the points array.
    # - onlypositive: set lower bound of plotting range at 0 (overrides xlims)
    # - xaxtitle and yaxtitle: titles for axes.

    # set plotting ranges and step sizes
    if( isinstance(xlims,int) ):
        xmean = np.mean(points[:,0])
        xstd = np.std(points[:,0])
        xlims = (xmean-xlims*xstd,xmean+xlims*xstd)
    if onlypositive:
        xlims = (0,xlims[1])
    xstep = (xlims[1]-xlims[0])/100.

    (fig,ax) = plt.subplots()

    if fitfunc is not None:

        # make a grid of points and evaluate the fitfunc
        x = np.arange(xlims[0], xlims[1], xstep)
        gridpoints = np.expand_dims(x, 1)
        evalpoints = fitfunc.pdf(gridpoints)
        if logprob: evalpoints = np.log(evalpoints)
        if clipprob: evalpoints = clip_scores(evalpoints)
        z = evalpoints

        # make a plot of probability contours
        contourplot = ax.plot(x, z)

    if not onlycontour:

        # make a plot of the data points
        ax.plot(points[:,0],[0]*len(points),'.b',markersize=2)

    ax.set_xlim(xlims)
    if title is not None: ax.set_title(title, fontsize=titlesize)
    if xaxtitle is not None: ax.set_xlabel(xaxtitle, fontsize=xaxtitlesize)
    if yaxtitle is not None: ax.set_ylabel(yaxtitle, fontsize=yaxtitlesize)
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))

    return (fig,ax)

def plot_fit_1d_clusters( points, clusters, labels=None, colors=None, **kwargs ):
    ### make a 1D scatter plot of a fitted contour with point clouds superimposed
    # input arguments: 
    # - points: numpy arrays of shape (npoints,ndims), where ndims is supposed to be 1,
    #           usually the points to which the fit was done
    #           note: only used to determine plotting range, these points are not plotted!
    # - clusters: list of numpy arrays of shape (npoints,ndims), where ndims is supposed to be 1,
    #             clouds of points to plot
    # - labels: list with legend entries (must be same length as clusters)
    # - colors: list with colors (must be same length as clusters)
    # - kwargs: passed down to plot_fit_1d
    #           note: onlycontour is set automatically and should not be in kwargs

    # first make contour plot
    fig,ax = plot_fit_1d(points, onlycontour=True, **kwargs )
    
    # other initializations
    dolegend = True
    if labels is None:
        dolegend = False
        labels = ['']*len(clusters)
    if colors is None:
        colors = ['b']*len(clusters)
    ymax = ax.get_ylim()[1]

    # make scatter plots
    for j in range(len(clusters)):
        cluster = clusters[j]
        label = labels[j]
        color = colors[j]
        yvalue = j*ymax/20
        ax.plot( cluster[:,0], [yvalue]*len(cluster), '.', color=color, markersize=4, label=label )
    if dolegend: ax.legend()
    return (fig,ax)
