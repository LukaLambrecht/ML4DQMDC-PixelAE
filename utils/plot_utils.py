#!/usr/bin/env python
# coding: utf-8



### imports

# external modules
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# local modules




# functions for plotting 
      
def plot_hists(histlist,colorlist=[],labellist=[],transparency=1,xlims=(0,-1),
              title=None,xaxtitle=None,yaxtitle=None):
    ### plot some histograms (in histlist) in one figure using specified colors and/or labels
    # - histlist is a list of 1D arrays containing the histograms (or a 2D array of shape (nhistograms,nbins))
    # - colorlist is a list or array containing colors (in string format)
    #   note: it can also be a single string representing a color (in pyplot), then all histograms will take this color
    # - labellist is a list or array containing labels for in legend
    # output: tuple of figure and axis objects, that can be used to further tune the look of the figure or save it
    dolabel = True; docolor = True
    if len(labellist)==0:
        labellist = ['']*len(histlist)
        dolabel = False
    if isinstance(colorlist,str):
        colorlist = [colorlist]*len(histlist)
    if len(colorlist)==0:
        docolor = False
    if xlims[1]<xlims[0]: xlims = (0,len(histlist[0]))
    xax = np.linspace(xlims[0],xlims[1],num=len(histlist[0]))
    fig,ax = plt.subplots()
    for i,row in enumerate(histlist):
        if docolor: ax.step(xax,row,color=colorlist[i],label=labellist[i],alpha=transparency)
        else: ax.step(xax,row,label=labellist[i],alpha=transparency)
    if dolabel: ax.legend()
    if title is not None: ax.set_title(title)
    if xaxtitle is not None: ax.set_xlabel(xaxtitle)
    if yaxtitle is not None: ax.set_ylabel(yaxtitle)
    plt.show()
    return (fig,ax)
    
def plot_hists_multi(histlist,colorlist=[],labellist=[],transparency=1,xlims=(0,-1),
                    title=None,xaxtitle=None,yaxtitle=None):
    ### plot many histograms (in histlist) in one figure using specified colors and/or labels
    # - histlist is a list of 1D arrays containing the histograms (or a 2D array of shape (nhistograms,nbins))
    # - colorlist is a list or array containing numbers to be mapped to colors
    # - labellist is a list or array containing labels for in legend
    # output: tuple of figure and axis objects, that can be used to further tune the look of the figure or save it
    dolabel = True; docolor = True
    if len(labellist)==0:
        labellist = ['']*len(histlist)
        dolabel = False
    if len(colorlist)==0:
        docolor = False
    if xlims[1]<xlims[0]: xlims = (0,len(histlist[0]))
    xax = np.linspace(xlims[0],xlims[1],num=len(histlist[0]))
    fig,ax = plt.subplots()
    if docolor:
        norm = mpl.colors.Normalize(vmin=np.min(colorlist),vmax=np.max(colorlist))
        cobject = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
        cobject.set_array([]) # ad-hoc bug fix
    for i,row in enumerate(histlist):
        if docolor: ax.step(xax,row,color=cobject.to_rgba(colorlist[i]),label=labellist[i],alpha=transparency)
        else: ax.step(xax,row,label=labellist[i],alpha=transparency)
    if docolor: fig.colorbar(cobject)
    if dolabel: ax.legend()
    if title is not None: ax.set_title(title)
    if xaxtitle is not None: ax.set_xlabel(xaxtitle)
    if yaxtitle is not None: ax.set_ylabel(yaxtitle)
    plt.show()
    return (fig,ax)
    
def plot_hists_from_df(df,histtype,nhists):
    ### plot a number of histograms in a dataframe
    # - df is the dataframe from which to plot
    # - histtype is the name of the histogram type (e.g. 'chargeInner_PXLayer_1')
    # - nhists is the number of histograms to plot
    dfs = select_histnames(df,[histtype])
    nhists = min(len(dfs),nhists)
    dfs = dfs[0:nhists+1]
    val = get_hist_values(dfs)[0]
    plot_hists(val)
    
def plot_sets(setlist,fig=None,ax=None,colorlist=[],labellist=[],transparencylist=[],xlims=(0,-1),
             title=None,xaxtitle=None,yaxtitle=None):
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
        ax.step(xax,row,color=colorlist[i],label=labellist[i],alpha=transparencylist[i])
        if len(histlist)<2: continue
        for j,row in enumerate(histlist[1:,:]):
            ax.step(xax,row,color=colorlist[i],alpha=transparencylist[i])
    if dolabel: ax.legend(loc='upper right')
    if title is not None: ax.set_title(title)
    if xaxtitle is not None: ax.set_xlabel(xaxtitle)
    if yaxtitle is not None: ax.set_ylabel(yaxtitle)
    return (fig,ax)

def plot_anomalous(histlist,ls,highlight=-1,hrange=-1):
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
        ax.step(xax,lshist,color='black',linewidth=2)
    plt.show()
    return (fig,ax)

def plot_moments(moments,ls,dims,fig=None,ax=None,markersize=10):
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
    plt.show()
    return (fig,ax)


def plot_distance(dists,ls=None,rmlargest=0.,doplot=True,
                 title=None,xaxtitle='lumisection number',yaxtitle='distance metric'):
    
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
              title=None,xaxtitle='epoch',yaxtitle='loss'):
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
def plot_mse(mse,rmlargest=0.,doplot=True,
            title=None,xaxtitle='lumisection number',yaxtitle='mse'):
    ### plot the mse's and return the mean and std
    # input args:
    # - mse is a 1D numpy array of mse scores
    # - doplot: boolean whether to make a plot or simply return mean and std
    # - rmlargest: fraction of largest mse's to remove (to avoid being too sensitive to outliers)
    
    (obj1,obj2) = plot_distance(mse,rmlargest=rmlargest,doplot=doplot,title=title,xaxtitle=xaxtitle,yaxtitle=yaxtitle)
    return (obj1,obj2)

### make a plot showing the distributions of the output scores for signal and background

def plot_score_dist( scores, labels, nbins=20, normalize=False,
                        title='output score distributions for signal and background',
                        xaxtitle='output score',yaxtitle=None):
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










