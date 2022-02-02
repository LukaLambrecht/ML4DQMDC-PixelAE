# plot utils  
  
**A collection of useful basic functions for plotting.**  
- - -
  
  
### make\_legend\_opaque  
full signature:  
```text  
def make_legend_opaque( leg )  
```  
comments:  
```text  
set the transparency of all entries in a legend to zero  
```  
  
  
### add\_text  
full signature:  
```text  
def add_text( ax, text, pos,  fontsize=10, horizontalalignment='left', verticalalignment='bottom', background_facecolor=None,  background_alpha=None,  background_edgecolor=None, **kwargs )  
```  
comments:  
```text  
add text to an axis at a specified position (in relative figure coordinates)  
input arguments:  
- ax: matplotlib axis object  
- text: string, can contain latex syntax such as /textbf{} and /textit{}  
- pos: tuple with relative x- and y-axis coordinates of bottom left corner  
```  
  
  
### add\_cms\_label  
full signature:  
```text  
def add_cms_label( ax, pos=(0.1,0.9), extratext=None, **kwargs )  
```  
comments:  
```text  
add the CMS label and extra text (e.g. 'Preliminary') to a plot  
special case of add_text, for convenience  
```  
  
  
### make\_text\_latex\_safe  
full signature:  
```text  
def make_text_latex_safe( text )  
```  
comments:  
```text  
make a string safe to process with matplotlib's latex parser in case no tex parsing is wanted  
(e.g. escape underscores)  
to be extended when the need arises!  
```  
  
  
### plot\_hists  
full signature:  
```text  
def plot_hists(histlist, fig=None, ax=None, colorlist=[], labellist=[], transparency=1, xlims=(-0.5,-1), title=None, xaxtitle=None, yaxtitle=None,  bkgcolor=None, bkgcmap='spring', bkgrange=None, bkgtitle=None)  
```  
comments:  
```text  
plot some histograms (in histlist) in one figure using specified colors and/or labels  
- histlist is a list of 1D arrays containing the histograms (or a 2D array of shape (nhistograms,nbins))  
- colorlist is a list or array containing colors (in string format), of length nhistograms  
  note: it can also be a single string representing a color (in pyplot), then all histograms will take this color  
- labellist is a list or array containing labels for in legend, of length nhistograms  
- xlims is a tuple of min and max for the x-axis labels, defaults to (-0.5,nbins-0.5)  
- title, xaxtitle, yaxtitle: strings for histogram title, x-axis title and y-axis title respectively  
- bkgcolor: 1D array representing background color for the plot   
            (color axis will be scaled between min and max in bkgcolor)  
  note: if bkgcolor does not have the same length as the x-axis, it will be compressed or stretched to fit the axis,  
        but this might be meaningless, depending on what you are trying to visualize!  
- bkgmap: name of valid pyplot color map for plotting the background color  
output: tuple of figure and axis objects, that can be used to further tune the look of the figure or save it  
```  
  
  
### plot\_hists\_from\_df  
full signature:  
```text  
def plot_hists_from_df(df, histtype, nhists)  
```  
comments:  
```text  
plot a number of histograms in a dataframe  
- df is the dataframe from which to plot  
- histtype is the name of the histogram type (e.g. 'chargeInner_PXLayer_1')  
- nhists is the number of histograms to plot  
```  
  
  
### plot\_hists\_multi  
full signature:  
```text  
def plot_hists_multi(histlist, fig=None, ax=None, colorlist=[], labellist=[], transparency=1, xlims=(-0.5,-1), title=None, titlesize=None, xaxtitle=None, xaxtitlesize=None, yaxtitle=None, yaxtitlesize=None, caxtitle=None, caxtitlesize=None, caxtitleoffset=None, remove_underflow=False, remove_overflow=False, ylims=None, ymaxfactor=None, legendsize=None, opaque_legend=False)  
```  
comments:  
```text  
plot many histograms (in histlist) in one figure using specified colors and/or labels  
- histlist is a list of 1D arrays containing the histograms (or a 2D array of shape (nhistograms,nbins))  
- colorlist is a list or array containing numbers to be mapped to colors  
- labellist is a list or array containing labels for in legend  
output: tuple of figure and axis objects, that can be used to further tune the look of the figure or save it  
```  
  
  
### plot\_sets  
full signature:  
```text  
def plot_sets(setlist, fig=None, ax=None, colorlist=[], labellist=[], transparencylist=[], title=None, titlesize=None,  xaxtitle=None, xaxtitlesize=None, xlims=(-0.5,-1),  remove_underflow=False, remove_overflow=False, yaxtitle=None, yaxtitlesize=None, ylims=None, ymaxfactor=None,  legendsize=None, opaque_legend=False)  
```  
comments:  
```text  
plot multiple sets of 1D histograms to compare the shapes  
- setlist is a list of 2D numpy arrays containing histograms  
- fig and ax: a pyplot figure and axis object (if one of both is none a new figure is created)  
- title is a string that will be used as the title for the ax object  
other parameters are lists of which each element applies to one list of histograms  
```  
  
  
### plot\_anomalous  
full signature:  
```text  
def plot_anomalous(histlist, ls, highlight=-1, hrange=-1)  
```  
comments:  
```text  
plot a range of 1D histograms and highlight one of them  
input arguments:  
- histlist and ls: a list of histograms and corresponding lumisection numbers  
- highlight: the lumisection number of the histogram to highlight  
- hrange: the number of histograms before and after lsnumber to plot (default: whole run)  
```  
  
  
### plot\_hist\_2d  
full signature:  
```text  
def plot_hist_2d(hist, fig=None, ax=None, title=None, titlesize=None, xaxtitle=None, xaxtitlesize=None, yaxtitle=None, yaxtitlesize=None, caxrange=None)  
```  
comments:  
```text  
plot a 2D histogram  
- hist is a 2D numpy array of shape (nxbins, nybins)  
notes:  
- if the histogram contains only nonnegative values, values below 1e-12 will not be plotted  
  (i.e. they will be shown as white spots in the plot) to discriminate zero from small but nonzero  
- if the histogram contains negative values, the color axis will be symmetrized around zero  
```  
  
  
### plot\_hists\_2d  
full signature:  
```text  
def plot_hists_2d(hists, ncols=4, axsize=5, title=None, titlesize=None, subtitles=None, subtitlesize=None, xaxtitles=None, yaxtitles=None, **kwargs)  
```  
comments:  
```text  
plot multiple 2D histograms next to each other  
input arguments  
- hists: list of 2D numpy arrays of shape (nxbins,nybins), or an equivalent 3D numpy array  
- ncols: number of columns to use  
- figsize: approximate size of a single axis in the figure  
           (will be modified by aspect ratio)  
- title, titlesize: properties of the super title for the entire figure  
- subtitles, subtitlesize: properties of the individual histogram titles  
- xaxtitles, yaxtitles: properties of axis titles of individual histograms  
- kwargs: passed down to plot_hist_2d  
```  
  
  
### plot\_hists\_2d\_gif  
full signature:  
```text  
def plot_hists_2d_gif(hists, titles=None, xaxtitle=None, yaxtitle=None, duration=0.3, figname='temp_gif.gif')  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### plot\_moments  
full signature:  
```text  
def plot_moments(moments, ls, dims=(0,1), fig=None, ax=None, markersize=10)  
```  
comments:  
```text  
plot the moments of a set of histograms  
input arguments:  
- moments: a numpy array of shape (nhists,nmoments)  
- dims: a tuple of two or three values between 0 and nmoments-1  
```  
  
  
### plot\_distance  
full signature:  
```text  
def plot_distance(dists, ls=None, rmlargest=0., doplot=True, title=None, xaxtitle='lumisection number', yaxtitle='distance metric')  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### plot\_loss  
full signature:  
```text  
def plot_loss(data, xlims=None, title=None, titlesize=None,  xaxtitle='Epoch', xaxtitlesize=None,  yaxtitle='Loss', yaxtitlesize=None, legendsize=None, legendloc='best', doshow=True)  
```  
comments:  
```text  
plot the training and validation loss of a keras/tensorflow model  
input arguments:  
- data: the object returned by the .fit method when called upon a keras model  
- other: plot layout options  
```  
  
  
### plot\_mse  
full signature:  
```text  
def plot_mse(mse, rmlargest=0., doplot=True, title=None, xaxtitle='lumisection number', yaxtitle='mse')  
```  
comments:  
```text  
plot the mse's and return the mean and std  
input args:  
- mse is a 1D numpy array of mse scores  
- doplot: boolean whether to make a plot or simply return mean and std  
- rmlargest: fraction of largest mse's to remove (to avoid being too sensitive to outliers)  
```  
  
  
### plot\_score\_dist  
full signature:  
```text  
def plot_score_dist( scores, labels, fig=None, ax=None, nbins=20, normalize=False, siglabel='Signal', sigcolor='g', bcklabel='Background', bckcolor='r', title=None, titlesize=12, xaxtitle=None, xaxtitlesize=12,  yaxtitle=None, yaxtitlesize=12, legendsize=None, legendloc='best', doshow=True)  
```  
comments:  
```text  
make a plot showing the distributions of the output scores for signal and background  
```  
  
  
### plot\_score\_ls  
full signature:  
```text  
def plot_score_ls( thisscore, refscores, fig=None, ax=None,  thislabel='This LS', thiscolor='black', reflabel='Reference LS', refcolor='dodgerblue', **kwargs )  
```  
comments:  
```text  
make a plot of the score for a single lumisection comparing to some reference distribution  
```  
  
  
### plot\_metric  
full signature:  
```text  
def plot_metric( wprange, metric, label=None, color=None, sig_eff=None, sig_label=None, sig_color=None, bck_eff=None, bck_label=None, bck_color=None, title=None, xaxtitle='working point', yaxlog=False, ymaxfactor=1.3, yaxtitle='metric' )  
```  
comments:  
```text  
plot a metric based on signal and background efficiencies.  
along with the metric, the actual signal and background efficiencies can be plotted as well.  
input arguments:  
- wprange, metric: equally long 1D-numpy arrays, x- and y-data respectively  
- label: label for the metric to put in the legend  
- color: color for the metric (default: blue)  
- sig_eff: 1D-numpy array of signal efficiencies corresponding to wprange  
- sig_label: label for sig_eff in the legend  
- color: color for sig_eff (default: green)  
- bck_eff, bck_label, bck_color: same as for signal  
- title, xaxtitle and yaxtitle: titles for the plot and axes  
- yaxlog: boolean whether to put y axis in log scale  
- ymaxfactor: factor to add extra space on the top of the plot (for the legend)  
```  
  
  
### plot\_roc  
full signature:  
```text  
def plot_roc( sig_eff, bkg_eff, auc=None, color='b', title=None, titlesize=None, xaxtitle='Background efficiency', xaxtitlesize=None, yaxtitle='Signal efficiency', yaxtitlesize=None, xaxlog=True, yaxlog=False, xlims='auto', ylims='auto', dogrid=True, doshow=True )  
```  
comments:  
```text  
note: automatic determination of xlims and ylims assumes log scale for x-axis and lin scale for y-axis;  
      might not work properly in other cases and ranges should be provided manually.  
```  
  
  
### clip\_scores  
full signature:  
```text  
def clip_scores( scores )  
```  
comments:  
```text  
clip +-inf values in scores  
local copy of the same functions in autoencoder_utils.py  
(need to copy here locally to use in plot_fit_2d and plot_fit_1d without circular import...)  
```  
  
  
### plot\_fit\_2d  
full signature:  
```text  
def plot_fit_2d( points, fitfunc=None, logprob=False, clipprob=False,  onlycontour=False, xlims=5, ylims=5, onlypositive=False, xaxtitle=None, xaxtitlesize=None, yaxtitle=None, yaxtitlesize=None,  title=None, titlesize=None, caxtitle=None, caxtitlesize=None, transparency=1 )  
```  
comments:  
```text  
make a 2D scatter plot of a point cloud with fitted contour  
input arguments:  
- points: a numpy array of shape (npoints,ndims), where ndims is supposed to be 2  
- fitfunc: an object of type CloudFitter (see src/cloudfitters)   
  or any other object that implements a pdf(points) method  
- logprob: boolean whether to plot log probability or normal probability  
- clipprob: boolean whether to replace +- inf values by (non-inf) max and min  
- onlycontour: a boolean whether to draw only the fit or include the data points  
- xlims and ylims: tuples of (low,high)  
  note: can be an integer, in which case the range will be determined automatically  
        from the formula low = mean-xlims*std, high = mean+xlims*std,  
        where mean and std are determined from the points array.  
- onlypositive: overrides previous argument to set lower bound of plotting range at 0 in both dimensions.  
- xaxtitle and yaxtitle: titles for axes.  
```  
  
  
### plot\_fit\_2d\_clusters  
full signature:  
```text  
def plot_fit_2d_clusters( points, clusters, labels=None, colors=None,  legendsize=10, legendloc='best', legendbbox=None, **kwargs )  
```  
comments:  
```text  
make a 2D scatter plot of a fitted contour with point clouds superimposed  
input arguments:   
- points: numpy arrays of shape (npoints,ndims), where ndims is supposed to be 2,  
          usually the points to which the fit was done  
          note: only used to determine plotting range, these points are not plotted!  
- clusters: list of numpy arrays of shape (npoints,ndims), where ndims is supposed to be 2,  
            clouds of points to plot  
- labels: list with legend entries (must be same length as clusters)  
- colors: list with colors (must be same length as clusters)  
- kwargs: passed down to plot_fit_2d   
          note: onlycontour is set automatically and should not be in kwargs  
```  
  
  
### plot\_fit\_1d  
full signature:  
```text  
def plot_fit_1d( points, fitfunc=None, logprob=False, clipprob=False, onlycontour=False, xlims=5, onlypositive=False, xaxtitle=None, xaxtitlesize=None, yaxtitle=None, yaxtitlesize=None, title=None, titlesize=None )  
```  
comments:  
```text  
make a 1D scatter plot of a point cloud with fitted contour  
input arguments:  
- points: a numpy array of shape (npoints,ndims), where ndims is supposed to be 1  
- fitfunc: an object of type CloudFitter (see src/cloudfitters)   
  or any other object that implements a pdf(points) method  
- logprob: boolean whether to plot log probability or normal probability  
- clipprob: boolean whether to replace +- inf values by (non-inf) max and min  
- onlycontour: a boolean whether to draw only the fit or include the data points  
- xlims: tuple of the form (low,high)  
  note: can be an integer, in which case the range will be determined automatically  
        from the formula low = mean-xlims*std, high = mean+xlims*std,  
        where mean and std are determined from the points array.  
- onlypositive: set lower bound of plotting range at 0 (overrides xlims)  
- xaxtitle and yaxtitle: titles for axes.  
```  
  
  
### plot\_fit\_1d\_clusters  
full signature:  
```text  
def plot_fit_1d_clusters( points, clusters, labels=None, colors=None, **kwargs )  
```  
comments:  
```text  
make a 1D scatter plot of a fitted contour with point clouds superimposed  
input arguments:   
- points: numpy arrays of shape (npoints,ndims), where ndims is supposed to be 1,  
          usually the points to which the fit was done  
          note: only used to determine plotting range, these points are not plotted!  
- clusters: list of numpy arrays of shape (npoints,ndims), where ndims is supposed to be 1,  
            clouds of points to plot  
- labels: list with legend entries (must be same length as clusters)  
- colors: list with colors (must be same length as clusters)  
- kwargs: passed down to plot_fit_1d  
          note: onlycontour is set automatically and should not be in kwargs  
```  
  
  
