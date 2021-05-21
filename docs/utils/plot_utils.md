# plot utils  
  
- - -    
## plot\_hists(histlist, colorlist=[], labellist=[], transparency=1, xlims=(0,-1), title=None, xaxtitle=None, yaxtitle=None, bkgcolor=None, bkgcmap='spring')  
**plot some histograms (in histlist) in one figure using specified colors and/or labels**  
- histlist is a list of 1D arrays containing the histograms (or a 2D array of shape (nhistograms,nbins))  
- colorlist is a list or array containing colors (in string format)  
note: it can also be a single string representing a color (in pyplot), then all histograms will take this color  
- labellist is a list or array containing labels for in legend  
output: tuple of figure and axis objects, that can be used to further tune the look of the figure or save it  
  
- - -    
## plot\_hists\_multi(histlist, colorlist=[], labellist=[], transparency=1, xlims=(0,-1), title=None, xaxtitle=None, yaxtitle=None)  
**plot many histograms (in histlist) in one figure using specified colors and/or labels**  
- histlist is a list of 1D arrays containing the histograms (or a 2D array of shape (nhistograms,nbins))  
- colorlist is a list or array containing numbers to be mapped to colors  
- labellist is a list or array containing labels for in legend  
output: tuple of figure and axis objects, that can be used to further tune the look of the figure or save it  
  
- - -    
## plot\_hist\_2d(hist, fig=None, ax=None, title=None, xaxtitle=None, yaxtitle=None, caxrange=None)  
**plot a 2D histogram**  
- hist is a 2D numpy array of shape (nxbins, nybins)  
notes:  
- if the histogram contains only nonnegative values, values below 1e-12 will not be plotted  
(i.e. they will be shown as white spots in the plot) to discriminate zero from small but nonzero  
- if the histogram contains negative values, the color axis will be symmetrized around zero  
  
- - -    
## plot\_hists\_2d(hists, ncols=4, title = None, subtitles=None, xaxtitle=None, yaxtitle=None, caxrange=None)  
**plot multiple 2D histograms next to each other**  
- hists: list of 2D numpy arrays of shape (nxbins,nybins), or an equivalent 3D numpy array  
- ncols: number of columns to use  
  
- - -    
## plot\_hists\_2d\_gif(hists, titles = None, xaxtitle=None, yaxtitle=None, duration=0.3, figname='temp\_gif.gif')  
(no valid documentation found)  
  
- - -    
## plot\_hists\_from\_df(df, histtype, nhists)  
**plot a number of histograms in a dataframe**  
- df is the dataframe from which to plot  
- histtype is the name of the histogram type (e.g. 'chargeInner\_PXLayer\_1')  
- nhists is the number of histograms to plot  
  
- - -    
## plot\_sets(setlist, fig=None, ax=None, colorlist=[], labellist=[], transparencylist=[], xlims=(0,-1), title=None, xaxtitle=None, yaxtitle=None)  
**plot multiple sets of histograms to compare the shapes**  
- setlist is a list of 2D numpy arrays containing histograms  
- fig and ax: a pyplot figure and axis object (if one of both is none a new figure is created)  
- title is a string that will be used as the title for the ax object  
other parameters are lists of which each element applies to one list of histograms  
  
- - -    
## plot\_anomalous(histlist, ls, highlight=-1, hrange=-1)  
histlist and ls are a list of histograms and corresponding lumisection numbers  
lsnumber is the lumisection number of the histogram to highlight  
hrange is the number of histograms before and after lsnumber to plot (default: whole run)  
  
- - -    
## plot\_moments(moments, ls, dims, fig=None, ax=None, markersize=10)  
moments is an (nhists,nmoments) array  
dims is a tuple of two or three values between 0 and nmoments-1  
  
- - -    
## plot\_distance(dists, ls=None, rmlargest=0., doplot=True, title=None, xaxtitle='lumisection number', yaxtitle='distance metric')  
(no valid documentation found)  
  
- - -    
## plot\_loss(data, xlims=None, title=None, xaxtitle='epoch', yaxtitle='loss')  
**plot the training and validation loss**  
data is the object returned by the .fit method when called upon a keras model  
e.g. history = <your autoencoder>.fit(<training params>)  
plot\_loss(history,'a title')  
  
- - -    
## plot\_mse(mse, rmlargest=0., doplot=True, title=None, xaxtitle='lumisection number', yaxtitle='mse')  
**plot the mse's and return the mean and std**  
input args:  
- mse is a 1D numpy array of mse scores  
- doplot: boolean whether to make a plot or simply return mean and std  
- rmlargest: fraction of largest mse's to remove (to avoid being too sensitive to outliers)  
  
- - -    
## plot\_score\_dist( scores, labels, nbins=20, normalize=False, title='output score distributions for signal and background', xaxtitle='output score', yaxtitle=None)  
**make a plot showing the distributions of the output scores for signal and background**  
  
- - -    
## plot\_fit\_2d( points, fitfunc=None, logprob=False, onlycontour=False, xlims=5, ylims=5, onlypositive=False, xaxtitle=None, yaxtitle=None )  
**make a scatter plot of a 2D point cloud with fitted contour**  
input arguments:  
- points: a numpy array of shape (npoints,ndims)  
- fitfunc: an object of type CloudFitter (see src/cloudfitters)  
or any other object that implements a pdf(points) method  
- logprob: boolean whether to plot log probability or normal probability  
- onlycontour: a boolean whether to draw only the fit or include the data points  
- xlims and ylims: tuples of (low,high)  
note: can be an integer, in which case the range will be determined automatically  
from the formula low = mean-xlims*std, high = mean+xlims*std,  
where mean and std are determined from the points array.  
- onlypositive: overrides previous argument to set lower bound of plotting range at 0 in both dimensions.  
- xaxtitle and yaxtitle: titles for axes.  
  
