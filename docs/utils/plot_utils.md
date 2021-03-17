# plot utils  
  
- - -    
## plot_hists(histlist,colorlist=[],labellist=[],transparency=1,xlims=(0,-1), title=None,xaxtitle=None,yaxtitle=None)  
**plot some histograms (in histlist) in one figure using specified colors and/or labels**  
- histlist is a list of 1D arrays containing the histograms (or a 2D array of shape (nhistograms,nbins))  
- colorlist is a list or array containing colors (in string format)  
note: it can also be a single string representing a color (in pyplot), then all histograms will take this color  
- labellist is a list or array containing labels for in legend  
output: tuple of figure and axis objects, that can be used to further tune the look of the figure or save it  
  
- - -    
## plot_hists_multi(histlist,colorlist=[],labellist=[],transparency=1,xlims=(0,-1), title=None,xaxtitle=None,yaxtitle=None)  
**plot many histograms (in histlist) in one figure using specified colors and/or labels**  
- histlist is a list of 1D arrays containing the histograms (or a 2D array of shape (nhistograms,nbins))  
- colorlist is a list or array containing numbers to be mapped to colors  
- labellist is a list or array containing labels for in legend  
output: tuple of figure and axis objects, that can be used to further tune the look of the figure or save it  
  
- - -    
## plot_hists_from_df(df,histtype,nhists)  
**plot a number of histograms in a dataframe**  
- df is the dataframe from which to plot  
- histtype is the name of the histogram type (e.g. 'chargeInner_PXLayer_1')  
- nhists is the number of histograms to plot  
  
- - -    
## plot_sets(setlist,fig=None,ax=None,colorlist=[],labellist=[],transparencylist=[],xlims=(0,-1), title=None,xaxtitle=None,yaxtitle=None)  
**plot multiple sets of histograms to compare the shapes**  
- setlist is a list of 2D numpy arrays containing histograms  
- fig and ax: a pyplot figure and axis object (if one of both is none a new figure is created)  
- title is a string that will be used as the title for the ax object  
other parameters are lists of which each element applies to one list of histograms  
  
- - -    
## plot_anomalous(histlist,ls,highlight=-1,hrange=-1)  
histlist and ls are a list of histograms and corresponding lumisection numbers  
lsnumber is the lumisection number of the histogram to highlight  
hrange is the number of histograms before and after lsnumber to plot (default: whole run)  
  
- - -    
## plot_moments(moments,ls,dims,fig=None,ax=None,markersize=10)  
moments is an (nhists,nmoments) array  
dims is a tuple of two or three values between 0 and nmoments-1  
  
- - -    
## plot_distance(dists,ls=None,rmlargest=0.,doplot=True, title=None,xaxtitle='lumisection number',yaxtitle='distance metric')  
(no valid documentation found)  
  
- - -    
## plot_loss(data, xlims=None, title=None,xaxtitle='epoch',yaxtitle='loss')  
**plot the training and validation loss**  
data is the object returned by the .fit method when called upon a keras model  
e.g. history = <your autoencoder>.fit(<training params>)  
plot_loss(history,'a title')  
  
- - -    
## plot_mse(mse,rmlargest=0.,doplot=True, title=None,xaxtitle='lumisection number',yaxtitle='mse')  
**plot the mse's and return the mean and std**  
input args:  
- mse is a 1D numpy array of mse scores  
- doplot: boolean whether to make a plot or simply return mean and std  
- rmlargest: fraction of largest mse's to remove (to avoid being too sensitive to outliers)  
  
- - -    
## plot_score_dist( scores, labels, nbins=20, normalize=False, title='output score distributions for signal and background', xaxtitle='output score',yaxtitle=None)  
(no valid documentation found)  
  
