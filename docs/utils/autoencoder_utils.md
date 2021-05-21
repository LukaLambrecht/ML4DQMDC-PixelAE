# autoencoder utils  
  
- - -    
## mseTop10(y\_true, y\_pred)  
**MSE top 10 loss function for autoencoder training**  
input arguments:  
- y\_true and y\_pred: two numpy arrays of equal shape,  
typically a histogram and its autoencoder reconstruction.  
if two-dimensional, the arrays are assumed to have shape (nhists,nbins)!  
output:  
- mean squared error between y\_true and y\_pred,  
where only the 10 bins with largest squared error are taken into account.  
if y\_true and y\_pred are 2D arrays, this function returns 1D array (mseTop10 for each histogram)  
  
- - -    
## mseTop10Raw(y\_true, y\_pred)  
**same as mseTop10 but without using tf or K**  
the version including tf or K seemed to cause randomly dying kernels, no clear reason could be found,  
but it was solved using this loss function instead.  
verified that it gives exactly the same output as the function above on some random arrays.  
contrary to mseTop10, this function only works for arrays with 2D shapes (so shape (nhists,nbins)), not for (nbins,).  
  
- - -    
## mseTopNRaw(y\_true, y\_pred, n=10)  
**generalization of mseTop10Raw to any number of bins to take into account**  
note: now generalized to also work for 2D histograms, i.e. arrays of shape (nhists,nybins,nxbins)!  
hence this is the most general method and preferred above mseTop10 and mseTop10Raw, which are only kept for reference  
input arguments:  
- y\_true, y\_pred: numpy arrays between which to calculate the mean square difference, of shape (nhists,nbins) or (nhists,nybins,nxbins)  
- n: number of largest elements to keep for averaging  
output:  
numpy array of shape (nhists)  
  
- - -    
## chiSquared(y\_true, y\_pred)  
**chi2 loss functionfor autoencoder training**  
input arguments:  
- y\_true and y\_pred: two numpy arrays of equal shape,  
typically a histogram and its autoencoder reconstruction.  
if two-dimensional, the arrays are assumed to have shape (nhists,nbins)!  
output:  
- relative mean squared error between y\_true and y\_pred,  
if y\_true and y\_pred are 2D arrays, this function returns 1D array (chiSquared for each histogram)  
  
- - -    
## chiSquaredTop10(y\_true, y\_pred)  
**same as chiSquared but take into account only 10 largest values in averaging.**  
  
- - -    
## calculate\_roc(scores, labels, scoreax)  
**calculate a roc curve**  
input arguments:  
- scores is a 1D numpy array containing output scores of any algorithm  
- labels is a 1D numpy array (equally long as scores) containing labels  
note that 1 for signal and 0 for background is assumed!  
this convention is only used to define what scores belong to signal or background;  
the scores itself can be anything (not limited to (0,1)),  
as long as the target for signal is higher than the target for background  
- scoreax is an array of score thresholds for which to compute the signal and background efficiency,  
assumed to be sorted in increasing order (i.e. from loose to tight)  
output:  
tuple of two np arrays (signal efficiency and background efficiency)  
  
- - -    
## get\_roc(scores, labels, mode='lin', npoints=100, doprint=False, doplot=True, plotmode='classic')  
**make a ROC curve**  
input arguments:  
- scores is a 1D numpy array containing output scores of any algorithm  
- labels is a 1D numpy array (equally long as scores) containing labels  
note that 1 for signal and 0 for background is assumed!  
this convention is only used to define what scores belong to signal or background;  
the scores itself can be anything (not limited to (0,1)),  
as long as the target for signal is higher than the target for background  
- mode: how to determine the points where to calculate signal and background efficiencies; options are:  
- 'lin': np.linspace between min and max score  
- 'geom': np. geomspace between min and max score  
- 'full': one point per score instance  
- npoints: number of points where to calculate the signal and background efficiencies  
(ignored if mode is 'full')  
- doprint: boolean whether to print score thresholds and corresponding signal and background efficiencies  
- doplot: boolean whether to make a plot or simply return the auc.  
- plotmode: how to plot the roc curve; options are:  
- 'classic' = signal efficiency afo background efficiency  
  
- - -    
## get\_roc\_from\_hists(hists, labels, predicted\_hists, mode='lin', npoints=100, doprint=False, doplot=True, plotmode='classic')  
**make a ROC curve without manually calculating the scores**  
the output score is the mseTop10Raw between the histograms and their reconstruction  
- input arguments:  
- hists and predicted\_hists are 2D numpy arrays of shape (nhistograms,nbins)  
- other arguments: see get\_roc  
  
- - -    
## get\_confusion\_matrix(scores, labels, wp)  
**plot a confusion matrix**  
scores and labels are defined in the same way as for get\_roc  
wp is the chosen working point  
(i.e. any score above wp is flagged as signal, any below is flagged as background)  
  
- - -    
## get\_confusion\_matrix\_from\_hists(hists, labels, predicted\_hists, msewp)  
**plot a confusion matrix without manually calculating the scores**  
the output score is the mse between the histograms and their reconstruction  
  
- - -    
## getautoencoder(input\_size,arch,act=[],opt='adam',loss=mseTop10)  
get a trainable autoencoder model  
input args:  
- input\_size: size of vector that autoencoder will operate on  
- arch: list of number of nodes per hidden layer (excluding input and output layer)  
- act: list of activations per layer (default: tanh)  
- opt: optimizer to use (default: adam)  
- loss: loss function to use (defualt: mseTop10)  
  
- - -    
## train\_simple\_autoencoder(hists,nepochs=-1,modelname='')  
**create and train a very simple keras model**  
the model consists of one hidden layer (with half as many units as there are input bins), tanh activation, adam optimizer and mseTop10 loss.  
input args:  
- hists is a 2D numpy array of shape (nhistograms, nbins)  
- nepochs is the number of epochs to use (has a default value if left unspecified)  
- modelname is a file name to save the model in (default: model is not saved to a file)  
  
