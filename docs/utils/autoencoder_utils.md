# autoencoder utils  
  
- - -    
## mseTop10(y_true, y_pred)  
(no valid documentation found)  
  
- - -    
## mseTop10Raw(y_true, y_pred)  
same as above but without using tf or K  
the version including tf or K seemed to cause randomly dying kernels, no clear reason could be found,  
but it was solved using this loss function instead.  
verified that it gives exactly the same output as the function above on some random arrays  
does only work for arrays with 2D shapes, not for (nbins,)  
  
- - -    
## mseTopNRaw(y_true, y_pred, n=10)  
generalization of the above  
  
- - -    
## chiSquared(y_true, y_pred)  
(no valid documentation found)  
  
- - -    
## chiSquaredTop10(y_true, y_pred)  
(no valid documentation found)  
  
- - -    
## get_roc(scores, labels, mode='classic', doplot=True)  
**make a ROC curve**  
input arguments:  
- scores is a 1D numpy array containing output scores of any algorithm  
- labels is a 1D numpy array (equally long as scores) containing labels  
note that 1 for signal and 0 for background is assumed!  
this convention is only used to define what scores belong to signal or background;  
the scores itself can be anything (not limited to (0,1)),  
as long as the target for signal is higher than the target for background  
- mode: how to plot the roc curve; options are:  
- 'classic' = signal efficiency afo background efficiency  
- doplot: boolean whether to make a plot or simply return the auc.  
  
- - -    
## get_roc_from_hists(hists, labels, predicted_hists, mode='classic', doplot=True)  
**make a ROC curve without manually calculating the scores**  
the output score is the mse between the histograms and their reconstruction  
hists and predicted_hists are 2D numpy arrays of shape (nhistograms,nbins)  
other arguments: see get_roc  
  
- - -    
## get_confusion_matrix(scores, labels, wp)  
**plot a confusion matrix**  
scores and labels are defined in the same way as for get_roc  
wp is the chosen working point  
(i.e. any score above wp is flagged as signal, any below is flagged as background)  
  
- - -    
## get_confusion_matrix_from_hists(hists, labels, predicted_hists, msewp)  
**plot a confusion matrix without manually calculating the scores**  
the output score is the mse between the histograms and their reconstruction  
  
- - -    
## getautoencoder(input_size,arch,act=[],opt='adam',loss=mseTop10)  
get a trainable autoencoder model  
input args:  
- input_size: size of vector that autoencoder will operate on  
- arch: list of number of nodes per hidden layer (excluding input and output layer)  
- act: list of activations per layer (default: tanh)  
- opt: optimizer to use (default: adam)  
- loss: loss function to use (defualt: mseTop10)  
  
- - -    
## train_simple_autoencoder(hists,nepochs=-1,modelname='')  
**create and train a very simple keras model**  
the model consists of one hidden layer (with half as many units as there are input bins), tanh activation, adam optimizer and mseTop10 loss.  
input args:  
- hists is a 2D numpy array of shape (nhistograms, nbins)  
- nepochs is the number of epochs to use (has a default value if left unspecified)  
- modelname is a file name to save the model in (default: model is not saved to a file)  
  
