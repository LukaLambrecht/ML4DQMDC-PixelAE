# autoencoder utils  
  
**Utilities related to the training and evaluation of autoencoder models with keras**

The functionality in this script includes:
- definition of loss functions (several flavours of MSE or chi-squared)
- calculating and plotting ROC curves and confusion matrices
- definition of very simple ready-to-use keras model architectures
- - -
  
  
### mseTop10  
full signature:  
```text  
def mseTop10(y_true, y_pred)  
```  
comments:  
```text  
MSE top 10 loss function for autoencoder training  
input arguments:  
- y_true and y_pred: two numpy arrays of equal shape,  
  typically a histogram and its autoencoder reconstruction.  
  if two-dimensional, the arrays are assumed to have shape (nhists,nbins)!  
output:  
- mean squared error between y_true and y_pred,  
  where only the 10 bins with largest squared error are taken into account.  
  if y_true and y_pred are 2D arrays, this function returns 1D array (mseTop10 for each histogram)  
```  
  
  
### mseTop10Raw  
full signature:  
```text  
def mseTop10Raw(y_true, y_pred)  
```  
comments:  
```text  
same as mseTop10 but without using tf or K  
the version including tf or K seemed to cause randomly dying kernels, no clear reason could be found,  
but it was solved using this loss function instead.  
verified that it gives exactly the same output as the function above on some random arrays.  
contrary to mseTop10, this function only works for arrays with 2D shapes (so shape (nhists,nbins)), not for (nbins,).  
```  
  
  
### mseTopNRaw  
full signature:  
```text  
def mseTopNRaw(y_true, y_pred, n=10)  
```  
comments:  
```text  
generalization of mseTop10Raw to any number of bins to take into account  
note: now generalized to also work for 2D histograms, i.e. arrays of shape (nhists,nybins,nxbins)!  
      hence this is the most general method and preferred above mseTop10 and mseTop10Raw, which are only kept for reference  
input arguments:  
- y_true, y_pred: numpy arrays between which to calculate the mean square difference, of shape (nhists,nbins) or (nhists,nybins,nxbins)  
- n: number of largest elements to keep for averaging  
output:  
numpy array of shape (nhists)  
```  
  
  
### chiSquared  
full signature:  
```text  
def chiSquared(y_true, y_pred)  
```  
comments:  
```text  
chi2 loss function for autoencoder training  
input arguments:  
- y_true and y_pred: two numpy arrays of equal shape,  
  typically a histogram and its autoencoder reconstruction.  
  if two-dimensional, the arrays are assumed to have shape (nhists,nbins)!  
output:  
- relative mean squared error between y_true and y_pred,  
  if y_true and y_pred are 2D arrays, this function returns 1D array (chiSquared for each histogram)  
```  
  
  
### chiSquaredTopNRaw  
full signature:  
```text  
def chiSquaredTopNRaw(y_true, y_pred, n=10)  
```  
comments:  
```text  
generalization of chiSquared to any number of bins to take into account  
note: should work for 2D histograms as well (i.e. arrays of shape (nhistograms,nybins,nxbins)),  
      but not yet tested!  
input arguments:  
- y_true, y_pred: numpy arrays between which to calculate the mean square difference, of shape (nhists,nbins) or (nhists,nybins,nxbins)  
- n: number of largest elements to keep for summing  
output:  
numpy array of shape (nhists)  
```  
  
  
### calculate\_roc  
full signature:  
```text  
def calculate_roc(scores, labels, scoreax)  
```  
comments:  
```text  
calculate a roc curve  
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
- tuple of two np arrays (signal efficiency and background efficiency)  
```  
  
  
### get\_roc  
full signature:  
```text  
def get_roc(scores, labels, mode='lin', npoints=100, doprint=False, doplot=True, plotmode='classic', doshow=True)  
```  
comments:  
```text  
make a ROC curve  
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
```  
  
  
### get\_roc\_from\_hists  
full signature:  
```text  
def get_roc_from_hists(hists, labels, predicted_hists, mode='lin', npoints=100, doprint=False, doplot=True, plotmode='classic')  
```  
comments:  
```text  
make a ROC curve without manually calculating the scores  
the output score is the mseTop10Raw between the histograms and their reconstruction  
- input arguments:  
- hists and predicted_hists are 2D numpy arrays of shape (nhistograms,nbins)  
- other arguments: see get_roc  
```  
  
  
### get\_confusion\_matrix  
full signature:  
```text  
def get_confusion_matrix(scores, labels, wp='maxauc', plotwp=True)  
```  
comments:  
```text  
plot a confusion matrix  
input arguments:  
- scores and labels: defined in the same way as for get_roc  
- wp: the chosen working point   
      (i.e. any score above wp is flagged as signal, any below is flagged as background)  
      note: wp can be a integer or float, in which case that value will be used directly,  
            or it can be a string in which case it will be used as the 'method' argument in get_wp!  
- plotwp: only relevant if wp is a string (see above), in which case plotwp will be used as the 'doplot' argument in get_wp  
```  
  
  
### get\_confusion\_matrix\_from\_hists  
full signature:  
```text  
def get_confusion_matrix_from_hists(hists, labels, predicted_hists, msewp=None)  
```  
comments:  
```text  
plot a confusion matrix without manually calculating the scores  
the output score is the mse between the histograms and their reconstruction  
```  
  
  
### get\_wp  
full signature:  
```text  
def get_wp(scores, labels, method='maxauc', doplot=False)  
```  
comments:  
```text  
automatically calculate a suitable working point  
input arguments:  
- scores, labels: equally long 1d numpy arrays of predictions and true labels respectively  
                  note: in all methods, the labels are assumed to be 0 (for background) or 1 (for signal)!  
- method: method to calculate the working point  
          currently supported: 'maxauc'  
- doplot: make a plot (if a plotting method exists for the chosen method)  
```  
  
  
### get\_wp\_maxauc  
full signature:  
```text  
def get_wp_maxauc(scores, labels, doplot=False)  
```  
comments:  
```text  
calculate the working point corresponding to maximum pseudo-AUC  
(i.e. maximize the rectangular area enclosed by the working point)  
```  
  
  
### getautoencoder  
full signature:  
```text  
def getautoencoder(input_size,arch,act=[],opt='adam',loss=mseTop10)  
```  
comments:  
```text  
get a trainable autoencoder model  
input args:  
- input_size: size of vector that autoencoder will operate on  
- arch: list of number of nodes per hidden layer (excluding input and output layer)  
- act: list of activations per layer (default: tanh)  
- opt: optimizer to use (default: adam)  
- loss: loss function to use (defualt: mseTop10)  
```  
  
  
### train\_simple\_autoencoder  
full signature:  
```text  
def train_simple_autoencoder(hists, nepochs=-1, modelname='',  batch_size=500, shuffle=False,  verbose=1, validation_split=0.1)  
```  
comments:  
```text  
create and train a very simple keras model  
the model consists of one hidden layer (with half as many units as there are input bins), tanh activation, adam optimizer and mseTop10 loss.  
input args:   
- hists is a 2D numpy array of shape (nhistograms, nbins)  
- nepochs is the number of epochs to use (has a default value if left unspecified)  
- modelname is a file name to save the model in (default: model is not saved to a file)  
```  
  
  
### clip\_scores  
full signature:  
```text  
def clip_scores( scores )  
```  
comments:  
```text  
clip +-inf values in scores  
+inf values in scores will be replaced by the maximum value (exclucing +inf) plus one  
-inf values in scores will be replaced by the minimim value (exclucing -inf) minus one  
input arguments:  
- scores: 1D numpy array  
returns  
- array with same length as scores with elements replaced as explained above  
```  
  
  
