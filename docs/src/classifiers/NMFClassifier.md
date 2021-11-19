# NMFClassifier  
  
**Histogram classification based on nonnegative matrix factorization**

Specifically intended for 2D histograms, but should in principle work for 1D as well.  
It is basically a wrapper for a sklearn.decomposition.NMF instance.
- - -
  
  
- - -
## [class] NMFClassifier  
comments:  
```text  
histogram classification based on nonnegative matrix factorization  
specifically intended for 2D histograms, but should in principle work for 1D as well.  
it is basically a wrapper for a sklearn.decomposition.NMF instance.  
```  
### &#10551; \_\_init\_\_  
full signature:  
```text  
def __init__( self, ncomponents=5, loss_type='mse', nmax=10 )  
```  
comments:  
```text  
initializer  
input arguments:  
- ncomponents: number of NMF components (aka clusters aka basis vectors) to use in the decomposition  
- loss_type: choose from 'mse' (mean-squared-error) or 'chi2' (chi squared error)  
- nmax: number of largest elements to keep in error calculation  
TODO: add keyword arguments to pass down to sklearn.decomposition.NMF  
```  
### &#10551; train  
full signature:  
```text  
def train( self, histograms )  
```  
comments:  
```text  
train the NMF model on a given set of input histograms  
input arguments:  
- histograms: a numpy array of shape (nhists,nbins) or (nhists,nybins,nxbins) that will be used to fit a NMF model  
```  
### &#10551; set\_nmax  
full signature:  
```text  
def set_nmax( self, nmax )  
```  
comments:  
```text  
set number of largest elements to keep in mean square error calculation  
useful to quickly re-evaluate the model with different nmax without retraining  
input arguments:  
- nmax: number of largest elements to keep in mean square error calculation  
```  
### &#10551; set\_loss\_type  
full signature:  
```text  
def set_loss_type( self, loss_type )  
```  
comments:  
```text  
set loss type  
useful to quickly re-evaluate the model with different loss without retraining  
input arguments:  
- loss_type: choose from 'mse' (mean-squared-error) or 'chi2' (chi squared error)  
```  
### &#10551; evaluate  
full signature:  
```text  
def evaluate( self, histograms )  
```  
comments:  
```text  
classify the given histograms based on the MSE with respect to their reconstructed version  
input arguments:  
- histograms: numpy array of shape (nhists,nbins) or (nhists,nybins,nxbins)  
```  
### &#10551; get\_components  
full signature:  
```text  
def get_components( self )  
```  
comments:  
```text  
return the NMF components (aka cluster centers aka basis vectors)  
output:  
a numpy array of shape (ncomponents,nbins) or (ncomponents,nybins,nxbins)  
```  
### &#10551; reconstruct  
full signature:  
```text  
def reconstruct( self, histograms )  
```  
comments:  
```text  
return the NMF reconstruction for a given set of histograms  
input arguments:  
- histograms: numpy array of shape (nhists,nbins) or (nhists,nybins,nxbins)  
```  
- - -  
  
