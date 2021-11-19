# PCAClassifier  
  
**Histogram classification based on principal component analysis**
 
It is basically a wrapper for a sklearn.decomposition.PCA instance.
- - -
  
  
- - -
## [class] PCAClassifier  
comments:  
```text  
histogram classification based on principal component analysis  
it is basically a wrapper for a sklearn.decomposition.PCA instance.  
```  
### &#10551; \_\_init\_\_  
full signature:  
```text  
def __init__( self, ncomponents=None, svd_solver='auto', loss_type='mse', nmax=10 )  
```  
comments:  
```text  
initializer  
input arguments:  
- ncomponents: number of PCA components (aka clusters aka basis vectors) to use in the decomposition  
- svd_solver: solver method to extract the PCA components  
  note: both ncomponents and svd_solver are arguments passed down to sklearn.decomposition.PCA,  
        see https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html  
- loss_type: choose from 'mse' (mean-squared-error) or 'chi2' (chi squared error)  
- nmax: number of largest elements to keep in error calculation  
TODO: add keyword arguments to pass down to sklearn.decomposition.PCA  
```  
### &#10551; train  
full signature:  
```text  
def train( self, histograms )  
```  
comments:  
```text  
train the PCA model on a given set of input histograms  
input arguments:  
- histograms: a numpy array of shape (nhists,nbins) or (nhists,nybins,nxbins) that will be used to fit a PCA model  
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
return the PCA components (aka cluster centers aka basis vectors)  
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
return the PCA reconstruction for a given set of histograms  
input arguments:  
- histograms: numpy array of shape (nhists,nbins) or (nhists,nybins,nxbins)  
```  
- - -  
  
