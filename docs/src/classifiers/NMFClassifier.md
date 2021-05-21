# NMFClassifier  
  
- - -    
## NMFClassifier(HistogramClassifier)  
**histogram classification based on nonnegative matrix factorization**  
specifically intended for 2D histograms, but should in principle work for 1D as well.  
it is basically a wrapper for a sklearn.decomposition.NMF instance.  
  
### \_\_init\_\_( self, histograms, ncomponents )  
**initializer from a collection of histograms**  
input arguments:  
- histograms: a numpy array of shape (nhists,nbins) or (nhists,nybins,nxbins) that will be used to fit a NMF model  
- ncomponents: number of NMF components (aka clusters aka basis vectors) to use in the decomposition  
TODO: add keyword arguments to pass down to sklearn.decomposition.NMF  
  
### evaluate( self, histograms, nmax )  
**classify the given histograms based on the MSE with respect to their reconstructed version**  
input arguments:  
- histograms: numpy array of shape (nhists,nbins) or (nhists,nybins,nxbins)  
- nmax: number of largest elements to keep in mean square error calculation  
  
### getcomponents( self )  
**return the NMF components (aka cluster centers aka basis vectors)**  
output:  
a numpy array of shape (ncomponents,nbins) or (ncomponents,nybins,nxbins)  
  
### reconstruct( self, histograms )  
**return the NMF reconstruction for a given set of histograms**  
input arguments:  
- histograms: numpy array of shape (nhists,nbins) or (nhists,nybins,nxbins)  
  
