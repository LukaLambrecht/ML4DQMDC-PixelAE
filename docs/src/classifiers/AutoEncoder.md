# AutoEncoder  
  
**Histogram classfier based on the MSE of an autoencoder reconstruction**

The AutoEncoder derives from the generic HistogramClassifier.  
For this specific classifier, the output score of a histogram is the mean-square-error (MSE) between the original histogram and its autoencoder reconstruction.  
In essence, it is just a wrapper for a tensorflow model.  
- - -
  
  
- - -
## [class] AutoEncoder(HistogramClassifier)  
```text  
histogram classfier based on the MSE of an autoencoder reconstruction  
the AutoEncoder derives from the generic HistogramClassifier.   
for this specific classifier, the output score of a histogram is the mean-square-error (MSE)   
between the original histogram and its autoencoder reconstruction.  
in essence, it is just a wrapper for a tensorflow model.  
```  
### &#10551; \_\_init\_\_( self, model=None )  
```text  
intializer from a tensorflow model  
the model is assumed to be a valid tensorflow model;  
it can be already trained before wrapping it in an AutoEncoder object,  
but if this is not the case, the AutoEncoder.train function can be called afterwards.  
```  
### &#10551; train( self, histograms, doplot=True, **kwargs )  
```text  
train the model on a given set of input histograms  
```  
### &#10551; evaluate( self, histograms )  
```text  
classification of a collection of histograms based on their autoencoder reconstruction  
```  
### &#10551; reconstruct( self, histograms )  
```text  
return the autoencoder reconstruction of a set of histograms  
```  
### &#10551; save( self, path )  
```text  
save the underlying tensorflow model to a tensorflow SavedModel or H5 format.  
note: depending on the extension specified in path, the SavedModel or H5 format is chosen,  
      see https://www.tensorflow.org/guide/keras/save_and_serialize  
```  
### &#10551; load( self, path, **kwargs )  
```text  
get an AutoEncoder instance from a saved tensorflow SavedModel or H5 file  
```  
- - -  
  
- - -
## [class] classifier = AutoEncoder( model=model )  
```text  
(no valid documentation found)  
```  
- - -  
  
