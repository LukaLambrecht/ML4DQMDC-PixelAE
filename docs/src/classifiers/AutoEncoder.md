# AutoEncoder  
  
**Histogram classfier based on the MSE of an autoencoder reconstruction**

The AutoEncoder derives from the generic HistogramClassifier.  
For this specific classifier, the output score of a histogram is the mean-square-error (MSE) between the original histogram and its autoencoder reconstruction.  
In essence, it is just a wrapper for a tensorflow model.  
- - -
  
  
- - -
## [class] AutoEncoder  
comments:  
```text  
histogram classfier based on the MSE of an autoencoder reconstruction  
the AutoEncoder derives from the generic HistogramClassifier.   
for this specific classifier, the output score of a histogram is the mean-square-error (MSE)   
between the original histogram and its autoencoder reconstruction.  
in essence, it is just a wrapper for a tensorflow model.  
```  
### &#10551; \_\_init\_\_  
full signature:  
```text  
def __init__( self, model=None, modelpath=None )  
```  
comments:  
```text  
intializer from a tensorflow model  
input arguments:  
- model: a valid tensorflow model;  
         it does not have to be trained already,  
         the AutoEncoder.train function will take care of this.  
- modelpath: path to a stored tensorflow model,  
             it does not have to be trained already,  
             the AutoEncoder.train function will take care of this.  
note: model and modelpath are alternative options, they should not both be used simultaneously.  
```  
### &#10551; train  
full signature:  
```text  
def train( self, histograms, doplot=True, epochs=10, batch_size=500, shuffle=False, verbose=1, validation_split=0.1, **kwargs )  
```  
comments:  
```text  
train the model on a given set of input histograms  
input arguments:  
- histograms: set of training histograms, a numpy array of shape (nhistograms,nbins)  
- doplot: boolean whether to make a plot of the loss value  
- others: see the keras fit function  
- kwargs: additional arguments passed down to keras fit function  
```  
### &#10551; evaluate  
full signature:  
```text  
def evaluate( self, histograms )  
```  
comments:  
```text  
classification of a collection of histograms based on their autoencoder reconstruction  
```  
### &#10551; reconstruct  
full signature:  
```text  
def reconstruct( self, histograms )  
```  
comments:  
```text  
return the autoencoder reconstruction of a set of histograms  
```  
### &#10551; save  
full signature:  
```text  
def save( self, path )  
```  
comments:  
```text  
save the underlying tensorflow model to a tensorflow SavedModel or H5 format.  
note: depending on the extension specified in path, the SavedModel or H5 format is chosen,  
      see https://www.tensorflow.org/guide/keras/save_and_serialize  
```  
### &#10551; load  
full signature:  
```text  
def load( self, path, **kwargs )  
```  
comments:  
```text  
get an AutoEncoder instance from a saved tensorflow SavedModel or H5 file  
```  
- - -  
  
