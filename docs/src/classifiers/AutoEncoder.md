# AutoEncoder  
  
- - -    
## AutoEncoder(HistogramClassifier)  
**histogram classfier based on the MSE of an autoencoder reconstruction**  
the AutoEncoder derives from the generic HistogramClassifier.  
for this specific classifier, the output score of a histogram is the mean-square-error (MSE)  
between the original histogram and its autoencoder reconstruction.  
in essence, it is just a wrapper for a tensorflow model.  
  
### \_\_init\_\_( self, model=None )  
**intializer from a tensorflow model**  
the model is assumed to be fully trained on a suitable training set and ready for use  
TODO: perhaps the functionality for initializing and training the model can be absorbed in the AutoEncoder class,  
but this is not yet supported currently  
  
### evaluate( self, histograms )  
**classification of a collection of histograms based on their autoencoder reconstruction**  
  
