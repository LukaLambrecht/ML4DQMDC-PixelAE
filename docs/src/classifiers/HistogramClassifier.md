# HistogramClassifier  
  
- - -    
## HistogramClassifier(ABC)  
**abstract base class for histogram classifying objects**  
note that all concrete histogram classifiers must inherit from HistogramClassifier!  
a HistogramClassifier can be any object that classifies a histogram; in more detail:  
- the input is a collection of histograms (of the same type), represented by a numpy array of shape (nhists,nbins).  
- the output is an array of numbers of shape (nhists).  
- the processing between input and output can in principle be anything, but usually some sort of discriminating power is assumed.  
how to make a concrete HistogramClassifier class:  
- define a class that inherits from HistogramClassifier  
- make sure all functions with @abstractmethod are implemented in your class  
- it is recommended to start each overriding function with a call to super(), but this is not strictly necessary  
see also the existing examples!  
  
### \_\_init\_\_( self )  
**empty intializer**  
  
### evaluate( self, histograms )  
**main function used to process a set of histograms**  
input arguments:  
- histograms: 2D numpy array of shape (nhists,nbins).  
output: 1D numpy array of shape (nhists), one number per histogram.  
  
