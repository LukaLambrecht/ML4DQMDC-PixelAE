# HistogramClassifier  
  
**Abstract base class for histogram classifying objects**  

Note that all concrete histogram classifiers must inherit from HistogramClassifier!
A HistogramClassifier can be any object that classifies a histogram; in more detail:
- the input is a collection of histograms (of the same type), represented by a numpy array of shape (nhists,nbins) for 1D histograms or (nhists,nybins,nxbins) for 2D histograms.
- the output is an array of numbers of shape (nhists).
- the processing between input and output can in principle be anything, but usually some sort of discriminating power is assumed.

How to make a concrete HistogramClassifier class:
- define a class that inherits from HistogramClassifier
- make sure all functions with @abstractmethod are implemented in your class
- it is recommended to start each overriding function with a call to super(), but this is not strictly necessary

See also the existing examples!
- - -
  
  
- - -
## [class] HistogramClassifier(ABC)  
```text  
abstract base class for histogram classifying objects  
note that all concrete histogram classifiers must inherit from HistogramClassifier!  
a HistogramClassifier can be any object that classifies a histogram; in more detail:  
- the input is a collection of histograms (of the same type), represented by a numpy array of shape (nhists,nbins) for 1D histograms or (nhists,nybins,nxbins) for 2D histograms.  
- the output is an array of numbers of shape (nhists).  
- the processing between input and output can in principle be anything, but usually some sort of discriminating power is assumed.  
how to make a concrete HistogramClassifier class:  
- define a class that inherits from HistogramClassifier  
- make sure all functions with @abstractmethod are implemented in your class  
- it is recommended to start each overriding function with a call to super(), but this is not strictly necessary  
see also the existing examples!  
```  
### &#10551; \_\_init\_\_( self )  
```text  
empty intializer  
this is an @abstractmethod and must be overridden in any concrete deriving class!  
```  
### &#10551; train( self, histograms )  
```text  
train the classifier on a set of input histograms  
this is an @abstractmethod and must be overridden in any concrete deriving class!  
input arguments:  
- histograms: numpy array of shape (nhists,nbins) or (nhists,nybins,nxbins).  
output: expected to be none.  
```  
### &#10551; evaluate( self, histograms )  
```text  
main function used to evaluate a set of histograms  
this is an @abstractmethod and must be overridden in any concrete deriving class!  
input arguments:  
- histograms: numpy array of shape (nhists,nbins) or (nhists,nybins,nxbins).  
output: expected to be a 1D numpy array of shape (nhists), one number per histogram.  
```  
### &#10551; save( self, path )  
```text  
save a classifier to disk  
specific implementation in concrete classes, here only path creation  
```  
### &#10551; load( self, path )  
```text  
load a classifier object from disk  
specific implementation in concrete classes, here only path checking  
```  
- - -  
  
