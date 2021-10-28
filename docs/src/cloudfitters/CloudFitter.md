# CloudFitter  
  
**Abstract base class for all point cloud fitting algorithms** 

Note that all concrete point cloud fitters must inherit from CloudFitter!  

How to make a concrete CloudFitter class:
- define a class that inherits from CloudFitter
- make sure all functions with @abstractmethod are implemented in your class
- it is recommended to start each overriding function with a call to super(), but this is not strictly necessary

See also the existing examples!
- - -
  
  
- - -
## [class] CloudFitter(ABC)  
```text  
abstract base class for all point cloud fitting algorithms  
note that all concrete point cloud fitters must inherit from CloudFitter!  
how to make a concrete CloudFitter class:  
- define a class that inherits from CloudFitter  
- make sure all functions with @abstractmethod are implemented in your class  
- it is recommended to start each overriding function with a call to super(), but this is not strictly necessary  
see also the existing examples!  
```  
### &#10551; \_\_init\_\_( self, points )  
```text  
default intializer  
this is an @abstractmethod and must be overridden in any concrete deriving class!  
input arguments:  
- points: 2D numpy array of shape (npoints,ndims)  
```  
### &#10551; pdf( self, points )  
```text  
evaluate the pdf (probability density function) at given points  
this is an @abstractmethod and must be overridden in any concrete deriving class!  
input arguments:  
- points: a 2D numpy array of shape (npoints,ndims)  
output: a 1D array of shape (npoints)  
```  
- - -  
  
