# ExponentialFitter  
  
**Class for fitting an exponential distribution to a point cloud**

An exponential distribution in N dimensions is fully determined by an N-dimensional vector, representing the N-dimensional decay parameter (or lambda parameter) of the distribution. 
- - -
  
  
- - -
## [class] ExponentialFitter  
comments:  
```text  
class for fitting an exponential distribution to a point cloud  
parameters  
- l: multidimensional lambda parameter of exponential  
```  
### &#10551; \_\_init\_\_  
full signature:  
```text  
def __init__(self, points)  
```  
comments:  
```text  
constructor  
input arguments:  
- points: a np array of shape (npoints,ndims)  
```  
### &#10551; pdf  
full signature:  
```text  
def pdf(self, points)  
```  
comments:  
```text  
get pdf at points  
```  
- - -  
  
