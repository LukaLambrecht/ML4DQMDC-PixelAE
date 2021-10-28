# ExponentialFitter  
  
**Class for fitting an exponential distribution to a point cloud**

An exponential distribution in N dimensions is fully determined by an N-dimensional vector, representing the N-dimensional decay parameter (or lambda parameter) of the distribution. 
- - -
  
  
- - -
## [class] ExponentialFitter(CloudFitter)  
```text  
class for fitting an exponential distribution to a point cloud  
parameters  
- l: multidimensional lambda parameter of exponential  
```  
### &#10551; \_\_init\_\_(self, points)  
```text  
constructor  
input arguments:  
- points: a np array of shape (npoints,ndims)  
```  
### &#10551; pdf(self, points)  
```text  
get pdf at points  
```  
- - -  
  
