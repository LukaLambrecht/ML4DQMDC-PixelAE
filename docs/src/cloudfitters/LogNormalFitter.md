# LogNormalFitter  
  
**Class for fitting a log-normal distribution to a point cloud**

A log-normal distribution is constructed by fitting a normal distribution to the logarithm of the point coordinates.
- - -
  
  
- - -
## [class] LogNormalFitter(CloudFitter)  
```text  
class for fitting a log-normal distribution to a point cloud  
parameters:  
- mean: multidim mean of underlying normal  
- cov: multidim covariance matrix of underlying normal  
- mvn: scipy.stats multivariate_normal object built from the mean and cov  
```  
### &#10551; \_\_init\_\_(self,points)  
```text  
constructor  
input arguments:  
- points: a np array of shape (npoints,ndims)  
```  
### &#10551; pdf(self,points)  
```text  
get pdf at points  
```  
- - -  
  
