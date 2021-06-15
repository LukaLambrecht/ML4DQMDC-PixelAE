# LogNormalFitter  
  
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
points is a np array of shape (npoints,ndims)  
```  
### &#10551; pdf(self,points)  
```text  
get pdf at points  
```  
- - -  
  
