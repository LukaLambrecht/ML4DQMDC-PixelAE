# LogNormalFitter  
  
**Class for fitting a log-normal distribution to a point cloud**

A log-normal distribution is constructed by fitting a normal distribution to the logarithm of the point coordinates.
- - -
  
  
- - -
## [class] LogNormalFitter  
comments:  
```text  
class for fitting a log-normal distribution to a point cloud  
parameters:  
- mean: multidim mean of underlying normal  
- cov: multidim covariance matrix of underlying normal  
- mvn: scipy.stats multivariate_normal object built from the mean and cov  
```  
### &#10551; \_\_init\_\_  
full signature:  
```text  
def __init__(self)  
```  
comments:  
```text  
empty constructor  
```  
### &#10551; fit  
full signature:  
```text  
def fit(self, points)  
```  
comments:  
```text  
fit to a set of points  
input arguments:  
- points: a np array of shape (npoints,ndims)  
```  
### &#10551; pdf  
full signature:  
```text  
def pdf(self,points)  
```  
comments:  
```text  
get pdf at points  
```  
- - -  
  
