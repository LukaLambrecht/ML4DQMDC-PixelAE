# GaussianKdeFitter  
  
**Class for fitting a gaussian kernel density to a point cloud**

Basically a wrapper for scipy.stats.gaussian_kde.  
A gaussian kernel density can be thought of as a sum of little (potentially multidimensional) gaussians, each one centered at one of the points in the cloud. Hence, the resulting distribution is a sort of smoothed version of the discrete point cloud.
- - -
  
  
- - -
## [class] GaussianKdeFitter  
comments:  
```text  
class for fitting a gaussian kernel density to a point cloud  
basically a wrapper for scipy.stats.gaussian_kde.  
parameters  
- kernel: scipy.stats.gaussian_kde object  
- cov: covariance matrix   
(use np.cov for now, maybe later replace by internal kernel.covariance)  
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
def fit(self, points, bw_method='scott', bw_scott_factor=None)  
```  
comments:  
```text  
fit to a set of points  
input arguments:  
- points: a np array of shape (npoints,ndims)  
- bw_method: method to calculate the bandwidth of the gaussians,  
  see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html  
- bw_scott_factor: additional multiplication factor applied to bandwidth in case it is set to 'scott'  
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
  
