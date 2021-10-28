# GaussianKdeFitter  
  
**Class for fitting a gaussian kernel density to a point cloud**

Basically a wrapper for scipy.stats.gaussian_kde.  
A gaussian kernel density can be thought of as a sum of little (potentially multidimensional) gaussians, each one centered at one of the points in the cloud. Hence, the resulting distribution is a sort of smoothed version of the discrete point cloud.
- - -
  
  
- - -
## [class] GaussianKdeFitter(CloudFitter)  
```text  
class for fitting a gaussian kernel density to a point cloud  
basically a wrapper for scipy.stats.gaussian_kde.  
parameters  
- kernel: scipy.stats.gaussian_kde object  
- cov: covariance matrix   
(use np.cov for now, maybe later replace by internal kernel.covariance)  
```  
### &#10551; \_\_init\_\_(self, points, bw\_method='scott')  
```text  
constructor  
input arguments:  
- points: a np array of shape (npoints,ndims)  
- bw_method: method to calculate the bandwidth of the gaussians,  
  see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html  
```  
### &#10551; pdf(self,points)  
```text  
get pdf at points  
```  
- - -  
  
