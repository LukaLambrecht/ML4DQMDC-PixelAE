# GaussianKdeFitter  
  
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
### &#10551; \_\_init\_\_(self,points,bw\_method='scott')  
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
  
