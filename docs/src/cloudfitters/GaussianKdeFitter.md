# GaussianKdeFitter  
  
- - -    
## GaussianKdeFitter(CloudFitter)  
**class for fitting a gaussian kernel density to a point cloud**  
basically a wrapper for scipy.stats.gaussian\_kde.  
parameters  
- kernel: scipy.stats.gaussian\_kde object  
- cov: covariance matrix  
(use np.cov for now, maybe later replace by internal kernel.covariance)  
  
### \_\_init\_\_(self,points,bw\_method='scott')  
**constructor**  
input arguments:  
- points: a np array of shape (npoints,ndims)  
- bw\_method: method to calculate the bandwidth of the gaussians,  
see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian\_kde.html  
  
### pdf(self,points)  
**get pdf at points**  
  
