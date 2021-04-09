# LogNormalFitter  
  
- - -    
## LogNormalFitter(CloudFitter)  
**class for fitting a log-normal distribution to a point cloud**  
parameters:  
- mean: multidim mean of underlying normal  
- cov: multidim covariance matrix of underlying normal  
- mvn: scipy.stats multivariate\_normal object built from the mean and cov  
  
### \_\_init\_\_(self,points)  
**constructor**  
points is a np array of shape (npoints,ndims)  
  
### pdf(self,points)  
**get pdf at points**  
  
