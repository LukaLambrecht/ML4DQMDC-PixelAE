# SeminormalFitter  
  
- - -    
## SeminormalFitter(CloudFitter)  
**class for fitting a 'seminormal' distribution to a point cloud**  
this is not strictly speaking a probability distribution,  
only the first quadrant of the result of fitting a normal distribution  
to the data + its mirror image wrt the origin.  
parameters  
- cov: multidim covariance matrix of normal distribution  
- mvn: scipy.stats multivariate\_normal object built from the cov  
  
### \_\_init\_\_(self,points)  
**constructor**  
points is a np array of shape (npoints,ndims)  
note: points can also be an array or list with length 0,  
in that case the object is initialized empty.  
use this followed by the 'load' method to load a previously saved fit!  
  
### pdf(self,points)  
**get pdf at points**  
  
### save(self,path)  
**save the covariance matrix as a .npy file specified by path**  
  
### load(self,path)  
**load a covariance matrix from a .npy file specified by path and build the fit from it**  
  
