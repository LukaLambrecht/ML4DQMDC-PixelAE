# SeminormalFitter  
  
**Class for fitting a 'seminormal' distribution to a point cloud**

This is not strictly speaking a probability distribution, only the first quadrant of the result of fitting a normal distribution to the data + its mirror image wrt the origin.  
- - -
  
  
- - -
## [class] SeminormalFitter  
comments:  
```text  
class for fitting a 'seminormal' distribution to a point cloud  
this is not strictly speaking a probability distribution,  
only the first quadrant of the result of fitting a normal distribution  
to the data + its mirror image wrt the origin.  
parameters  
- cov: multidim covariance matrix of normal distribution  
- mvn: scipy.stats multivariate_normal object built from the cov  
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
def fit(self,points)  
```  
comments:  
```text  
make the fit  
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
input arguments:  
- points: a np array of shape (npoints,ndims)  
```  
### &#10551; save  
full signature:  
```text  
def save(self,path)  
```  
comments:  
```text  
save the covariance matrix as a .npy file specified by path  
```  
### &#10551; load  
full signature:  
```text  
def load(self,path)  
```  
comments:  
```text  
load a covariance matrix from a .npy file specified by path and build the fit from it  
```  
- - -  
  
