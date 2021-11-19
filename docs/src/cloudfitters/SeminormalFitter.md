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
def __init__(self,points)  
```  
comments:  
```text  
constructor  
input arguments:  
- points: a np array of shape (npoints,ndims)  
  note: points can also be an array or list with length 0,  
        in that case the object is initialized empty.  
        use this followed by the 'load' method to load a previously saved fit!  
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
  
