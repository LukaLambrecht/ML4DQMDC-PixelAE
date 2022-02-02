# PCAGaussianFitter  
  
**Class for fitting a multidimensional gaussian distribution to a PCA-reduced point cloud**

Instead of fitting the full (high-dimensional) point cloud,
a PCA-based dimensionality reduction is first applied on it.
This has the advantage that the fit can be visualised correctly (in case of 2 reduced dimensions),
instead of only projections of it.
The potential disadvantage is that the PCA reduction might distort the relative separations.
- - -
  
  
- - -
## [class] PCAGaussianFitter  
comments:  
```text  
class for fitting a gaussian distribution to a PCA-reduced point cloud  
parameters  
- pca: sklearn.decomposition.pca object  
- mean: multidim mean of normal distribution  
- cov: multidim covariance matrix of normal distribution  
- mvn: scipy.stats multivariate_normal object built from mean and cov  
```  
### &#10551; \_\_init\_\_  
full signature:  
```text  
def __init__(self, points, npcadims=2)  
```  
comments:  
```text  
constructor  
input arguments:  
- points: a np array of shape (npoints,ndims)  
- npcadims: number of PCA compoments to keep  
```  
### &#10551; pdf  
full signature:  
```text  
def pdf(self, points)  
```  
comments:  
```text  
get pdf at points  
note: points can be both of shape (npoints,ndims) or of shape (npoints,npcadims);  
      in the latter case it is assumed that the points are already PCA-transformed,  
      and only the gaussian kernel density is applied on them.  
```  
### &#10551; transform  
full signature:  
```text  
def transform(self, points)  
```  
comments:  
```text  
perform PCA transformation  
```  
- - -  
  
