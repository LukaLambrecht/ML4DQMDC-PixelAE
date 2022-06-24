# PCAGaussianKdeFitter  
  
**Class for fitting a gaussian kernel density to a PCA-reduced point cloud**

Extension of GaussianKdeFitter: instead of fitting the full point cloud,
a PCA-based dimensionality reduction is first applied on it.
This has the advantage that the fit can be visualised correctly (in case of 2 reduced dimensions),
instead of only projections of it.
The potential disadvantage is that the PCA reduction might distort the relative separations.
- - -
  
  
- - -
## [class] PCAGaussianKdeFitter  
comments:  
```text  
class for fitting a gaussian kernel density to a PCA-reduced point cloud  
basically a wrapper for sklean.decomposition.PCA + scipy.stats.gaussian_kde.  
parameters  
- pca: sklearn.decomposition.pca object  
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
def fit(self, points, npcadims=2, bw_method='scott', bw_scott_factor=None)  
```  
comments:  
```text  
fit to a set of points  
input arguments:  
- points: a np array of shape (npoints,ndims)  
- npcadims: number of PCA compoments to keep  
- bw_method: method to calculate the bandwidth of the gaussians,  
  see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html  
- bw_scott_factor: additional multiplication factor applied to bandwidth in case it is set to 'scott'  
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
  
