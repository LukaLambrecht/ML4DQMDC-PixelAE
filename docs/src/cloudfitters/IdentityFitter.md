# IdentityFitter  
  
**Class for using classifier scores directly as global scores** 
- - -
  
  
- - -
## [class] IdentityFitter  
comments:  
```text  
class for propagating classifier output scores (e.g. MSE) to global lumisection score  
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
def fit(self, points)  
```  
comments:  
```text  
fit to a set of points  
input arguments:  
- points: a numpy array of shape (npoints,ndims)   
          note that ndims is supposed to be 1,   
          else this type of classifier is not well defined.  
```  
### &#10551; pdf  
full signature:  
```text  
def pdf(self, points)  
```  
comments:  
```text  
get pdf at points  
```  
### &#10551; save  
full signature:  
```text  
def save(self, path)  
```  
comments:  
```text  
save this fitter (dummy for now since nothing to be saved)  
```  
### &#10551; load  
full signature:  
```text  
def load(self, path)  
```  
comments:  
```text  
load this fitter (dummy for now since nothing to be loaded)  
```  
- - -  
  
