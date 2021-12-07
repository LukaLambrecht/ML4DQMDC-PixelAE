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
def __init__(self, points)  
```  
comments:  
```text  
constructor  
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
- - -  
  
