# LandauClassifier  
  
### landaufun  
full signature:  
```text  
def landaufun(x, landaumax, landauwidth, norm)  
```  
comments:  
```text  
see https://en.wikipedia.org/wiki/Landau_distribution  
```  
  
  
### langaufun  
full signature:  
```text  
def langaufun(x, landaumax, landauwidth, norm, gausswidth)  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
- - -
## [class] LandauClassifier  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; \_\_init\_\_  
full signature:  
```text  
def __init__( self, dogauss=False )  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; train  
full signature:  
```text  
def train( self )  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; fit  
full signature:  
```text  
def fit( self, histogram )  
```  
comments:  
```text  
find initial guess for the parameters  
```  
### &#10551; evaluate  
full signature:  
```text  
def evaluate( self, histograms )  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; reconstruct  
full signature:  
```text  
def reconstruct( self, histograms )  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; save  
full signature:  
```text  
def save( self, path )  
```  
comments:  
```text  
save the classifier  
```  
### &#10551; load  
full signature:  
```text  
def load( self, path, **kwargs )  
```  
comments:  
```text  
get a LandauClassifier instance from a pkl file  
```  
- - -  
  
