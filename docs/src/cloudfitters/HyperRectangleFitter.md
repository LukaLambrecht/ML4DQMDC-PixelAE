# HyperRectangleFitter  
  
**Simple fitter making a hard cut in each dimension**


- - -
  
  
### calculate\_cut\_values  
full signature:  
```text  
def calculate_cut_values( values, quantile, side='both' )  
```  
comments:  
```text  
calculate the appropriate cut values to discard a given quantile of values  
input arguments:  
- values: a 1D numpy array  
- quantile: quantile of values to discard, a float between 0 and 1  
  (or between 0 and 0.5 for side='both')  
- side: either 'both', 'down' or 'up'  
  for 'up', the cut will discard the quantile highest values,  
  for 'down', cut will discard the quantile lowest values,  
  for 'both', the cut(s) will discard the quantile values both at the high and low end.  
returns:  
- a tuple of shape (lower cut, upper cut), with None entries if not applicable  
```  
  
  
- - -
## [class] HyperRectangleFitter  
comments:  
```text  
Simple fitter making a hard cut in each dimension  
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
def fit(self, points, quantiles=0, side='both', verbose=False)  
```  
comments:  
```text  
fit to a set of points  
input arguments:  
- points: a np array of shape (npoints,ndims)  
- quantiles: quantiles of values to discard.  
  can either be a float between 0 and 1 (applied in all dimensions),  
  or a list of such floats with same length as number of dimensions in points.  
  (note: for side='both', quantiles above 0.5 will discard everything)  
- side: either 'both', 'down' or 'up'  
  for 'up', the cut will discard the quantile highest values,  
  for 'down', cut will discard the quantile lowest values,  
  for 'both', the cut(s) will discard the quantile values both at the high and low end.  
```  
### &#10551; apply\_cuts  
full signature:  
```text  
def apply_cuts(self, point)  
```  
comments:  
```text  
apply the cuts to a point and return whether it passes them  
input arguments:  
- point: a 1D numpy array of shape (ndims,)  
returns:  
- boolean  
```  
### &#10551; pdf  
full signature:  
```text  
def pdf(self, points)  
```  
comments:  
```text  
get pdf at points  
note that the pdf is either 0 (does not pass cuts) or 1 (passes cuts)  
```  
- - -  
  
