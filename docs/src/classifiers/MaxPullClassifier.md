# MaxPullClassifier  
  
**Histogram classification based on maximum pull between test histogram and reference histogram.**

Specifically intended for 2D histograms, but should in principle work for 1D as well.  
Ssee static function 'pull' for definition of bin-per-bin pull and other notes.
- - -
  
  
### pull  
full signature:  
```text  
def pull( testhist, refhist )  
```  
comments:  
```text  
calculate bin-per-bin pull between two histograms  
bin-per-bin pull is defined here preliminarily as (testhist(bin)-refhist(bin))/sqrt(refhist(bin))  
notes:   
- bins in the denominator where refhist is < 1 are set to one! This is for histograms with absolute counts, and they should not be normalized!  
- instead another normalization is applied: the test histogram is multiplied by sum(refhist)/sum(testhist) before computing the pulls  
input arguments:  
- testhist, refhist: numpy arrays of the same shape  
output:  
numpy array of same shape as testhist and refhist  
```  
  
  
### maxabspull  
full signature:  
```text  
def maxabspull( testhist, refhist, n=1 )  
```  
comments:  
```text  
calculate maximum of bin-per-bin pulls (in absolute value) between two histograms  
see definition of bin-per-bin pull in function pull (above)  
input arguments:  
- testhist, refhist: numpy arrays of the same shape  
- n: nubmer of largest pull values to average over (default: 1, just take single maximum)  
output:  
a float  
```  
  
  
- - -
## [class] MaxPullClassifier  
comments:  
```text  
histogram classification based on maximum pull between test histogram and reference histogram.  
specifically intended for 2D histograms, but should in principle work for 1D as well.  
see static function pull (above) for definition of bin-per-bin pull and other notes.  
```  
### &#10551; \_\_init\_\_  
full signature:  
```text  
def __init__( self, nmaxpulls=1 )  
```  
comments:  
```text  
initializer  
input arguments:  
- nmaxpulls: number of largest pull values to average over   
  (default: 1, just take single maximum)  
```  
### &#10551; set\_nmaxpulls  
full signature:  
```text  
def set_nmaxpulls( self, nmaxpulls )  
```  
comments:  
```text  
set the nmaxpulls parameter (see also initializer)  
```  
### &#10551; train  
full signature:  
```text  
def train( self, refhist )  
```  
comments:  
```text  
'train' the classifier, i.e. set the reference histogram.  
input arguments:  
- refhist: a numpy array of shape (1,nbins) or (1,nybins,nxbins)  
```  
### &#10551; evaluate  
full signature:  
```text  
def evaluate( self, histograms )  
```  
comments:  
```text  
classify the histograms based on their max bin-per-bin pull (in absolute value) with respect to a reference histogram  
```  
### &#10551; getpull  
full signature:  
```text  
def getpull( self, histogram )  
```  
comments:  
```text  
get the pull histogram for a given test histogram  
input arguments:  
histogram: a single histogram, i.e. numpy array of shape (nbins) for 1D or (nybins,nxbins) for 2D.  
output:  
numpy array of same shape as histogram containing bin-per-bin pull w.r.t. reference histogram  
```  
- - -  
  
