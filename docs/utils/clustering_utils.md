# clustering utils  
  
**A collection of functions used for performing clustering tasks**  

This collection of tools is a little deprecated at this moment but kept for reference; it contains functionality for pre-filtering the histograms in the training set based on their moments (e.g. mean, rms).  
Note that the functions here have not been used in a long time and might need some maintenance before they work properly again.
- - -
  
  
### vecdist  
full signature:  
```text  
def vecdist(moments, index)  
```  
comments:  
```text  
calculate the vectorial distance between a set of moments  
input arguments:  
- moments: 2D numpy array of shape (ninstances,nmoments)  
- index: index for which instance to calculate the distance relative to the other instances  
returns:  
- a distance measure for the given index w.r.t. the other instances in 'moments'  
notes:  
- for this distance measure, the points are considered as vectors and the point at index is the origin.  
  with respect to this origin, the average vector before index and the average vector after index are calculated.  
  the distance is then defined as the norm of the difference of these vectors,   
  normalized by the norms of the individual vectors.  
```  
  
  
### costhetadist  
full signature:  
```text  
def costhetadist(moments, index)  
```  
comments:  
```text  
calculate the costheta distance between a set of moments  
input arguments:  
- moments: 2D numpy array of shape (ninstances,nmoments)  
- index: index for which instance to calculate the distance relative to the other instances  
returns:  
- a distance measure for the given index w.r.t. the other instances in 'moments'  
notes:  
- this distance measure takes the cosine of the angle between the point at index  
  and the one at index-1 (interpreted as vectors from the origin).  
```  
  
  
### avgnndist  
full signature:  
```text  
def avgnndist(moments, index, nn)  
```  
comments:  
```text  
calculate average euclidean distance to neighbouring points  
input arguments:  
- moments: 2D numpy array of shape (ninstances,nmoments)  
- index: index for which instance to calculate the distance relative to the other instances  
- nn: (half-) window size  
returns:  
- a distance measure for the given index w.r.t. the other instances in 'moments'  
notes:  
- for this distance measure, the average euclidean distance is calculated between the point at 'index'  
  and the points at index-nn and index+nn (e.g. the nn previous and next lumisections).  
```  
  
  
### getavgnndist  
full signature:  
```text  
def getavgnndist(hists, nmoments, xmin, xmax, nbins, nneighbours)  
```  
comments:  
```text  
apply avgnndist to a set of histograms  
```  
  
  
### filteranomalous  
full signature:  
```text  
def filteranomalous(df, nmoments=3, rmouterflow=True, rmlargest=0., doplot=True)  
```  
comments:  
```text  
do a pre-filtering, removing the histograms with anomalous moments  
```  
  
  
