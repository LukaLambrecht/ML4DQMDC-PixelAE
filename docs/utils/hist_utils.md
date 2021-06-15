# hist utils  
  
# cropping of hisograms
- - -
  
  
### crophists(hists, slices)  
```text  
perform cropping on a set of histograms  
input arguments:  
- hists: a numpy array of shape (nhistograms,nbins) for 1D or (nhistograms,nybins,nxbins) for 2D  
- slices is a list of slice objects (builtin python type) of length 1 (for 1D) or 2 (for 2D)  
  note: a slice can be created using the builtin python syntax 'slice(start,stop,step)',   
        and the syntax 'list[slice]' is equivalent to 'list[start:stop:step]'.  
        use 'None' to ignore one of the arguments for slice creation (equivalent to ':' in direct slicing)  
example usage:  
- see tutorials/plot_histograms_2d.ipynb  
returns:  
- a numpy array containing the same histograms as input but cropped according to the slices argument  
```  
  
  
### rebinhists(hists, factor)  
```text  
perform rebinning on a set of histograms  
input arguments:  
- hists: a numpy array of shape (nhistograms,nbins) for 1D or (nhistograms,nybins,nxbins) for 2D  
- factor: the rebinning factor (for 1D), or a tuple of (y axis rebinning factor, x axis rebinning factor) (for 2D)   
  note: the rebinning applied here is simple summing of bin contents,  
        and the rebinning factors must be divisors of the respective number of bins!  
example usage:  
- see tutorials/plot_histograms_2d.ipynb  
returns:  
- a numpy array containing the same histograms as input but rebinned according to the factor argument  
```  
  
  
### normalizehists(hists)  
```text  
perform normalization on a set of histograms  
note:   
- for 1D histograms, the sum of bin contents is set equal one for each histogram  
- for 2D histograms, the bin contents are scaled so the maximum is 1 for each histogram  
- maybe later make more flexible by adding normalization stragy as argument...  
input arguments:  
- hists: a numpy array of shape (nhistograms,nbins) for 1D or (nhistograms,nybins,nxbins) for 2D  
returns:  
- a numpy array containing the same histograms as input but normalized  
```  
  
  
### averagehists(hists, nout)  
```text  
partition a set of histograms into equal parts and take the average histogram of each part  
input arguments:  
- hists: a numpy array of shape (nhistograms,nbins) for 1D or (nhistograms,nybins,nxbins) for 2D  
- nout: number of partitions / output histograms  
  note: nout=1 corresponds to simply taking the average of all histograms in hists.  
returns:  
- a numpy array of shape (nout,<input number of bins>)  
```  
  
  
### running\_average\_hists(hists, window, weights=None)  
```text  
replace each histogram in a collection of histograms by its running average  
input arguments:  
- hists: a numpy array of shape (nhistograms,nbins) for 1D or (nhistograms,nybins,nxbins) for 2D  
- nwindow: number of histograms to consider for the averaging  
  if window is an integer, it is the number of previous histograms in hists used for averaging  
  (so window=0 would correspond to no averaging)  
  if window is a tuple, it corresponds to (nprevious,nnext), and the nprevious previous and nnext next histograms in hists are used for averaging  
  (so window=(0,0) would correspond to no averaging)  
- weights: a list or numpy array containing the relative weights of the histograms in the averaging procedure.  
  note: the weights can be any number, but they will be normalized to have unit sum.  
  note: weights must have length nwindow+1 or nprevious+1+nnext.  
  note: the default behaviour is a uniform array with values 1./(window+1) (or 1./(nprevious+1+nnext))  
returns:  
- a numpy array with same shape as input but where each histogram is replaced by its running average  
notes:  
- at the edges, the weights are cropped to match the input array and renormalized  
- this function will crash when the length of the set of histograms is smaller than the total window length,  
  maybe extend later (although this is not normally needed)  
```  
  
  
### moment(bins, counts, order)  
```text  
get n-th central moment of a histogram  
input arguments:  
- bins: a 1D or 2D np array holding the bin centers  
  (shape (nbins) or (nhistograms,nbins))  
- counts: a 2D np array containing the bin counts  
  (shape (nhistograms,nbins))  
- order: the order of the moment to calculate  
  (0 = maximum value, 1 = mean value)  
returns:  
- an array of shape (nhistograms) holding the requested moment per histogram  
notes:   
- for now only 1D histograms are supported!  
```  
  
  
### histmean(bins, counts)  
```text  
special case of moment calculation (with order=1)  
```  
  
  
### histrms(bins, counts)  
```text  
special case of moment calculation  
```  
  
  
### histmoments(bins, counts, orders)  
```text  
apply moment calculation for a list of orders  
input arguments:  
- see function moment(bins, counts, order),  
  the only difference being that orders is a list instead of a single number  
returns:  
- a numpy array of shape (nhistograms,nmoments)  
```  
  
  
### preparedatafromnpy(dataname, cropslices=None, rebinningfactor=None, donormalize=True, doplot=False)  
```text  
read a .npy file and output the histograms  
input arguments:   
- see e.g. preparedatafromdf  
notes:   
- not yet tested for 2D histograms, but is expected to work...  
```  
  
  
### preparedatafromdf(df, returnrunls=False, cropslices=None, rebinningfactor=None, donormalize=False, doplot=False)  
```text  
prepare the data contained in a dataframe in the form of a numpy array  
input arguments:  
- returnrunls: boolean whether to return a tuple of (histograms, run numbers, lumisection numbers).  
  (default: return only histograms)  
- cropslices: list of slices by which to crop the historams (default: no cropping)  
- rebinningfactor: an integer (or tuple of integers for 2D histograms) to downsample/rebin the histograms (default: no rebinning)  
- donormalize: boolean whether to normalize the data  
- doplot: if True, some example plots are made showing the histograms  
```  
  
  
### preparedatafromcsv(dataname, returnrunls=False, cropslices=None, rebinningfactor=None, donormalize=True, doplot=False)  
```text  
prepare the data contained in a dataframe csv file in the form of a numpy array  
input arguments:  
- returnrunls: boolean whether to return a tuple of (histograms, run numbers, lumisection numbers).  
  (default: return only histograms)  
- cropslices: list of slices by which to crop the historams (default: no cropping)  
- rebinningfactor: an integer (or tuple of integers for 2D histograms) to downsample/rebin the histograms (default: no rebinning)  
- donormalize: boolean whether to normalize the data  
- doplot: if True, some example plots are made showing the histograms  
```  
  
  
