# hist utils  
  
**A collection of useful basic functions for processing histograms.**  

Functionality includes:
- rebinning, cropping and normalization
- moment calculation
- averaging over neighbouring histograms
- smoothing over neighbouring bins
- higher-level functions preparing data for ML training, 
  starting from a dataframe or input csv file.
- - -
  
  
### crophists  
full signature:  
```text  
def crophists(hists, slices=None)  
```  
comments:  
```text  
perform cropping on a set of histograms  
input arguments:  
- hists: a numpy array of shape (nhistograms,nbins) for 1D or (nhistograms,nybins,nxbins) for 2D  
- slices: a slice object (builtin python type) or a list of two slices (for 2D)  
  notes:   
    - a slice can be created using the builtin python syntax 'slice(start,stop,step)',   
      and the syntax 'list[slice]' is equivalent to 'list[start:stop:step]'.  
      use 'None' to ignore one of the arguments for slice creation (equivalent to ':' in direct slicing)  
    - for 1D histograms, slices can be either a slice object or a list of length 1 containing a single slice.  
example usage:  
- see tutorials/plot_histograms_2d.ipynb  
returns:  
- a numpy array containing the same histograms as input but cropped according to the slices argument  
```  
  
  
### get\_cropslices\_from\_str  
full signature:  
```text  
def get_cropslices_from_str(slicestr)  
```  
comments:  
```text  
get a collection of slices from a string (e.g. argument in gui)  
note: the resulting slices are typically passed to crophists (see above)  
input arguments:  
- slicestr: string representation of slices  
            e.g. '0:6:2' for slice(0,6,2)  
            e.g. '0:6:2,1:5:2' for [slice(0,6,2),slice(1,5,2)]  
```  
  
  
### rebinhists  
full signature:  
```text  
def rebinhists(hists, factor=None)  
```  
comments:  
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
  
  
### get\_rebinningfactor\_from\_str  
full signature:  
```text  
def get_rebinningfactor_from_str(factstr)  
```  
comments:  
```text  
get a valid rebinning factor (int or tuple) from a string (e.g. argument in gui)  
note: the resulting factor is typically passed to rebinhists (see above)  
input arguments:  
- factstr: string representation of rebinning factor  
            e.g. '4' for 4 (for 1D histograms)  
            e.g. '4,4' for (4,4) (for 2D histograms)  
```  
  
  
### normalizehists  
full signature:  
```text  
def normalizehists(hists)  
```  
comments:  
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
  
  
### averagehists  
full signature:  
```text  
def averagehists(hists, nout=None)  
```  
comments:  
```text  
partition a set of histograms into equal parts and take the average histogram of each part  
input arguments:  
- hists: a numpy array of shape (nhistograms,nbins) for 1D or (nhistograms,nybins,nxbins) for 2D  
- nout: number of partitions, i.e. number of output histograms  
  note: nout=1 corresponds to simply taking the average of all histograms in hists.  
  note: if nout is negative or if nout is larger than number of input histograms, the original set of histograms is returned.  
returns:  
- a numpy array of shape (nout,nbins)  
```  
  
  
### running\_average\_hists  
full signature:  
```text  
def running_average_hists(hists, window=None, weights=None)  
```  
comments:  
```text  
replace each histogram in a collection of histograms by its running average  
input arguments:  
- hists: a numpy array of shape (nhistograms,nbins) for 1D or (nhistograms,nybins,nxbins) for 2D  
- window: number of histograms to consider for the averaging  
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
- this function will throw an error when the length of the set of histograms is smaller than the total window length,  
  maybe extend later (although this is not normally needed)  
```  
  
  
### select\_random  
full signature:  
```text  
def select_random(hists, nselect=10)  
```  
comments:  
```text  
select nselect random examples from a set of histograms  
input arguments:  
- hists: a numpy array of shape (nhistograms, nbins) for 1D  
         or (nhistograms, nybins, nxbins) for 2D.  
- nselect: number of random instances to draw  
```  
  
  
### smoothhists  
full signature:  
```text  
def smoothhists(hists, halfwindow=None, weights=None)  
```  
comments:  
```text  
perform histogram smoothing by averaging over neighbouring bins  
input arguments:  
- hists: a numpy array of shape (nhistograms, nbins) for 1D  
         or (nhistograms, nybins, nxbins) for 2D.  
- halfwindow: number of bins to consider for the averaging;  
              for 1D histograms, must be an int, corresponding to the number of bins  
              before and after the current bin to average over;  
              for 2D histograms, must be a tuple of (halfwindow_y, halfwindow_x).  
- weights: numpy array containing the relative weights of the bins for the averaging;  
           for 1D histograms, must have length 2*halfwindow+1;  
           for 2D histograms, must have shape (2*halfwindow_y+1, 2*halfwindow_x+1).  
           note: the weights can be any number, but they will be normalized to have unit sum.  
           note: the default behaviour is a uniform array  
returns:  
- a numpy array with same shape as input but where each histogram is replaced   
  by its smoothed version  
```  
  
  
### get\_smoothinghalfwindow\_from\_str  
full signature:  
```text  
def get_smoothinghalfwindow_from_str(windowstr)  
```  
comments:  
```text  
get a valid smoothing half window (int or tuple) from a string (e.g. argument in gui)  
note: the resulting factor is typically passed to smoothhists (see above)  
input arguments:  
- windowstr: string representation of smoothing window  
              e.g. '4' for 4 (for 1D histograms)  
              e.g. '4,4' for (4,4) (for 2D histograms)  
```  
  
  
### moment  
full signature:  
```text  
def moment(bins, counts, order)  
```  
comments:  
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
  
  
### histmean  
full signature:  
```text  
def histmean(bins, counts)  
```  
comments:  
```text  
special case of moment calculation (with order=1)  
```  
  
  
### histrms  
full signature:  
```text  
def histrms(bins, counts)  
```  
comments:  
```text  
special case of moment calculation  
```  
  
  
### histmoments  
full signature:  
```text  
def histmoments(bins, counts, orders)  
```  
comments:  
```text  
apply moment calculation for a list of orders  
input arguments:  
- see function moment(bins, counts, order),  
  the only difference being that orders is a list instead of a single number  
returns:  
- a numpy array of shape (nhistograms,nmoments)  
```  
  
  
### preparedatafromnpy  
full signature:  
```text  
def preparedatafromnpy(dataname, cropslices=None, rebinningfactor=None,  smoothinghalfwindow=None, smoothingweights=None, averagewindow=None, averageweights=None, donormalize=True, doplot=False)  
```  
comments:  
```text  
read a .npy file and output the histograms  
input arguments:   
- see e.g. preparedatafromdf  
notes:   
- not yet tested for 2D histograms, but is expected to work...  
```  
  
  
### preparedatafromdf  
full signature:  
```text  
def preparedatafromdf(df, returnrunls=False, cropslices=None, rebinningfactor=None,  smoothinghalfwindow=None, smoothingweights=None, averagewindow=None, averageweights=None, donormalize=False, doplot=False)  
```  
comments:  
```text  
prepare the data contained in a dataframe in the form of a numpy array  
input arguments:  
- returnrunls: boolean whether to return a tuple of   
  (histograms, run numbers, lumisection numbers).  
  (default: return only histograms)  
- cropslices: list of slices (one per dimension) by which to crop the historams   
  (default: no cropping)  
- rebinningfactor: an integer (or tuple of integers for 2D histograms)   
  to downsample/rebin the histograms (default: no rebinning)  
- smoothinghalfwindow: int or tuple (for 1D/2D histograms) used for smoothing the histograms  
- smoothingweights: 1D or 2D array (for 1D/2D histograms) with weights for smoothing  
- donormalize: boolean whether to normalize the data  
- doplot: if True, some example plots are made showing the histograms  
```  
  
  
### preparedatafromcsv  
full signature:  
```text  
def preparedatafromcsv(dataname, returnrunls=False, cropslices=None, rebinningfactor=None,  smoothinghalfwindow=None, smoothingweights=None, averagewindow=None, averageweights=None, donormalize=True, doplot=False)  
```  
comments:  
```text  
prepare the data contained in a dataframe csv file in the form of a numpy array  
input arguments:  
- returnrunls: boolean whether to return a tuple of (histograms, run numbers, lumisection numbers).  
  (default: return only histograms)  
- cropslices: list of slices (one per dimension) by which to crop the historams   
  (default: no cropping)  
- rebinningfactor: an integer (or tuple of integers for 2D histograms)   
  to downsample/rebin the histograms (default: no rebinning)  
- smoothinghalfwindow: int or tuple (for 1D/2D histograms) used for smoothing the histograms  
- smoothingweights: 1D or 2D array (for 1D/2D histograms) with weights for smoothing  
- donormalize: boolean whether to normalize the data  
- doplot: if True, some example plots are made showing the histograms  
```  
  
  
