# hist utils  
  
- - -    
## crophists(hists, slices)  
**perform cropping on a sit of histograms**  
input arguments:  
- hists is a numpy array of shape (nhistograms,nbins) for 1D or (nhistograms,nybins,nxbins) for 2D  
- slices is a list of slice objects (builtin python type) of length 1 (for 1D) or 2 (for 2D)  
  
- - -    
## rebinhists(hists, factor)  
**perform rebinning on a set of histograms**  
input arguments:  
- hists is a numpy array of shape (nhistograms,nbins) for 1D or (nhistograms,nybins,nxbins) for 2D  
- factor is the rebinning factor, or a tuple (y axis rebinning factor, x axis rebinning factor),  
which must be a divisors of the respective number of bins.  
  
- - -    
## normalizehists(hists)  
**perform normalization**  
for 1D histograms, the sum of bin contents is set equal one for each histogram  
for 2D histograms, the bin contents are scaled so the maximum is 1 for each histogram  
(maybe later make more flexible by adding normalization stragy as argument)  
  
- - -    
## averagehists(hists, nout)  
**partition hists (of shape (nhistograms,nbins) or (nhistograms,nybins,nxbins)) into nout parts and take the average histogram of each part**  
  
- - -    
## moment(bins, counts, order)  
**get n-th central moment of a histogram**  
- bins is a 1D or 2D np array holding the bin centers  
(shape (nbins) or (nhistograms,nbins))  
- array is a 2D np array containing the bin counts  
(shape (nhistograms,nbins))  
- order is the order of the moment to calculate  
(0 = maximum, 1 = mean value)  
note: for now only 1D histograms are supported!  
  
- - -    
## histmean(bins, counts)  
**special case of moment calculation (with order=1)**  
  
- - -    
## histrms(bins, counts)  
**special case of moment calculation**  
  
- - -    
## histmoments(bins, counts, orders)  
**apply moment calculation for a list of orders**  
the return type is a numpy array of shape (nhistograms,nmoments)  
  
- - -    
## preparedatafromnpy(dataname, cropslices=None, rebinningfactor=None, donormalize=True, doplot=False)  
**read a .npy file and output the histograms**  
args: see e.g. preparedatafromdf  
note: not yet tested for 2D histograms, but is expected to work...  
  
- - -    
## preparedatafromdf(df, returnrunls=False, cropslices=None, rebinningfactor=None, donormalize=False, doplot=False)  
prepare the data contained in a dataframe in the form of a numpy array  
args:  
- returnrunls: boolean whether to return a tuple of (histograms, run numbers, lumisection numbers).  
(default: return only histograms)  
- cropslices: list of slices by which to crop the historams (default: no cropping)  
- rebinningfactor: an integer (or tuple of integers for 2D histograms) to downsample/rebin the histograms (default: no rebinning)  
- donormalize: boolean whether to normalize the data  
- doplot: if True, some example plots are made showing the histograms  
  
- - -    
## preparedatafromcsv(dataname, returnrunls=False, cropslices=None, rebinningfactor=None, donormalize=True, doplot=False)  
**prepare the data contained in a dataframe csv file in the form of a numpy array**  
args:  
- returnrunls: boolean whether to return a tuple of (histograms, run numbers, lumisection numbers).  
(default: return only histograms)  
- cropslices: list of slices by which to crop the historams (default: no cropping)  
- rebinningfactor: an integer (or tuple of integers for 2D histograms) to downsample/rebin the histograms (default: no rebinning)  
- donormalize: boolean whether to normalize the data  
- doplot: if True, some example plots are made showing the histograms  
  
