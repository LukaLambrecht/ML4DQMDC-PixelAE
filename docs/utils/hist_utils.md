# hist utils  
  
- - -    
## rebinhists(hists, factor)  
**perform rebinning on a set of histograms**  
hists is a numpy array of shape (nhistograms,nbins)  
factor is the rebinning factor, which must be a divisor of nbins.  
  
- - -    
## normalizehists(hists)  
**perform normalization (i.e. sum of bin contents equals one for each histogram)**  
  
- - -    
## averagehists(hists, nout)  
**partition hists (of shape (nhistograms,nbins)) into nout parts and take the average histogram of each part**  
  
- - -    
## moment(bins, counts, order)  
**get n-th central moment of a histogram**  
- bins is a 1D or 2D np array holding the bin centers  
(shape (nbins) or (nhistograms,nbins))  
- array is a 2D np array containing the bin counts  
(shape (nhistograms,nbins))  
- order is the order of the moment to calculate  
(0 = maximum, 1 = mean value)  
  
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
## preparedatafromnpy(dataname, rebinningfactor=1, donormalize=True, doplot=False)  
**read a .npy file and output the histograms**  
  
- - -    
## preparedatafromdf(df, returnrunls=False, rebinningfactor=1, donormalize=False, doplot=False)  
prepare the data contained in a dataframe in the form of a numpy array  
args:  
- returnrunls: boolean whether to return a tuple of (histograms, run numbers, lumisection numbers).  
(default: return only histograms)  
- rebinningfactor: an integer number to downsample the histograms in the dataframe  
- donormalize: if True, data are normalized  
- doplot: if True, some example plots are made showing the histograms  
  
- - -    
## preparedatafromcsv(dataname, returnrunls=False, rebinningfactor=1, donormalize=True, doplot=False)  
**prepare the data contained in a dataframe csv file in the form of a numpy array**  
args:  
- returnrunls: wether to return only a histogram array or 1D arrays of run and lumisection numbers as well  
- rebinningfactor: an integer number to downsample the histograms in the dataframe  
- donormalize: if True, data are normalized  
- doplot: if True, some example plots are made showing the histograms  
  
