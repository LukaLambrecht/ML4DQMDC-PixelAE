# HistStruct  
  
- - -
## [class] HistStruct(object)  
```text  
main data structure used within this framework  
a HistStruct object basically consists of a mutually consistent collection of numpy arrays,  
where each numpy array corresponds to one histogram type, with dimensions (number of histograms, number of bins).  
the HistStruct has functions to easily perform the following common tasks (among others):  
- select a subset of runs and/or lumisections (e.g. using a json file formatted selector),  
- prepare the data for machine learning training  
- evaluate classifiers (machine learning types or other)  
```  
### &#10551; \_\_init\_\_( self )  
```text  
empty initializer, setting all containers to empty defaults  
a HistStruct object has the following properties:  
histnames: list of histogram names  
histograms: dict mapping histogram name to 2D numpy array of histograms (shape (nhists,nbins))  
nentries: dict mapping histogram name to 1D numpy array of number of entries per histogram (same length as histograms)  
runnbs: 1D numpy array of run numbers (same length as histograms)  
lsnbs: 1D numpy array of lumisection numbers (same length as histograms)  
classifiers: dict mapping histogram name to object of type HistogramClassifier  
scores: dict mapping histogram name to 1D numpy array of values associated to the histograms (same length as histograms)  
masks: dict mapping name to 1D numpy array of booleans (same length as histograms) that can be used for masking  
exthistograms: dict similar to histograms for additional (e.g. artificially generated) histograms  
```  
### &#10551; save( self, path )  
```text  
save a HistStruct object to a pkl file  
input arguments:  
- path where to store the file (appendix .pkl is automatically appended)  
```  
### &#10551; load( self, path )  
```text  
load a HistStruct object from a pkl file  
input arguments:  
- path to a pkl file containing a HistStruct object  
```  
### &#10551; add\_dataframe( self, df, cropslices=None, donormalize=True, rebinningfactor=None )  
```text  
add a dataframe to a HistStruct  
input arguments:  
- df: a pandas dataframe as read from the input csv files  
- cropslices: list of slices (one per dimension) by which to crop the histograms  
- donormalize: boolean whether to normalize the histograms  
- rebinningfactor: factor by which to group bins together  
for more details on cropslices, donormalize and rebinningfactor, see hist_utils.py / preparedatafromdf!  
notes:  
- the new dataframe can contain one or multiple histogram types  
- the new dataframe must contain the same run and lumisection numbers (for each histogram type in it)  
  as already present in the HistStruct, except if it is the first one to be added  
- alternative to adding the dataframe with the options cropslices, donormalize and rebinningfactor  
  (that will be passed down to preparedatafromdf), one can also call preparedatafromdf manually and add it  
  with add_histograms, allowing for more control over complicated preprocessing.  
```  
### &#10551; add\_histograms( self, histname, histograms, runnbs, lsnbs, nentries=None )  
```text  
add a set of histograms to a HistStruct  
input arguments:  
- histname: name of the histogram type to be added  
- histograms: a numpy array of shape (nhistograms,nbins), assumed to be of a single type  
- runnbs: a 1D list or array of length nhistograms containing the run number per histogram  
- lsnbs: a 1D list or array of length nhistograms containing the lumisection number per histogram  
- nentries: a 1D list or array of length nhistograms containing the number of entries per histogram  
  notes:  
  - must be provided explicitly since histograms might be normalized,   
    in which case the number of entries cannot be determined from the sum of bin contents.  
  - used for (de-)selecting histograms with sufficient statistics;   
    if you don't need that type of selection, nentries can be left at default.  
  - default is None, meaning all entries will be set to zero.  
notes:  
- no preprocessing is performed, this is assumed to have been done manually (if needed) before adding the histograms  
- runnbs and lsnbs must correspond to what is already in the current HistStruct, except if this is the first set of histogram to be added  
- see also add_dataframe for an alternative way of adding histograms  
```  
### &#10551; add\_mask( self, name, mask )  
```text  
add a mask to a HistStruct  
input arguments:  
- name: a name for the mask  
- mask: a 1D np array of booleans  with same length as number of lumisections in HistStruct  
```  
### &#10551; remove\_mask( self, name )  
```text  
inverse operation of add_mask  
```  
### &#10551; add\_json\_mask( self, name, jsondict )  
```text  
add a mask corresponding to a json dict  
input arguments:  
- name: a name for the mask  
- jsondict: a dictionary in typical json format (see the golden json file for inspiration)  
all lumisections present in the jsondict will be masked True, the others False.  
```  
### &#10551; add\_goldenjson\_mask( self, name )  
```text  
add a mask corresponding to the golden json file  
input arguments:  
- name: a name for the mask  
```  
### &#10551; add\_dcsonjson\_mask( self, name )  
```text  
add a mask corresponding to the DCS-bit on json file  
input arguments:  
- name: a name for the mask  
```  
### &#10551; add\_hightstat\_mask( self, name, histnames=None, entries\_to\_bins\_ratio=100 )  
```text  
add a mask corresponding to lumisections where all histograms have sufficient statistics  
input arguments:  
- histnames: list of histogram names to take into account for making the mask (default: all in the HistStruct)  
- entries_to_bins_ratio: criterion to determine if a histogram has sufficient statistics, number of entries divided by number of bins  
```  
### &#10551; get\_combined\_mask( self, names )  
```text  
get a combined mask given multiple mask names  
mostly for internal use; externally you can use get_histograms( histname, <list of mask names>) directly  
```  
### &#10551; get\_runnbs( self, masknames=None )  
```text  
get the array of run numbers, optionally after masking  
input arguments:  
- masknames: list of names of masks (default: no masking, return full array)  
```  
### &#10551; get\_lsnbs( self, masknames=None )  
```text  
get the array of lumisection numbers, optionally after masking  
input arguments:  
- masknames: list of names of masks (default: no masking, return full array)  
```  
### &#10551; get\_index( self, runnb, lsnb )  
```text  
get the index in the current HistStruct of a given run and lumisection number  
input arguments:  
- runnb and lsnb: run and lumisection number respectively  
```  
### &#10551; get\_scores( self, histname=None, masknames=None )  
```text  
get the array of scores for a given histogram type, optionally after masking  
input arguments:  
- histname: name of the histogram type for which to retrieve the score.   
  if None, return a dict matching histnames to arrays of scores  
- masknames: list of names of masks (default: no masking, return full array)  
notes:  
- this method takes the scores from the HistStruct.scores attribute;  
  make sure to have evaluated the classifiers before calling this method,  
  else an exception will be thrown.  
```  
### &#10551; get\_scores\_ls( self, runnb, lsnb, histnames=None, suppresswarnings=False )  
```text  
get the scores for a given run/lumisection number and for given histogram names  
input arguments:  
- runnb: run number  
- lsnb: lumisection number  
- histnames: names of the histogram types for which to retrieve the score.   
returns:  
- a dict matching each name in histnames to a score (or None if no valid score)  
notes:  
- this method takes the scores from the HistStruct.scores attribute;  
  make sure to have evaluated the classifiers before calling this method,  
  else the returned scores will be None.  
```  
### &#10551; get\_histograms( self, histname=None, masknames=None )  
```text  
get the array of histograms for a given type, optionally after masking  
input arguments:  
- histname: name of the histogram type to retrieve   
  if None, return a dict matching histnames to arrays of histograms  
- masknames: list of names of masks (default: no masking, return full array)  
```  
### &#10551; add\_classifier( self, histname, classifier, evaluate=False )  
```text  
add a histogram classifier for a given histogram name to the HistStruct  
input arguments:  
- histname: a valid histogram name present in the HistStruct to which this classifier applies  
- classifier: an object of type HistogramClassifier (i.e. of any class that derives from it)  
- evaluate: a bool whether to evaluate the classifier (and store the result in the 'scores' attribute)  
  if set to True, the result is both returned and stored in the 'scores' attribute.  
```  
### &#10551; evaluate\_classifier( self, histname )  
```text  
evaluate a histogram classifier for a given histogram name in the HistStruct  
input arguments:  
- histname: a valid histogram name present in the HistStruct for which to evaluate the classifier  
notes:  
- the result is both returned and stored in the 'scores' attribute  
```  
### &#10551; plot\_ls( self, runnb, lsnb, histnames=None, recohist=None, recohistlabel='reco', refhists=None, refhistslabel='reference')  
```text  
plot the histograms in a HistStruct for a given run/ls number versus their references and/or their reconstruction  
note: so far only for 1D histograms.  
      case of 2D histograms requires different plotting method since they cannot be clearly overlaid.  
      if a HistStruct contains both 1D and 2D histograms, the 1D histograms must be selected with the histnames argument.  
input arguments:  
- runnb: run number  
- lsnb: lumisection number  
- histnames: names of histogram types to plot (default: all)  
- recohist: dict matching histogram names to reconstructed histograms  
  notes: - 'reconstructed histograms' refers to e.g. autoencoder or NMF reconstructions;  
           some models (e.g. simply looking at histogram moments) might not have this kind of reconstruction  
         - in principle one histogram per key is expected, but still the the shape must be 2D (i.e. (1,nbins))  
         - in case recohist is set to 'auto', the reconstruction is calculated on the fly for the input histograms  
- recohistlabel: legend entry for the reco histograms  
- refhists: dict matching histogram names to reference histograms  
  notes: - multiple histograms (i.e. a 2D array) per key are expected;  
           in case there is only one reference histogram, it must be reshaped into (1,nbins)  
- refhistslabel: legend entry for the reference histograms  
```  
### &#10551; plot\_run( self, runnb, recohist=None, recohistlabel='reco', refhists=None, refhistslabel='reference', doprint=False)  
```text  
call plot_ls for all lumisections in a given run  
```  
- - -  
  
