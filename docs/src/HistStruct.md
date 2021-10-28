# HistStruct  
  
**HistStruct: consistent treatment of multiple histogram types**  

The HistStruct class is intended to be the main data structure used within this framework.
A HistStruct object basically consists of a mutually consistent collection of numpy arrays, where each numpy array corresponds to one histogram type, with dimensions (number of histograms, number of bins). The HistStruct has functions to easily perform the following common tasks (among others):  
- select a subset of runs and/or lumisections (e.g. using a custom or predefined json file formatted selector),
- prepare the data for machine learning training, with all kinds of preprocessing,
- evaluate classifiers (machine learning types or other).

Up to now the HistStruct is not used in many places, the main reason being that most of the tutorials for example were written (or at leasted started) before this class.  
When only processing a single histogram type, the HistStruct might be a bit of an overkill and one could choose to operate on the dataframe directly.  
However, especially when using multiple histogram types, the HistStruct is very handy to keep everything consistent.  

See the tutorial autoencoder_combine.ipynb for an important example!
- - -
  
  
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
globalscores: 1D numpy array of global score per lumisection (same length as histograms)  
classifiers: dict mapping histogram name to object of type HistogramClassifier  
scores: dict mapping histogram name to 1D numpy array of values associated to the histograms (same length as histograms)  
masks: dict mapping name to 1D numpy array of booleans (same length as histograms) that can be used for masking  
exthistograms: dict of dicts similar to histograms for additional (e.g. artificially generated) histograms  
extscores: dict of dicts similar to scores for additional (e.g. artificially generated) histograms  
extglobalscores: dict of lists similar to scores for additional (e.g. artificially generated) histograms  
```  
### &#10551; \_\_str\_\_( self )  
```text  
get a printable representation of a HistStruct  
```  
### &#10551; save( self, path, save\_classifiers=True )  
```text  
save a HistStruct object to a pkl file  
input arguments:  
- path where to store the file (appendix .zip is automatically appended)  
- save_classifiers: a boolean whether to include the classifiers if present in the HistStruct  
```  
- - -  
  
- - -
## [class] classifiers = dict(self.classifiers)  
```text  
(no valid documentation found)  
```  
- - -  
  
- - -
## [class] classifier.save( os.path.join(cpath,histname) )  
```text  
(no valid documentation found)  
```  
### &#10551; load( self, path, load\_classifiers=True, verbose=False )  
```text  
load a HistStruct object from a pkl file  
input arguments:  
- path to a zip file containing a HistStruct object  
- load_classifiers: a boolean whether to load the classifiers if present  
- verbose: boolean whether to print some information  
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
### &#10551; add\_globalscores( self, globalscores )  
```text  
add an array of global scores (one per lumisection)  
input arguments:  
- globalscores: 1D numpy array of scores (must have same length as lumisection and run numbers)  
```  
### &#10551; add\_extglobalscores( self, extname, globalscores )  
```text  
add an array of global scores (one per lumisection) for a specified extra set of histograms in the HistStruct  
input arguments:  
- extname: name of extra histogram set  
- globalscores: 1D numpy array of scores  
note: this function checks if all histogram types in this set contain the same number of histograms,  
      (and that this number corresponds to the length of globalscores)  
      else adding globalscores is meaningless  
```  
### &#10551; add\_exthistograms( self, extname, histname, histograms, overwrite=False )  
```text  
add a set of extra histograms to a HistStruct  
these histograms are not assumed to correspond to physical run/lumisections numbers (e.g. resampled ones),  
and no consistency checks are done  
input arguments:  
- extname: name of the extra histogram set (you can add multiple, e.g. resampled_good, resampled_bad and/or resampled_training)  
- histname: name of the histogram type  
- histograms: a numpy array of shape (nhistograms,nbins)  
- overwrite: boolean whether to overwrite a set of histograms of the same name if present (default: raise exception)  
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
### &#10551; add\_stat\_mask( self, name, histnames=None, min\_entries\_to\_bins\_ratio=-1, max\_entries\_to\_bins\_ratio=-1 )  
```text  
add a mask corresponding to lumisections where all histograms have statistics within given bounds  
input arguments:  
- histnames: list of histogram names to take into account for making the mask (default: all in the HistStruct)  
- min_entries_to_bins_ratio: number of entries divided by number of bins, lower boundary for statistics (default: no lower boundary)  
- max_entries_to_bins_ratio: same but upper boundary instead of lower boundary (default: no upper boundary)  
```  
### &#10551; add\_highstat\_mask( self, name, histnames=None, entries\_to\_bins\_ratio=100 )  
```text  
shorthand call to add_stat_mask with only lower boundary and no upper boundary for statistics  
input arguments:  
- entries_to_bins_ratio: number of entries divided by number of bins, lower boundary for statistics  
others: see add_stat_mask  
```  
### &#10551; get\_combined\_mask( self, names )  
```text  
get a combined mask given multiple mask names  
mostly for internal use; externally you can use get_histograms( histname, <list of mask names>) directly  
```  
### &#10551; get\_masknames( self )  
```text  
return a simple list of all mask names in the current HistStruct  
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
### &#10551; get\_scores\_array( self, masknames=None )  
```text  
similar to get_scores, but with different return type:  
np array of shape (nhistograms, nhistogramtypes)  
```  
### &#10551; get\_extscores( self, extname, histname=None )  
```text  
get the array of scores for a given histogram type in a given extra set.  
input arguments:  
- extname: name of the extra set (see also add_exthistograms)  
- histname: name of the histogram type for which to retrieve the score.   
  if None, return a dict matching histnames to arrays of scores  
notes:  
- this method takes the scores from the HistStruct.extscores attribute;  
  make sure to have evaluated the classifiers before calling this method,  
  else an exception will be thrown.  
```  
### &#10551; get\_extscores\_array( self, extname )  
```text  
similar to get_extscores, but with different return type:  
np array of shape (nhistograms, nhistogramtypes)  
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
### &#10551; get\_globalscores( self, masknames=None )  
```text  
get the array of global scores, optionally after masking  
input arguments:  
- masknames: list of names of masks (default: no masking, return full array)  
notes:  
- this method takes the scores from the HistStruct.globalscores attribute;  
  make sure to have set this attribute with add_globalscores,  
  else an exception will be thrown.  
```  
### &#10551; get\_globalscore\_ls( self, runnb, lsnb )  
```text  
get the global score for a given run/lumisection number  
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
### &#10551; get\_extglobalscores( self, extname )  
```text  
get the array of global scores for one of the extra histogram sets  
input arguments:  
- extname: name of the extra histogram set  
notes:  
- this method takes the scores from the HistStruct.extglobalscores attribute;  
  make sure to have set this attribute with add_extglobalscores,  
  else an exception will be thrown.  
```  
### &#10551; get\_histograms( self, histname=None, masknames=None )  
```text  
get the array of histograms for a given type, optionally after masking  
input arguments:  
- histname: name of the histogram type to retrieve   
  if None, return a dict matching histnames to arrays of histograms  
- masknames: list of names of masks (default: no masking, return full array)  
```  
### &#10551; get\_exthistograms( self, extname, histname=None )  
```text  
get the array of extra histograms for a given set name and type name  
input arguments:  
- extname: name of the set of extra histograms (see also add_exthistograms)  
- histname: name of the histogram type to retrieve   
  if None, return a dict matching histnames to arrays of histograms  
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
### &#10551; evaluate\_classifier( self, histname, extname=None )  
```text  
evaluate a histogram classifier for a given histogram name in the HistStruct  
input arguments:  
- histname: a valid histogram name present in the HistStruct for which to evaluate the classifier  
- extname: name of a set of extra histograms (see add_exthistograms)  
           if None, will evaluate the classifer for the main set of histograms  
notes:  
- the result is both returned and stored in the 'scores' attribute  
```  
### &#10551; plot\_histograms( self, histnames=None, masknames=None, colorlist=[], labellist=[], transparencylist=[] )  
```text  
plot the histograms in a HistStruct, optionally after msking  
note: so far only for 1D hsitograms.  
      case of 2D histograms requires different plotting method since they cannot be clearly overlaid.  
      if a HistStruct contains both 1D and 2D histograms, the 1D histograms must be selected with the histnames argument.  
input arguments:  
- histnames: list of names of the histogram types to plot (default: all)  
- masknames: list of list of mask names  
  note: each element in masknames represents a set of masks to apply;   
        the histograms passing different sets of masks are plotted in different colors  
- colorlist: list of matplotlib colors, must have same length as masknames  
- labellist: list of labels for the legend, must have same legnth as masknames  
- transparencylist: list of transparency values, must have same length as masknames  
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
### &#10551; plot\_run( self, runnb, masknames=None, recohist=None, recohistlabel='reco', refhists=None, refhistslabel='reference', doprint=False)  
```text  
call plot_ls for all lumisections in a given run  
```  
- - -  
  
