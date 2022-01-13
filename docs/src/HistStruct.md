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
## [class] HistStruct  
comments:  
```text  
main data structure used within this framework  
a HistStruct object basically consists of a mutually consistent collection of numpy arrays,  
where each numpy array corresponds to one histogram type, with dimensions (number of histograms, number of bins).  
the HistStruct has functions to easily perform the following common tasks (among others):  
- select a subset of runs and/or lumisections (e.g. using a json file formatted selector),  
- prepare the data for machine learning training  
- evaluate classifiers (machine learning types or other)  
```  
### &#10551; \_\_init\_\_  
full signature:  
```text  
def __init__( self )  
```  
comments:  
```text  
empty initializer, setting all containers to empty defaults  
a HistStruct object has the following properties:  
histnames: list of histogram names  
histograms: dict mapping histogram name to 2D numpy array of histograms (shape (nhists,nbins))  
nentries: dict mapping histogram name to 1D numpy array of number of entries per histogram (same length as histograms)  
histranges: dict mapping histogram name to tuple with (xmin, xmax)  
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
### &#10551; \_\_str\_\_  
full signature:  
```text  
def __str__( self )  
```  
comments:  
```text  
get a printable representation of a HistStruct  
```  
### &#10551; save  
full signature:  
```text  
def save( self, path, save_classifiers=True )  
```  
comments:  
```text  
save a HistStruct object to a pkl file  
input arguments:  
- path where to store the file (appendix .zip is automatically appended)  
- save_classifiers: a boolean whether to include the classifiers if present in the HistStruct  
```  
- - -  
  
- - -
## [class] classifiers = dict  
comments:  
```text  
(no valid documentation found)  
```  
- - -  
  
- - -
## [class] classifier.save  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; load  
full signature:  
```text  
def load( self, path, load_classifiers=True, verbose=False )  
```  
comments:  
```text  
load a HistStruct object from a pkl file  
input arguments:  
- path to a zip file containing a HistStruct object  
- load_classifiers: a boolean whether to load the classifiers if present  
- verbose: boolean whether to print some information  
```  
### &#10551; add\_dataframe  
full signature:  
```text  
def add_dataframe( self, df, cropslices=None, rebinningfactor=None,  smoothinghalfwindow=None, smoothingweights=None, donormalize=True )  
```  
comments:  
```text  
add a dataframe to a HistStruct  
input arguments:  
- df: a pandas dataframe as read from the input csv files  
- cropslices: list of slices (one per dimension) by which to crop the histograms  
- rebinningfactor: factor by which to group bins together  
- smoothinghalfwindow: half window (int for 1D, tuple for 2D) for doing smoothing of histograms  
- smoothingweights: weight array (1D for 1D, 2D for 2D) for smoothing of histograms  
- donormalize: boolean whether to normalize the histograms  
for more details on cropslices, rebinningfactor, smoothingwindow, smoothingweights  
and donormalize: see hist_utils.py!  
notes:  
- the new dataframe can contain one or multiple histogram types  
- the new dataframe must contain the same run and lumisection numbers (for each histogram type in it)  
  as already present in the HistStruct, except if it is the first one to be added  
- alternative to adding the dataframe with the options cropslices, donormalize and rebinningfactor  
  (that will be passed down to preparedatafromdf), one can also call preparedatafromdf manually and add it  
  with add_histograms, allowing for more control over complicated preprocessing.  
```  
### &#10551; add\_histograms  
full signature:  
```text  
def add_histograms( self, histname, histograms, runnbs, lsnbs, nentries=None )  
```  
comments:  
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
### &#10551; preprocess  
full signature:  
```text  
def preprocess( self, masknames=None, cropslices=None, rebinningfactor=None, smoothinghalfwindow=None, smoothingweights=None, donormalize=False )  
```  
comments:  
```text  
do preprocessing  
input arguments:  
- masknames: names of masks to select histograms to which to apply the preprocessing  
              (histograms not passing the masks are simply copied)  
the other input arguments are equivalent to those given in add_dataframe,  
but this function allows to do preprocessing after the dataframes have already been loaded  
note: does not work on extended histograms sets!  
      one needs to apply preprocessing before generating extra histograms.  
```  
### &#10551; add\_globalscores  
full signature:  
```text  
def add_globalscores( self, globalscores )  
```  
comments:  
```text  
add an array of global scores (one per lumisection)  
input arguments:  
- globalscores: 1D numpy array of scores (must have same length as lumisection and run numbers)  
```  
### &#10551; add\_extglobalscores  
full signature:  
```text  
def add_extglobalscores( self, extname, globalscores )  
```  
comments:  
```text  
add an array of global scores (one per lumisection) for a specified extra set of histograms in the HistStruct  
input arguments:  
- extname: name of extra histogram set  
- globalscores: 1D numpy array of scores  
note: this function checks if all histogram types in this set contain the same number of histograms,  
      (and that this number corresponds to the length of globalscores)  
      else adding globalscores is meaningless  
```  
### &#10551; get\_globalscores\_jsonformat  
full signature:  
```text  
def get_globalscores_jsonformat( self, working_point=None )  
```  
comments:  
```text  
make a json format listing all lumisections in this histstruct  
the output list has entries for global score, pass/fail given working point, and masks  
input arguments:  
- working_point: if present, an entry will be made for each lumisection whether it passes this working point  
```  
### &#10551; add\_exthistograms  
full signature:  
```text  
def add_exthistograms( self, extname, histname, histograms, overwrite=False )  
```  
comments:  
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
### &#10551; add\_mask  
full signature:  
```text  
def add_mask( self, name, mask )  
```  
comments:  
```text  
add a mask to a HistStruct  
input arguments:  
- name: a name for the mask  
- mask: a 1D np array of booleans  with same length as number of lumisections in HistStruct  
```  
### &#10551; remove\_mask  
full signature:  
```text  
def remove_mask( self, name )  
```  
comments:  
```text  
inverse operation of add_mask  
```  
### &#10551; add\_json\_mask  
full signature:  
```text  
def add_json_mask( self, name, jsondict )  
```  
comments:  
```text  
add a mask corresponding to a json dict  
input arguments:  
- name: a name for the mask  
- jsondict: a dictionary in typical json format (see the golden json file for inspiration)  
all lumisections present in the jsondict will be masked True, the others False.  
```  
### &#10551; add\_goldenjson\_mask  
full signature:  
```text  
def add_goldenjson_mask( self, name )  
```  
comments:  
```text  
add a mask corresponding to the golden json file  
input arguments:  
- name: a name for the mask  
```  
### &#10551; add\_dcsonjson\_mask  
full signature:  
```text  
def add_dcsonjson_mask( self, name )  
```  
comments:  
```text  
add a mask corresponding to the DCS-bit on json file  
input arguments:  
- name: a name for the mask  
```  
### &#10551; add\_stat\_mask  
full signature:  
```text  
def add_stat_mask( self, name, histnames=None, min_entries_to_bins_ratio=-1, max_entries_to_bins_ratio=-1 )  
```  
comments:  
```text  
add a mask corresponding to lumisections where all histograms have statistics within given bounds  
input arguments:  
- histnames: list of histogram names to take into account for making the mask (default: all in the HistStruct)  
- min_entries_to_bins_ratio: number of entries divided by number of bins, lower boundary for statistics (default: no lower boundary)  
- max_entries_to_bins_ratio: same but upper boundary instead of lower boundary (default: no upper boundary)  
```  
### &#10551; add\_highstat\_mask  
full signature:  
```text  
def add_highstat_mask( self, name, histnames=None, entries_to_bins_ratio=100 )  
```  
comments:  
```text  
shorthand call to add_stat_mask with only lower boundary and no upper boundary for statistics  
input arguments:  
- entries_to_bins_ratio: number of entries divided by number of bins, lower boundary for statistics  
others: see add_stat_mask  
```  
### &#10551; get\_combined\_mask  
full signature:  
```text  
def get_combined_mask( self, names )  
```  
comments:  
```text  
get a combined mask given multiple mask names  
mostly for internal use; externally you can use get_histograms( histname, <list of mask names>) directly  
```  
### &#10551; get\_masknames  
full signature:  
```text  
def get_masknames( self )  
```  
comments:  
```text  
return a simple list of all mask names in the current HistStruct  
```  
### &#10551; get\_runnbs  
full signature:  
```text  
def get_runnbs( self, masknames=None )  
```  
comments:  
```text  
get the array of run numbers, optionally after masking  
input arguments:  
- masknames: list of names of masks (default: no masking, return full array)  
```  
### &#10551; get\_runnbs\_unique  
full signature:  
```text  
def get_runnbs_unique( self )  
```  
comments:  
```text  
get a list of unique run numbers  
```  
### &#10551; get\_lsnbs  
full signature:  
```text  
def get_lsnbs( self, masknames=None )  
```  
comments:  
```text  
get the array of lumisection numbers, optionally after masking  
input arguments:  
- masknames: list of names of masks (default: no masking, return full array)  
```  
### &#10551; get\_index  
full signature:  
```text  
def get_index( self, runnb, lsnb )  
```  
comments:  
```text  
get the index in the current HistStruct of a given run and lumisection number  
input arguments:  
- runnb and lsnb: run and lumisection number respectively  
```  
### &#10551; get\_scores  
full signature:  
```text  
def get_scores( self, histname=None, masknames=None )  
```  
comments:  
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
### &#10551; get\_scores\_array  
full signature:  
```text  
def get_scores_array( self, masknames=None )  
```  
comments:  
```text  
similar to get_scores, but with different return type:  
np array of shape (nhistograms, nhistogramtypes)  
```  
### &#10551; get\_extscores  
full signature:  
```text  
def get_extscores( self, extname, histname=None )  
```  
comments:  
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
### &#10551; get\_extscores\_array  
full signature:  
```text  
def get_extscores_array( self, extname )  
```  
comments:  
```text  
similar to get_extscores, but with different return type:  
np array of shape (nhistograms, nhistogramtypes)  
```  
### &#10551; get\_scores\_ls  
full signature:  
```text  
def get_scores_ls( self, runnb, lsnb, histnames=None, suppresswarnings=False )  
```  
comments:  
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
### &#10551; get\_globalscores  
full signature:  
```text  
def get_globalscores( self, masknames=None )  
```  
comments:  
```text  
get the array of global scores, optionally after masking  
input arguments:  
- masknames: list of names of masks (default: no masking, return full array)  
notes:  
- this method takes the scores from the HistStruct.globalscores attribute;  
  make sure to have set this attribute with add_globalscores,  
  else an exception will be thrown.  
```  
### &#10551; get\_globalscore\_ls  
full signature:  
```text  
def get_globalscore_ls( self, runnb, lsnb )  
```  
comments:  
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
### &#10551; get\_extglobalscores  
full signature:  
```text  
def get_extglobalscores( self, extname )  
```  
comments:  
```text  
get the array of global scores for one of the extra histogram sets  
input arguments:  
- extname: name of the extra histogram set  
notes:  
- this method takes the scores from the HistStruct.extglobalscores attribute;  
  make sure to have set this attribute with add_extglobalscores,  
  else an exception will be thrown.  
```  
### &#10551; get\_histograms  
full signature:  
```text  
def get_histograms( self, histname=None, masknames=None )  
```  
comments:  
```text  
get the array of histograms for a given type, optionally after masking  
input arguments:  
- histname: name of the histogram type to retrieve   
  if None, return a dict matching histnames to arrays of histograms  
- masknames: list of names of masks (default: no masking, return full array)  
```  
### &#10551; get\_exthistograms  
full signature:  
```text  
def get_exthistograms( self, extname, histname=None )  
```  
comments:  
```text  
get the array of extra histograms for a given set name and type name  
input arguments:  
- extname: name of the set of extra histograms (see also add_exthistograms)  
- histname: name of the histogram type to retrieve   
  if None, return a dict matching histnames to arrays of histograms  
```  
### &#10551; get\_histogramsandscores  
full signature:  
```text  
def get_histogramsandscores( self, extname=None, masknames=None, nrandoms=-1, nfirst=-1 )  
```  
comments:  
```text  
combination of get_histograms, get_scores and get_globalscores with additional options  
- extname: use an extended histogram set  
- nrandoms: if > 0, number of random instances to draw  
- nfirst: if > 0, number of first instances to keep  
return type:  
dict with keys 'histograms', 'scores' and 'globalscores'  
note that the values of scores and globalscores may be None if not initialized  
```  
### &#10551; add\_classifier  
full signature:  
```text  
def add_classifier( self, histname, classifier, evaluate=False )  
```  
comments:  
```text  
add a histogram classifier for a given histogram name to the HistStruct  
input arguments:  
- histname: a valid histogram name present in the HistStruct to which this classifier applies  
- classifier: an object of type HistogramClassifier (i.e. of any class that derives from it)  
- evaluate: a bool whether to evaluate the classifier (and store the result in the 'scores' attribute)  
  if set to True, the result is both returned and stored in the 'scores' attribute.  
```  
### &#10551; evaluate\_classifier  
full signature:  
```text  
def evaluate_classifier( self, histname, extname=None )  
```  
comments:  
```text  
evaluate a histogram classifier for a given histogram name in the HistStruct  
input arguments:  
- histname: a valid histogram name present in the HistStruct for which to evaluate the classifier  
- extname: name of a set of extra histograms (see add_exthistograms)  
           if None, will evaluate the classifer for the main set of histograms  
notes:  
- the result is both returned and stored in the 'scores' attribute  
```  
### &#10551; plot\_histograms  
full signature:  
```text  
def plot_histograms( self, histnames=None, masknames=None, histograms=None,  colorlist=[], labellist=[], transparencylist=[],  titledict=None, xaxtitledict=None, physicalxax=False,  yaxtitledict=None, **kwargs )  
```  
comments:  
```text  
plot the histograms in a HistStruct, optionally after masking  
input arguments:  
- histnames: list of names of the histogram types to plot (default: all)  
- masknames: list of list of mask names  
  note: each element in masknames represents a set of masks to apply;   
        the histograms passing different sets of masks are plotted in different colors  
- histograms: list of dicts of histnames to 2D arrays of histograms,  
              can be used to plot a given collection of histograms directly,  
              and bypass the histnames and masknames arguments  
              (note: for use in the gui, not recommended outside of it)  
- colorlist: list of matplotlib colors, must have same length as masknames  
- labellist: list of labels for the legend, must have same legnth as masknames  
- transparencylist: list of transparency values, must have same length as masknames  
- titledict: dict mapping histogram names to titles for the subplots (default: title = histogram name)  
- xaxtitledict: dict mapping histogram names to x-axis titles for the subplots (default: no x-axis title)  
- yaxtitledict: dict mapping histogram names to y-axis titles for the subplots (default: no y-axis title)  
- physicalxax: bool whether to use physical x-axis range or simply use bin number (default)  
- kwargs: keyword arguments passed down to plot_utils.plot_sets   
```  
### &#10551; plot\_histograms\_1d  
full signature:  
```text  
def plot_histograms_1d( self, histnames=None, masknames=None, histograms=None,  colorlist=[], labellist=[], transparencylist=[], titledict=None, xaxtitledict=None, physicalxax=False, yaxtitledict=None,  **kwargs )  
```  
comments:  
```text  
plot the histograms in a histstruct, optionally after masking  
internal helper function, use only via plot_histograms  
```  
### &#10551; plot\_histograms\_2d  
full signature:  
```text  
def plot_histograms_2d( self, histnames=None, masknames=None, histograms=None, labellist=[], titledict=None, xaxtitledict=None, yaxtitledict=None, **kwargs )  
```  
comments:  
```text  
plot the histograms in a histstruct, optionally after masking  
internal helper function, use only via plot_histograms  
```  
### &#10551; plot\_ls  
full signature:  
```text  
def plot_ls( self, runnb, lsnb, histnames=None, histlabel=None,  recohist=None, recohistlabel='Reconstruction',  refhists=None, refhistslabel='Reference histograms', refhiststransparency=None, titledict=None, xaxtitledict=None, physicalxax=False, yaxtitledict=None, **kwargs)  
```  
comments:  
```text  
plot the histograms in a HistStruct for a given run/ls number versus their references and/or their reconstruction  
note: so far only for 1D histograms.  
      case of 2D histograms requires different plotting method since they cannot be clearly overlaid.  
      if a HistStruct contains both 1D and 2D histograms, the 1D histograms must be selected with the histnames argument.  
input arguments:  
- runnb: run number  
- lsnb: lumisection number  
- histnames: names of histogram types to plot (default: all)  
- histlabel: legend entry for the histogram (default: run and lumisection number)  
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
- titledict: dict mapping histogram names to titles for the subplots (default: title = histogram name)  
- xaxtitledict: dict mapping histogram names to x-axis titles for the subplots (default: no x-axis title)  
- yaxtitledict: dict mapping histogram names to y-axis titles for the subplots (default: no y-axis title)  
- physicalxax: bool whether to use physical x-axis range or simply use bin number (default)  
- kwargs: keyword arguments passed down to plot_utils.plot_sets   
```  
### &#10551; plot\_run  
full signature:  
```text  
def plot_run( self, runnb, masknames=None, recohist=None, recohistlabel='reco', refhists=None, refhistslabel='reference', doprint=False)  
```  
comments:  
```text  
call plot_ls for all lumisections in a given run  
```  
### &#10551; plot\_ls\_1d  
full signature:  
```text  
def plot_ls_1d( self, runnb, lsnb, histnames=None, histlabel=None, recohist=None, recohistlabel='Reconstruction', refhists=None, refhistslabel='Reference histograms', refhiststransparency=None, titledict=None, xaxtitledict=None, physicalxax=False, yaxtitledict=None, **kwargs)  
```  
comments:  
```text  
plot the histograms in a HistStruct for a given run/ls number versus their references and/or their reconstruction  
internal helper function, use only via plot_ls  
```  
### &#10551; plot\_ls\_2d  
full signature:  
```text  
def plot_ls_2d( self, runnb, lsnb, histnames=None, histlabel=None, recohist=None, recohistlabel='Reconstruction', titledict=None, xaxtitledict=None, yaxtitledict=None, **kwargs)  
```  
comments:  
```text  
plot the histograms in a HistStruct for a given run/ls number versus their reconstruction  
internal helper function, use only via plot_ls  
```  
- - -  
  
