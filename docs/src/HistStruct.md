# HistStruct  
  
**HistStruct: consistent treatment of multiple histogram types**  

The HistStruct class is the main data structure used within this framework.  
A HistStruct object basically consists of a mutually consistent collection of numpy arrays,  
where each numpy array corresponds to one histogram type, with dimensions (number of histograms, number of bins).  
The HistStruct has functions to easily perform the following common tasks (among others):  

- select a subset of runs and/or lumisections (e.g. using a custom or predefined json file formatted selector),  
- prepare the data for machine learning training, with all kinds of preprocessing,  
- evaluate classifiers (machine learning types or other),  
- go from per-histogram scores to per-lumisection scores.  
 
When only processing a single histogram type, the HistStruct might be a bit of an overkill.  
One could instead choose to operate on the dataframe directly.  
However, especially when using multiple histogram types, the HistStruct is very handy to keep everything consistent.  
- - -
  
  
- - -
## [class] HistStruct  
comments:  
```text  
(no valid documentation found)  
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
masks: dict mapping name to 1D numpy array of booleans (same length as histograms) that can be used for masking  
exthistograms: dict of dicts similar to histograms for additional (e.g. artificially generated) histograms  
setnames: list of names of extended sets  
models: dict mapping model names to ModelInterfaces  
modelnames: list of model names  
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
def save( self, path, save_models=False, save_classifiers=True, save_fitter=True )  
```  
comments:  
```text  
save a HistStruct object to a pkl file  
input arguments:  
- path where to store the file (appendix .zip is automatically appended)  
- save_models: a boolean whether to include the models if present in the HistStruct  
- save_classifiers: a boolean whether to include the classifiers if present in the ModelInterfaces  
- save_fitter: a boolean whether to include the fitter if present in the ModelInterfaces  
```  
### &#10551; load  
full signature:  
```text  
def load( self, path, load_models=True, load_classifiers=True, load_fitter=True, verbose=False )  
```  
comments:  
```text  
load a HistStruct object  
input arguments:  
- path to a zip file containing a HistStruct object  
- load_models: a boolean whether to load the models if present  
- load_classifiers: a boolean whether to load the classifiers if present  
- load_fitter: a boolean whether to load the fitter if present  
- verbose: boolean whether to print some information  
```  
### &#10551; add\_dataframe  
full signature:  
```text  
def add_dataframe( self, df, cropslices=None, rebinningfactor=None,  smoothinghalfwindow=None, smoothingweights=None, averagewindow=None, averageweights=None, donormalize=True )  
```  
comments:  
```text  
add a dataframe to a HistStruct  
input arguments:  
- df: a pandas dataframe as read from the input csv files  
- cropslices: list of slices (one per dimension) by which to crop the histograms  
               see hist_utils.py / crophists for more info.  
- rebinningfactor: factor by which to group bins together  
                   see hist_utils.py / rebinhists for more info.  
- smoothinghalfwindow: half window (int for 1D, tuple for 2D) for doing smoothing of histograms  
- smoothingweights: weight array (1D for 1D, 2D for 2D) for smoothing of histograms  
                    see hist_utils.py / smoothhists for more info.  
- averagewindow: window (int or tuple) for averaging each histogram with its neighbours  
- averageweights: weights for averaging each histogram with its neighbours  
                  see hist_utils.py / running_average_hists for more info.  
- donormalize: boolean whether to normalize the histograms  
               see hist_utils.py / normalizehists for more info.  
notes:  
- the new dataframe can contain one or multiple histogram types  
- the new dataframe must contain the same run and lumisection numbers (for each histogram type in it)  
  as already present in the HistStruct, except if it is the first one to be added  
- alternative to adding the dataframe with the preprocessing options,   
  one can also apply the preprocessing at a later stage using the preprocess() function  
  with the same arguments.  
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
def preprocess( self, masknames=None, cropslices=None, rebinningfactor=None, smoothinghalfwindow=None, smoothingweights=None, averagewindow=None, averageweights=None, donormalize=False )  
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
DEPRECATED, DO NOT USE ANYMORE  
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
DEPRECATED, DO NOT USE ANYMORE  
input arguments:  
- extname: name of extra histogram set  
- globalscores: 1D numpy array of scores  
note: this function checks if all histogram types in this set contain the same number of histograms,  
      (and that this number corresponds to the length of globalscores)  
      else adding globalscores is meaningless  
```  
### &#10551; add\_exthistograms  
full signature:  
```text  
def add_exthistograms( self, setname, histname, histograms, overwrite=False )  
```  
comments:  
```text  
add a set of extra histograms to a HistStruct  
these histograms are not assumed to correspond to physical run/lumisections numbers (e.g. resampled ones),  
and no consistency checks are done  
input arguments:  
- setname: name of the extra histogram set (you can add multiple, e.g. resampled_good, resampled_bad and/or resampled_training)  
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
- mask: a 1D np array of booleans with same length as number of lumisections in HistStruct  
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
### &#10551; add\_index\_mask  
full signature:  
```text  
def add_index_mask( self, name, indices )  
```  
comments:  
```text  
add a mask corresponding to predefined indices  
input arguments:  
- name: a name for the mask  
- indices: a 1D np array of integer indices, between 0 and the number of lumisections in HistStruct  
```  
### &#10551; add\_run\_mask  
full signature:  
```text  
def add_run_mask( self, name, runnb )  
```  
comments:  
```text  
add a mask corresponding to a given run number  
input arguments:  
- name: a name for the mask  
- runnb: run number  
```  
### &#10551; add\_multirun\_mask  
full signature:  
```text  
def add_multirun_mask( self, name, runnbs )  
```  
comments:  
```text  
add a mask corresponding to a given list of run numbers  
input arguments:  
- name: a name for the mask  
- runnbs: a list of run numbers  
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
- min_entries_to_bins_ratio: number of entries divided by number of bins, lower boundary for statistics   
  (default: no lower boundary)  
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
### &#10551; pass\_masks  
full signature:  
```text  
def pass_masks( self, masknames, runnbs=None, lsnbs=None )  
```  
comments:  
```text  
get a list of booleans of lumisections whether they pass a given set of masks  
input arguments:  
- masknames: list of mask names  
- runnbs: list of run numbers (default: all in histstruct)  
- lsnbs: list of lumisection numbers (equally long as runnbs) (default: al in histstruct)  
```  
### &#10551; get\_masknames  
full signature:  
```text  
def get_masknames( self )  
```  
comments:  
```text  
return a list of all mask names in the current HistStruct  
```  
### &#10551; get\_mask  
full signature:  
```text  
def get_mask( self, name )  
```  
comments:  
```text  
return a mask in the current HistStruct  
```  
### &#10551; get\_combined\_mask  
full signature:  
```text  
def get_combined_mask( self, names )  
```  
comments:  
```text  
get a combined (intersection) mask given multiple mask names  
mostly for internal use; externally you can use get_histograms( histname, <list of mask names>) directly  
```  
### &#10551; get\_union\_mask  
full signature:  
```text  
def get_union_mask( self, names )  
```  
comments:  
```text  
get a combined (union) mask given multiple mask names  
mostly for internal use  
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
def get_runnbs_unique( self, masknames=None )  
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
def get_scores( self, modelname, histname=None, setnames=None, masknames=None )  
```  
comments:  
```text  
get the array of scores for a given model and for a given histogram type, optionally after masking  
input arguments:  
- modelname: name of the model for which to retrieve the scores  
- histname: name of the histogram type for which to retrieve the score.   
  if None, return a dict matching histnames to arrays of scores  
- setnames: list of names of the histogram sets (use None for standard set)  
- masknames: list of names of masks (default: no masking, return full array)  
notes:  
- do not use setnames and masknames simultaneously, this is not defined  
- if multiple masks are given, the intersection is taken;  
  if multiple sets are given, the union is taken  
- the classifiers in the appropriate model must have been evaluated before calling this method!  
```  
### &#10551; get\_scores\_array  
full signature:  
```text  
def get_scores_array( self, modelname, setnames=None, masknames=None )  
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
DEPRECATED, DO NOT USE ANYMORE  
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
DEPRECATED, DO NOT USE ANYMORE  
```  
### &#10551; get\_scores\_ls  
full signature:  
```text  
def get_scores_ls( self, modelname, runnb, lsnb, histnames=None )  
```  
comments:  
```text  
get the scores for a given run/lumisection number and for given histogram names  
input arguments:  
- modelname: name of the model for which to retrieve the score  
- runnb: run number  
- lsnb: lumisection number  
- histnames: names of the histogram types for which to retrieve the score.   
returns:  
- a dict matching each name in histnames to a score (or None if no valid score)  
```  
### &#10551; get\_globalscores  
full signature:  
```text  
def get_globalscores( self, modelname, setnames=None, masknames=None )  
```  
comments:  
```text  
get the array of global scores, optionally after masking  
input arguments:  
- modelname: name of the model for which to retrieve the global score  
- setnames: list of names of the histogram sets (use None for standard set)  
- masknames: list of names of masks (default: no masking, return full array)  
notes:  
- do not use setnames and masknames simultaneously, this is not defined  
- if multiple masks are given, the intersection is taken;  
  if multiple sets are given, the union is taken  
- the classifiers in the appropriate model must have been evaluated before calling this method!  
```  
### &#10551; get\_globalscores\_jsonformat  
full signature:  
```text  
def get_globalscores_jsonformat( self, modelname=None )  
```  
comments:  
```text  
make a json format listing all lumisections in this histstruct  
the output list has entries for global scores and masks  
input arguments:  
- modelname: name of the model for wich to retrieve the global score  
  if None, all available models will be used  
```  
### &#10551; get\_globalscore\_ls  
full signature:  
```text  
def get_globalscore_ls( self, modelname, runnb, lsnb )  
```  
comments:  
```text  
get the global score for a given run/lumisection number  
input arguments:  
- modelname: name of the model for which to retrieve the global score  
- runnb: run number  
- lsnb: lumisection number  
- histnames: names of the histogram types for which to retrieve the score.   
returns:  
- a dict matching each name in histnames to a score (or None if no valid score)  
```  
### &#10551; get\_globalscores\_mask  
full signature:  
```text  
def get_globalscores_mask( self, modelname, masknames=None, setnames=None, score_up=None, score_down=None )  
```  
comments:  
```text  
get the mask for global score between specified boundaries  
input arguments:  
- modelname: name of the model for which to consider the global scores  
- masknames: list of additional masks (on top of score boundaries) to consider  
- setnames: list of set names for which to retrieve the global scores  
- score_up and score_down are upper and lower thresholds  
    if both are not None, the mask for global scores between the boundaries are returned  
    if score_up is None, the mask for global score > score_down are returned  
    if score_down is None, the mask for global score < score_up are returned  
```  
### &#10551; get\_globalscores\_indices  
full signature:  
```text  
def get_globalscores_indices( self, modelname, masknames=None, setnames=None, score_up=None, score_down=None )  
```  
comments:  
```text  
get the indices with a global score between specified boundaries  
input arguments: see get_globalscore_mask  
```  
### &#10551; get\_globalscores\_runsls  
full signature:  
```text  
def get_globalscores_runsls( self, modelname, masknames=None, setnames=None, score_up=None, score_down=None )  
```  
comments:  
```text  
get the run and lumisection numbers with a global score between specified boundaries  
input arguments: see get_globalscore_mask  
```  
### &#10551; get\_extglobalscores  
full signature:  
```text  
def get_extglobalscores( self, extname )  
```  
comments:  
```text  
get the array of global scores for one of the extra histogram sets  
DEPRECATED, DO NOT USE ANYMORE  
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
def get_histograms( self, histname=None, masknames=None, setnames=None )  
```  
comments:  
```text  
get the array of histograms for a given type, optionally after masking  
input arguments:  
- histname: name of the histogram type to retrieve   
  if None, return a dict matching histnames to arrays of histograms  
- masknames: list of names of masks (default: no masking, return full array)  
- setnames: list of names of the sets of extra histograms (see also add_exthistograms)  
  if multiple setnames are provided, the union/concatenation is returned  
```  
### &#10551; get\_histogramsandscores  
full signature:  
```text  
def get_histogramsandscores( self, modelname=None, setnames=None, masknames=None, nrandoms=-1, nfirst=-1 )  
```  
comments:  
```text  
combination of get_histograms, get_scores and get_globalscores with additional options  
- modelname: name of the model for which to retrieve the score  
  if None, no scores will be retrieved (only histograms)  
- setnames: list of names of histogram sets (use None for default set)  
- masknames: list of names of masks  
- nrandoms: if > 0, number of random instances to draw  
- nfirst: if > 0, number of first instances to keep  
return type:  
dict with keys 'histograms', 'scores' and 'globalscores'  
note that the values of scores and globalscores may be None if not initialized  
```  
### &#10551; add\_model  
full signature:  
```text  
def add_model( self, modelname, model )  
```  
comments:  
```text  
add a model to the HistStruct  
input arguments:  
- modelname: a name for the model  
- model: an instance of ModelInterface class with histnames corresponding to the ones for this HistStruct  
```  
### &#10551; check\_model  
full signature:  
```text  
def check_model( self, modelname )  
```  
comments:  
```text  
check if a given model name is present in the HistStruct  
input arguments:  
- modelname: name of the model to check  
```  
### &#10551; remove\_model  
full signature:  
```text  
def remove_model( self, modelname )  
```  
comments:  
```text  
remove a model  
input arguments:  
- modelname: name of the model to remove  
```  
### &#10551; train\_classifier  
full signature:  
```text  
def train_classifier( self, modelname, histname, masknames=None, setnames=None, **kwargs )  
```  
comments:  
```text  
train a histogram classifier  
input arguments:  
- modelname: name of the model for which to train the classifiers  
- histname: a valid histogram name present in the HistStruct for which to train the classifier  
- masknames: list of masks the classifiers should be trained on  
- setnames: list of names of sets of extra histograms on which the classifiers should be trained  
- kwargs: additional keyword arguments for training  
```  
### &#10551; train\_classifiers  
full signature:  
```text  
def train_classifiers( self, modelname, masknames=None, setnames=None, **kwargs )  
```  
comments:  
```text  
train histogram classifiers for all histogram types  
input arguments:  
- modelname: name of the model for which to train the classifiers  
- masknames: list of masks the classifiers should be trained on  
- setnames: list of names of sets of extra histograms on which the classifiers should be trained  
- kwargs: additional keyword arguments for training  
```  
### &#10551; evaluate\_classifier  
full signature:  
```text  
def evaluate_classifier( self, modelname, histname, masknames=None, setnames=None )  
```  
comments:  
```text  
evaluate a histogram classifier  
input arguments:  
- modelname: name of the model for wich to evaluate the classifiers  
- histname: a valid histogram name present in the HistStruct for which to evaluate the classifier  
- masknames: list of masks if the classifiers should be evaluated on a subset only (e.g. for speed)  
- setnames: list of names of sets of extra histograms for which the classifiers should be evaluated  
```  
### &#10551; evaluate\_classifiers  
full signature:  
```text  
def evaluate_classifiers( self, modelname, masknames=None, setnames=None )  
```  
comments:  
```text  
evaluate histogram classifiers for all histogram types  
input arguments:  
- modelname: name of the model for wich to evaluate the classifiers  
- masknames: list of masks if the classifiers should be evaluated on a subset only (e.g. for speed)  
- setnames: list of names of a set of extra histograms for which the classifiers should be evaluated  
```  
### &#10551; set\_fitter  
full signature:  
```text  
def set_fitter( self, modelname, fitter )  
```  
comments:  
```text  
set the fitter for a given model  
```  
### &#10551; train\_fitter  
full signature:  
```text  
def train_fitter( self, modelname, masknames=None, setnames=None, verbose=False, **kwargs )  
```  
comments:  
```text  
train the fitter for a given model  
input arguments:  
- modelname: name of the model to train  
- masknames: list of mask names for training set  
- setnames: list of set names for training set  
- kwargs: additional keyword arguments for fitting  
note: use either masksnames or setnames, not both!  
```  
### &#10551; train\_partial\_fitters  
full signature:  
```text  
def train_partial_fitters( self, modelname, dimslist, masknames=None, setnames=None, **kwargs )  
```  
comments:  
```text  
train partial fitters for a given model  
input arguments:  
- modelname: name of the model to train  
- dimslist: list of tuples with integer dimension numbers  
- masknames: list of mask names for training set  
- setnames: list of set names for training set  
- kwargs: additional keyword arguments for fitting  
note: use either masksnames or setnames, not both!  
note: see also plot_partial_fit for a convenient plotting method!  
```  
### &#10551; evaluate\_fitter  
full signature:  
```text  
def evaluate_fitter( self, modelname, masknames=None, setnames=None, verbose=False )  
```  
comments:  
```text  
evaluate the fitter for a given model  
input arguments:  
- modelname: name of the model for which to evaluate the fitter  
- masknames: list of mask names if the fitter should be evaluated on a subset only (e.g. for speed)  
- setnames: list of set names of extra histograms for which the fitter should be evaluated  
```  
### &#10551; evaluate\_fitter\_on\_point  
full signature:  
```text  
def evaluate_fitter_on_point( self, modelname, point )  
```  
comments:  
```text  
evaluate the fitter on a given points  
input arguments:  
- modelname: name of the model for which to evaluate the fitter  
- points: dict matching histnames to scores (one float per histogram type)  
  (e.g. as returned by get_scores_ls)  
returns:  
- the global score for the provided point (a float)  
```  
### &#10551; evaluate\_fitter\_on\_points  
full signature:  
```text  
def evaluate_fitter_on_points( self, modelname, points )  
```  
comments:  
```text  
evaluate the fitter on a given set of points  
input arguments:  
- modelname: name of the model for which to evaluate the fitter  
- points: dict matching histnames to scores (np array of shape (nhistograms))  
returns:  
- the global scores for the provided points  
```  
### &#10551; plot\_histograms  
full signature:  
```text  
def plot_histograms( self, histnames=None, masknames=None, histograms=None, ncols=4, colorlist=[], labellist=[], transparencylist=[],  titledict=None, xaxtitledict=None, physicalxax=False,  yaxtitledict=None, **kwargs )  
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
- ncols: number of columns (only relevant for 1D histograms)  
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
def plot_histograms_1d( self, histnames=None, masknames=None, histograms=None, ncols=4, colorlist=[], labellist=[], transparencylist=[], titledict=None, xaxtitledict=None, physicalxax=False, yaxtitledict=None,  **kwargs )  
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
### &#10551; plot\_histograms\_run  
full signature:  
```text  
def plot_histograms_run( self, histnames=None, masknames=None, histograms=None, ncols=4,  titledict=None, xaxtitledict=None, physicalxax=False,  yaxtitledict=None, **kwargs )  
```  
comments:  
```text  
plot a set of histograms in a HistStruct with a smooth color gradient.  
typical use case: plot a single run.  
note: only for 1D histograms!  
input arguments:  
- histnames: list of names of the histogram types to plot (default: all)  
- masknames: list mask names (typically should contain a run number mask)  
- histograms: dict of histnames to 2D arrays of histograms,  
              can be used to plot a given collection of histograms directly,  
              and bypass the histnames and masknames arguments  
              (note: for use in the gui, not recommended outside of it.  
- titledict: dict mapping histogram names to titles for the subplots (default: title = histogram name)  
- xaxtitledict: dict mapping histogram names to x-axis titles for the subplots (default: no x-axis title)  
- yaxtitledict: dict mapping histogram names to y-axis titles for the subplots (default: no y-axis title)  
- physicalxax: bool whether to use physical x-axis range or simply use bin number (default)  
- kwargs: keyword arguments passed down to plot_utils.plot_hists_multi  
```  
### &#10551; plot\_histograms\_run\_1d  
full signature:  
```text  
def plot_histograms_run_1d( self, histnames=None, masknames=None, histograms=None, ncols=4, titledict=None, xaxtitledict=None, physicalxax=False, yaxtitledict=None,  **kwargs )  
```  
comments:  
```text  
plot the histograms in a histstruct, optionally after masking  
internal helper function, use only via plot_histograms_run  
```  
### &#10551; plot\_ls  
full signature:  
```text  
def plot_ls( self, runnb, lsnb, histnames=None, histlabel=None, ncols=4, recohist=None, recohistlabel='Reconstruction',  refhists=None, refhistslabel='Reference histograms', refhiststransparency=None, titledict=None, xaxtitledict=None, physicalxax=False, yaxtitledict=None, **kwargs)  
```  
comments:  
```text  
plot the histograms in a HistStruct for a given run/ls number versus their references and/or their reconstruction  
input arguments:  
- runnb: run number  
- lsnb: lumisection number  
- histnames: names of histogram types to plot (default: all)  
- histlabel: legend entry for the histogram (default: run and lumisection number)  
- recohist: dict matching histogram names to reconstructed histograms  
  notes: - 'reconstructed histograms' refers to e.g. autoencoder or NMF reconstructions;  
           some models (e.g. simply looking at histogram moments) might not have this kind of reconstruction  
         - in principle one histogram per key is expected, but still the the shape must be 2D (i.e. (1,nbins))  
         - in case recohist is set to a valid model name present in the current HistStruct,   
           the reconstruction is calculated on the fly for the input histograms  
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
def plot_run( self, runnb, masknames=None, ncols=4,  recohist=None, recohistlabel='reco',  refhists=None, refhistslabel='reference', doprint=False)  
```  
comments:  
```text  
call plot_ls for all lumisections in a given run  
```  
### &#10551; plot\_ls\_1d  
full signature:  
```text  
def plot_ls_1d( self, runnb, lsnb, histnames=None, histlabel=None, ncols=4, recohist=None, recohistlabel='Reconstruction', refhists=None, refhistslabel='Reference histograms', refhiststransparency=None, titledict=None, xaxtitledict=None, physicalxax=False, yaxtitledict=None, **kwargs)  
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
### &#10551; plot\_ls\_score  
full signature:  
```text  
def plot_ls_score( self, modelname, runnb, lsnb, ncols=4, masknames=None, setnames=None, **kwargs )  
```  
comments:  
```text  
plot the score of a given lumisection for each histogram type compared to reference scores  
input arguments:  
- modelname: name of the model for which to retrieve the score  
- runnb: run number  
- lsnb: lumisection number  
- masknames: list of mask names for the reference scores  
- setnames: list of set names for the reference scores  
- kwargs: additional keyword arguments passed down to pu.plot_score_dist  
```  
### &#10551; plot\_partial\_fit  
full signature:  
```text  
def plot_partial_fit( self, modelname, dims, clusters, **kwargs)  
```  
comments:  
```text  
plot a partial fit calculated with train_partial_fitters  
input arguments:  
- modelname: name of the model for which to plot the partial fits  
- dims: a tuple of length 1 or 2 with integer dimension indices  
  note: the partial fit for this dimension must have been calculated with train_partial_fitters first  
- clusters: a list of the different point clusters to plot  
            each element in the list should be a dict of the form   
            {'masknames': [list of mask names]} or {'setnames': [list of set names]}  
- kwargs: plot options passed down to pu.plot_fit_1d_clusters or pu.plot_fit_2d_clusters;  
          some of them have to have the same length as clusters (e.g. colors and labels)  
```  
### &#10551; plot\_score\_dist  
full signature:  
```text  
def plot_score_dist( self, modelname, histname=None,  masknames_sig=None, setnames_sig=None,  masknames_bkg=None, setnames_bkg=None, **kwargs )  
```  
comments:  
```text  
plot a 1D score distribution  
input arguments:  
- modelname: name of the model for which to retrieve the scores  
- histname: type of histogram for which to retrieve the scores  
            if None, the global scores will be retrieved  
- masknames_sig, setnames_sig: lists of mask or set names for signal distribution  
- masknames_bkg, setnames_bkg: lists of mask or set names for background distribution  
  note: in case of multiple masks, the intersection is taken (as usual);  
        in case of multiple sets, the union is taken!  
- kwargs: additional keyword arguments passed down to pu.plot_score_dist  
```  
- - -  
  
