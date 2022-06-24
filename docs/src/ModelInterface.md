# ModelInterface  
  
**ModelInterface: extension of Model class interfaced by HistStruct**  

This class is the interface between a Model (holding classifiers and fitters)  
and a HistStruct (holding histogram data).  
It stores the classifier and model scores for the histograms in a HistStruct.  
- - -
  
  
- - -
## [class] ModelInterface  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; \_\_init\_\_  
full signature:  
```text  
def __init__( self, histnames )  
```  
comments:  
```text  
initializer  
input arguments:  
- histnames: list of the histogram names for this Model(Interface).  
```  
### &#10551; \_\_str\_\_  
full signature:  
```text  
def __str__( self )  
```  
comments:  
```text  
get a printable representation of a ModelInterface  
```  
### &#10551; add\_setname  
full signature:  
```text  
def add_setname( self, setname )  
```  
comments:  
```text  
initialize empty scores for extended set  
input arguments:  
- setname: name of extended set  
```  
### &#10551; check\_setname  
full signature:  
```text  
def check_setname( self, setname )  
```  
comments:  
```text  
check if a setname is present  
input arguments:  
- setname: name of the set to check  
```  
### &#10551; check\_setnames  
full signature:  
```text  
def check_setnames( self, setnames )  
```  
comments:  
```text  
check if all names in a list of set names are present  
```  
### &#10551; check\_scores  
full signature:  
```text  
def check_scores( self, histnames=None, setnames=None )  
```  
comments:  
```text  
check if scores are present for a given set name  
input arguments:  
- histnames: list of histogram names for which to check the scores (default: all)  
- setname: list of set names for which to check the scores (default: standard set)  
```  
### &#10551; check\_globalscores  
full signature:  
```text  
def check_globalscores( self, setnames=None )  
```  
comments:  
```text  
check if global scores are present for a given set name  
input arguments:  
- setname: list of set names for which to check the scores (default: standard set)  
```  
### &#10551; evaluate\_store\_classifier  
full signature:  
```text  
def evaluate_store_classifier( self, histname, histograms, mask=None, setname=None )  
```  
comments:  
```text  
same as Model.evaluate_classifier but store the result internally  
input arguments:  
- histname: histogram name for which to evaluate the classifier  
- histograms: the histograms for evaluation, np array of shape (nhistograms,nbins)  
- mask: a np boolean array masking the histograms to be evaluated  
- setname: name of extended set (default: standard set)  
```  
### &#10551; evaluate\_store\_classifiers  
full signature:  
```text  
def evaluate_store_classifiers( self, histograms, mask=None, setname=None )  
```  
comments:  
```text  
same as Model.evaluate_classifiers but store the result internally  
input arguments:  
- histograms: dict of histnames to histogram arrays (shape (nhistograms,nbins))  
- mask: a np boolean array masking the histograms to be evaluated  
- setname: name of extended set (default: standard set)  
```  
### &#10551; evaluate\_store\_fitter  
full signature:  
```text  
def evaluate_store_fitter( self, points, mask=None, setname=None, verbose=False )  
```  
comments:  
```text  
same as Model.evaluate_fitter but store the result internally  
input arguments:  
- points: dict matching histnames to scores (np array of shape (nhistograms))  
- mask: a np boolean array masking the histograms to be evaluated  
- setname: name of extended set (default: standard set)  
```  
### &#10551; get\_scores  
full signature:  
```text  
def get_scores( self, setnames=None, histname=None )  
```  
comments:  
```text  
get the scores stored internally  
input arguments:  
- setnames: list of names of extended sets (default: standard set)  
- histname: name of histogram type for which to get the scores  
  if specified, an array of scores is returned.  
  if not, a dict matching histnames to arrays of scores is returned.  
```  
### &#10551; get\_globalscores  
full signature:  
```text  
def get_globalscores( self, setnames=None )  
```  
comments:  
```text  
get the global scores stored internally  
input arguments:  
- setnames: list of name of extended sets (default: standard set)  
```  
### &#10551; get\_globalscores\_mask  
full signature:  
```text  
def get_globalscores_mask( self, setnames=None, score_up=None, score_down=None )  
```  
comments:  
```text  
get a mask of global scores within boundaries  
input arguments:  
- setnames: list of name of extended sets (default: standard set)  
- score_up and score_down are upper and lower thresholds  
    if both are not None, the mask for global scores between the boundaries are returned  
    if score_up is None, the mask for global score > score_down are returned  
    if score_down is None, the mask for global score < score_up are returned  
```  
### &#10551; get\_globalscores\_indices  
full signature:  
```text  
def get_globalscores_indices( self, setnames=None, score_up=None, score_down=None )  
```  
comments:  
```text  
get the indices of global scores within boundaries  
input arguments:  
- setnames: list of name of extended sets (default: standard set)  
- score_up and score_down are upper and lower thresholds  
    if both are not None, the indices with global scores between the boundaries are returned  
    if score_up is None, the indices with global score > score_down are returned  
    if score_down is None, the indices with global score < score_up are returned  
```  
### &#10551; train\_partial\_fitters  
full signature:  
```text  
def train_partial_fitters( self, dimslist, points, **kwargs )  
```  
comments:  
```text  
train partial fitters on a given set of dimensions  
input arguments:  
- dimslist: list of tuples with integer dimension numbers  
- points: dict matching histnames to scores (np array of shape (nhistograms))  
- kwargs: additional keyword arguments for fitting  
```  
### &#10551; save  
full signature:  
```text  
def save( self, path, save_classifiers=True, save_fitter=True )  
```  
comments:  
```text  
save a ModelInterface object to a pkl file  
input arguments:  
- path where to store the file  
- save_classifiers: a boolean whether to include the classifiers (alternative: only scores)  
- save_fitter: a boolean whether to include the fitter (alternative: only scores)  
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
get all files to store in the zip file  
```  
### &#10551; load  
full signature:  
```text  
def load( self, path, load_classifiers=True, load_fitter=True, verbose=False )  
```  
comments:  
```text  
load a ModelInterface object  
input arguments:  
- path to a zip file containing a ModelInterface object  
- load_classifiers: a boolean whether to load the classifiers if present  
- load_fitter: a boolean whether to load the fitter if present  
- verbose: boolean whether to print some information  
```  
- - -  
  
