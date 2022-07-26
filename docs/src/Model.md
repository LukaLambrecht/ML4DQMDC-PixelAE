# Model  
  
**Model: grouping classifiers for different histogram types**  

This class represents a general model for assigning a score to a lumisection.  
It consists of two distinct parts:  

- a collection of classifiers acting on individual histogramgs (one for each type).  
- a fitter to assign a probability density to the output scores obtained in the previous step.  
The types of histograms, classifiers, and fitter can be freely chosen.  
 
This class does not contain the histograms or other data; it only contains the classifiers and fitter.  
Use the derived class ModelInterface to make the bridge between a HistStruct (holding the data)  
and a Model (holding the classifiers and fitter).  
- - -
  
  
- - -
## [class] Model  
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
- histnames: list of the histogram names for this Model.  
  this is the only argument needed for initialization,  
  use the functions set_classifiers and set_fitter   
  to set the classifiers and fitter respectively.  
```  
### &#10551; set\_classifiers  
full signature:  
```text  
def set_classifiers( self, classifiers )  
```  
comments:  
```text  
set the classifiers for this Model  
input arguments:  
- classifiers: dict of histnames to classifiers.  
  the histnames must match the ones used to initialize this Model,  
  the classifiers must be a subtype of HistogramClassifier.  
```  
### &#10551; set\_fitter  
full signature:  
```text  
def set_fitter( self, fitter )  
```  
comments:  
```text  
set the fitter for this Model  
input arguments:  
- fitter: an (untrained) object of type CloudFitter  
```  
### &#10551; check\_classifier  
full signature:  
```text  
def check_classifier( self, histname )  
```  
comments:  
```text  
check if a classifier was initialized  
input arguments:  
- histname: type of histogram for which to check the classifier  
```  
### &#10551; check\_fitter  
full signature:  
```text  
def check_fitter( self )  
```  
comments:  
```text  
check if a fitter was initialized  
```  
### &#10551; train\_classifier  
full signature:  
```text  
def train_classifier( self, histname, histograms, **kwargs )  
```  
comments:  
```text  
train a classifier  
input arguments:  
- histname: histogram name for which to train the classifier  
- histograms: the histograms for training, np array of shape (nhistograms,nbins)  
- kwargs: additional keyword arguments for training  
```  
### &#10551; train\_classifiers  
full signature:  
```text  
def train_classifiers( self, histograms, **kwargs )  
```  
comments:  
```text  
train classifiers for all histograms in this Model  
input arguments:  
- histograms: the histograms for training, dict of histnames to np arrays of shape (nhistograms,nbins)  
- kwargs: additional keyword arguments for training  
```  
### &#10551; evaluate\_classifier  
full signature:  
```text  
def evaluate_classifier( self, histname, histograms, mask=None )  
```  
comments:  
```text  
evaluate a classifier and return the score  
input arguments:  
- histname: histogram name for which to evaluate the classifier  
- histograms: the histograms for evaluation, np array of shape (nhistograms,nbins)  
- mask: a np boolean array masking the histograms to be evaluated  
returns:  
- a np array of shape (nhistograms) with the scores  
note: masked-out indices are set to np.nan!  
```  
### &#10551; evaluate\_classifiers  
full signature:  
```text  
def evaluate_classifiers( self, histograms, mask=None )  
```  
comments:  
```text  
evaluate the classifiers and return the scores  
input arguments:  
- histograms: dict of histnames to histogram arrays (shape (nhistograms,nbins))  
- mask: a np boolean array masking the histograms to be evaluated  
returns:  
- dict of histnames to scores (shape (nhistograms))  
note: masked-out indices are set to np.nan!  
```  
### &#10551; get\_point\_array  
full signature:  
```text  
def get_point_array( self, points )  
```  
comments:  
```text  
for internal use in train_fitter and evaluate_fitter  
input arguments:  
- points: dict matching histnames to scores (np array of shape (nhistograms))  
```  
### &#10551; train\_fitter  
full signature:  
```text  
def train_fitter( self, points, verbose=False, **kwargs )  
```  
comments:  
```text  
train the fitter  
input arguments:  
- points: dict matching histnames to scores (np array of shape (nhistograms))  
- kwargs: additional keyword arguments for fitting  
```  
### &#10551; evaluate\_fitter  
full signature:  
```text  
def evaluate_fitter( self, points, mask=None, verbose=False )  
```  
comments:  
```text  
evaluate the fitter and return the scores  
input arguments:  
- points: dict matching histnames to scores (np array of shape (nhistograms))  
- mask: a np boolean array masking the histograms to be evaluated  
returns:  
- a np array of shape (nhistograms) with the scores  
note: masked-out indices are set to np.nan!  
```  
- - -  
  
