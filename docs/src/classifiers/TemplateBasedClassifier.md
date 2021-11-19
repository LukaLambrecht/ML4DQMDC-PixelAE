# TemplateBasedClassifier  
  
**Histogram classifier based on a direct comparison with templates (i.e. reference histograms)**
- - -
  
  
### mseTopN\_templates  
full signature:  
```text  
def mseTopN_templates( histograms, templates, n=-1 )  
```  
comments:  
```text  
calculate the mse between each histogram in histograms and each histogram in templates  
input arguments:  
- histograms: 2D numpy array of shape (nhistograms, nbins)  
- templates: 2D numpy array of shape (ntemplates,nbins)  
- n: integer representing the number of (sorted) bin squared errors to take into account (default: all)  
output:  
2D numpy array of shape (nhistograms,ntemplates) holding the mseTopN between each  
```  
  
  
### mseTopN\_min  
full signature:  
```text  
def mseTopN_min( histograms, templates, n=-1 )  
```  
comments:  
```text  
calculate the mse betwee a histogram and each template and return the minimum  
input arguments:  
- histograms: 2D numpy array of shape (nhistograms, nbins)  
- templates: 2D numpy array of shape (ntemplates,nbins)  
- n: integer representing the number of (sorted) bin squared errors to take into account (default: all)  
output:  
1D numpy array of shape (nhistograms) holding the minimum mseTopN for each histogram  
```  
  
  
### mseTop10\_min  
full signature:  
```text  
def mseTop10_min( histograms, templates )  
```  
comments:  
```text  
special case of above with n=10  
```  
  
  
### mseTopN\_avg  
full signature:  
```text  
def mseTopN_avg( histograms, templates, n=-1 )  
```  
comments:  
```text  
calculate the mse betwee a histogram and each template and return the average  
input arguments:  
- histograms: 2D numpy array of shape (nhistograms, nbins)  
- templates: 2D numpy array of shape (ntemplates,nbins)  
- n: integer representing the number of (sorted) bin squared errors to take into account (default: all)  
output:  
1D numpy array of shape (nhistograms) holding the average mseTopN for each histogram  
```  
  
  
### mseTop10\_avg  
full signature:  
```text  
def mseTop10_avg( histograms, templates )  
```  
comments:  
```text  
special case of above with n=10  
```  
  
  
- - -
## [class] TemplateBasedClassifier  
comments:  
```text  
histogram classifier based on a direct comparison with templates (i.e. reference histograms)  
```  
### &#10551; \_\_init\_\_  
full signature:  
```text  
def __init__( self, comparemethod='minmse' )  
```  
comments:  
```text  
initializer  
input arguments:  
- comparemethod: string representing the method by which to compare a histogram with a set of templates  
  currently supported methods are:  
  - minmse: minimum mean square error between histogram and all templates  
  - avgmse: average mean square error between histogram and all templates  
```  
### &#10551; train  
full signature:  
```text  
def train( self, templates )  
```  
comments:  
```text  
'train' the classifier, i.e. set the templates (reference histograms)  
input arguments:  
- templates: a 2D numpy array of shape (nhistograms,nbins)  
```  
### &#10551; evaluate  
full signature:  
```text  
def evaluate( self, histograms )  
```  
comments:  
```text  
classification of a collection of histograms based on their deviation from templates  
```  
- - -  
  
