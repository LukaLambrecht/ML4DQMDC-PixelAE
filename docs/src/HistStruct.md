# HistStruct  
  
- - -    
## HistStruct(object)  
**main data structure used within this framework**  
a HistStruct object basically consists of a mutually consistent collection of numpy arrays,  
where each numpy array corresponds to one histogram type, with dimensions (number of histograms, number of bins).  
the HistStruct has functions to easily perform the following common tasks (among others):  
- select a subset of runs and/or lumisections (e.g. using a json file formatted selector),  
- prepare the data for machine learning training  
- evaluate classifiers (machine learning types or other)  
  
### \_\_init\_\_( self )  
**empty initializer, setting all containers to empty defaults**  
  
### save( self, path )  
**save a HistStruct object to a pkl file**  
  
### load( self, path )  
(no valid documentation found)  
  
### add\_dataframe( self, df, donormalize=True, rebinningfactor=1 )  
**add a dataframe to a HistStruct**  
input arguments:  
- df is a pandas dataframe as read from the input csv files.  
- donormalize: boolean whether to normalize the histograms  
- rebinningfactor: factor by which to group bins together  
notes:  
- the new dataframe can contain one or more histogram types  
- the new dataframe must contain the same run and lumisection numbers (for each histogram type in it)  
as already present in the HistStruct, except if it is the first one to be added  
  
### add\_mask( self, name, mask )  
**add a mask to a HistStruct**  
input arguments:  
- name: a name for the mask  
- mask: a 1D np array of booleans  with same length as number of lumisections in HistStruct  
  
### remove\_mask( self, name )  
**inverse operation of add\_mask**  
  
### add\_json\_mask( self, name, jsondict )  
**add a mask corresponding to a json dict**  
  
### add\_goldenjson\_mask( self, name )  
**add a mask corresponding to the golden json file**  
  
### add\_dcsonjson\_mask( self, name )  
**add a mask corresponding to the DCS-bit on json file**  
  
### add\_hightstat\_mask( self, name, histnames=None, entries\_to\_bins\_ratio=100 )  
**add a mask corresponding to lumisections where all histograms have sufficient statistics**  
input arguments:  
- histnames: list of histogram names to take into account for making the mask (default: all in the HistStruct)  
- entries\_to\_bins\_ratio: criterion to determine if a histogram has sufficient statistics, number of entries divided by number of bins  
  
### get\_combined\_mask( self, names )  
**get a combined mask given multiple mask names**  
mostly for internal use; externally you can use get\_histograms( histname, <list of mask names>) directly  
  
### get\_runnbs( self, masknames=None )  
**get the array of run numbers, optionally after masking**  
  
### get\_lsnbs( self, masknames=None )  
**get the array of lumisection numbers, optionally after masking**  
  
### get\_scores( self, histname=None, masknames=None )  
**get the array of scores for a given histogram type, optionally after masking**  
if histname is None, return a dict matching histnames to arrays of scores  
  
### get\_histograms( self, histname=None, masknames=None )  
**get the array of histograms for a given type, optionally after masking**  
if histname is None, return a dict matching histnames to arrays of histograms  
  
### add\_classifier( self, histname, classifier, evaluate=False )  
**add a histogram classifier for a given histogram name to the HistStruct**  
classifier must be an object of type HistogramClassifier (i.e. of any class that derives from it)  
evaluate is a bool whether to evaluate the classifier (and store the result in the 'scores' attribute)  
  
### evaluate\_classifier( self, histname )  
**evaluate a histogram classifier for a given histogram name in the HistStruct**  
the result is both returned and stored in the 'scores' attribute  
  
### plot\_ls( self, run, ls, recohist=None, recolabel='reco', refhists=None, refhistslabel='reference', doprint=False)  
**plot the histograms for a given run/ls number versus their references and/or their reconstruction**  
  
### plot\_run( self, run, recohist=None, recolabel='reco', refhists=None, refhistslabel='reference', doprint=False)  
**call plot\_ls for all lumisections in a given run**  
  
