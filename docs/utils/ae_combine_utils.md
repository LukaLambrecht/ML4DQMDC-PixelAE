# ae combine utils  
  
- - -    
## histstructure  
(no valid documentation found)  
  
### __init__(self)  
(no valid documentation found)  
  
### create(self,year,histnames,jsonselector=None,highstatonly=False,dcsononly=False)  
**create the histstructure given the arguments provided**  
most arguments are self-explanatory  
remarks:  
- if jsonselector is None, no selection will be done on run/ls number, i.e. all runs and lumisections are kept  
- if jsonselector contains a single negative run number as key, templates will be used (e.g. averaging the dataset) instead of actual ls from the data  
for example, if jsonselector = {"-15":[[-1]]}, the dataset will be split in 15 parts and each part will be averaged to yield a single histogram (per type)  
  
### get_golden_mask(self)  
return a boolean mask on the lumisections whether or not they belong to the golden json  
  
### get_golden_indices(self)  
return an array of indices of lumisections that belong to the golden json  
  
### get_perrun_indices(self)  
return a list of arrays of indices of lumisections, one element in the list represents one run  
  
- - -    
## get_mse_array(histstruct,valkey,dims=[])  
(no valid documentation found)  
  
- - -    
## fitseminormal(histstruct,valkey,dims=[],fitnew=True,savefit=False)  
(no valid documentation found)  
  
- - -    
## fitgaussiankde(histstruct,valkey,dims=[],maxnpoints=-1)  
(no valid documentation found)  
  
- - -    
## plotfit2d(histstruct,valkey,dims,fitfunc,doinitialplot=True,onlycontour=False,rangestd=30)  
(no valid documentation found)  
  
- - -    
## msenormalizer  
(no valid documentation found)  
  
### __init__(self)  
(no valid documentation found)  
  
### fit(self,array)  
(no valid documentation found)  
  
### apply(self,array)  
(no valid documentation found)  
  
