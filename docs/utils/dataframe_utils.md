# dataframe utils  
  
getter and selector for histogram names 
- - -
  
  
### get\_histnames(df)  
```text  
get a list of (unique) histogram names present in a df  
df is a dataframe read from an input csv file.  
```  
  
  
### select\_histnames(df, histnames)  
```text  
keep only a subset of histograms in a df  
histnames is a list of histogram names to keep in the df.  
```  
  
  
### get\_runs(df)  
```text  
return a list of (unique) run numbers present in a df  
df is a dataframe read from an input csv file.  
```  
  
  
### select\_runs(df, runnbs)  
```text  
keep only a subset of runs in a df  
runnbs is a list of run numbers to keep in the df.  
```  
  
  
### get\_ls(df)  
```text  
return a list of ls numbers present in a df  
note that the numbers are not required to be unique!  
note: no check is done on the run number!  
```  
  
  
### select\_ls(df, lsnbs)  
```text  
keep only a subset of lumisection numbers in a df  
lsnbs is a list of lumisection numbers to keep in the df.  
note: no check is done on the run number!  
```  
  
  
### get\_runsls(df)  
```text  
return a dictionary with runs and lumisections in a dataframe (same format as e.g. golden json)  
```  
  
  
### select\_json(df, jsonfile)  
```text  
keep only lumisections that are in the given json file  
```  
  
  
### select\_runsls(df, jsondict)  
```text  
equivalent to select_json but using a pre-loaded json dict instead of a json file on disk  
```  
  
  
### select\_golden(df)  
```text  
keep only golden lumisections in df  
```  
  
  
### select\_notgolden(df)  
```text  
keep all but golden lumisections in df  
```  
  
  
### select\_dcson(df)  
```text  
keep only lumisections in df that have DCS-bit on  
```  
  
  
### select\_dcsoff(df)  
```text  
keep only lumisections in df that have DCS-bit off  
```  
  
  
### select\_pixelgood(df)  
```text  
keep only lumisections in df that are in good pixel json  
```  
  
  
### select\_pixelbad(df)  
```text  
keep only lumisections in df that are in bad pixel json  
```  
  
  
### get\_highstat(df, entries\_to\_bins\_ratio=100)  
```text  
return a select object of runs and ls of histograms with high statistics  
```  
  
  
### select\_highstat(df, entries\_to\_bins\_ratio=100)  
```text  
keep only lumisection in df with high statistics  
```  
  
  
### get\_hist\_values(df)  
```text  
same as builtin "df['histo'].values" but convert strings to np arrays  
input arguments:  
- df: a dataframe containing histograms (assumed to be of a single type!)  
note: this function works for both 1D and 2D histograms,  
      the distinction is made based on whether or not 'Ybins' is present as a column in the dataframe  
      update: 'Ybins' is also present for 1D histograms, but has value 1!  
output:  
a tuple containing the following elements:  
- np array of shape (nhists,nbins) (for 1D) or (nhists,nybins,nxbins) (for 2D)  
- np array of run numbers of length nhists  
- np array of lumisection numbers of length nhists  
warning: no check is done to assure that all histograms are of the same type!  
```  
  
  
