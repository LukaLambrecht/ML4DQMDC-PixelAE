# dataframe utils  
  
**A collection of useful basic functions for manipulating pandas dataframes.**  

Functionality includes (among others):
- selecting DCS-bit on data or golden json data.
- selecting specific runs, lumisections, or types of histograms
- - -
  
  
### get\_histnames  
full signature:  
```text  
def get_histnames(df)  
```  
comments:  
```text  
get a list of (unique) histogram names present in a df  
df is a dataframe read from an input csv file.  
```  
  
  
### select\_histnames  
full signature:  
```text  
def select_histnames(df, histnames)  
```  
comments:  
```text  
keep only a subset of histograms in a df  
histnames is a list of histogram names to keep in the df.  
```  
  
  
### get\_runs  
full signature:  
```text  
def get_runs(df)  
```  
comments:  
```text  
return a list of (unique) run numbers present in a df  
df is a dataframe read from an input csv file.  
```  
  
  
### select\_runs  
full signature:  
```text  
def select_runs(df, runnbs)  
```  
comments:  
```text  
keep only a subset of runs in a df  
runnbs is a list of run numbers to keep in the df.  
```  
  
  
### get\_ls  
full signature:  
```text  
def get_ls(df)  
```  
comments:  
```text  
return a list of ls numbers present in a df  
note that the numbers are not required to be unique!  
note: no check is done on the run number!  
```  
  
  
### select\_ls  
full signature:  
```text  
def select_ls(df, lsnbs)  
```  
comments:  
```text  
keep only a subset of lumisection numbers in a df  
lsnbs is a list of lumisection numbers to keep in the df.  
note: no check is done on the run number!  
```  
  
  
### get\_runsls  
full signature:  
```text  
def get_runsls(df)  
```  
comments:  
```text  
return a dictionary with runs and lumisections in a dataframe (same format as e.g. golden json)  
```  
  
  
### select\_json  
full signature:  
```text  
def select_json(df, jsonfile)  
```  
comments:  
```text  
keep only lumisections that are in the given json file  
```  
  
  
### select\_runsls  
full signature:  
```text  
def select_runsls(df, jsondict)  
```  
comments:  
```text  
equivalent to select_json but using a pre-loaded json dict instead of a json file on disk  
```  
  
  
### select\_golden  
full signature:  
```text  
def select_golden(df)  
```  
comments:  
```text  
keep only golden lumisections in df  
```  
  
  
### select\_notgolden  
full signature:  
```text  
def select_notgolden(df)  
```  
comments:  
```text  
keep all but golden lumisections in df  
```  
  
  
### select\_dcson  
full signature:  
```text  
def select_dcson(df)  
```  
comments:  
```text  
keep only lumisections in df that have DCS-bit on  
```  
  
  
### select\_dcsoff  
full signature:  
```text  
def select_dcsoff(df)  
```  
comments:  
```text  
keep only lumisections in df that have DCS-bit off  
```  
  
  
### select\_pixelgood  
full signature:  
```text  
def select_pixelgood(df)  
```  
comments:  
```text  
keep only lumisections in df that are in good pixel json  
```  
  
  
### select\_pixelbad  
full signature:  
```text  
def select_pixelbad(df)  
```  
comments:  
```text  
keep only lumisections in df that are in bad pixel json  
```  
  
  
### get\_highstat  
full signature:  
```text  
def get_highstat(df, entries_to_bins_ratio=100)  
```  
comments:  
```text  
return a select object of runs and ls of histograms with high statistics  
```  
  
  
### select\_highstat  
full signature:  
```text  
def select_highstat(df, entries_to_bins_ratio=100)  
```  
comments:  
```text  
keep only lumisection in df with high statistics  
```  
  
  
### get\_hist\_values  
full signature:  
```text  
def get_hist_values(df)  
```  
comments:  
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
  
  
