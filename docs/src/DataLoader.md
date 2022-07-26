# DataLoader  
  
**Class for loading histograms from disk into a pandas dataframe.**  

Typically, the input consists of a single file per histogram type,  
prepared from the nanoDQMIO format (accessed via DAS) and converted into another file format.  
see the tools in the 'dqmio' folder for more info on preparing the input files.  
Currently supported input file formats:  

- csv  
- parquet  

Example usage:  
```  
from DataLoader import DataLoader  
dl = DataLoader()  
df = dl.get_dataframe_from_file( <path to input file> )  
```  

Alternatively, support is available to read the legacy per-LS csv files  
(deprecated approach for run-II data, before nanoDQMIO in run-III).  
In this case, the needed input consists of:  

- a set of histogram names to load  
- a specification in terms of eras or years  

Example usage:  
```
from DataLoader import DataLoader  
dl = DataLoader()  
csvfiles = dl.get_default_csv_files( year=<year>, dim=<histogram dimension> )  
df = dl.get_dataframe_from_files( csvfiles, histnames=<histogram names> )  
```

The output consists of a pandas dataframe containing the requested histograms.  
- - -
  
  
- - -
## [class] DataLoader  
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
initializer  
initialization of valid years and eras for which legacy csv files exist  
(note: only relevant for legacy csv approach, else empty initializer)  
```  
### &#10551; check\_year  
full signature:  
```text  
def check_year( self, year )  
```  
comments:  
```text  
check if a provided year is valid  
(note: only relevant for legacy csv approach)  
(note: internal helper function, no need to call)  
input arguments:  
- year: year in string format  
```  
### &#10551; check\_eras  
full signature:  
```text  
def check_eras( self, eras, year )  
```  
comments:  
```text  
check if a list of provided eras is valid  
(note: only relevant for legacy csv approach)  
(note: internal helper function, no need to call)  
input arguments:  
- eras: list of eras in string format, e.g. ['B','C']  
- year: year in string format  
```  
### &#10551; check\_dim  
full signature:  
```text  
def check_dim( self, dim )  
```  
comments:  
```text  
check if a histogram dimension is valid  
(note: only 1D and 2D histograms are supported for now)  
(note: internal helper function, no need to call)  
```  
### &#10551; check\_eos  
full signature:  
```text  
def check_eos( self )  
```  
comments:  
```text  
check if the /eos directory exists and is accessible  
(note: only relevant for legacy csv approach)  
(note: internal helper function, no need to call)  
```  
### &#10551; get\_default\_data\_dirs  
full signature:  
```text  
def get_default_data_dirs( self, year='2017', eras=[], dim=1 )  
```  
comments:  
```text  
get the default data directories for the data for this project  
(note: only relevant for legacy csv approach)  
(note: internal helper function, no need to call)  
note: this returns the directories where the data is currently stored;  
      might change in future reprocessings of the data,  
      and should be extended for upcoming Run-III data.  
note: default directories are on the /eos file system.  
      this function will throw an exception if it does not have access to /eos.  
input arguments:  
- year: data-taking year, should be '2017' or '2018' so far (default: 2017)  
- eras: list of valid eras for the given data-taking year (default: all eras)  
- dim: dimension of requested histograms (1 or 2)  
  note: need to provide the dimension at this stage since the files for 1D and 2D histograms  
        are stored in different directories.  
returns:  
a list of directories containing the legacy csv files with the requested data.  
```  
### &#10551; get\_csv\_files\_in\_dir  
full signature:  
```text  
def get_csv_files_in_dir( self, inputdir, sort=True )  
```  
comments:  
```text  
get a (optionally sorted) list of csv files in a given input directory  
(note: only relevant for legacy csv approach)  
(note: internal helper function, no need to call)  
input arguments:  
- inputdir: directory to scan for csv files  
- sort: boolean whether to sort the files  
returns:  
a list of csv files in the given directory.  
```  
### &#10551; get\_csv\_files\_in\_dirs  
full signature:  
```text  
def get_csv_files_in_dirs( self, inputdirs, sort=True )  
```  
comments:  
```text  
find the csv files in a set of input directories and return them in one list.  
(note: only relevant for legacy csv approach)  
(note: internal helper function, no need to call)  
this function simply loops over the directories given in inputdirs,  
calls get_csv_files_in_dir for each of them, and concatenates the results.  
input arguments:  
- list of input directories where to look for csv files  
- sort: see get_csv_files_in_dir  
returns:  
a list of csv files in the given directories.  
```  
### &#10551; get\_default\_csv\_files  
full signature:  
```text  
def get_default_csv_files( self, year='2017', eras=[], dim=1, sort=True )  
```  
comments:  
```text  
read the csv files from the default directories with input data for this project  
(note: only relevant for legacy csv approach)  
note: default directories are on the /eos file system.  
      this function will throw an exception if it has not access to /eos.  
input arguments:   
- year, eras, dim: see get_default_data_dirs!  
- sort: see get_csv_files_in_dir!  
returns:  
a list of csv files with the data corresponding to the provided year, eras and dimension.  
```  
### &#10551; get\_dataframe\_from\_file  
full signature:  
```text  
def get_dataframe_from_file( self, dfile, histnames=[], sort=True, verbose=True )  
```  
comments:  
```text  
load histograms from a given file into a dataframe  
input arguments:  
- dfile: file containing the data.  
  currently supported formats: csv, parquet.  
- histnames: list of histogram names to keep  
  (default: keep all histograms present in the input file).  
- sort: whether to sort the dataframe by run and lumisection number  
  (note: requires keys 'fromrun' and 'fromlumi' to be present in the dataframe).  
- verbose: whether to print info messages.  
returns:  
a pandas dataframe  
```  
### &#10551; get\_dataframe\_from\_files  
full signature:  
```text  
def get_dataframe_from_files( self, dfiles, histnames=[], sort=True, verbose=True )  
```  
comments:  
```text  
load histograms from a given set of files into a single dataframe  
input arguments:  
- dfiles: list of files containing the data.  
  currently supported formats: csv, parquet.  
- histnames: list of histogram names to keep  
  (default: keep all histograms present in the input file).  
- sort: whether to sort the dataframe by run and lumisection number  
  (note: requires keys 'fromrun' and 'fromlumi' to be present in the dataframe).  
- verbose: whether to print info messages.  
returns:  
a pandas dataframe  
```  
### &#10551; write\_dataframe\_to\_file  
full signature:  
```text  
def write_dataframe_to_file( self, df, dfile, overwrite=False, verbose=True )  
```  
comments:  
```text  
write a dataframe to a file  
input arguments:  
- df: a pandas dataframe.  
- dfile: file name to write.  
  currently supported formats: csv, parquet.  
- overwrite: whether to overwrite if a file with the given name already exists.  
- verbose: whether to print info messages.  
```  
- - -  
  
