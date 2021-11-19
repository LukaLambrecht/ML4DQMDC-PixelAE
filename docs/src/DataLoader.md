# DataLoader  
  
- - -
## [class] DataLoader  
comments:  
```text  
class for loading histograms from disk  
the input usually consists of:  
- a csv file or a folder containing csv files in the correct format  
- a set of histogram names to load  
- a specification in terms of eras or years  
the output typically consists of pandas dataframes containing the requested histograms.  
```  
### &#10551; \_\_init\_\_  
full signature:  
```text  
def __init__( self )  
```  
comments:  
```text  
initializer  
```  
### &#10551; check\_year  
full signature:  
```text  
def check_year( self, year )  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; check\_eras  
full signature:  
```text  
def check_eras( self, eras, year )  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; check\_dim  
full signature:  
```text  
def check_dim( self, dim )  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; check\_eos  
full signature:  
```text  
def check_eos( self )  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; get\_default\_data\_dirs  
full signature:  
```text  
def get_default_data_dirs( self, year='2017', eras=[], dim=1 )  
```  
comments:  
```text  
get the default data directories for the data for this project  
note: this returns the directories where the data is currently stored;  
      might change in future reprocessings of the data,  
      and should be extended for upcoming Run-III data.  
note: default directories are on the /eos file system.  
      this function will throw an exception if it has not access to /eos.  
input arguments:  
- year: data-taking year, should be '2017' or '2018' so far (default: 2017)  
- eras: list of valid eras for the given data-taking year (default: all eras)  
- dim: dimension of requested histograms (1 or 2)  
```  
### &#10551; get\_csv\_files\_in\_dir  
full signature:  
```text  
def get_csv_files_in_dir( self, inputdir, sort=True )  
```  
comments:  
```text  
get a (sorted) list of csv files in a given input directory  
input arguments:  
- inputdir: directory to scan for csv files  
- sort: boolean whether to sort the files  
```  
### &#10551; get\_csv\_files\_in\_dirs  
full signature:  
```text  
def get_csv_files_in_dirs( self, inputdirs, sort=True )  
```  
comments:  
```text  
find the csv files in a set of input directories and return them in one list  
input arguments:  
- list of input directories where to look for csv files  
- sort: see get_csv_files_in_dir  
```  
### &#10551; get\_default\_csv\_files  
full signature:  
```text  
def get_default_csv_files( self, year='2017', eras=[], dim=1, sort=True )  
```  
comments:  
```text  
read the csv files from the default directories with input data for this project  
note: default directories are on the /eos file system.  
      this function will throw an exception if it has not access to /eos.  
input arguments:   
- year, eras, dim: see get_default_data_dirs!  
- sort: see get_csv_files_in_dir!  
```  
### &#10551; get\_dataframe\_from\_file  
full signature:  
```text  
def get_dataframe_from_file( self, csvfile, histnames=[] )  
```  
comments:  
```text  
load histograms from a given file  
```  
### &#10551; get\_dataframe\_from\_files  
full signature:  
```text  
def get_dataframe_from_files( self, csvfiles, histnames=[] )  
```  
comments:  
```text  
load histograms from a given set of files  
```  
### &#10551; write\_dataframe\_to\_file  
full signature:  
```text  
def write_dataframe_to_file( self, df, csvfile )  
```  
comments:  
```text  
write a dataframe to a csv file  
```  
- - -  
  
