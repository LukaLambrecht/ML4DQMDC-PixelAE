# DataLoader  
  
- - -
## [class] DataLoader(object)  
```text  
class for loading histograms from disk  
the input usually consists of:  
- a csv file or a folder containing csv files in the correct format  
- a set of histogram names to load  
- a specification in terms of eras or years  
the output typically consists of pandas dataframes containing the requested histograms.  
```  
### &#10551; \_\_init\_\_( self )  
```text  
initializer  
```  
### &#10551; check\_year( self, year )  
```text  
(no valid documentation found)  
```  
### &#10551; check\_eras( self, eras, year )  
```text  
(no valid documentation found)  
```  
### &#10551; check\_dim( self, dim )  
```text  
(no valid documentation found)  
```  
### &#10551; check\_eos( self )  
```text  
(no valid documentation found)  
```  
### &#10551; get\_default\_data\_dirs( self, year='2017', eras=[], dim=1 )  
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
### &#10551; get\_csv\_files\_in\_dir( self, inputdir, sort=True )  
```text  
get a (sorted) list of csv files in a given input directory  
input arguments:  
- inputdir: directory to scan for csv files  
- sort: boolean whether to sort the files  
```  
### &#10551; get\_csv\_files\_in\_dirs( self, inputdirs, sort=True )  
```text  
find the csv files in a set of input directories and return them in one list  
input arguments:  
- list of input directories where to look for csv files  
- sort: see get_csv_files_in_dir  
```  
### &#10551; get\_default\_csv\_files( self, year='2017', eras=[], dim=1, sort=True )  
```text  
read the csv files from the default directories with input data for this project  
note: default directories are on the /eos file system.  
      this function will throw an exception if it has not access to /eos.  
input arguments:   
- year, eras, dim: see get_default_data_dirs!  
- sort: see get_csv_files_in_dir!  
```  
### &#10551; get\_dataframe\_from\_file( self, csvfile, histnames=[] )  
```text  
load histograms from a given file  
```  
### &#10551; get\_dataframe\_from\_files( self, csvfiles, histnames=[] )  
```text  
load histograms from a given set of files  
```  
### &#10551; write\_dataframe\_to\_file( self, df, csvfile )  
```text  
write a dataframe to a csv file  
```  
- - -  
  
