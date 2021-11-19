# csv utils  
  
**A collection of useful basic functions for reading and processing the input csv files.**  

Functionality includes:
- reading the raw input csv files and producing more manageable csv files (grouped per histogram type).
- reading csv files into pandas dataframes and writing pandas dataframes back to csv files.

**Note: the functionality of these utils has been absorbed into the DataLoader class, which is now the recommended way to read the data!**
- - -
  
  
### get\_data\_dirs  
full signature:  
```text  
def get_data_dirs(year='2017', eras=[], dim=1)  
```  
comments:  
```text  
yield all data directories  
note that the location of the data is hard-coded;  
this function might break for newer or later reprocessings of the data.  
- year is a string, either '2017' or '2018'  
- era is a list containing a selection of era names  
  (default empty list = all eras)  
- dim is either 1 or 2 (for 1D or 2D plots)  
```  
  
  
### get\_csv\_files  
full signature:  
```text  
def get_csv_files(inputdir)  
```  
comments:  
```text  
yields paths to all csv files in input directory  
note that the output paths consist of input_dir/filename  
this function is only meant for 1-level down searching,  
i.e. the .csv files listed directly under input_dir.  
```  
  
  
### sort\_filenames  
full signature:  
```text  
def sort_filenames(filelist)  
```  
comments:  
```text  
sort filenames in numerical order (e.g. 2 before 10)  
note that the number is supposed to be in ..._<number>.<extension> format  
```  
  
  
### read\_csv  
full signature:  
```text  
def read_csv(csv_file)  
```  
comments:  
```text  
read csv file into pandas dataframe  
csv_file is the path to the csv file to be read  
```  
  
  
### write\_csv  
full signature:  
```text  
def write_csv(dataframe,csvfilename)  
```  
comments:  
```text  
write a dataframe to a csv file  
note: just a wrapper for builtin dataframe.to_csv  
```  
  
  
### read\_and\_merge\_csv  
full signature:  
```text  
def read_and_merge_csv(csv_files, histnames=[], runnbs=[])  
```  
comments:  
```text  
read and merge list of csv files into a single df  
csv_files is a list of paths to files to merge into a df  
histnames is a list of the types of histograms to keep (default: all)  
runnbs is a list of run numbers to keep (default: all)  
```  
  
  
### write\_skimmed\_csv  
full signature:  
```text  
def write_skimmed_csv(histnames, year, eras=['all'], dim=1)  
```  
comments:  
```text  
read all available data for a given year/era and make a file per histogram type  
input arguments:  
- histnames: list of histogram names for which to make a separate file  
- year: data-taking year (in string format)  
- eras: data-taking eras for which to make a separate file (in string format)  
        use 'all' to make a file with all eras merged, i.e. a full data taking year  
- dim: dimension of histograms (1 or 2), needed to retrieve the correct folder containing input files  
output:  
- one csv file per year/era and per histogram type  
note: this function can take quite a while to run!  
```  
  
  
