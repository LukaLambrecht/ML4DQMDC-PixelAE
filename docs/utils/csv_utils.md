# csv utils  
  
### get\_data\_dirs(year='2017', eras=[], dim=1)  
```text  
yield all data directories  
note that the location of the data is hard-coded;  
this function might break for newer or later reprocessings of the data.  
- year is a string, either '2017' or '2018'  
- era is a list containing a selection of era names  
  (default empty list = all eras)  
- dim is either 1 or 2 (for 1D or 2D plots)  
```  
  
  
### get\_csv\_files(inputdir)  
```text  
yields paths to all csv files in input directory  
note that the output paths consist of input_dir/filename  
this function is only meant for 1-level down searching,  
i.e. the .csv files listed directly under input_dir.  
```  
  
  
### sort\_filenames(filelist)  
```text  
sort filenames in numerical order (e.g. 2 before 10)  
note that the number is supposed to be in ..._<number>.<extension> format  
```  
  
  
### read\_csv(csv\_file)  
```text  
read csv file into pandas dataframe  
csv_file is the path to the csv file to be read  
```  
  
  
### write\_csv(dataframe,csvfilename)  
```text  
write a dataframe to a csv file  
note: just a wrapper for builtin dataframe.to_csv  
```  
  
  
### read\_and\_merge\_csv(csv\_files, histnames=[], runnbs=[])  
```text  
read and merge list of csv files into a single df  
csv_files is a list of paths to files to merge into a df  
histnames is a list of the types of histograms to keep (default: all)  
runnbs is a list of run numbers to keep (default: all)  
```  
  
  
### write\_skimmed\_csv(histnames, year, eras=['all'], dim=1)  
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
  
  
