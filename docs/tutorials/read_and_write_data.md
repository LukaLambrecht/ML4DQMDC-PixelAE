# read and write data  
  
**Basic reading and writing of csv files as a first data processing**  

This script has two functions:
   * Serve as an example on the usage of the DataLoader class.
   * Put the raw input files in a more workable format (see more info below).
   
This script starts from the raw csv files provided by central DQM as an ultimate input.  
These files are difficult to work with since they contain a fixed number of lines, not grouped by e.g. run number, and they contain a large number of histogram types together.  
This script (of which basically all the functionality is in the 'utils' folder, interfaced by the DataLoader class) puts them into a more useful form, i.e. one file per histogram type and per year, containing all runs and lumisections for that type for that year.  

It might be a good idea to run this code, where you change the histogram types to the ones that you intend to use in your study.  
Options are also available (although not shown in this small tutorial) to make files per era instead of per year, if you prefer that.

For more information, check the documentation of src/DataLoader, utils/csv_utils and utils/dataframe_utils! See also the comments in the code below for some more explanation.
```python
### imports

# external modules
import sys
import os
import importlib

# local modules
sys.path.append('../utils')
import csv_utils as csvu
import dataframe_utils as dfu
importlib.reload(csvu)
importlib.reload(dfu)
sys.path.append('../src')
import DataLoader
importlib.reload(DataLoader)
```
Output:
```text

```
```python
# find csv files for a given data-taking year and set of eras

# settings
year = '2017' # data-taking year
eras = ['B'] # list of eras
dim = 1 # dimension of histograms (1 or 2)

# create a DataLoader instance
dloader = DataLoader.DataLoader()

# get the default directories where the data are stored
# (this requires access to the /eos filesystem!)
datadirs = dloader.get_default_data_dirs( year=year, eras=eras, dim=dim )
print('some example data directories:')
print(datadirs[:10])

# get the csv files located in those directories
csvfiles = dloader.get_csv_files_in_dirs( datadirs )
print('number of csv files: {}'.format(len(csvfiles)))

# read an example csv file
csvfile = csvfiles[0]
df = dloader.get_dataframe_from_file(csvfile) 
# uncomment the following two lines to get a printout of the dataframe before any further processing.
# comment them back again to have a better view of the rest of the printouts in this cell.
print('example data frame:')
print(df)
```
Output:
```text

```
```python
# select a specific type of histogram

histname = 'chargeInner_PXLayer_1'
# option 1: use the already loaded dataframe
dftest = dfu.select_histnames(df, [histname])

# option 2: directly load only the needed histograms
df = dloader.get_dataframe_from_file( csvfile, histnames=[histname])

# compare the output
print(dftest)
print(df)
```
Output:
```text

```
```python
# do some printouts for the example dataframe loaded in previous cell

print('--- available runs present in this file: ---')
for r in dfu.get_runs(df): print(r)
print('--- available histogram types in this file ---')
for h in dfu.get_histnames(df): print(h)
```
Output:
```text

```
```python
# main reformatting of input csv files into smaller files,
# with one type of histogram each (for a full data-taking year)
# note: this cell can take quite a while to run!

importlib.reload(DataLoader)

# settings
outputdir = '../data_test'
histnames = ([
    'chargeInner_PXLayer_1'
])
year = '2017'
dim = 1

# load all input files
dloader = DataLoader.DataLoader()
csvfiles = dloader.get_default_csv_files( year=year, dim=dim )
df = dloader.get_dataframe_from_files( csvfiles, histnames=histnames )

# loop over histnames and write one file per histogram type
for histname in histnames:
    thisdf = dfu.select_histnames(df, [histname])
    outputfile = 'DF_'+year+'_'+histname+'.csv'
    dloader.write_dataframe_to_file( thisdf, os.path.join(outputdir,outputfile) )
```
Output:
```text

```
```python
### same as cell above, but now writing one file per era and per histogram type

# settings
outputdir = '../data_test'
histnames = ([
    'chargeInner_PXLayer_1'
])
year = '2017'
eras = ['B']
dim = 1

for era in eras:
    
    # load all input files
    dloader = DataLoader.DataLoader()
    csvfiles = dloader.get_default_csv_files( year=year, eras=[era], dim=dim )
    df = dloader.get_dataframe_from_files( csvfiles, histnames=histnames )

    # loop over histnames and write one file per histogram type
    for histname in histnames:
        thisdf = dfu.select_histnames(df, [histname])
        outputfile = 'DF_'+year+era+'_'+histname+'.csv'
        dloader.write_dataframe_to_file( thisdf, os.path.join(outputdir,outputfile) )
```
Output:
```text

```
```python
# extra: for 2D histograms, even the files per histogram type and per era might be too big to easily work with.
# this cell writes even smaller files for quicker testing

# settings
outputdir = '../data_test'
histname = 'clusterposition_zphi_ontrack_PXLayer_1'
year = '2017'
era = 'B'
dim = 2

dloader = DataLoader.DataLoader()
csvfiles = dloader.get_default_csv_files( year=year, eras=[era], dim=dim)
# just pick one (or a few) csv file(s)
csvfiles = [csvfiles[0]]
df = dloader.get_dataframe_from_files( csvfiles, histnames=[histname] )
outputfile = 'DF'+year+era+'subset_'+histname+'.csv'
dloader.write_dataframe_to_file( thisdf, os.path.join(outputdir,outputfile) )
```
Output:
```text

```
