# plot histograms  
  
**Plot the histograms for general investigation and visual inspection**  
Functionality for selecting a single run and plotting all lumisections belonging to that run.
Functionality for plotting the moments of the distributions as a function of LS number.
```python
### imports

# external modules
import sys
import os
import numpy as np
import importlib

# local modules
sys.path.append('../utils')
import dataframe_utils as dfu
import plot_utils as pu
import hist_utils as hu
importlib.reload(dfu)
importlib.reload(pu)
importlib.reload(hu)
sys.path.append('../src')
import DataLoader
importlib.reload(DataLoader)
```
Output:
```text

```
```python
### read the data
# note: this cell assumes you have a csv file stored at the specified location,
#       containing only histograms of the specified type;
#       see the tutorial read_and_write_data for examples on how to create such files!

histname = 'chargeInner_PXLayer_2'
filename = 'DF2017_'+histname+'.csv'
datadir = '../data'

dloader = DataLoader.DataLoader()
df = dloader.get_dataframe_from_file( os.path.join(datadir, filename) )
print('raw input data shape: {}'.format( dfu.get_hist_values(df)[0].shape ))

# select a single run
runs = dfu.get_runs(df)
print('number of runs: '+str(len(runs)))
#print(runs) # uncomment this to see a printed list of available runs
runnbs = [305351] # you can also add multiple runs to the list to plot them all together
df = dfu.select_runs(df,runnbs)

# select DCS-bit on data
#df = dfu.select_dcson(df)
```
Output:
```text

```
```python
# make a plot of the histograms
xmin = df.at[0,'Xmin']
xmax = df.at[0,'Xmax']
nbins = df.at[0,'Xbins']
(values,_,ls) = dfu.get_hist_values(df)
# (note: get_hist_values returns the raw histograms as stored in the dataframe;
# check out utils/hist_utils.py/preparedatafromdf for more advanced data loading, e.g. normalizing)
print('shape of histogram array: '+str(values.shape))
# just plot all the histograms:
pu.plot_hists_multi(values,xlims=(xmin,xmax))
# plot the histograms with a color according to their lumisection number:
pu.plot_hists_multi(values,colorlist=ls,xlims=(xmin,xmax))
# same as before but normalizing each histogram:
pu.plot_hists_multi(hu.normalizehists(values),colorlist=ls,xlims=(xmin,xmax))
```
Output:
```text

```
```python
# select a single lumisection and plot it on top of all the other lumisections
lsnumber = 869
pu.plot_anomalous(values,ls,highlight=lsnumber,hrange=10)
```
Output:
```text

```
```python
# make a plot of the moments of the (normalized) histograms
# use xmin = 0 and xmax = 1 as a kind of normalization
hists = values[:,1:-1]
bins = np.linspace(0,1,num=hists.shape[1],endpoint=True)
moments = hu.histmoments(bins,hists,orders=[1,2,3])
_ = pu.plot_moments(moments,ls,(0,1))
```
Output:
```text

```
```python

```
Output:
```text

```
