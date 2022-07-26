# plot histograms loop  
  
**Plot histograms in a loop, making a plot per run**  
This allows to relatively quickly scan all runs or a collection of runs for either general characteristics of the histograms under investigation or spot anomalies by eye
```python
### imports

# external modules
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import importlib

# internal modules
sys.path.append('../utils')
import hist_utils as hu
import dataframe_utils as dfu
import plot_utils as pu
import clustering_utils as cu
importlib.reload(hu)
importlib.reload(dfu)
importlib.reload(pu)
importlib.reload(cu)
sys.path.append('../src')
import DataLoader
importlib.reload(DataLoader)
```
Output:
```text

```
```python
# global settings
plot_type1 = True
plot_type2 = True
plot_moment = True

# read the data
# note: this cell assumes you have a csv file stored at the specified location,
#       containing only histograms of the specified type;
#       see the tutorial read_and_write_data for examples on how to create such files!
histname = 'chargeInner_PXLayer_2'
filename = 'DF2017_'+histname+'.csv'
datadir = '../data'

dloader = DataLoader.DataLoader()
df = dloader.get_dataframe_from_file( os.path.join(datadir, filename) )
print('raw input data shape: {}'.format( dfu.get_hist_values(df)[0].shape ))

# first select a set of reference histograms (for plot type 2)
# note: depending on the type of histogram you are looking at, the runs hard-coded below might not be good reference runs at all!
#       these runs are chosen just to show the principle of how to do a selection and make the plots shown below.
refhists = hu.preparedatafromdf(dfu.select_dcson(dfu.select_runs(df,[297056,297177,301449])),donormalize=True)

# filter the data
#df = dfu.select_golden(df)
#df = dfu.select_notgolden(df)
df = dfu.select_dcson(df)
#df = dfu.select_dcsoff(df)
print('filtered number of LS: '+str(len(df)))

# start loop over runs
runs = dfu.get_runs(df)
print('number of runs: '+str(len(runs)))
runs = runs[:10]
print('will loop over following runs: '+str(runs))
for run in runs:
    print('run '+str(run))
    dfr = dfu.select_runs(df,[run])
    # get histograms
    (hists,_,ls) = dfu.get_hist_values(dfr)
    # plot type 1
    if plot_type1:
        pu.plot_hists_multi(hists.tolist(),colorlist=ls)
        plt.show()
    # plot type 2
    if plot_type2:
        normhists = hu.normalizehists(hists)
        pu.plot_sets([refhists,normhists],colorlist=['blue','red'],labellist=['reference runs','this run'],transparencylist=[0.1,1.])
        plt.show()
    # get moments
    if plot_moment:
        nmoments = 3
        moments = np.zeros((len(hists),nmoments))
        xmin = 0. # some sort of normalization
        xmax = 1. # some sort of normalization
        nbins = hists.shape[1]
        binwidth = (xmax-xmin)/nbins
        bins = np.linspace(xmin+binwidth/2,xmax-binwidth/2,num=nbins,endpoint=True)
        for i in range(1,nmoments+1):
            moments[:,i-1] = hu.moment(bins,hists,i)
        pu.plot_moments(moments,ls,(0,1))
        dists = np.zeros(len(ls))
        for i in range(len(ls)):
            dists[i] = cu.avgnndist(moments,i,2)
        pu.plot_distance(dists,ls)
        plt.show()
```
Output:
```text

```
```python

```
Output:
```text

```
