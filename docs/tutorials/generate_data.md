# generate data  
  
## Exploration of data generation ('resampling') methods

This notebook lists and plots some of the implemented methods for resampling histograms in order to artificially increase the statistics of training or testing sets.
```python
### imports

# external modules
import sys
import os
import matplotlib.pyplot as plt

# local modules
sys.path.append('../utils')
import hist_utils as hu
import dataframe_utils as dfu
import generate_data_utils as gdu
import plot_utils as pu
sys.path.append('../src')
import DataLoader
```
Output:
```text

```
```python
### load the data
# note: this cell assumes you have a csv file stored at the specified location,
#       containing only histograms of the specified type;
#       see the tutorial read_and_write_data for examples on how to create such files!

histname = 'chargeInner_PXLayer_2'
filename = 'DF2017_'+histname+'.csv'
datadir = '../data'

dloader = DataLoader.DataLoader()
df = dloader.get_dataframe_from_file( os.path.join(datadir, filename) )
print('raw input data shape: {}'.format( dfu.get_hist_values(df)[0].shape ))
allhists = hu.preparedatafromdf(df,donormalize=True)

# note: depending on which histogram you are looking at, the 'good' and 'bad' runs defined below might not be good or bad at all!
#       you will need to find a set of clearly good and bad runs for you type(s) of histogram.
goodrunsls = {'2017':
              {
                "297056":[[-1]],
                "297177":[[-1]],
                "301449":[[-1]]
              }} 

badrunsls = {'2017':
                {
                "297287":[[-1]],
                "297288":[[-1]],
                "297289":[[-1]],
                "299316":[[-1]],
                "299324":[[-1]],
                }}

goodhists = hu.preparedatafromdf(dfu.select_runsls(df,goodrunsls['2017']),donormalize=True)
badhists = hu.preparedatafromdf(dfu.select_runsls(df,badrunsls['2017']),donormalize=True)

# plot some together
pu.plot_sets([goodhists,badhists],colorlist=['b','r'],labellist=['"good" histograms','"bad" histograms'])
```
Output:
```text

```
```python
### also select a seed

seed = dfu.select_runsls(df,{"297056":[[100,100]]})
run = dfu.select_runs(df,[297056])
seedhist = hu.preparedatafromdf(seed,donormalize=True)
runhists = hu.preparedatafromdf(run,donormalize=True)

# plot some together
pu.plot_sets([runhists,seedhist],colorlist=['lightblue','k'],labellist=['histograms','seed histogram'])
```
Output:
```text

```
```python
### testing section for fourier_noise_on_mean

(reshists,_,_) = gdu.fourier_noise_on_mean(allhists, nresamples=10, nonnegative=True, doplot=True)
print('size of original set: {}'.format(allhists.shape))
print('size of resampled set: {}'.format(reshists.shape))
pu.plot_sets([hu.select_random(allhists, nselect=3), hu.select_random(reshists, nselect=3)],
             colorlist=['k','b'],
             labellist=['original histograms','resampled histograms'],
             transparencylist=[0.5,0.5])
```
Output:
```text

```
```python
### testing section for fourier_noise

(greshists,_,_) = gdu.fourier_noise(goodhists, nresamples=10, nonnegative=True, doplot=True)
(breshists,_,_) = gdu.fourier_noise(badhists, nresamples=9, nonnegative=True, stdfactor=3., doplot=True)
print('size of resampled good set: {}'.format(greshists.shape))
print('size of resampled bad set: {}'.format(breshists.shape))
pu.plot_sets([hu.select_random(greshists, nselect=100), hu.select_random(breshists, nselect=100)],
             colorlist=['b','r'],
             labellist=['resampled good histograms','resampled bad histograms'],
             transparencylist=[0.5,0.5])
```
Output:
```text

```
```python
### testing section for resample_bin_per_bin

(reshists,_,_) = gdu.resample_bin_per_bin(allhists, nresamples=10, nonnegative=True, smoothinghalfwidth=0, doplot=True)
print('size of original set: {}'.format(allhists.shape))
print('size of resampled set: {}'.format(reshists.shape))
pu.plot_sets([hu.select_random(allhists, nselect=3), hu.select_random(reshists, nselect=3)],
             colorlist=['k','b'],
             labellist=['original histograms','resampled histograms'],
             transparencylist=[0.5,0.5])
```
Output:
```text

```
```python
### testing section for resample_similar_bin_per_bin

(greshists,_,_) = gdu.resample_similar_bin_per_bin(allhists, goodhists, nresamples=3, nonnegative=True, keeppercentage=0.005, doplot=True)
(breshists,_,_) = gdu.resample_similar_bin_per_bin(allhists, badhists, nresamples=3, nonnegative=True, keeppercentage=0.003, doplot=True)
print('size of resampled good set: {}'.format(greshists.shape))
print('size of resampled bad set: {}'.format(breshists.shape))
pu.plot_sets([hu.select_random(greshists, nselect=100), hu.select_random(breshists, nselect=100)],
             colorlist=['b','r'],
             labellist=['resampled good histograms','resampled bad histograms'],
             transparencylist=[0.5,0.5])
```
Output:
```text

```
```python
### testing section for resample_similar_fourier_noise

(greshists,_,_) = gdu.resample_similar_fourier_noise(allhists, goodhists, nresamples=3, nonnegative=True, keeppercentage=0.001, doplot=True)
(breshists,_,_) = gdu.resample_similar_fourier_noise(allhists, badhists, nresamples=3, nonnegative=True, keeppercentage=0.001, doplot=True)
print('size of resampled good set: {}'.format(greshists.shape))
print('size of resampled bad set: {}'.format(breshists.shape))
pu.plot_sets([hu.select_random(greshists, nselect=100), hu.select_random(breshists, nselect=100)],
             colorlist=['b','r'],
             labellist=['resampled good histograms','resampled bad histograms'],
             transparencylist=[0.5,0.5])
```
Output:
```text

```
```python
### testing section for resample_similar_lico

(greshists,_,_) = gdu.resample_similar_lico(allhists,goodhists,nresamples=10,nonnegative=True,keeppercentage=0.1, doplot=True)
(breshists,_,_) = gdu.resample_similar_lico(allhists,badhists,nresamples=1,nonnegative=False,keeppercentage=0.001, doplot=True)
print('size of resampled good set: {}'.format(greshists.shape))
print('size of resampled bad set: {}'.format(breshists.shape))
pu.plot_sets([hu.select_random(greshists, nselect=100), hu.select_random(breshists, nselect=100)],
             colorlist=['b','r'],
             labellist=['resampled good histograms','resampled bad histograms'],
             transparencylist=[0.5,0.5])
```
Output:
```text

```
```python
### testing section for mc_sampling

(reshists,_,_) = gdu.mc_sampling(seedhist, nresamples=10, nMC=10000, doplot=True)
print('size of resampled set: {}'.format(reshists.shape))
pu.plot_sets([seedhist, hu.select_random(reshists, nselect=3)],
             colorlist=['k','b'],
             labellist=['original histogram','resampled histograms'],
             transparencylist=[0.5,0.5])
```
Output:
```text

```
```python
### testing section for white_noise

(greshists,_,_) = gdu.white_noise(goodhists, stdfactor=15, doplot=True)
(breshists,_,_) = gdu.white_noise(badhists, stdfactor=3., doplot=True)
print('size of resampled good set: {}'.format(greshists.shape))
print('size of resampled bad set: {}'.format(breshists.shape))
pu.plot_sets([hu.select_random(greshists, nselect=100), hu.select_random(breshists, nselect=100)],
             colorlist=['b','r'],
             labellist=['resampled good histograms','resampled bad histograms'],
             transparencylist=[0.5,0.5])
```
Output:
```text

```
```python

```
Output:
```text

```
```python

```
Output:
```text

```
