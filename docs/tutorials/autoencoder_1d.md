# autoencoder 1d  
  
## Train and test an autoencoder on a set of 1D monitoring elements  

This notebook walks you through the basics of the autoencoder approach to detecting anomalies for 1D monitoring elements.  
It consists of the following steps:
   - Loading the data
   - Applying selections (e.g. DCS-bit on and sufficient statistics)
   - Preprocessing (e.g. normalizing)
   - Building an autoencoder model with keras
   - Investigate the output
```python
### imports

# external modules
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import load_model

# local modules
sys.path.append('../utils')
import dataframe_utils as dfu
import hist_utils as hu
import autoencoder_utils as aeu
import plot_utils as pu
import generate_data_utils as gdu
sys.path.append('../src')
import DataLoader
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
```
Output:
```text

```
```python
### filtering: select only DCS-bit on data and filter out low statistics

df = dfu.select_dcson(df)
print('number of passing lumisections after DCS selection: {}'.format( len(df) ))

df = dfu.select_highstat(df, entries_to_bins_ratio=100)
print('number of passing lumisections after high statistics selection: {}'.format( len(df) ))
```
Output:
```text

```
```python
### preprocessing of the data: rebinning and normalizing

rebinningfactor = 1

X_train = hu.preparedatafromdf(df, rebinningfactor=rebinningfactor,
                               donormalize=True, doplot=True)
(ntrain,nbins) = X_train.shape
print('size of training set: '+str(X_train.shape))
```
Output:
```text

```
```python
### build the model and train it

input_size = X_train.shape[1]
arch = [int(X_train.shape[1]/2.)]
act = ['tanh']*len(arch)
opt = 'adam'
loss = aeu.mseTop10
autoencoder = aeu.getautoencoder(input_size,arch,act,opt,loss) 
history = autoencoder.fit(X_train, X_train, epochs=20, batch_size=500, shuffle=False, verbose=1, validation_split=0.1)
pu.plot_loss(history, title = 'model loss')
```
Output:
```text

```
```python
### evaluate the model on the training set

prediction_train = autoencoder.predict(X_train)
mse_train = aeu.mseTop10Raw(X_train, prediction_train)
```
Output:
```text

```
```python
### plot the global MSE trend

pu.plot_mse(mse_train, rmlargest=0.005)
(mean,std) = pu.plot_mse(mse_train, doplot=False, rmlargest=0.005)
print('mean mse: {}'.format(mean))
print('std mse: {}'.format(std))
```
Output:
```text

```
```python
### impose a mse upper boundary and plot random examples of passing and failing histograms
# note: at this point, only the training set is considered!
# for a test set: see cell below.

cutvalue = mean + 3*std
print('The mse threshold is: '+str(cutvalue))
goodindices = np.arange(0,len(mse_train))[mse_train<cutvalue]
badindices = np.arange(0,len(mse_train))[mse_train>cutvalue]

print('Number of passing histograms: '+str(len(goodindices)))
print('Number of failing histograms: '+str(len(badindices)))

nplot = 5
print('examples of good histograms and reconstruction:')
randint = np.random.choice(goodindices,size=nplot,replace=False)
for i in randint: 
    histlist = [X_train[int(i),:],prediction_train[int(i),:]]
    labellist = ['data','reconstruction']
    colorlist = ['black','blue']
    pu.plot_hists(histlist,colorlist=colorlist,labellist=labellist)
    plt.show()

print('examples of bad histograms and reconstruction:')
randint = np.random.choice(badindices,size=nplot,replace=False)
for i in randint:
    histlist = [X_train[int(i),:],prediction_train[int(i),:]]
    labellist = ['data','reconstruction']
    colorlist = ['black','blue']
    pu.plot_hists(histlist,colorlist=colorlist,labellist=labellist)
    plt.show()
```
Output:
```text

```
```python
### get a test set and evaluate the model

goodrunsls = { "297056":[[-1]],
                "297177":[[-1]],
                "301449":[[-1]] 
             }
badrunsls = {
                "297287":[[-1]],
                "297288":[[-1]],
                "297289":[[-1]],
                "299316":[[-1]],
                "299317":[[-1]],
                "299318":[[-1]],
                "299324":[[-1]],
            }

# re-read the dataframe
# (in case the selections are different than for the training set)
dloader = DataLoader.DataLoader()
df = dloader.get_dataframe_from_file( os.path.join(datadir, filename) )
df = dfu.select_dcson(df)
df = dfu.select_highstat(df,entries_to_bins_ratio=100)

# good histograms option 1: predefined runs/lumisections
#X_test_good = hu.preparedatafromdf( dfu.select_runsls(df,goodrunsls),donormalize=True )
# good histograms option 2: averages of total set
X_test_good = hu.averagehists( hu.preparedatafromdf(df, donormalize=True), 15 )
# bad histograms: predefined runs/lumisections
(X_test_bad, runnbs_bad,lsnbs_bad) = hu.preparedatafromdf( 
                                    dfu.select_runsls(df,badrunsls),
                                    donormalize=True,
                                    returnrunls = True )
print('shape of good test set: {}'.format(X_test_good.shape))
print('shape of bad test set: {}'.format(X_test_bad.shape))

pu.plot_sets([X_test_good,X_test_bad],colorlist=['b','r'],
             labellist=['Histograms in test set labeled "good"','Histograms in test set labeled "bad"'])
plt.show()

prediction_test_good = autoencoder.predict(X_test_good)
mse_test_good = aeu.mseTopNRaw(X_test_good, prediction_test_good, n=10 )
prediction_test_bad = autoencoder.predict(X_test_bad)
mse_test_bad = aeu.mseTopNRaw(X_test_bad, prediction_test_bad, n=10 )

print('average mse on good set: '+str(np.mean(mse_test_good)))
print('average mse on bad set: '+str(np.mean(mse_test_bad)))

nplot = 10
print('examples of good histograms and reconstruction:')
randint = np.random.choice(np.arange(len(X_test_good)),size=nplot,replace=False)
for i in randint: 
    histlist = [X_test_good[int(i),:],prediction_test_good[int(i),:]]
    labellist = ['data','reconstruction']
    colorlist = ['black','blue']
    pu.plot_hists(histlist,colorlist=colorlist,labellist=labellist)
    plt.show()

print('examples of bad histograms and reconstruction:')
randint = np.random.choice(np.arange(len(X_test_bad)),size=nplot,replace=False)
for i in randint:
    histlist = [X_test_bad[int(i),:],prediction_test_bad[int(i),:]]
    labellist = ['data','reconstruction']
    colorlist = ['black','blue']
    pu.plot_hists(histlist,colorlist=colorlist,labellist=labellist)
    plt.show()
```
Output:
```text

```
```python
### use artificial data to assess the model performance

(goodhists,_,_) = gdu.upsample_hist_set( X_test_good, ntarget=5e3, fourierstdfactor=20., doplot=True )
(badhists,_,_) = gdu.upsample_hist_set( X_test_bad, ntarget=5e3, fourierstdfactor=20., doplot=True )
print('number of good histograms: '+str(len(goodhists)))
print('number of bad histograms: '+str(len(badhists)))

validation_data = np.vstack((goodhists,badhists))
labels = np.hstack((np.zeros(len(goodhists)),np.ones(len(badhists))))
prediction = autoencoder.predict(validation_data)
mse = aeu.mseTopNRaw(validation_data, prediction, n=10 )
shuffled_indices = np.arange(len(validation_data))
_ = np.random.shuffle(shuffled_indices)
validation_data = validation_data[shuffled_indices]
labels = labels[shuffled_indices]
prediction = prediction[shuffled_indices]
mse = mse[shuffled_indices]

# distribution of output scores
pu.plot_score_dist(mse, labels, 
                   siglabel='anomalous', sigcolor='r',
                   bcklabel='good', bckcolor='g',
                   nbins=200, normalize=True)
print('minimum mse on bad set: {}'.format(np.amin(mse[np.where(labels==1)])))
print('maximum mse on good set: {}'.format(np.amax(mse[np.where(labels==0)])))
# classical ROC curve: signal efficiency (good data marked as good) vs background efficiency (bad data marked as good)
auc = aeu.get_roc(mse, labels, npoints=500, bootstrap_samples=100)
```
Output:
```text

```
```python
### continution of previous cell: choose wp and plot confusion matrix

aeu.get_confusion_matrix_from_hists( validation_data, labels, prediction, msewp='maxauc' )
```
Output:
```text

```
```python
### plot some histograms in the bad test set with their reconstruction

inds = np.random.choice( np.arange(len(lsnbs_bad)), 10, replace=False )
for i in inds:
    runnb = runnbs_bad[i]
    lsnb = lsnbs_bad[i]
    histogram = X_test_bad[i:i+1,:]
    reco = autoencoder.predict(histogram)
    mse = aeu.mseTopNRaw(histogram, reco, n=10 )
    pu.plot_sets([histogram,reco],
                    labellist=['hist {}/{}'.format(runnb,lsnb),'reco'],
                    colorlist=['black','red'],
                    )
    plt.show()
    print('MSE: {}'.format(mse))
```
Output:
```text

```
```python
### plot some histograms in the good test set with their reconstruction
# note: depends on whether the good test set was obtained from real lumisections,
#       or from averages from entire set.

inds = np.random.choice( np.arange(len(X_test_good)), 10, replace=False )
for i in inds:
    try:
        runnb = runnbs_good[i]
        lsnb = lsnbs_good[i]
        histlabel = 'hist {}/{}'.format(runnb,lsnb)
    except:
        runnb = 0
        lsnb = 0
        histlabel = 'hist (artificial)'
    histogram = X_test_good[i:i+1,:]
    reco = autoencoder.predict(histogram)
    mse = aeu.mseTopNRaw(histogram, reco, n=10 )
    pu.plot_sets([histogram,reco],
                    labellist=[histlabel,'reco'],
                    colorlist=['black','red'],
                    )
    plt.show()
    print('MSE: {}'.format(mse))
```
Output:
```text

```
```python

```
Output:
```text

```
