# autoencoder iterative  
  
## Train and test an autoencoder iteratively on local datasets

This notebook investigates the possibility to perform local autoencoder training, i.e. training on a small number of runs instead of training on a large dataset (e.g. a full year of data taking).  
This notebook consists of three parts:
   - Reading and preparing the data (common to part 2 and 3) 
   - Train an autoencoder on the first 5, 10, 15 etc. runs of 2017 data taking.  
   - Choose a random run to test on, use 5 previous runs for training. This is a first step naive attempt towards using dedicated reference runs for training.
### Part 1: Reading and preparing the data
```python
### imports

# external modules
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# local modules
sys.path.append('../utils')
import dataframe_utils as dfu
import hist_utils as hu
import plot_utils as pu
import autoencoder_utils as aeu
import generate_data_utils as gdu
sys.path.append('../src')
import DataLoader
```
Output:
```text

```
```python
### read the data and perform some selections
# note: this cell assumes you have a csv file stored at the specified location,
#       containing only histograms of the specified type;
#       see the tutorial read_and_write_data for examples on how to create such files!

histname = 'chargeInner_PXLayer_2'
filename = 'DF2017_'+histname+'.csv'
datadir = '../data'

dloader = DataLoader.DataLoader()
df = dloader.get_dataframe_from_file( os.path.join(datadir, filename) )
print('raw input data shape: {}'.format( dfu.get_hist_values(df)[0].shape ))
df = dfu.select_dcson(df)
df = dfu.select_highstat(df)
print('number of passing lumisections after selection: {}'.format( len(df) ))
runs_all = dfu.get_runs(df)
(hists_all,runnbs_all,lsnbs_all) = hu.preparedatafromdf(df,returnrunls=True,rebinningfactor=1,donormalize=True)
print('shape of histogram array: {}'.format(hists_all.shape))
```
Output:
```text

```
### Part 2: Updating the training set
```python
### get a test set

goodrunsls = {'2017':
              {
                "297056":[[-1]],
              }}

badrunsls = {'2017':
            {
                "297287":[[-1]],
                "297288":[[-1]],
                "297289":[[-1]],
                "299316":[[-1]],
                "299324":[[-1]],
            }}

# select the correct data-taking year relevant for the file chosen above
year = '2017'

# load good and bad sets from df
(hists_good,runnbs_good,lsnbs_good) = hu.preparedatafromdf( 
                                        dfu.select_runsls(df,goodrunsls[year]),
                                        returnrunls=True, donormalize=True)
(hists_bad,runnbs_bad,lsnbs_bad) = hu.preparedatafromdf( 
                                        dfu.select_runsls(df,badrunsls[year]),
                                        returnrunls=True, donormalize=True)
print('shape of good test set '+str(hists_good.shape))
print('shape of bad test set '+str(hists_bad.shape))

# make plot
pu.plot_sets([hists_good,hists_bad],
             colorlist=['b','r'],
             labellist=['good','bad'],
             transparencylist=[],
             xlims=(0,-1))

# use resampling tool to upsample and add more variation
(hists_good,_,_) = gdu.upsample_hist_set(hists_good,ntarget=2e3,fourierstdfactor=15., doplot=False)
(hists_bad,_,_) = gdu.upsample_hist_set(hists_bad,ntarget=2e3,fourierstdfactor=5., doplot=False)
print('shape of good test set '+str(hists_good.shape))
print('shape of bad test set '+str(hists_bad.shape))
# make plot

pu.plot_sets([hists_good,hists_bad],
             colorlist=['b','r'],
             labellist=['good','bad'],
             transparencylist=[0.1,0.1],
             xlims=(0,-1))
```
Output:
```text

```
```python
### function to purify training set by removing a given fraction of high mse histograms

def purify_training_set(hists,model,rmfraction):
    mse = aeu.mseTop10Raw(hists,model.predict(hists))
    threshold = np.quantile(mse,1-rmfraction)
    keepindices = np.where(mse<threshold)
    return hists[keepindices]

### functions to test performance on test set

def test_autoencoder(hists_good,hists_bad,model):
    mse_good = aeu.mseTop10Raw(hists_good,model.predict(hists_good))
    mse_bad = aeu.mseTop10Raw(hists_bad,model.predict(hists_bad))
    labels_good = np.zeros(len(mse_good))
    labels_bad = np.ones(len(mse_bad))

    labels = np.concatenate(tuple([labels_good,labels_bad]))
    scores = np.concatenate(tuple([mse_good,mse_bad]))
    maxnoninf = np.max(np.where(scores==np.inf,np.min(scores),scores))
    scores = np.where(scores==np.inf,maxnoninf,scores)

    auc = aeu.get_roc(scores, labels, mode='full', bootstrap_samples=100)
    plt.show()

def plot_examples(hists_good,hists_bad,model):
    # set parameters
    nexamples = 6
    fig,axs = plt.subplots(2,nexamples,figsize=(24,12))
    inds_good = np.random.choice(range(len(hists_good)),nexamples)
    inds_bad = np.random.choice(range(len(hists_bad)),nexamples)
    inds_ref = np.random.choice(range(len(hists_good)),20)
    # determine whether to show run/lumi number in label (not possible when using resampled sets)
    truelabel = True
    if( len(hists_good)!=len(runnbs_good) or len(hists_bad)!=len(runnbs_good) ): truelabel = False
    # plot examples
    for i in range(nexamples):
        hist_good = hists_good[inds_good[i]:inds_good[i]+1]
        reco_good = model.predict(hist_good)
        hist_bad = hists_bad[inds_bad[i]:inds_bad[i]+1]
        reco_bad = model.predict(hist_bad)
        hist_good_label = hist_bad_label = 'hist'
        if truelabel: 
            hist_good_label += ' (run: '+str(int(runnbs_good[inds_good[i]]))+', ls: '+str(int(lsnbs_good[inds_good[i]]))+')'
            hist_bad_label += ' (run: '+str(int(runnbs_bad[inds_bad[i]]))+', ls: '+str(int(lsnbs_bad[inds_bad[i]]))+')'
        pu.plot_sets([hist_good,reco_good,hists_good[inds_ref]],
                  fig=fig,ax=axs[0,i],
                  title='',
                  colorlist=['black','red','blue'],
                  labellist=[hist_good_label,'reco','good hists'],
                  transparencylist=[1.,1.,0.1])
        pu.plot_sets([hist_bad,reco_bad,hists_good[inds_ref]],
                  fig=fig,ax=axs[1,i],
                  title='',
                  colorlist=['black','red','blue'],
                  labellist=[hist_bad_label,'reco','good hists'],
                  transparencylist=[1.,1.,0.1])
    plt.show()
```
Output:
```text

```
```python
### iterate over growing amount of data

nruns = [5,10]

# first iteration manually
X_train = hists_all[np.where(runnbs_all<runs_all[nruns[0]])]
print('size of training set (intial): '+str(len(X_train)))
(X_train_ext,_,_) = gdu.upsample_hist_set(X_train, 1e5)
#X_train_ext = X_train
model = aeu.train_simple_autoencoder(X_train_ext,nepochs=10)
print('evaluating model on test set')
test_autoencoder(hists_good,hists_bad,model)
plot_examples(hists_good,hists_bad,model)

# next iterations in a loop
for i in range(1,len(nruns)):
    
    this_upperbound = runs_all[nruns[i]]
    this_lowerbound = runs_all[nruns[i-1]]
    print('adding runs {} to {}'.format(this_lowerbound,this_upperbound))
    print('(training on {} runs in total)'.format(nruns[i]))
    newhists = hists_all[np.where( (runnbs_all<this_upperbound) & (runnbs_all>=this_lowerbound) )]
    print('number of new histograms added to training set: '+str(len(newhists)))
    X_train = np.concatenate( (X_train,newhists), axis=0 )
    X_train = purify_training_set(X_train,model,0.1)
    print('size of training set (intial): '+str(len(X_train)))
    (X_train_ext,_,_) = gdu.upsample_hist_set(X_train, 1e5)
    #X_train_ext = X_train
    model = aeu.train_simple_autoencoder(X_train_ext)
    print('size of training set (after training and purifying): '+str(len(X_train)))
    print('evaluating model on test set')
    test_autoencoder(hists_good,hists_bad,model)
    plot_examples(hists_good,hists_bad,model)
```
Output:
```text

```
### Part 3: Local training on nearby runs
```python
### choose random run for testing, train on previous runs
# (testing: see next cell)

# choose runs
print('number of available runs: '+str(len(runs_all)))
runindex = np.random.choice(range(5,len(runs_all)))
#runindex = runs_all.index(305364)
test_run = runs_all[runindex]
print('chosen run index: '+str(runindex)+', corresponding to run: '+str(test_run))
training_runs = runs_all[runindex-5:runindex]
print('runs used for training: '+str(training_runs))

# get the data
df = dfu.select_dcson(df)
(hists_train,runnbs_train,lsnbs_train) = hu.preparedatafromdf( dfu.select_runs(df,training_runs),returnrunls=True,donormalize=True)
(hists_test,runnbs_test,lsnbs_test) = hu.preparedatafromdf( dfu.select_runs(df,[test_run]),returnrunls=True,donormalize=True)
print('shape of training set '+str(hists_train.shape))
print('shape of test set '+str(hists_test.shape))
# make plot
pu.plot_sets([hists_train,hists_test],ax=None,title='',colorlist=['b','g'],labellist=['train','test'],transparencylist=[0.5,0.5],xlims=(0,-1))
plt.show()

# train the autoencoder
(X_train,_,_) = gdu.upsample_hist_set(hists_train, 1e5)
#X_train = hists_train
model = aeu.train_simple_autoencoder(X_train)
```
Output:
```text

```
```python
### test the autoencoder trained in the previous cell

# note: do not simply use hists_good for a good test set, 
# as the model might not be trained on all shape variations that are supposed to be 'good',
# resulting in artificially bad performance.
# we can use hists_test instead, but it is not at all guaranteed that all of them are good.
# hence the label 'good hists' in the plot below is not necessarily correct.

test_autoencoder(hists_test,hists_bad,model)
plot_examples(hists_test,hists_bad,model)
```
Output:
```text

```
```python

```
Output:
```text

```
