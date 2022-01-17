# generate data 2d utils  
  
**Extension of generate\_data\_utils.py towards 2D histograms** 
- - -
  
  
### goodnoise\_nd  
full signature:  
```text  
def goodnoise_nd(shape, fstd=None, kmaxscale=0.25, ncomponents=3)  
```  
comments:  
```text  
generate one sample of 'good' noise consisting of fourier components  
generalization of goodnoise (see generate_data_utils) to arbitrary number of dimensions  
input args:  
- shape: a tuple, shape of the noise array to be sampled  
  note: in case of 1D, a comma is needed e.g. shape = (30,) else it will be automatically parsed to int and raise an error  
- fstd: an array of shape given by shape argument,   
  used for scaling of the amplitude of the noise bin-by-bin  
  (default: no scaling).  
- kmaxscale: scale factor to limit maximum frequency (lower kmaxscale means smoother noise)  
  note: can be a tuple with same length as shape, to scale differently in different dimensions.  
- ncomponents: number of random sines to add per dimension  
  note: can be a tuple with same length as shape, to use a different number of components in different dimensions.  
output:   
- numpy array of shape detailed by shape argument containing the noise  
```  
  
  
### whitenoise\_nd  
full signature:  
```text  
def whitenoise_nd(shape, fstd=None)  
```  
comments:  
```text  
generate one sample of white noise (standard normally distributed, uncorrelated between bins)  
generalization of whitenoise (see generate_data_utils) to arbitrary number of dimensions  
input args:  
- shape: a tuple, shape of the noise array to be sampled  
  note: in case of 1D, a comma is needed e.g. shape = (30,) else it will be automatically parsed to int and raise an error  
- fstd: an array of shape given by shape argument,   
  used for scaling of the amplitude of the noise bin-by-bin  
  (default: no scaling).  
output:   
- numpy array of shape detailed by shape argument containing the noise  
```  
  
  
### random\_lico\_nd  
full signature:  
```text  
def random_lico_nd(hists)  
```  
comments:  
```text  
generate one linear combination of histograms with random coefficients in (0,1) summing to 1.  
generalization of random_lico (see generate_data_utils) to arbitrary number of dimensions.  
input args:   
- numpy array of shape (nhists,<arbitrary number of additional dimensions>)  
output:  
- numpy array of shape (<same dimensions as input>), containing the new histogram  
```  
  
  
### fourier\_noise\_nd  
full signature:  
```text  
def fourier_noise_nd(hists, outfilename=None, doplot=False, ntarget=None, nresamples=1, nonnegative=True,  stdfactor=15., kmaxscale=0.25, ncomponents=3)  
```  
comments:  
```text  
apply fourier noise on random histograms with simple flat amplitude scaling.  
generalization of fourier_noise (see generate_data_utils) to arbitrary number of dimensions.  
input args:   
- hists: numpy array of shape (nhists,<arbitrary number of dimensions>) used for seeding  
- outfilename: path to csv file to write results to (default: no writing)  
- doplot: boolean whether to make a plot of some examples (only for 2D histograms!)  
- ntarget: total target number of histograms (default: use nresamples instead)  
- nresamples: number of samples to draw per input histogram  
  (note: ignored if ntarget is not None)  
- nonnegative: boolean whether to set all bins to minimum zero after applying noise  
- stdfactor: factor to scale magnitude of noise (larger factor = smaller noise)  
- kmaxscale and ncomponents: see goodnoise_nd  
```  
  
  
### white\_noise\_nd  
full signature:  
```text  
def white_noise_nd(hists, doplot=False, ntarget=None, nresamples=1, nonnegative=True, stdfactor=15.)  
```  
comments:  
```text  
apply white noise to the histograms in hists.  
generalization of white_noise (see generate_data_utils) to arbitrary number of dimensions.  
input args:  
- hists: np array (nhists,<arbitrary number of dimensions>) containing input histograms  
- doplot: boolean whether to plot some examples (only for 2D histograms!)  
- ntarget: total target number of histograms (default: use nresamples instead)  
- nresamples: number of samples to draw per input histogram  
  (note: ignored if ntarget is not None)  
- nonnegative: boolean whether to set all bins to minimum zero after applying noise  
- stdfactor: scaling factor of white noise amplitude (higher factor = smaller noise)  
```  
  
  
### resample\_lico\_nd  
full signature:  
```text  
def resample_lico_nd(hists, doplot=False, ntarget=None, nonnegative=True)  
```  
comments:  
```text  
take random linear combinations of input histograms  
generalization of fourier_noise (see generate_data_utils) to arbitrary number of dimensions.  
input args:   
- hists: numpy array of shape (nhists,<arbitrary number of dimensions>) used for seeding  
- doplot: boolean whether to plot some examples (only for 2D histograms!)  
- ntarget: total target number of histograms (default: same as number of input histograms)  
- nonnegative: boolean whether to set all bins to minimum zero after applying noise  
  note: coefficients in linear combination are always nonnegative,   
        so this setting is superfluous is input histograms are all nonnegative  
```  
  
  
