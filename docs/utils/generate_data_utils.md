# generate data utils  
  
**A collection of functions for artificially creating a labeled dataset.**  

See the function documentation below for more details on the implemented methods.  
Also check the tutorial generate\_data.ipynb for examples!
- - -
  
  
### goodnoise  
full signature:  
```text  
def goodnoise(nbins, fstd=None)  
```  
comments:  
```text  
generate one sample of 'good' noise consisting of fourier components  
input args:  
- nbins: number of bins, length of noise array to be sampled  
- fstd: an array of length nbins used for scaling of the amplitude of the noise  
        bin-by-bin.  
output:   
- numpy array of length nbins containing the noise  
```  
  
  
### badnoise  
full signature:  
```text  
def badnoise(nbins, fstd=None)  
```  
comments:  
```text  
generate one sample of 'bad' noise consisting of fourier components  
(higher frequency and amplitude than 'good' noise)  
input args and output: simlar to goodnoise  
WARNING: NOT NECESSARILY REPRESENTATIVE OF ANOMALIES TO BE EXPECTED, DO NOT USE  
```  
  
  
### whitenoise  
full signature:  
```text  
def whitenoise(nbins, fstd=None)  
```  
comments:  
```text  
generate one sample of white noise (uncorrelated between bins)  
input args and output: similar to goodnoise  
```  
  
  
### random\_lico  
full signature:  
```text  
def random_lico(hists)  
```  
comments:  
```text  
generate one linear combination of histograms with random coefficients in (0,1) summing to 1  
input args:   
- numpy array of shape (nhists,nbins), the rows of which will be linearly combined  
output:  
- numpy array of shape (nbins), containing the new histogram  
```  
  
  
### smoother  
full signature:  
```text  
def smoother(inarray, halfwidth=1)  
```  
comments:  
```text  
smooth the rows of a 2D array using the 2*halfwidth+1 surrounding values.  
```  
  
  
### mse\_correlation\_vector  
full signature:  
```text  
def mse_correlation_vector(hists, index)  
```  
comments:  
```text  
calculate mse of a histogram at given index wrt all other histograms  
input args:  
- hists: numpy array of shape (nhists,nbins) containing the histograms  
- index: the index (must be in (0,len(hists)-1)) of the histogram in question  
output:  
- numpy array of length nhists containing mse of the indexed histogram with respect to all other histograms  
WARNING: can be slow if called many times on a large collection of histograms with many bins.  
```  
  
  
### moments\_correlation\_vector  
full signature:  
```text  
def moments_correlation_vector(moments, index)  
```  
comments:  
```text  
calculate moment distance of hist at index wrt all other hists  
very similar to mse_correlation_vector but using histogram moments instead of full histograms for speed-up  
```  
  
  
### plot\_data\_and\_gen  
full signature:  
```text  
def plot_data_and_gen(datahists, genhists, nplot=10, figname='fig.png')  
```  
comments:  
```text  
plot a couple of random examples from data and generated histograms  
input arguments:  
- datahists, genhists: numpy arrays of shape (nhists,nbins)  
- nplot: integer, maximum number of examples to plot  
- figname: name of figure to plot  
```  
  
  
### plot\_seed\_and\_gen  
full signature:  
```text  
def plot_seed_and_gen(seedhists, genhists, figname='fig.png')  
```  
comments:  
```text  
plot seed and generated histograms  
input arguments:  
- seedhists, genhists: numpy arrays of shape (nhists,nbins)  
- figname: name of figure to plot  
```  
  
  
### plot\_noise  
full signature:  
```text  
def plot_noise(noise, histstd=None, figname='fig.png')  
```  
comments:  
```text  
plot histograms in noise (numpy array of shape (nhists,nbins))  
optional argument histstd plots +- histstd as boundaries  
```  
  
  
### fourier\_noise\_on\_mean  
full signature:  
```text  
def fourier_noise_on_mean(hists, outfilename='', figname='', nresamples=0, nonnegative=True)  
```  
comments:  
```text  
apply fourier noise on the bin-per-bin mean histogram, with amplitude scaling based on bin-per-bin std histogram.  
input args:  
- hists: numpy array of shape (nhists,nbins) used for determining mean and std  
- outfilename: path to csv file to write results to (default: no writing)  
- figname: path to figure plotting examples (default: no plotting)  
- nresamples: number of samples to draw (default: number of input histograms / 10)  
- nonnegative: boolean whether to set all bins to minimum zero after applying noise  
MOSTLY SUITABLE AS HELP FUNCTION FOR RESAMPLE_SIMILAR_FOURIER_NOISE, NOT AS GENERATOR IN ITSELF  
advantages: mean histogram is almost certainly 'good' because of averaging, eliminate bad histograms  
disadvantages: deviations from mean are small, does not model systematic shifts by lumi.  
```  
  
  
### fourier\_noise  
full signature:  
```text  
def fourier_noise(hists, outfilename='', figname='', nresamples=1, nonnegative=True, stdfactor=15.)  
```  
comments:  
```text  
apply fourier noise on random histograms with simple flat amplitude scaling.  
input args:   
- hists: numpy array of shape (nhists,nbins) used for seeding  
- outfilename: path to csv file to write results to (default: no writing)  
- figname: path to figure plotting examples (default: no plotting)  
- nresamples: number of samples to draw per input histogram  
- nonnegative: boolean whether to set all bins to minimum zero after applying noise  
- stdfactor: factor to scale magnitude of noise (larger factor = smaller noise)  
advantages: resampled histograms will have statistically same features as original input set  
disadvantages: also 'bad' histograms will be resampled if included in hists  
```  
  
  
### upsample\_hist\_set  
full signature:  
```text  
def upsample_hist_set(hists, ntarget=-1, fourierstdfactor=15., figname='f')  
```  
comments:  
```text  
wrapper for fourier_noise allowing for a fixed target number of histograms instead of a fixed resampling factor  
useful function for quickly generating a fixed number of resampled histograms,  
without bothering too much about what exact resampling technique or detailed settings would be most appropriate.  
input arguments:  
hists: input histogram set  
ntarget: targetted number of resampled histograms (default: equally many as in hists)  
fourierstdfactor: see fourier_noise  
```  
  
  
### white\_noise  
full signature:  
```text  
def white_noise(hists, figname='', stdfactor=15.)  
```  
comments:  
```text  
apply white noise to the histograms in hists.  
input args:  
- hists: np array (nhists,nbins) containing input histograms  
- figname: path to figure plotting examples (default: no plotting)  
- stdfactor: scaling factor of white noise amplitude (higher factor = smaller noise)  
```  
  
  
### resample\_bin\_per\_bin  
full signature:  
```text  
def resample_bin_per_bin(hists, outfilename='', figname='', nresamples=0, nonnegative=True, smoothinghalfwidth=2)  
```  
comments:  
```text  
do resampling from bin-per-bin probability distributions  
input args:  
- hists: np array (nhists,nbins) containing the histograms to draw new samples from  
- outfilename: path to csv file to write results to (default: no writing)  
- figname: path to figure plotting examples (default: no plotting)  
- nresamples: number of samples to draw (default: 1/10 of number of input histograms)  
- nonnegative: boolean whether or not to put all bins to minimum zero after applying noise  
- smoothinghalfwidth: halfwidth of smoothing procedure to apply on the result (default: no smoothing)  
advantages: no arbitrary noise modeling  
disadvantages: bins are considered independent, shape of historams not taken into account,  
               does not work well on small number of input histograms,   
               does not work well on histograms with systematic shifts  
```  
  
  
### resample\_similar\_bin\_per\_bin  
full signature:  
```text  
def resample_similar_bin_per_bin( allhists, selhists, outfilename='', figname='', nresamples=1, nonnegative=True, keeppercentage=1.)  
```  
comments:  
```text  
resample from bin-per-bin probability distributions, but only from similar looking histograms.  
input args:  
- allhists: np array (nhists,nbins) containing all available histograms (to determine mean)  
- selhists: np array (nhists,nbins) conataining selected histograms used as seeds (e.g. 'good' histograms)  
- outfilename: path of csv file to write results to (default: no writing)  
- figname: path to figure plotting examples (default: no plotting)  
- nresamples: number of samples per input histogram in selhists  
- nonnegative: boolean whether or not to put all bins to minimum zero after applying noise  
- keeppercentage: percentage (between 1 and 100) of histograms in allhists to use per input histogram  
advantages: no assumptions on shape of noise,  
            can handle systematic shifts in histograms  
disadvantages: bins are treated independently from each other  
```  
  
  
### resample\_similar\_fourier\_noise  
full signature:  
```text  
def resample_similar_fourier_noise( allhists, selhists, outfilename='', figname='', nresamples=1, nonnegative=True, keeppercentage=1.)  
```  
comments:  
```text  
apply fourier noise on mean histogram,   
where the mean is determined from a set of similar-looking histograms  
input args:  
- allhists: np array (nhists,nbins) containing all available histograms (to determine mean)  
- selhists: np array (nhists,nbins) conataining selected histograms used as seeds (e.g. 'good' histograms)  
- outfilename: path of csv file to write results to (default: no writing)  
- figname: path to figure plotting examples (default: no plotting)  
- nresamples: number of samples per input histogram in selhists  
- nonnegative: boolean whether or not to put all bins to minimum zero after applying noise  
- keeppercentage: percentage (between 1 and 100) of histograms in allhists to use per input histogram  
advantages: most of fourier_noise_on_mean but can additionally handle shifting histograms,  
            apart from fourier noise, also white noise can be applied.  
disadvantages: does not filter out odd histograms as long as enough other odd histograms look more or less similar  
```  
  
  
### resample\_similar\_lico  
full signature:  
```text  
def resample_similar_lico( allhists, selhists, outfilename='', figname='', nresamples=1, nonnegative=True, keeppercentage=1.)  
```  
comments:  
```text  
take linear combinations of similar histograms  
input arguments:  
- allhists: 2D np array (nhists,nbins) with all available histograms, used to take linear combinations  
- selhists: 2D np array (nhists,nbins) with selected hists used for seeding (e.g. 'good' histograms)  
- outfilename: path to csv file to write result to (default: no writing)  
- figname: path to figure plotting examples (defautl: no plotting)  
- nresamples: number of combinations to make per input histogram  
- nonnegative: boolean whether to make all final histograms nonnegative  
- keeppercentage: percentage (between 0. and 100.) of histograms in allhists to use per input histogram  
advantages: no assumptions on noise  
disadvantages: sensitive to outlying histograms (more than with averaging)  
```  
  
  
### mc\_sampling  
full signature:  
```text  
def mc_sampling(hists, nMC=10000 , nresamples=10)  
```  
comments:  
```text  
resampling of a histogram using MC methods  
Drawing random points from a space defined by the range of the histogram in all axes.  
Points are "accepted" if the fall under the sampled histogram:  
f(x) - sampled distribution  
x_r, y_r -> randomly sampled point  
if y_r<=f(x_r), fill the new distribution at bin corresponding to x_r with weight:  
weight = (sum of input hist)/(#mc points accepted)  
this is equal to   
weight = (MC space volume)/(all MC points)  
```  
  
  
