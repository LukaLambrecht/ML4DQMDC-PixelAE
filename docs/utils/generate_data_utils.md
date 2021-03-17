# generate data utils  
  
- - -    
## goodnoise(nbins, fstd=None)  
generate one sample of 'good' noise consisting of fourier components  
input args:  
- nbins: number of bins, length of noise array to be sampled  
- fstd: an array of length nbins used for scaling of the amplitude of the noise  
bin-by-bin.  
output:  
- numpy array of length nbins containing the noise  
  
- - -    
## badnoise(nbins,fstd=None)  
generate one sample of 'bad' noise consisting of fourier components  
(higher frequency and amplitude than 'good' noise)  
input args and output: simlar to goodnoise  
WARNING: NOT NECESSARILY REPRESENTATIVE OF ANOMALIES TO BE EXPECTED, DO NOT USE  
  
- - -    
## whitenoise(nbins,fstd=None)  
generate one sample of white noise (uncorrelated between bins)  
input args and output: similar to goodnoise  
  
- - -    
## random_lico(hists)  
generate one linear combination of histograms with random coefficients in (0,1) summing to 1  
input args:  
- numpy array of shape (nhists,nbins), the rows of which will be linearly combined  
output:  
- numpy array of shape (nbins), containing the new histogram  
  
- - -    
## smoother(inarray,halfwidth)  
smooth the rows of a 2D array using the 2*halfwidth+1 surrounding values.  
  
- - -    
## mse_correlation_vector(hists,index)  
calculate mse of a histogram at given index wrt all other histograms  
input args:  
- hists: numpy array of shape (nhists,nbins) containing the histograms  
- index: the index (must be in (0,len(hists)-1)) of the histogram in question  
output:  
- numpy array of length nhists containing mse of the indexed histogram with respect to all other histograms  
WARNING: can be slow if called many times on a large collection of histograms with many bins.  
  
- - -    
## moments_correlation_vector(moments,index)  
calculate moment distance of hist at index wrt all other hists  
very similar to mse_correlation_vector but using histogram moments instead of full histograms for speed-up  
  
- - -    
## plot_data_and_gen(nplot,datahist,genhist,figname='fig.png')  
plot a couple of random examples from rhist (data), ghist (resampled 'good') and bhist (resampled 'bad')  
input arguments:  
- nplot: integer, maximum number of examples to plot  
- datahist, genhist: numpy arrays of shape (nhists,nbins)  
- figname: name of figure to plot  
  
- - -    
## plot_seed_and_gen(seedhist,genhist,figname='fig.png')  
plot a couple of random examples from rhist (data), ghist (resampled 'good') and bhist (resampled 'bad')  
input arguments:  
- datahist, genhist: numpy arrays of shape (nhists,nbins)  
- figname: name of figure to plot  
  
- - -    
## plot_noise(noise,histstd=None,figname='fig.png')  
plot histograms in noise (numpy array of shape (nhists,nbins))  
optional argument histstd plots +- histstd as boundaries  
  
- - -    
## fourier_noise_on_mean(hists,outfilename='',figname='',nresamples=0,nonnegative=True)  
apply fourier noise on the bin-per-bin mean histogram,  
with amplitude scaling based on bin-per-bin std histogram.  
input args:  
- hists: numpy array of shape (nhists,nbins) used for determining mean and std  
- outfilename: path to csv file to write results to (default: no writing)  
- figname: path to figure plotting examples (default: no plotting)  
- nresamples: number of samples to draw (default: number of input histograms / 10)  
- nonnegative: boolean whether to set all bins to minimum zero after applying noise  
MOSTLY SUITABLE AS HELP FUNCTION FOR RESAMPLE_SIMILAR_FOURIER_NOISE, NOT AS GENERATOR IN ITSELF  
  
- - -    
## fourier_noise(hists,outfilename='',figname='',nresamples=1,nonnegative=True,stdfactor=15.)  
apply fourier noise on random histograms with simple flat amplitude scaling.  
input args:  
- hists: numpy array of shape (nhists,nbins) used for seeding  
- outfilename: path to csv file to write results to (default: no writing)  
- figname: path to figure plotting examples (default: no plotting)  
- nresamples: number of samples to draw per input histogram  
- nonnegative: boolean whether to set all bins to minimum zero after applying noise  
- stdfactor: factor to scale magnitude of noise (larger factor = smaller noise)  
  
- - -    
## upsample_hist_set(hists,ntarget,fourierstdfactor=15.,figname='f')  
(no valid documentation found)  
  
- - -    
## white_noise(hists,figname='',stdfactor=15.)  
apply white noise to the histograms in hists.  
input args:  
- hists: np array (nhists,nbins) containing input histograms  
- figname: path to figure plotting examples (default: no plotting)  
- stdfactor: scaling factor of white noise amplitude (higher factor = smaller noise)  
  
- - -    
## resample_bin_per_bin(hists,outfilename='',figname='',nresamples=0,nonnegative=True,smoothinghalfwidth=2)  
do resampling from bin-per-bin probability distributions  
input args:  
- hists: np array (nhists,nbins) containing the histograms to draw new samples from  
- outfilename: path to csv file to write results to (default: no writing)  
- figname: path to figure plotting examples (default: no plotting)  
- nresamples: number of samples to draw (default: 1/10 of number of input histograms)  
- nonnegative: boolean whether or not to put all bins to minimum zero after applying noise  
- smoothinghalfwidth: halfwidth of smoothing procedure to apply on the result (default: no smoothing)  
  
- - -    
## resample_similar_bin_per_bin( allhists, selhists, outfilename='', figname='', nresamples=1, nonnegative=True, keeppercentage=1.)  
resample from bin-per-bin probability distributions, but only from similar looking histograms.  
input args:  
- allhists: np array (nhists,nbins) containing all available histograms (to determine mean)  
- selhists: np array (nhists,nbins) conataining selected histograms used as seeds (e.g. 'good' histograms)  
- outfilename: path of csv file to write results to (default: no writing)  
- figname: path to figure plotting examples (default: no plotting)  
- nresamples: number of samples per input histogram in selhists  
- nonnegative: boolean whether or not to put all bins to minimum zero after applying noise  
- keeppercentage: percentage (between 1 and 100) of histograms in allhists to use per input histogram  
  
- - -    
## resample_similar_fourier_noise( allhists, selhists, outfilename='', figname='', nresamples=1, nonnegative=True, keeppercentage=1.)  
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
  
- - -    
## resample_similar_lico( allhists, selhists, outfilename='', figname='', nresamples=1, nonnegative=True, keeppercentage=1.)  
take linear combinations of similar histograms  
input arguments:  
- allhists: 2D np array (nhists,nbins) with all available histograms, used to take linear combinations  
- selhists: 2D np array (nhists,nbins) with selected hists used for seeding (e.g. 'good' histograms)  
- outfilename: path to csv file to write result to (default: no writing)  
- figname: path to figure plotting examples (defautl: no plotting)  
- nresamples: number of combinations to make per input histogram  
- nonnegative: boolean whether to make all final histograms nonnegative  
- keeppercentage: percentage (between 0. and 100.) of histograms in allhists to use per input histogram  
  
- - -    
## mc_sampling(hists, nMC=10000 , nresamples=10)  
**resampling of a histogram using MC methods**  
**(Marek's method)**  
Drawing random points from a space defined by the range of the histogram in all axes.  
Points are "accepted" if the fall under the sampled histogram:  
f(x) - sampled distribution  
x_r, y_r -> randomly sampled point  
if y_r<=f(x_r), fill the new distribution at bin corresponding to x_r with weight:  
weight = (sum of input hist)/(#mc points accepted)  
this is equal to  
weight = (MC space volume)/(# all MC points)  
  
