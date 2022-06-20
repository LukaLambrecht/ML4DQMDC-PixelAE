## Run the ML4DQM functionality using the GUI on SWAN

### How to start the program
Simply download the repository and open the notebook nbgui.ipynb in SWAN. 

#### Downloading the repository
Go to the [GitHub page](https://github.com/LukaLambrecht/ML4DQM-DC) and clone or download the repository via git.  
Dependencies:  
- All of the required dependencies are installed by default on SWAN!  
To do: make list of optional dependencies that are not installed by default (e.g. imageio?)

### Where to start?
The starting screen of the GUI shows a number of tabs for several actions one can perform.  
Usually one would start by loading a previously stored HistStruct (a structure containing all relevant histograms and classifiers) via the 'Load' tab, or create a new one via the 'New' tab. Many of the other tabs will not work if you have not loaded or created a valid HistStruct first. To create a HistStruct, the GUI expects input files in a specific format, see below.

### Input files
Some example input files are provided on this [CERNBox location](https://cernbox.cern.ch/index.php/s/E9GzJ4WMZs3jbPd).  
They are skimmed versions of the centrally provided csv files with monintoring elements (both 1D and 2D histograms) with a per-lumisection granularity. The central files are available on EOS: `/eos/project/c/cmsml4dc/ML_2020`.  
How to produce your own skimmed input files from the centrally provided csv files is still under discussion. The code to do so is available in this repository (if run on SWAN, which has direct access to EOS). To do: update to nanoDQMIO input.

### The HistStruct object
The 'HistStruct' is the central object used in this program. It contains all relevant histograms (depending on the target run, training runs and histogram types) and their respective classifiers in a structured way.  

Selecting subsets of histograms, e.g. for classifier training or evaluation, is managed by using masks, which you can initialize when creating the HistStruct. There are three types of masks:  
- run masks: select lumisections belonging to a given run number.  
- statistics masks: select lumisections with high or low statistics. Currently the criterion is simply based on the number of histogram entries divided by the number of bins. Might be extended in the future.  
- json masks: select lumisections with a custom json file in the same format as the typical so-called golden json file. Any json file in the correct format can be uploaded, which allows full flexibility in selecting lumisections.  

### Typical workflow
After having loaded or created a new HistStruct object, you can start the training and evaluation of histogram classifiers. A typical workflow would consist of (most of) the following steps:  
- preprocessing the histograms (i.e. rebinning, normalizing and/or cropping). See the 'Preprocessing' button.  
- make plots of the training set, target set and other histograms that might be relevant. See the 'Plotting' button.  
- resample the training set to increase statistics (might not be needed for some types of classifiers). See the 'Resampling' button.  
- adding classifiers for each histogram type. See the 'Add classifiers' button.  
- train all classifiers. See the 'Train classifiers' button.
- fit a probability density function to the resulting classifier scores in order to get from a per-histogram score to a per-lumisection score. See the 'Fit' button.
- evaluate the model. See 'Evaluate model' button.  
- take a closer look at some lumisections. For example in the case of an autoencoder one might be interested to compare the original histogram to its autoencoder reconstruction. Or more generally one could be interested to see what histogram types are causing the high lumisection score. See 'Plot lumisection' button.  

### Example usage
Some examples on how to use the GUI are put in slides and collected on this [CERNBox location](https://cernbox.cern.ch/index.php/s/QEUAG7eKmgSeRns). To do: update!  