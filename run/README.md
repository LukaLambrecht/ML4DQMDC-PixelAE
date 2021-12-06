**Run the ML4DQM functionality using the GUI**

### How to start the program
The GUI cannot be run in SWAN, it needs to be run locally on your computer.  
Download the repository, navigate to this folder and run via 'python3 gui.py'.  
Dependencies:  
- The GUI will not work with python 2, it is required to use python 3.  
- Apart from tools within this repository, there are quite a few other python dependencies.  
- To do: make executable with all dependencies included and update this part of documentation.  

### Where to start?
The starting screen of the GUI shows a number of buttons and text boxes.  
Usually one would start by loading a previously stored HistStruct (a structure containing all relevant histograms and classifiers) via the 'Load' button, or create a new one via the 'New' button. Many of the other buttons will not work if you have not loaded or created a valid HistStruct first. To create a HistStruct, the GUI expects input files in a specific format, see below.

### Input files
To do: update this part of the documentation after discussing with Abhit about how to integrate this in the ML4DQM playground.

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
