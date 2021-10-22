# Some utilities and example notebooks for ML4DQM/DC
  
This repository contains example code for the ML4DQM/DC project.  
It was developed with the idea of training autoencoders on the per-lumisection histograms stored in dedicated DQMIO files in order to (partially) automate the DQM and/or DC process. However, it is intended to be generally useful for any ML4DQM/DC study (i.e. any subsystem, any type of histogram, any classification method).  

In more detail:  

- The framework has been developed based on a number of 1D histograms related to the status of the pixel tracker. Support for 2D histograms was added later and tested preliminarily, but some parts of the code may not yet support them. Please feel free to contact me if you notice a place in the code where this is the case.
- Likewise, most of the development effort went into using an autoencoder as classification algorithm, i.e. looking at the mean-squared-difference between a histogram and its reconstruction as a measure for anomality. Support for other algorithms is also present, one just needs to define a class deriving from src/classifiers/HistogramClassifier. Working examples for NMF, PCA and a couple of other classifiers are present in the tutorials.

### Structure of this repository:  
There are five important directories: tutorials, utils, src, run and omsapi.  
The other directories in the repository contain either data or meta-information (e.g. for documentation).  
In more detail:  

- utils: contains a number of python notebooks and equivalent scripts with static utility functions for general use. They are being called and used in various places throughout the code.  
- src: contains the classes for this repository:
    - DataLoader: class for convenient loading of histograms from the input csv files.
    - HistStruct: a histogram container for easy handling of multiple types of histograms simultaneously and consistently.
    - classifiers: folder containing an abstract base HistogramClassifier class and derived classes representing concrete examples of histogram classification algorithms.
    - cloudfitters: folder containing an abstract base CloudFitter class and derived classes representing concrete examples of point cloud fitting algorithms (used when working with multiple types of histograms simultaneously).
- tutorials: contains a number of notebooks that can be used to get familiar with the code and its capabilities.  
- run: contains code for alternative ways of running the workflow, e.g. with configuration files or with a GUI. Note: under development, recommended to start with script-based workflows first (as exemplified in the tutorials).
- omsapi: standalone API for retrieving information from OMS.

### Tutorials:  
Some tutorials are located in the tutorials folder in this repository, that should help you get started with the code. They can be grouped into different steps:  

- Step 1: put the data in a more manageable format. The raw csv files that are our common input are not very easy to work with. Therefore you would probably first want to do something similar to what's done in the notebook read\_and\_write\_data.ipynb. See the code and inline comments in that script and the functions it refers to for more detailed explanation. Its output is one single csv file per histogram type and per year, which is often much more convenient than the original csv files (which contain all histogram types together and are split per number of lines, not per run). All other functions and notebooks presuppose this first step.  
- Step 2: plot the data. Next, you can run plot\_histograms.ipynb and plot\_histograms\_loop.ipynb. These notebooks should help you get a feeling of what your histogram looks like in general, and perhaps help you find some anomalies that you can use for testing. For 2D histograms, look at plot\_histograms\_2d.ipynb instead.  
- Step 3: train an autoencoder. The scripts autoencoder.ipynb and autoencoder\_iterative.ipynb are used to train an autoencoder on the whole dataset or a particular subset respectively. Finally, autoencoder\_combine.ipynb trains autoencoders on multiple types of histograms and combines the mse's for each. An example on how to implement another classification method is shown in template\_combine.ipynb.  
  
### Other remarks:  

- The repository contains no data files. I was planning to put some example data files in a data folder, but the files are too big for github. However, the tutorial read\_and\_write\_data.ipynb should help you get the data from where it is stored and put it in a useful format for further processing.  
- Another way to get started is to get them from my [CERNBox](https://cernbox.cern.ch/index.php/s/E9GzJ4WMZs3jbPd)
- Disclaimer: the whole repository is still in development stage. Feel free to contact me in case you found bugs or if you have other suggestions.  
  
### To get the tutorial notebooks running in SWAN  
#### (preferred method):  

- Log in to SWAN.  
- Go to Projects.  
- Click the cloud icon that says 'Download Project from git'  
- Paste the following url: [https://github.com/LukaLambrecht/ML4DQM-DC.git](https://github.com/LukaLambrecht/ML4DQM-DC.git).

#### (alternative method):  

- Log in to SWAN.
- Click on the leftmost icon on the top right ('new terminal').
- Navigate to where you want this repository (the starting place is your CERNBox home directory).
- Paste this command: git clone https://github.com/LukaLambrecht/ML4DQM-DC.git (or however you usually clone a repository).    
- Exit the terminal.  
- The folder should now be where you cloned it, and you can open and run the notebooks in it in SWAN. 
 
### Further documentation:  

- Documentation for all the class definitions and functions in the relevant code directories: https://LukaLambrecht.github.io/ML4DQM-DC/ (note: this documentation is generated automatically from comments in the code and currently not yet in perfect shape, both regarding content and layout).  
- Note that the website above does not include documentation for the tutorials (yet?). However, some comments in the tutorial notebooks should provide (enough?) explanation to follow along.  
