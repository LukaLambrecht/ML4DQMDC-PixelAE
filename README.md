# Some utilities and example notebooks for ML4DQM/DC
  
This repository contains example code for the ML4DQM/DC project.  
It was developed with the idea of training autoencoders on the per-lumisection histograms stored in dedicated DQMIO files in order to (partially) automate the DQM and/or DC process. However, it is intended to be generally useful for any ML4DQM/DC study (i.e. any subsystem, any type of histogram, any classification method).  

In more detail:  

- The framework has been developed based on a number of 1D histograms related to the status of the pixel tracker. Support for 2D histograms was added later.  
- Likewise, most of the development effort went into using an autoencoder as classification algorithm, looking at the mean-squared-difference between a histogram and its reconstruction as a measure for anomality. Support for other algorithms is also present, one just needs to define a class deriving from `src/classifiers/HistogramClassifier`. Working examples for NMF, PCA and a couple of other classifiers are present in the tutorials.  
- Furthermore, the idea is to train a separate classifier on each histogram type, and then combine the output scores in some way. Using instead only one histogram type is of course also possible (examples are present in the tutorials). Combining histograms in different ways (e.g. appending them together and training an autoencoder on the result) is not explicitly supported yet, though many parts of the code should still be perfectly usable.


### Structure of this repository  
There are six important directories: `dqmio`, `tutorials`, `utils`, `src`, `runswan` and `omsapi`.  
The other directories in the repository contain either data or meta-information (e.g. for documentation).  
In more detail (in order of most likely appearance):  

- `dqmio`: contains tools and scripts to read (nano)DQMIO files from DAS and put them in a more useful format. Before starting with further steps, it is recommended to generate small files with only one monitor element per file. See the README in the `dqmio` folder for more info. Please use the conversion into csv files for now, as this is the expected input for the downstream steps (for historical reasons). This might be made more flexible in the future.  
Note 1: the nanoDQMIO files are intended to be the new starting point for any ML4DQM-DC study, as of 2022 data-taking. Previously, for 2017 and 2018 data, we had large csv files stored on `\eos` that basically contained the full DQMIO content in csv format. They can also still be used as input, for studies on legacy data. But in that case you also need to run a prepocessing step. See the tutorial `read_and_write_data.ipynb` (more info below).  
Note 2: The difficulties with using the nanoDQMIO files directly are: 
    - that they still contain a relatively large number of monitoring elements (even though reduced with respect to the standard DQMIO format). This makes reading and operating on these files rather slow. Furthermore, many of the tools in the downstream steps assume one monitoring element per file as an input (at least for now, might be made more flexible in the future).  
    - that the lumisections belonging to a given data-taking period are split (arbitrarily?) over multiple files. It is often more useful to have all lumisections of a data-taking period in one file.  
- `tutorials`: contains a number of notebooks that can be used to get familiar with the code and its capabilities (see more info below).  
- `utils`: contains a number of python files with static utility functions for general use. They are being called and used in various places throughout the code. See the tutorials for examples.  
- `src`: contains a number of useful classes for this project. See the tutorials for examples of how to use them.
    - `DataLoader`: class for convenient loading of histograms from the input csv files.  
    - `HistStruct`: a histogram container for easy handling of multiple types of histograms simultaneously and consistently.  
    - `Model` and `ModelInterface`: a model container that can hold classifiers acting on several histogram types and a combination method to provide a global per-lumisection score.
    - `classifiers`: folder containing an abstract base `HistogramClassifier` class and derived classes representing concrete examples of histogram classification algorithms.  
    - `cloudfitters`: folder containing an abstract base `CloudFitter` class and derived classes representing concrete examples of point cloud fitting algorithms (used when working with multiple types of histograms simultaneously).  
- `runswan`: contains code for a graphical interface (see more info below).  
- `omsapi`: standalone API for retrieving information from OMS.  


### Tutorials  
Some tutorials are located in the tutorials folder in this repository, that should help you get started with the code. They can be grouped into different steps:  

- Put the data in a more manageable format. This step is no longer needed if you start from the (nano)DQMIO files and if you have prepared the data with the scripts in the `dqmio` folder of this repository. If you start from the legacy csv files however, follow these steps. The raw csv files that are (were) our common input are not very easy to work with. Therefore you would probably first want to do something similar to what's done in the notebook `read_and_write_data.ipynb`. See the code and inline comments in that script and the functions it refers to for more detailed explanation. Its output is one single csv file per histogram type and per year, which is often much more convenient than the original csv files (which contain all histogram types together and are split per number of lines, not per run). All other functions and notebooks presuppose this first step.  
- Plot the data. Next, you can run `plot_histograms.ipynb` and `plot_histograms_loop.ipynb`. These notebooks should help you get a feeling of what your histogram looks like in general, and perhaps help you find some anomalies that you can use for testing. For 2D histograms, look at `plot_histograms_2d.ipynb` instead.  
- Generate some artificial data for model training and testing, as exemplified in `generate_data.ipynb`.  
- Train an autoencoder (or any other type of classifier). The scripts `autoencoder.ipynb` and `autoencoder_iterative.ipynb` are used to train an autoencoder on the whole dataset or a particular subset respectively. `maxpull.ipynb`, `nmf_1d.ipynb` and `nmf_2.ipynb` are showcases of some other classifiers (a maximum pull with respect to a reference histogram, and an NMF model in 1D and in 2D respectively).  
- Finally, `global_combined_training.ipynb` shows a more complete example of training a model on several histogram types and combine the output.  


### Graphical interface
See more info in the dedicated README in the corresponding folder.
  
  
### Other remarks  
- The repository contains no data files. I was planning to put some example data files in a data folder, but the files are too big for github. You can produce your own input files starting from the (new) nanoDQMIO or the (legacy) csv files as explained above. Another way to get started is to get some example files from my [CERNBox](https://cernbox.cern.ch/index.php/s/E9GzJ4WMZs3jbPd)
- Disclaimer: the whole repository is still in development stage. Feel free to contact me (at [luka.lambrecht@cern.ch](luka.lambrecht@cern.ch)) in case you found bugs or if you have other suggestions.  
  
  
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
 
 
### Further documentation  

- Documentation for all the class definitions and functions in the relevant code directories can be found [here](https://lukalambrecht.github.io/ML4DQMDC-PixelAE/). The documentation is generated automatically from annotations in the source code, so the formatting might behave oddly in some cases.  
- Note that the website above does not include documentation for the tutorials (yet?). However, some comments in the tutorial notebooks should provide (enough?) explanation to follow along.  
