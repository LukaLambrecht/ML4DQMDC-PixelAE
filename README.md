**Some utilities and example notebooks for ML4DQM/DC**  
  
This repository contains example code for the ML4DQM/DC project.  
It was developed with the idea of training autoencoders on the per-lumisection histograms stored in dedicated DQMIO files in order to (partially) automate the DQM and/or DC process. In the future, this repository is intended to be generally useful for any ML4DQM/DC study (i.e. any subsystem, any type of histogram, any classification method), but for historical reasons it is at this moment focused on the following subtopics:  
- some 1D histograms related to the status of the pixel tracker.
- using an autoencoder as classification algorithm, i.e. looking at the mean-squared-error between a histogram and its reconstruction as a measure for anomality.  

Structure of this repository:  
There are three important directories: tutorials, utils and src.  
- utils contains a number of python notebooks and equivalent scripts with static functions for general use.
- src contains a class structure that should in principle allow a modular approach and easy extensions, for example towards other single-histogram classification algorithms (see subfolder classifiers).
- tutorials contains a number of notebooks that can be used to get familiar with the code and its possibilities. 

Tutorials:  
Some tutorials are located in the tutorials folder in this repository, that should help you get started with the code. They can be grouped into different steps:
- Step 1: put the data in a more manageable format. The raw csv files that are our common input are not very easy to work with. Therefore you would probably first want to do something similar to what's done in the notebook read\_and\_write\_data.ipynb. See the code and inline comments in that script and the functions it refers to for more detailed explanation. Its output is one single csv file per histogram type and per year, which is often much more convenient than the original csv files (which contain all histogram types together and are split per number of lines, not per run). All other functions and notebooks presuppose this first step.  
- Step 2: plot the data. Next, you can run plot\_histograms.ipynb and plot\_histograms\_loop.ipynb. These notebooks should help you get a feeling of what your histogram looks like in general, and perhaps help you find some anomalies that you can use for testing.  
- Step 3: train an autoencoder. The scripts autoencoder.ipynb and autoencoder\_iterative.ipynb are used to train an autoencoder on the whole dataset or a particular subset respectively. Finally, autoencoder\_combine.ipynb trains autoencoders on multiple types of histograms and combines the mse's for each.  
  
Other remarks:  
- The repository contains not data files. I was planning to put some example data files in a data folder, but the files are too big for github. However, the tutorial read\_and\_write\_data.ipynb should help you get the data from where it is stored and put it in a useful format for further processing. 
  
To get the tutorial notebooks running in SWAN (preferred method):  
- Log in to SWAN.
- Go to Projects.
- Click the cloud icon that says 'Download Project from git'
- Paste the following url: https://github.com/LukaLambrecht/ML4DQM-DC.git. 

(alternative method):
- Log in to SWAN.
- Click on the leftmost icon on the top right ('new terminal').
- Navigate to where you want this repository (the starting place is your CERNBox home directory).
- Paste this command: git clone https://github.com/LukaLambrecht/ML4DQM-DC.git (or however you usually clone a repository).
- Exit the terminal.
- The folder should now be where you cloned it, and you can open and run the notebooks in it in SWAN.  

Further documentation: 
