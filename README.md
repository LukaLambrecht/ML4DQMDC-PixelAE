**Some utilities and example notebooks for ML4DQM/DC**  
  
Step 1: put the data in a more manageable format.  
The raw csv files that are our common input are not very easy to work with. Therefore you would probably first want to do something similar to what's done in the notebook read_and_write_data.ipynb. See the code and inline comments in that script and the functions it refers to for more detailed explanation. Its output is one single csv file per histogram type and per year, which is often much more convenient than the original csv files (which contain all histogram types together and are split per number of lines, not per run). All other functions and notebooks presuppose this first step.  
  
Step 2: plot the data.  
Next, you can run plot_histograms.ipynb and plot_histograms_loop.ipynb. These notebooks should help you get a feeling of what your histogram looks like in general, and perhaps help you find some anomalies that you can use for testing.  
  
Step 3: train an autoencoder.  
The scripts autoencoder.ipynb and autoencoder_iterative.ipynb are used to train an autoencoder on the whole dataset or a particular subset respectively. Finally, autoencoder_combine.ipynb trains autoencoders on multiple types of histograms and combines the mse's for each.  
  
General remarks:  
- All the example notebooks are in the main folder, while most of the functionality is situated in the utils folder. This folder contains notebooks that only contain functions, no executing code. These functions are imported in the example notebooks by doing %run utils/(notebook_name).ipynb.
- I was planning to put some example data files (the output of read_and_write_data.ipynb) in the data folder, but the files are too big for github. So you'll have to run that script yourself.
- Some of the example notebooks contain cells that I haven't used in a long time, and that will probably not run or not make sense, but that I did not want to delete yet. It should be indicated in the comments where that is the case.  
  
To get the notebooks running in SWAN:  
- Log in to SWAN.
- Click on the leftmost icon on the top right ('new terminal').
- Navigate to where you want this repository (the starting place is your CERNBox home directory).
- Paste this command: git clone https://github.com/LukaLambrecht/SudokuSolver.git (or however you usually clone a repository).
- Exit the terminal.
- The folder should now be where you cloned it, and you can open and run the notebooks in it in SWAN.
