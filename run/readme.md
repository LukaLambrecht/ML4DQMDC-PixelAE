**Available parameters for the configuration file:**

**General**

   * HISTSTRUCT\_FILE\_NAME: file name of the pickled file containing a HistStruct object. Usually the output of the create_histstruct.ipynb script. 
   
   * TRAINING\_MASKS: mask names defining the training set. The masks should be present in the HistStruct object (see create_histstruct.ipynb for examples on how to write them). Example: suppose the HistStruct contains masks with names "dcson" (for DCS-bit on selection) and "highstat" (to filter out low-statistics histograms), you can define TRAINING\_MASKS = \["dcson","highstat"\] to select all lumisections with DCS-bit on and sufficient statistics in all histograms for training.
   
   * TEST\_GOOD\_MASKS: mask names defining the test set labeled as good. The masks should be present in the HistStruct object (see create_histstruct.ipynb for examples on how to write them). This parameter should be defined as a list of lists, in order to allow multiple good test sets. Example: suppose the HistStruct contains masks with names "dcson" (for DCS-bit on selection), "good1" and "good2", you can define TRAINING\_MASKS = \[\["dcson","good1"\],\["dcson","good2"\]\].

   * TEST\_BAD\_MASKS: similar to TEST\_GOOD\_MASKS but for test sets labeled bad.
   
**Parameters for making a plot of the training and/or testing sets**

   * DO\_INITIAL\_PLOT: boolean, set to True to enable this plotting.

   * INITIAL\_PLOT\_SETTINGS: list of dicts containing plot settings. A separate plot will be made for each element in the list, and each element is supposed to be a dict containing options for that plot. The plotting function is HistStruct.plot_histograms (see the corresponding documentation for all available options). Example: INITIAL\_PLOT\_SETTINGS = \[ {"masknames": \[\["training"\], \["testing"\]\], "labellist": \["lumisections for training", "lumisections for testing"\], "colorlist": \["blue", "green"\]}\]
   
**Parameters for extending the training set**

   * EXTEND\_TRAINING: boolean, set to True to enable training set extension
  
   * EXTEND\_TRAINING\_FUNCTION: name of a function that takes a set of histograms and returns a resampled/extended set of histograms. Example: EXTEND\_TRAINING\_FUNCTION = gdu.upsample_hist_set (need to import utils/generate_data_utils as gdu).
   
   * EXTEND\_TRAINING\_OPTIONS: dict of options passed to the function used for extending the training set. Depends on the chosen EXTEND\_TRAINING\_FUNCTION. Example: if EXTEND\_TRAINING\_FUNCTION = gdu.upsample_hist_set, EXTEND\_TRAINING\_OPTIONS could be {"figname":"f","ntarget":5e4}.
   
   * EXTEND\_TRAINING\_PARTITIONS: integer defining the partitioning and averaging that will be performed before passing the histograms to EXTEND\_TRAINING\_FUNCTION. Example: EXTEND\_TRAINING\_PARTITIONS = 100 means that the training set will be split into 100 equal parts and each part will be averaged to a single histogram. If EXTEND\_TRAINING\_PARTITIONS is negative or larger than the number of input histograms, it will be ignored. If EXTEND\_TRAINING\_FUNCTION is None or not defined, the averaged histograms will be used directly for training.
   
**Parameters for the classifier training**

   * DO\_TRAINING: boolean, set to True to enable training. If False, you should either do the training manually in the configuration file, use classifiers that do not require training, or load previously trained classifiers.

   * TRAINING\_OPTIONS: dict of options passed to the HistogramClassifier.train function. See the documentation for the specific type you are using for all available options. If the classifier you use does not have a train function, you should put DO\_TRAINING = False and make the classifier ready for use within the configuration file itself.
   
   * SAVE\_MODELS: boolean, set to True to save the trained models to a specified directory (see below). Note: the models are also saved by default to a temporary directory where they can be read automatically when re-loading the HistStruct in following steps.

   * SAVE\_MODELS\_DIR: directory where to save the models if SAVE\_MODELS is True. 

   * SAVE\_MODELS\_BASENAME: base name of the models. The full path to a saved model is composed as follows: SAVE\_MODELS\_DIR / SAVE\_MODELS\_BASENAME \_ histogram type name . extension (depends on classifier type). 

**Parameters for combining the output scores of several histogram types**

   * CLOUDFITTER\_TYPE: class name of fitter to use. Example: CLOUDFITTER\_TYPE = GaussianKdeFitter (requires from src/cloudfitters/GaussianKdeFitter import GaussianKdeFitter)
   
   * CLOUDFITTER\_PLOT\_TRAINING: boolean, set to True to make 2D projection plots with training set.
   
   * CLOUDFITTER\_PLOT\_TEST: boolean, set to True to make 2D projection plots with test sets.

   * CLOUDFITTER\_OPTIONS: dict of options passed to the fitter constructor. See the documentation of the specific fitter you are using fol all available options.
   
**Parameters for extending the test set**

Available options are similar to those for extending the training set (see above for more explanation). Note that the partitions should be a list of integers in this case, one for each training/testing set.

   * EXTEND\_TEST\_GOOD
   * EXTEND\_TEST\_GOOD\_FUNCTION
   * EXTEND\_TEST_GOOD\_OPTIONS
   * EXTEND\_TEST_GOOD\_PARTITIONS 

   * EXTEND\_TEST\_BAD 
   * EXTEND\_TEST\_BAD_FUNCTION 
   * EXTEND\_TEST\_BAD_OPTIONS
   * EXTEND\_TEST\_BAD_PARTITIONS

**Parameters for ROC curve and related plots**

   * PLOT\_SCORE\_DIST: boolean, set to True to make a plot of the distributions of scores for the good and bad set. 
   * PLOT\_SCORE\_DIST\_OPTIONS: dict of options passed to the plotting function. See the function plot\_score\_dist in plot\_utils.py for all available options.

   * PLOT\_ROC\_CURVE: boolean, set to True to make a plot of the ROC curve.
   * PLOT\_ROC\_CURVE\_OPTIONS: dict of options passed to the plotting function. See the function get\_roc in autoencoder\_utils.py for all available options.

   * PLOT\_CONFUSION\_MATRIX: boolean, set to True to make a confusion matrix for a given working point.
   * PLOT\_CONFUSION\_MATRIX\_OPTIONS: dict of options passed to the plotting function. See the function get\_confusion\_matrix in autoencoder\_utils.py for all available options.

**Parameters for inspecting a run or lumisection in detail**

   * INSPECT\_MODE: specify whether to check a specific lumisection ('ls') or an entire run ('run'). Must be 'ls', 'run' or None. A plot will be made for each histogram type (and for each lumisection if INSPECT\_MODE is set to 'run') showing the histogram, its reconstruction (optional), and a set of reference histograms (optional).

   * INSPECT\_RUN: run number of the run or lumisection to inspect. 

   * INSPECT\_LS: lumisection number of the lumisection to inspect. Ignored if INSPECT\_MODE is set to 'run'.

   * INSPECT\_MASKS: lis of mask names. In case INSPECT\_MODE is set to 'run', the specified masks will be applied to the run before plotting (e.g. if you want to plot only the lumisections in a run where the DCS-bit was on). In case INSPECT\_MODE is set to 'ls', the masks specify which lumisections to take into account to make a reference score distribution.

   * INSPECT\_PLOT\_SCORE: boolean, set to True to make a plot per histogram showing the score for the given lumisection compared to a reference distribution. Ignored if INSPECT\_MODE is set to 'run'. The reference distribution of scores is defined by INSPECT\_MASKS, i.e. the distribution of scores for all lumisections passing INSPECT\_MASKS will be plotted as reference.
   
   * INSPECT\_RECO\_MODE: specify the histogram plotted as the reconstruction of the input histogram. This parameter can be set to 'auto', in which case the reconstruction will be created using the reconstruct method of the appropriate classifier (but this will fail if the HistStruct does not contain a classifier for this histogram type or if this type of classifier does not have a reconstruct function). If None, no reconstructed histogram will be plotted. Finally, you can also pass a dict with histogram names as keys and reconstructed histograms (numpy arrays of shape (nbins,)) as values.

   * INSPECT\_REFERENCE\_MASKS: list of mask names used to define the set of reference histograms plotted in the background.
   
   * INSPECT\_REFERENCE\_PARTITIONS: integer defining the partitioning and averaging that will be performed on the reference histograms before plotting. See EXTEND\_TRAIN\_PARTITIONS for an example. If neither INSPECT\_REFERENCE\_MASKS or INSPECT\_REFERENCE\_PARTITIONS are defined (or if INSPECT\_REFERENCE\_MASKS is not defined and INSPECT\_REFERENCE\_PARTITIONS is negative), no reference histograms will be plotted.
   
**Parameters for evaluating the classifier on an external test set (e.g. golden json)**

   * DO\_EVAL: boolean, set to True to enable this test.

   * EVAL\_MASKS: list of mask names that define the evaluation set. 

   * EVAL\_SCORE\_UP: upper value for the score. Can be None, in which case there is no upper value.

   * EVAL\_SCORE\_DOWN: lower value for the score. Can be None, in which case there is no lower value.

   * EVAL\_NMAXPLOTS: maximum number of plots to make. A plot will be made for each lumisection that passes EVAL\_MASKS whose score is between EVAL\_SCORE\_UP and EVAL\_SCORE\_DOWN until all these lumisections have been plotted or EVAL\_NMAXPLOTS is reached.

   * EVAL\_OUTFILENAME: name of output file (pdf format) where the plots will be written to. If None, the plots will not be saved to a file.
   
   * EVAL\_RECO\_MODE: see INSPECT\_RECO\_MODE

   * EVAL\_REFERENCE\_MASKS: see INSPECT\_REFERENCE\_MASKS
   
   * EVAL\_REFERENCE\_PARTITIONS: see INSPECT\_REFERENCE\_PARTITIONS