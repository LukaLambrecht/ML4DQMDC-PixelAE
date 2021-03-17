# get_histnames(df)  
**get a list of (unique) histogram names present in a df**  
df is a dataframe read from an input csv file.  
  
# select_histnames(df,histnames)  
**keep only a subset of histograms in a df**  
histnames is a list of histogram names to keep in the df.  
  
# get_runs(df)  
**return a list of (unique) run numbers present in a df**  
df is a dataframe read from an input csv file.  
  
# select_runs(df,runnbs)  
**keep only a subset of runs in a df**  
runnbs is a list of run numbers to keep in the df.  
  
# get_ls(df)  
**return a list of ls numbers present in a df**  
note that the numbers are not required to be unique!  
note: no check is done on the run number!  
  
# select_ls(df,lsnbs)  
**keep only a subset of lumisection numbers in a df**  
lsnbs is a list of lumisection numbers to keep in the df.  
note: no check is done on the run number!  
  
# get_runsls(df)  
**return a dictionary with runs and lumisections in a dataframe (same format as e.g. golden json)**  
  
# select_json(df,jsonfile)  
**keep only lumisections that are in the given json file**  
  
# select_runsls(df,jsondict)  
**equivalent to select_json but using a pre-loaded json dict instead of a json file on disk**  
  
# select_golden(df)  
**keep only golden lumisections in df**  
  
# select_notgolden(df)  
**keep all but golden lumisections in df**  
  
# select_dcson(df)  
**keep only lumisections in df that have DCS-bit on**  
  
# select_dcsoff(df)  
**keep only lumisections in df that have DCS-bit off**  
  
# select_pixelgood(df)  
**keep only lumisections in df that are in good pixel json**  
  
# select_pixelbad(df)  
**keep only lumisections in df that are in bad pixel json**  
  
# get_highstat(df,entries_to_bins_ratio=100)  
**return a select object of runs and ls of histograms with high statistics**  
  
# select_highstat(df,entries_to_bins_ratio=100)  
(no valid documentation found)  
  
# get_hist_values(df)  
**same as builtin "df['histo'].values" but convert strings to np arrays**  
also an array of run and LS numbers is returned  
warning: no check is done to assure that all histograms are of the same type!  
  
