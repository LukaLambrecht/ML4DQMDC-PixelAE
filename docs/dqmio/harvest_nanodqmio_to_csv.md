# harvest nanodqmio to csv  
  
**A script for reading (nano)DQMIO files and storing a ME in a CSV file format**  

Run with `python harvest_nanodqmio_to_csv.py -h` for a list of available options.  

The output is stored in a CSV file similar to the ones for the RunII legacy campaign.  
The file format is targeted to be as close as possible to the RunII legacy files,  
with the same columns, data types and naming conventions.  
The only difference is that there are no duplicate columns.  

While this file format may be far from optimal,  
it has the advantage that much of the existing code was developed to run on those files,  
so this is implemented to at least have the option to run on new DQMIO files   
without any code change.  
It was tested that the output files from this script can indeed be read correctly  
by the already existing part of the framework without any code change.  
Note: need to do definitive check (both for 1D and 2D) with collision data  
in order to verify that the shapes are correct (hard to tell with cosmics...)  
- - -
  
  
