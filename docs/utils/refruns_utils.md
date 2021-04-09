# refruns utils  
  
- - -    
## get\_reference\_run( runnb, jsonlist=None, jsonfile='json\_allRunsRefRuns.json' )  
**get the reference run for a given run number**  
input arguments:  
- runnb: integer representing a run number.  
- jsonlist: list matching run numbers to reference run numbers.  
note: the list is supposed to contain dicts with keys 'run\_number' and 'reference\_run\_number',  
this convention is based on the json file provided by the tracker group.  
note: if jsonlist is None, jsonfile (see below) will be opened and a jsonlist read from it.  
- jsonfile: path to json file matching run numbers to reference run numbers.  
note: the json file must contain a list of dicts with keys 'run\_number' and 'reference\_run\_number', as explained above.  
note: ignored if jsonlist is not None.  
output:  
integer representing the reference run number for the given run.  
if the given run is not in the json, -1 is returned.  
  
