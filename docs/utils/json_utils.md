# json utils  
  
### loadjson( jsonfile )  
```text  
load the content of a json file into a python object  
input arguments:  
- jsonfile: the name (or full path if needed) to the json file to be read  
output:  
- an dict object as specified in the note below  
note: the json file is supposed to contain an object like this example:  
      { "294927": [ [ 55,85 ], [ 95,105] ], "294928": [ [1,33 ] ] }  
      although no explicit checking is done in this function,   
      objects that don't have this structure will probably lead to errors further in the code  
```  
  
  
### writejson( jsondict, outputfile, overwrite=False )  
```text  
inverse function of loadjson  
input arguments  
- jsondict: dict object to be written to a json file  
- outputfile: output file to be written, extension '.json' will be appended automatically  
- overwrite: boolean whether to overwrite outputfile if it exists (default: throw exception)  
```  
  
  
### injson\_single( run, lumi, jsondict )  
```text  
helper function for injson, only for internal use  
input arguments:  
- run and lumi are integers  
- jsondict is an object loaded from a json file  
output:  
- boolean whether the run/lumi combination is in the json dict  
```  
  
  
### injson( run, lumi, jsonfile=None, jsondict=None )  
```text  
find if a run and lumi combination is in a given json file  
input arguments:  
- run and lumi: integers or (equally long) arrays of integers  
- jsonfile: a path to a json file  
- jsondict: a dict loaded from a json file  
  note: either jsonfile or jsondict must not be None!  
output:   
boolean or array of booleans (depending on run and lumi)  
```  
  
  
### isgolden(run, lumi)  
```text  
find if a run and lumi combination is in the golden json file  
input arguments:  
- run and lumi: either integers or (equally long) arrays of integers  
```  
  
  
### isdcson(run, lumi)  
```text  
find if a run and lumi combination is in DCS-only json file  
input arguments:  
- run and lumi: either integers or (equally long) arrays of integers  
```  
  
  
### ispixelgood(run, lumi)  
```text  
find if a run and lumi combination is in the json with good pixel flag  
note: this json was custom generated in run regisitry and not official!  
```  
  
  
### ispixelbad(run, lumi)  
```text  
find if a run and lumi combination is in the json with bad pixel flag  
note: this json was custom generated in run registry and not official!  
note: not simply the negation of ispixelgood! json has more relaxed conditions on DCS-like criteria.  
```  
  
  
### plainlist\_to\_rangelist( plainlist )  
```text  
helper function for tuplelist_to_jsondict, only for internal use  
input arguments:  
- plainlist: a list of integers in increasing order, must have length >= 2  
output:  
- a list lists representing ranges  
example: [1,2,3,5,6] -> [ [1,3], [5,6] ]  
```  
  
  
### rangelist\_to\_plainlist( rangelist )  
```text  
inverse function of plainlist_to_rangelist, for internal use only  
```  
  
  
### tuplelist\_to\_jsondict( tuplelist )  
```text  
convert a list of tuples of format (run number, [lumisection numbers]) to json dict  
```  
  
  
### jsondict\_to\_tuplelist( jsondict )  
```text  
inverse function of tuplelist_to_jsondict  
```  
  
  
### get\_lcs( jsonlist )  
```text  
return a jsondict object that is the largest common subset (LCS) between the jsondict objects in jsonlist  
input arguments:  
- jsonlist: a list of dicts in the conventional json format,   
  so each element in jsonlist must be e.g. { "294927": [ [ 55,85 ], [ 95,105] ], "294928": [ [1,33 ] ] }  
remark: this is probably not the most efficient implementation, open for improvement...   
```  
  
  
