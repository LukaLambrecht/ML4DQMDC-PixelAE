# json utils  
  
**A collection of useful basic functions for manipulating json files.**  
Functionality includes:
- reading and writing json files for given sets of run numbers and lumisection numbers
- checking if a given run number, lumisection number or combination is present in a given json file

Note that the json files are always assumed to contain the following structure:  
- dict  
  - run number (in string format)  
    - list  
      - list of two elements  
        - starting lumisection number, ending lumisection number  

Example:  
{ "294927": \[ \[ 55,85 \], \[ 95,105\] \] } 

There is one exception to this rule: instead of \[ start, stop \], the lumisection list can also be \[ -1 \], which is short for all lumisections within that run.
- - -
  
  
### loadjson  
full signature:  
```text  
def loadjson( jsonfile )  
```  
comments:  
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
  
  
### writejson  
full signature:  
```text  
def writejson( jsondict, outputfile, overwrite=False )  
```  
comments:  
```text  
inverse function of loadjson  
input arguments  
- jsondict: dict object to be written to a json file  
- outputfile: output file to be written, extension '.json' will be appended automatically  
- overwrite: boolean whether to overwrite outputfile if it exists (default: throw exception)  
```  
  
  
### injson\_single  
full signature:  
```text  
def injson_single( run, lumi, jsondict )  
```  
comments:  
```text  
helper function for injson, only for internal use  
input arguments:  
- run and lumi are integers  
- jsondict is an object loaded from a json file  
output:  
- boolean whether the run/lumi combination is in the json dict  
```  
  
  
### injson  
full signature:  
```text  
def injson( run, lumi, jsonfile=None, jsondict=None )  
```  
comments:  
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
  
  
### getjsondir  
full signature:  
```text  
def getjsondir()  
```  
comments:  
```text  
internal helper function returning the path to where json files are stored  
```  
  
  
### isgolden  
full signature:  
```text  
def isgolden(run, lumi)  
```  
comments:  
```text  
find if a run and lumi combination is in the golden json file  
input arguments:  
- run and lumi: either integers or (equally long) arrays of integers  
```  
  
  
### isdcson  
full signature:  
```text  
def isdcson(run, lumi)  
```  
comments:  
```text  
find if a run and lumi combination is in DCS-only json file  
input arguments:  
- run and lumi: either integers or (equally long) arrays of integers  
```  
  
  
### plainlist\_to\_rangelist  
full signature:  
```text  
def plainlist_to_rangelist( plainlist )  
```  
comments:  
```text  
helper function for tuplelist_to_jsondict, only for internal use  
input arguments:  
- plainlist: a list of integers in increasing order, must have length >= 2  
output:  
- a list lists representing ranges  
example: [1,2,3,5,6] -> [ [1,3], [5,6] ]  
```  
  
  
### rangelist\_to\_plainlist  
full signature:  
```text  
def rangelist_to_plainlist( rangelist )  
```  
comments:  
```text  
inverse function of plainlist_to_rangelist, for internal use only  
```  
  
  
### tuplelist\_to\_jsondict  
full signature:  
```text  
def tuplelist_to_jsondict( tuplelist )  
```  
comments:  
```text  
convert a list of tuples of format (run number, [lumisection numbers]) to json dict  
```  
  
  
### jsondict\_to\_tuplelist  
full signature:  
```text  
def jsondict_to_tuplelist( jsondict )  
```  
comments:  
```text  
inverse function of tuplelist_to_jsondict  
```  
  
  
### get\_lcs  
full signature:  
```text  
def get_lcs( jsonlist )  
```  
comments:  
```text  
return a jsondict object that is the largest common subset (LCS) between the jsondict objects in jsonlist  
input arguments:  
- jsonlist: a list of dicts in the conventional json format,   
  so each element in jsonlist must be e.g. { "294927": [ [ 55,85 ], [ 95,105] ], "294928": [ [1,33 ] ] }  
remark: this is probably not the most efficient implementation, open for improvement...   
```  
  
  
