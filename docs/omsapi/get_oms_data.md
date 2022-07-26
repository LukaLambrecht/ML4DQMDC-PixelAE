# get oms data  
  
**Functionality to call the OMS API with the correct query based on input parameters**  

How to use?  
Check the readme file in this directory for the required setup!  
In particular, you will need an application ID and client secret to authenticate.  

Once this is ready, you can do the following:  

- Import this module, for example via `from get_oms_data import get_oms_api, get_oms_data, get_oms_response_attribute`  
- Create an instance of the OMS API class using `omsapi = get_oms_api()`  
  This instance can be re-used for all consecutive calls to OMS, no need to recreate it for every call.  
- Make a call to `get_oms_data`, where the first argument is the instance you just created.  
  Other arguments: see the function documentation below.  
- The returned object is a complicated dictionary containing all information.  
  Simply print it to find out its exact structure and how to access exactly the values you need.  
  The function `get_oms_response_attribute` is a small helper function to retrieve a specific attribute from this dictionary.  
  
See the notebook example.ipynb in this directory for some examples!  
- - -
  
  
### get\_oms\_api  
full signature:  
```text  
def get_oms_api()  
```  
comments:  
```text  
get an OMSAPI instance  
takes no input arguments, as the configuration parameters are unlikely to change very often  
if needed, these parameters can be changed in the file urls.py  
```  
  
  
### get\_oms\_data  
full signature:  
```text  
def get_oms_data( omsapi, api_endpoint, runnb=None, fillnb=None, extrafilters=[], extraargs={}, sort=None, attributes=[], limit_entries=1000)  
```  
comments:  
```text  
query some data from OMS  
input arguments:  
- omsapi: an OMSAPI instance, e.g. created by get_oms_api()  
- api_endpoint: string, target information, e.g. 'runs' or 'lumisections'  
  (see the readme for a link where the available endpoints are listed)  
- runnb: run number(s) to retrieve the info for,  
  either integer (for single run) or tuple or list of two elements (first run and last run)  
  (can also be None to not filter on run number but this is not recommended)  
- fillnb: runnb but for fill number instead of run number  
- extrafilters: list of extra filters (apart from run number),  
  each filter is supposed to be a dict of the form {'attribute_name':<name>,'value':<value>,'operator':<operator>}  
  where <name> must be a valid field name in the OMS data, <value> its value, and <operator> chosen from "EQ", "NEQ", "LT", "GT", "LE", "GE" or "LIKE"  
- extraargs: dict of custom key/value pairs to add to the query  
  (still experimental, potentially usable for changing the granularity from 'run' to 'lumisection' for e.g. L1 trigger rates, see example.ipynb)  
- sort: valid field name in the OMS data by which to sort  
- attributes: list of valid field names in the OMS data to return (if not specified, all information is returned)  
- limit_entries: entry limit for output json object  
```  
  
  
### get\_oms\_response\_attribute  
full signature:  
```text  
def get_oms_response_attribute( omsresponse, attribute )  
```  
comments:  
```text  
small helper function to retrieve a list of values for a single attribute  
input arguments:  
- omsresponse: the json-like object returned by get_oms_data  
- attribute: name of one of the attributes present in omsresponse  
```  
  
  
