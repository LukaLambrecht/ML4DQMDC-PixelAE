# get oms data  
  
### get\_oms\_api()  
```text  
get an OMSAPI instance  
takes no input arguments, as the configuration parameters are unlikely to change very often  
if needed, these parameters can be changed in the file urls.py  
```  
  
  
### get\_oms\_data( omsapi, api\_endpoint, runnb, extrafilters=[], extraargs={}, sort=None, attributes=[])  
```text  
query some data from OMS  
input arguments:  
- omsapi: an OMSAPI instance, e.g. created by get_oms_api()  
- api_endpoint: string, target information, e.g. 'runs' or 'lumisections'  
  (see the readme for a link where the available endpoints are listed)  
- runnb: run number(s) to retrieve the info for,  
  either integer (for single run) or tuple or list of two elements (first run and last run)  
  (can also be None to not filter on run number but this is not recommended)  
- extrafilters: list of extra filters (apart from run number),  
  each filter is supposed to be a dict of the form {'attribute_name':<name>,'value':<value>,'operator':<operator>}  
  where <name> must be a valid field name in the OMS data, <value> its value, and <operator> chosen from "EQ", "NEQ", "LT", "GT", "LE", "GE" or "LIKE"  
- extraargs: dict of custom key/value pairs to add to the query  
  (still experimental, potentially usable for changing the granularity from 'run' to 'lumisection' for e.g. L1 trigger rates, see example.ipynb)  
- sort: valid field name in the OMS data by which to sort  
- attributes: list of valid field names in the OMS data to return (if not specified, all information is returned)  
```  
  
  
### get\_oms\_response\_attribute( omsresponse, attribute )  
```text  
small helper function to retrieve a list of values for a single attribute  
input arguments:  
- omsresponse: the json-like object returned by get_oms_data  
- attribute: name of one of the attributes present in omsresponse  
```  
  
  
