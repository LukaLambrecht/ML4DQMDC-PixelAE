# omstools  
  
**Tools for accessing the OMS database**  

The functions in this script are not my own, but largely based on the wbmcrawler and cernrequests packages.  
See the readme file in this directory for more information.  

For normal users these functions should not be called directly, everything is handled by a single call to get_oms_data.py / get_oms_data.  
See get_oms_data.py in this directory for more information.
- - -
  
  
### check\_oms\_connectivity  
full signature:  
```text  
def check_oms_connectivity()  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### get\_oms\_cookies  
full signature:  
```text  
def get_oms_cookies( authmode, **kwargs )  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### \_get\_oms\_resource\_within\_cern\_gpn  
full signature:  
```text  
def _get_oms_resource_within_cern_gpn(relative_url)  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### \_get\_oms\_resource\_authenticated  
full signature:  
```text  
def _get_oms_resource_authenticated(relative_url, cookies)  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### get\_oms\_resource  
full signature:  
```text  
def get_oms_resource(table, parameters, **kwargs)  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### \_get\_single\_resource  
full signature:  
```text  
def _get_single_resource(table, parameters, **kwargs)  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### get\_run  
full signature:  
```text  
def get_run(run_number, **kwargs)  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### get\_fill  
full signature:  
```text  
def get_fill(fill_number, **kwargs)  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### \_get\_resources\_page  
full signature:  
```text  
def _get_resources_page(table, parameters, page, page_size, **kwargs)  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### get\_resources  
full signature:  
```text  
def get_resources(table, parameters, page_size=PAGE_SIZE, silent=False, **kwargs)  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### get\_runs  
full signature:  
```text  
def get_runs(begin, end, **kwargs)  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### get\_fills  
full signature:  
```text  
def get_fills(begin, end, **kwargs)  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### get\_lumisection\_count  
full signature:  
```text  
def get_lumisection_count(run_number, **kwargs)  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### get\_lumisections  
full signature:  
```text  
def get_lumisections( run_number=None, fill_number=None, start_time=None, end_time=None, **kwargs)  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### get\_hltpathinfos  
full signature:  
```text  
def get_hltpathinfos(run_number, **kwargs)  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### get\_hltpathrates  
full signature:  
```text  
def get_hltpathrates(run_number, path_name, **kwargs)  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### get\_all\_hltpathrates  
full signature:  
```text  
def get_all_hltpathrates(run_number, silent=False, **kwargs)  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### calc\_page\_count  
full signature:  
```text  
def calc_page_count(resource_count, page_size)  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### flatten\_resource  
full signature:  
```text  
def flatten_resource(response)  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### progress\_bar  
full signature:  
```text  
def progress_bar(current, total, text="", filler="#")  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### print\_progress  
full signature:  
```text  
def print_progress(current, total, text="", *args, **kwargs)  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
