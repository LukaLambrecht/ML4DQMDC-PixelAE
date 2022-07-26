# omsapi  
  
**OMS API class**

Copied from the OMS developers  
- - -
  
  
- - -
## [class] OMSApiException  
comments:  
```text  
(no valid documentation found)  
```  
- - -  
  
- - -
## [class] OMSQuery  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; \_\_init\_\_  
full signature:  
```text  
def __init__(self, base_url, resource, verbose, cookies, oms_auth, cert_verify, retry_on_err_sec, proxies)  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; \_attr\_exists  
full signature:  
```text  
def _attr_exists(self, attr)  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; \_load\_meta  
full signature:  
```text  
def _load_meta(self)  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; \_warn  
full signature:  
```text  
def _warn(self, message, raise_exc=False)  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; set\_verbose  
full signature:  
```text  
def set_verbose(self, verbose)  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; set\_validation  
full signature:  
```text  
def set_validation(self, attribute_validation)  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; attrs  
full signature:  
```text  
def attrs(self, attributes=None)  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; filters  
full signature:  
```text  
def filters(self, filters)  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; filter  
full signature:  
```text  
def filter(self, attribute, value, operator="EQ")  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; clear\_filter  
full signature:  
```text  
def clear_filter(self)  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; sort  
full signature:  
```text  
def sort(self, attribute, asc=True)  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; paginate  
full signature:  
```text  
def paginate(self, page=1, per_page=10)  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; include  
full signature:  
```text  
def include(self, key)  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; custom  
full signature:  
```text  
def custom(self, key, value=None)  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; data\_query  
full signature:  
```text  
def data_query(self)  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; data  
full signature:  
```text  
def data(self)  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; meta  
full signature:  
```text  
def meta(self)  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; get\_request  
full signature:  
```text  
def get_request(self, url, verify=False)  
```  
comments:  
```text  
(no valid documentation found)  
```  
- - -  
  
- - -
## [class] OMSAPIOAuth  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; \_\_init\_\_  
full signature:  
```text  
def __init__(self, client_id, client_secret, audience="cmsoms-prod", cert_verify=True, proxies={}, retry_on_err_sec=0)  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; auth\_oidc  
full signature:  
```text  
def auth_oidc(self)  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; auth\_oidc\_req  
full signature:  
```text  
def auth_oidc_req(self)  
```  
comments:  
```text  
(no valid documentation found)  
```  
- - -  
  
- - -
## [class] OMSAPI  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; \_\_init\_\_  
full signature:  
```text  
def __init__(self, api_url="https://cmsoms.cern.ch/agg/api", api_version="v1", verbose=True, cert_verify=True, retry_on_err_sec=0, proxies={})  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; query  
full signature:  
```text  
def query(self, resource, query_validation=True)  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; auth\_oidc  
full signature:  
```text  
def auth_oidc(self, client_id, client_secret, audience="cmsoms-prod", proxies={})  
```  
comments:  
```text  
(no valid documentation found)  
```  
### &#10551; auth\_krb  
full signature:  
```text  
def auth_krb(self, cookie_path="ssocookies.txt")  
```  
comments:  
```text  
(no valid documentation found)  
```  
- - -  
  
### rm\_file  
full signature:  
```text  
def rm_file(filename)  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
