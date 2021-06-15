# omsapi  
  
pylint: disable=W0702,R0902
- - -
  
  
- - -
## [class] OMSApiException(Exception)  
```text  
(no valid documentation found)  
```  
- - -  
  
- - -
## [class] OMSQuery(object)  
```text  
(no valid documentation found)  
```  
### &#10551; \_\_init\_\_(self, base\_url, resource, verbose, cookies, oms\_auth, cert\_verify, retry\_on\_err\_sec, proxies)  
```text  
(no valid documentation found)  
```  
### &#10551; \_attr\_exists(self, attr)  
```text  
(no valid documentation found)  
```  
### &#10551; \_load\_meta(self)  
```text  
(no valid documentation found)  
```  
### &#10551; \_warn(self, message, raise\_exc=False)  
```text  
(no valid documentation found)  
```  
### &#10551; set\_verbose(self, verbose)  
```text  
(no valid documentation found)  
```  
### &#10551; set\_validation(self, attribute\_validation)  
```text  
(no valid documentation found)  
```  
### &#10551; attrs(self, attributes=None)  
```text  
(no valid documentation found)  
```  
### &#10551; filters(self, filters)  
```text  
(no valid documentation found)  
```  
### &#10551; filter(self, attribute, value, operator="EQ")  
```text  
(no valid documentation found)  
```  
### &#10551; clear\_filter(self)  
```text  
(no valid documentation found)  
```  
### &#10551; sort(self, attribute, asc=True)  
```text  
(no valid documentation found)  
```  
### &#10551; paginate(self, page=1, per\_page=10)  
```text  
(no valid documentation found)  
```  
### &#10551; include(self, key)  
```text  
(no valid documentation found)  
```  
### &#10551; custom(self, key, value=None)  
```text  
(no valid documentation found)  
```  
### &#10551; data\_query(self)  
```text  
(no valid documentation found)  
```  
### &#10551; data(self)  
```text  
(no valid documentation found)  
```  
### &#10551; meta(self)  
```text  
(no valid documentation found)  
```  
### &#10551; get\_request(self, url, verify=False)  
```text  
(no valid documentation found)  
```  
- - -  
  
- - -
## [class] OMSAPIOAuth(object)  
```text  
(no valid documentation found)  
```  
### &#10551; \_\_init\_\_(self, client\_id, client\_secret, audience="cmsoms-prod", cert\_verify=True, proxies={}, retry\_on\_err\_sec=0)  
```text  
(no valid documentation found)  
```  
### &#10551; auth\_oidc(self)  
```text  
(no valid documentation found)  
```  
### &#10551; auth\_oidc\_req(self)  
```text  
(no valid documentation found)  
```  
- - -  
  
- - -
## [class] OMSAPI(object)  
```text  
(no valid documentation found)  
```  
### &#10551; \_\_init\_\_(self, api\_url="https://cmsoms.cern.ch/agg/api", api\_version="v1", verbose=True, cert\_verify=True, retry\_on\_err\_sec=0, proxies={})  
```text  
(no valid documentation found)  
```  
### &#10551; query(self, resource, query\_validation=True)  
```text  
(no valid documentation found)  
```  
### &#10551; auth\_oidc(self, client\_id, client\_secret, audience="cmsoms-prod", proxies={})  
```text  
(no valid documentation found)  
```  
### &#10551; auth\_krb(self, cookie\_path="ssocookies.txt")  
```text  
(no valid documentation found)  
```  
- - -  
  
### rm\_file(filename)  
```text  
(no valid documentation found)  
```  
  
  
