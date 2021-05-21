# omsapi  
  
- - -    
## OMSApiException(Exception)  
(no valid documentation found)  
  
- - -    
## OMSQuery(object)  
(no valid documentation found)  
  
### \_\_init\_\_(self, base\_url, resource, verbose, cookies, oms\_auth, cert\_verify, retry\_on\_err\_sec, proxies)  
(no valid documentation found)  
  
### \_attr\_exists(self, attr)  
(no valid documentation found)  
  
### \_load\_meta(self)  
(no valid documentation found)  
  
### \_warn(self, message, raise\_exc=False)  
(no valid documentation found)  
  
### set\_verbose(self, verbose)  
(no valid documentation found)  
  
### set\_validation(self, attribute\_validation)  
(no valid documentation found)  
  
### attrs(self, attributes=None)  
(no valid documentation found)  
  
### filters(self, filters)  
(no valid documentation found)  
  
### filter(self, attribute, value, operator="EQ")  
(no valid documentation found)  
  
### clear\_filter(self)  
(no valid documentation found)  
  
### sort(self, attribute, asc=True)  
(no valid documentation found)  
  
### paginate(self, page=1, per\_page=10)  
(no valid documentation found)  
  
### include(self, key)  
(no valid documentation found)  
  
### custom(self, key, value=None)  
(no valid documentation found)  
  
### data\_query(self)  
(no valid documentation found)  
  
### data(self)  
(no valid documentation found)  
  
### meta(self)  
(no valid documentation found)  
  
### get\_request(self, url, verify=False)  
(no valid documentation found)  
  
- - -    
## OMSAPIOAuth(object)  
(no valid documentation found)  
  
### \_\_init\_\_(self, client\_id, client\_secret, audience="cmsoms-prod", cert\_verify=True, proxies={}, retry\_on\_err\_sec=0)  
(no valid documentation found)  
  
### auth\_oidc(self)  
(no valid documentation found)  
  
### auth\_oidc\_req(self)  
(no valid documentation found)  
  
- - -    
## OMSAPI(object)  
(no valid documentation found)  
  
### \_\_init\_\_(self, api\_url="https://cmsoms.cern.ch/agg/api", api\_version="v1", verbose=True, cert\_verify=True, retry\_on\_err\_sec=0, proxies={})  
(no valid documentation found)  
  
### query(self, resource, query\_validation=True)  
(no valid documentation found)  
  
### auth\_oidc(self, client\_id, client\_secret, audience="cmsoms-prod", proxies={})  
(no valid documentation found)  
  
### auth\_krb(self, cookie\_path="ssocookies.txt")  
(no valid documentation found)  
  
### rm\_file(filename)  
(no valid documentation found)  
  
