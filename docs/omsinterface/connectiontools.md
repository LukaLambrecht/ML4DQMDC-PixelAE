# connectiontools  
  
**Tools for checking connectivity to specific URLs and obtaining cookies**  

The functions in this script are not my own, but largely based on the wbmcrawler and cernrequests packages.  
See the readme file in this directory for more information.  

For normal users these functions should not be called directly, everything is handled by a single call to get_oms_data.py / get_oms_data.  
See get_oms_data.py in this directory for more information.
- - -
  
  
### check\_connectivity  
full signature:  
```text  
def check_connectivity(url)  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### get\_cookies  
full signature:  
```text  
def get_cookies(url, authmode, **kwargs)  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### get\_cookies\_from\_certificate  
full signature:  
```text  
def get_cookies_from_certificate(url, certificate, **kwargs)  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### get\_cookies\_from\_login  
full signature:  
```text  
def get_cookies_from_login(url, login, **kwargs)  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### \_construct\_certificate\_authentication\_url  
full signature:  
```text  
def _construct_certificate_authentication_url(login_redirect_url)  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### \_extract\_login\_form  
full signature:  
```text  
def _extract_login_form( xml_response_content )  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
### \_modify\_xml\_content  
full signature:  
```text  
def _modify_xml_content( xml_response_content )  
```  
comments:  
```text  
(no valid documentation found)  
```  
  
  
