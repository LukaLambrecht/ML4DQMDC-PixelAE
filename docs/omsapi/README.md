**OMS API: retrieve information from the OMS database**

Collection of tools for obtaining OMS information in json-like format.  
Note: this functionality supersedes the older version in the omsinterface folder!  

References:  
The code is based on the oms api repository here: [https://gitlab.cern.ch/cmsoms/oms-api-client](https://gitlab.cern.ch/cmsoms/oms-api-client). The file omsapi.py in this folder is a direct copy of the omsapi/\_\_init\_\_.py file in that repository, as recommended by the developers to get it running on SWAN. See also these [slides](https://indico.cern.ch/event/997758/contributions/4191705/attachments/2173881/3670409/OMS%20CERN%20OpenID%20migration%20-%20update.pdf) for further info on the setup of the app and this [site](https://cmsoms.cern.ch/agg/api/v1/version/endpoints) for the available endpoints.

How to use:  

- You will need to authenticate through an application registered with the OMS developer team. Either contact me on llambrec@cern.ch so I can send you my application ID and client secret, or create your own as explained below.  
- Open example.ipynb for some examples. You need to import get\_oms\_api.py, then create an OMSAPI instance via get\_oms\_api() (only once, can be re-used for multiple queries) and then query the information via get\_oms\_data( \<arguments\> ). See example.ipynb or get\_oms\_data.py for details.

How to create a personal application for authentication:  

- You will need to register a personal application ID and client secret with the OMS developer team. See the slides linked above on how to do that (only slide 4-6 are relevant, the rest has been taken care of). You will receive an application ID and client secret (both are just string-like variables).   
- Create a new python file in this folder called clientid.py and define two variables in there:  
API\_CLIENT\_ID = '\<your application ID\>'  
API\_CLIENT\_SECRET = '\<your client secret\>'  
- That should be all!  
