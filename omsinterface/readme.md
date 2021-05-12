**Tools for obtaining OMS information in json format**

References:  
The code is largely based on / copied from the wbmcrawlr tool ([https://github.com/CMSTrackerDPG/wbmcrawlr](https://github.com/CMSTrackerDPG/wbmcrawlr)) and the cernrequests package ([https://github.com/CMSTrackerDPG/cernrequests](https://github.com/CMSTrackerDPG/cernrequests)) by the Tracker DPG. All credits to the Tracker DPG group, all mistakes in copying or modifying are of course my own.

How to use:
- Open example.ipynb
- The configuration should be quite self-explanatory: choose the mode ('run' for run information, 'lumsisections' for per-lumisection information), enter the run number for which to retrieve the information, choose the mode for authentication (via CERN username and password or via CERN grid certificate).
- In case you choose to use a certificate, first edit the file 'cert.py' to set the correct paths to where you stored your certificate and key. Else you will be prompted to enter your username and password.
- Run the cells below, the requested information should now be stored in the specified output json file. Instead, you could also directly use the resulting object in your script without writing it and loading it to a json file.

Notes:
- Preliminary implementation, will be extended to e.g. filter on specific data fields only (for example only keep pileup or luminosity per lumisection), access trigger rates, etc.
- The authentication with a certificate still has an issue. For now, you need both a certificate and a username/password to make the data retrieval work. The method using username and password seems to work without additional certificate.
- For more information on how to obtain a certificate, follow this link: [https://github.com/CMSTrackerDPG/cernrequests#prerequisites](https://github.com/CMSTrackerDPG/cernrequests#prerequisites)
- One could also install the wbmcrawlr tool (from [https://github.com/CMSTrackerDPG/wbmcrawlr](here)) and use it directly. Both methods are equivalent up to now, except these notebooks don't require additional installation of the wbmcrawler tool and cernrequests module.