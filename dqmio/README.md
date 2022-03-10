**Tools for reading (nano)DQMIO files**

The tools in this folder can be used to:  
- read a DQMIO or nanoDQMIO files and extract the monitoring elements.  
- access CMS DAS and retrieve all files in a given dataset.  
- convert a collection of nanoDQMIO files to other formats, both interactively and in job submission on local T2 cluster or on lxplus (experimental).  

Things to keep in mind when trying to read files from DAS:  
- You will need a valid grid certificate. Create one using `voms-proxy-init --voms cms`.  

Special instructions for job submission:  
- You will need a valid grid certificate if accessing DAS files (see above).  
- Copy the proxy to a location that is accessible from the cluster nodes (e.g. somewhere in you home folder) and make sure to set the path to it correctly in the submission script.
