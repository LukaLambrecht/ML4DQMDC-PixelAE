### Tools for reading (nano)DQMIO files

#### Use cases
The tools in this folder can be used to:  

- read a DQMIO or nanoDQMIO files and extract the monitoring elements.  
- access CMS DAS and retrieve all files in a given dataset.  
- convert a collection of nanoDQMIO files to other formats, both interactively and in job submission on lxplus and your local T2 cluster.  

#### Where to start  
- The scripts `harvest_nanodqmio_to_*.py` read one or more (nano)DQMIO files, select a single monitoring element, and write it to a different format. You can run each of these scripts with the option `-h` to see the available options.  
- The script `harvest_nanodqmio_submit.py` is a job submission script that wraps any of the scripts above in a Condor job. Run with the option `-h` to see the available options. 
- The script `harvest_nanodqmio_submitmultiple.py` has similar functionality to `harvest_nanodqmio_submit.py` but you can specify multiple monitoring elements at once that will each be written to their own output file. Run with the option `-h` to see the available options.  
- The folder `copydastolocal` contains a few scripts to copy remote files or even entire datasets to a local directory, to be used as a backup in case the remote file reading does not work. See the `README` there for more info. 
- The folder `jsons` contains an example json file needed as input for `harvest_nanodqmio_submitmultiple.py`, specifying the monitoring elements to read and their respective output files. You can put your own files with the monitoring elements of your choice in the same directory and specify the file as an input argument to `harvest_nanodqmio_submitmultiple.py`.  
- The folder `src` contains the actual (nano)DQMIO reader class and some other tools. You would probably not need to go here unless you found a bug.
- The folder 'test' contains some testing notebooks that can run e.g. on SWAN. Not needed anymore.  

#### Things to keep in mind when trying to read files from DAS:  
- You will need a valid grid certificate. Create one using `voms-proxy-init --voms cms`.  
- The scripts in this folder should contain the correct export command. If you still get an error concerning X509\_USER\_PROXY, you can run the command `export X509_USER_PROXY=path` (where `path` should be replaced by the path to where you stored the proxy created in the previous step) and try again.

#### Special instructions for job submission:  
- You will need a valid grid certificate if accessing remote files via DAS (see above). Copy the proxy to a location that is accessible from the cluster nodes (e.g. somewhere in you home folder) and make sure to pass the path to it as the `proxy` argument to the job submission script.  
- You might also need to set a CMSSW environment, depending on the configuration of your cluster. At least on lxplus this appears to be needed. You can do this using the `cmssw` argument to the job submission script.  
