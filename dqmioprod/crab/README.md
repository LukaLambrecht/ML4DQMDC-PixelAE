### Tools for producing nanoDQMIO via CRAB submission

#### How to use
First, a CMSSW configuration file is needed. You can produce it via the script `makeconf.py`. Run `makeconf.py -h` for a list of required and available options. This script is basically a wrapper around a `cmsDriver` command (provided by the user in a `.txt` file, see the `cmsdriver` folder for an example), with additional post-processing of the configuration file to allow for command-line arguments (needed for CRAB).

It is advisable to test the resulting configuration locally before CRAB submission. This can be done via the script `testconf.py`. This script is a wrapper around the `cmsRun` command, but also supplying the required input file, output file and number of events as command-line arguments in the correct format.

Next, check the `crabconf.py` configuration file for CRAB and change some settings if needed. 

Finally, use the script `crabsubmit.py` to submit CRAB jobs. Run `crabsubmit.py` for a list of all required and available options. The most notable of these are a CMSSW configuration file (produced with `makeconf.py` as explained above) and a sample list containing the CMS DAS names of the data sets to be processed.

#### Status
CRAB submission and processing was successful for a test run, output files are produced correctly. However, many jobs are failing due to too much RAM usage. Need to either increase the memory limit further or reduce the number of threads and/or cores.
