# get oms data  
  
**Main function in omsinterface to retrieve information from OMS**  

How to use?  
See the readme file in this directory and the notebook example.ipynb!
- - -
  
  
### get\_oms\_data( mode, run, hltpathname='', authmode='login' )  
```text  
main function for retrieving information from the OMS database  
input arguments:  
- mode: a string representing the type of information to retrieve.  
  the following options are currently supported:  
  'run' -> retrieve information per run  
  'lumisections' -> retrieve information per lumisection  
  'hltpathinfos' -> get information on the available HLT paths for a given run,   
                    in particular their names,   
  'hltrate' -> get the trigger rate of a specified HLT path    
  'hltrates' -> get the trigger rate for all available HLT paths  
- run: a single run number (integer format)  
  note: in case mode is 'run', the run argument can also be a tuple  
  representing a range of runs.  
- hltpathname: the name of a HLT path for which to retrieve the trigger rate.  
  ignored if mode is not 'hltrate'  
- authmode: string representing mode of authentication.  
  choose from 'login' (you will be prompted for your cern username and password)  
  or 'certificate' (requires you to have set up the path to a valid certificate)  
returns:  
- a list or dict (depending on the specifications) containing all information.  
  simply print it to see how to access the exact values you need.  
```  
  
  
