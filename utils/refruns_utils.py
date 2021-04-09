#!/usr/bin/env python
# coding: utf-8



### imports

# external modules
import os
import json

# local modules
import json_utils as jsonu




def get_reference_run( runnb, jsonlist=None, jsonfile='json_allRunsRefRuns.json' ):
    ### get the reference run for a given run number
    # input arguments:
    # - runnb: integer representing a run number.
    # - jsonlist: list matching run numbers to reference run numbers.
    #   note: the list is supposed to contain dicts with keys 'run_number' and 'reference_run_number',
    #         this convention is based on the json file provided by the tracker group.
    #   note: if jsonlist is None, jsonfile (see below) will be opened and a jsonlist read from it.
    # - jsonfile: path to json file matching run numbers to reference run numbers.
    #   note: the json file must contain a list of dicts with keys 'run_number' and 'reference_run_number', as explained above.
    #   note: ignored if jsonlist is not None.
    # output:
    # integer representing the reference run number for the given run.
    # if the given run is not in the json, -1 is returned.
    
    if jsonlist is None:
        if not os.path.exists(jsonfile):
            raise Exception('ERROR in refruns_utils.py / get_reference_run: file {} does not exist'.format(jsonfile))
        jsonlist = jsonu.loadjson(jsonfile)
        
    return next((item['reference_run_number'] for item in jsonlist if item['run_number'] == runnb), -1)





