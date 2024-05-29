#!/usr/bin/env python
# coding: utf-8

# **A collection of useful basic functions for manipulating json files.**  
# Functionality includes:
# - reading and writing json files for given sets of run numbers and lumisection numbers
# - checking if a given run number, lumisection number or combination is present in a given json file
# 
# Note that the json files are always assumed to contain the following structure:  
# - dict  
#   - run number (in string format)  
#     - list  
#       - list of two elements  
#         - starting lumisection number, ending lumisection number  
# 
# Example:  
# { "294927": \[ \[ 55,85 \], \[ 95,105\] \] } 
# 
# There is one exception to this rule: instead of \[ start, stop \], the lumisection list can also be \[ -1 \], which is short for all lumisections within that run.


### imports

# external modules
import os
import sys
import json
import numpy as np


### reading and writing json files

def loadjson( jsonfile ):
    ### load the content of a json file into a python object
    # input arguments:
    # - jsonfile: the name (or full path if needed) to the json file to be read
    # output:
    # - an dict object as specified in the note below
    # note: the json file is supposed to contain an object like this example:
    #       { "294927": [ [ 55,85 ], [ 95,105] ], "294928": [ [1,33 ] ] }
    #       although no explicit checking is done in this function, 
    #       objects that don't have this structure will probably lead to errors further in the code
    if not os.path.exists(jsonfile):
        raise Exception('ERROR in json_utils.py / loadjson: requested json file {} does not seem to exist...'.format(jsonfile))
    with open(jsonfile) as f: jsondict = json.load(f)
    return jsondict

def writejson( jsondict, outputfile, overwrite=False ):
    ### inverse function of loadjson
    # input arguments
    # - jsondict: dict object to be written to a json file
    # - outputfile: output file to be written, extension '.json' will be appended automatically
    # - overwrite: boolean whether to overwrite outputfile if it exists (default: throw exception)
    if not overwrite and os.path.exists(outputfile):
        raise Exception('ERROR in json_utils.py / writejson: requested output file already exists.'
                       +' You can suppress this error by giving "overwrite=True" as additional argument')
    with open(outputfile,'w') as f: json.dump(jsondict,f)


### checking if given run/lumi values are in a given json object

def injson_single( run, lumi, jsondict ):
    ### helper function for injson, only for internal use
    # input arguments:
    # - run and lumi are integers
    # - jsondict is an object loaded from a json file
    # output:
    # - boolean whether the run/lumi combination is in the json dict
    run = str(run)
    if not run in jsondict: return False
    lumiranges = jsondict[run]
    for lumirange in lumiranges:
        if( len(lumirange)==1 and lumirange[0]<0 ):
            return True
        if( lumi>=lumirange[0] and lumi<=lumirange[1] ): 
            return True
    return False

def injson( run, lumi, jsonfile=None, jsondict=None ):
    ### find if a run and lumi combination is in a given json file
    # input arguments:
    # - run and lumi: integers or (equally long) arrays of integers
    # - jsonfile: a path to a json file
    # - jsondict: a dict loaded from a json file
    #   note: either jsonfile or jsondict must not be None!
    # output: 
    # boolean or array of booleans (depending on run and lumi)
    
    # check the json object to use
    if( jsonfile is None and jsondict is None ):
        raise Exception('ERROR in json_utils.py / injson: both arguments jsonfile and jsondict are None. Specify one of both!')
    if( jsonfile is not None and jsondict is not None ):
        raise Exception('ERROR in json_utils.py / injson: both arguments jsonfile and jsondict are given, which leads to ambiguities. Omit one of both!')
    if jsondict is None:
        jsondict = loadjson( jsonfile )
        
    # check if single or multiple run/lumi combinations need to be assessed    
    if not hasattr(run,'__len__') and not isinstance(run,str):
        run = [run]; lumi = [lumi]
    res = np.zeros(len(run),dtype=np.int8)
    
    # check for all run/lumi combinations if they are in the json object
    for i,(r,l) in enumerate(zip(run,lumi)):
        if injson_single( r, l, jsondict ): res[i]=1
    res = res.astype(bool)
    if len(res)==1: res = res[0]
    return res

def getjsondir():
    ### internal helper function returning the path to where json files are stored
    thisdir = os.path.abspath(os.path.dirname(__file__))
    jsondir = os.path.join(thisdir,'../jsons')
    # the above method does not work in compiled executable version;
    # need different approach in that case.
    # note: the json directory must be added as 'ML4DQM-DC/jsons' under the temporary directory 
    #       created by the executable, see the gui.spec file.
    if not os.path.exists(jsondir):
        jsondir = sys._MEIPASS
        jsondir = os.path.join(jsondir,'ML4DQM-DC/jsons')
    return jsondir

def isgolden(run, lumi, year=None):
    ### find if a run and lumi combination is in the golden json file
    # input arguments:
    # - run and lumi: either integers or (equally long) arrays of integers
    
    jsonloc2017 = os.path.join( getjsondir(), 'json_GOLDEN_2017.txt' ) 
    # legacy reprocessing; from: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt
    jsonloc2018 = os.path.join( getjsondir(), 'json_GOLDEN_2018.txt' )
    # legacy reprocessing; from: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/Legacy_2018/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt
    jsonloc2022 = os.path.join( getjsondir(), 'json_GOLDEN_2022.txt' )
    # from: /eos/user/c/cmsdqm/www/CAF/certification/Collisions22/Cert_Collisions2022_355100_362760_Golden.json
    jsonloc2024 = 'ERROR: golden json for 2024 not yet defined.'
    if year=='2017': return injson(run,lumi,jsonfile=jsonloc2017)
    elif year=='2018': return injson(run,lumi,jsonfile=jsonloc2018)
    elif year=='2022': return injson(run,lumi,jsonfile=jsonloc2022)
    elif year=='2024': return injson(run,lumi,jsonfile=jsonloc2024)
    else: return (injson(run,lumi,jsonfile=jsonloc2017)
                  + injson(run,lumi,jsonfile=jsonloc2018)
                  + injson(run,lumi,jsonfile=jsonloc2022)
                  + injson(run,lumi,jsonfile=jsonloc2024))

def isdcson(run, lumi, year=None):
    ### find if a run and lumi combination is in DCS-only json file
    # input arguments:
    # - run and lumi: either integers or (equally long) arrays of integers
    
    jsonloc2017 = os.path.join( getjsondir(), 'json_DCSONLY_2017.txt' )
    # from: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/DCSOnly/json_DCSONLY.txt
    jsonloc2018 = os.path.join( getjsondir(), 'json_DCSONLY_2018.txt' )
    # from: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/DCSOnly/json_DCSONLY.txt
    jsonloc2022 = os.path.join( getjsondir(), 'json_DCSONLY_2022.txt' )
    # from: /eos/user/c/cmsdqm/www/CAF/certification/Collisions22/DCSOnly_JSONS/Cert_Collisions2022_352416_362760_eraABCDEFG_DCSOnly_TkPx.json
    jsonloc2024 = os.path.join( getjsondir(), 'json_DCSONLY_2024.txt' )
    # from: /eos/user/c/cmsdqm/www/CAF/certification/Collisions24/DCSOnly_JSONS/dailyDCSOnlyJSON/Collisions24_13p6TeV_378981_381309_DCSOnly_TkPx.json
    # to be updated as 2024 data taking progresses
    if year=='2017': return injson(run,lumi,jsonfile=jsonloc2017)
    elif year=='2018': return injson(run,lumi,jsonfile=jsonloc2018)
    elif year=='2022': return injson(run,lumi,jsonfile=jsonloc2022)
    elif year=='2024': return injson(run,lumi,jsonfile=jsonloc2024)
    else: return (injson(run,lumi,jsonfile=jsonloc2017) 
                  + injson(run,lumi,jsonfile=jsonloc2018)
                  + injson(run,lumi,jsonfile=jsonloc2022)
                  + injson(run,lumi,jsonfile=jsonloc2024))


### conversions with other useful formats

def plainlist_to_rangelist( plainlist ):
    ### helper function for tuplelist_to_jsondict, only for internal use
    # input arguments:
    # - plainlist: a list of integers in increasing order, must have length >= 2
    # output:
    # - a list lists representing ranges
    # example: [1,2,3,5,6] -> [ [1,3], [5,6] ]
    
    if len(plainlist)==0: return []
    if len(plainlist)==1: return [[plainlist[0],plainlist[0]]]
    start_index = 0
    stop_index = 1
    rangelist = []
    while stop_index < len(plainlist):
        if plainlist[stop_index]==plainlist[stop_index-1]+1:
            stop_index += 1
        else:
            rangelist.append( [ plainlist[start_index],plainlist[stop_index-1] ] )
            start_index = stop_index
            stop_index = stop_index+1
    rangelist.append( [ plainlist[start_index],plainlist[stop_index-1] ] )
    return rangelist
    

def rangelist_to_plainlist( rangelist ):
    ### inverse function of plainlist_to_rangelist, for internal use only
    plainlist = []
    for el in rangelist:
        if len(el)!=2:
            raise Exception('ERROR in json_utils.py / rangelist_to_plainlist: found range specifier with length {}'.format(len(el))
                           +' while 2 is required [first, last]')
        for number in range(el[0],el[1]+1):
            plainlist.append(number)
    return plainlist


def tuplelist_to_jsondict( tuplelist ):
    ### convert a list of tuples of format (run number, [lumisection numbers]) to json dict
    jsondict = {}
    for el in tuplelist:
        runnb = el[0]
        lslist = el[1]
        lumiranges = []
        if( len(lslist)<1 ): continue
        if( len(lslist)==1 and lslist[0]<0 ): lumiranges = [[lslist[0]]]
        elif( len(lslist)==1 ): lumiranges = [[lslist[0],lslist[0]]]
        else: lumiranges = plainlist_to_rangelist( lslist )
        jsondict[str(runnb)] = lumiranges
    return jsondict

def jsondict_to_tuplelist( jsondict ):
    ### inverse function of tuplelist_to_jsondict
    tuplelist = []
    for runnb in jsondict.keys():
        lumiranges = jsondict[runnb]
        lslist = []
        if( len(lumiranges)==1 and len(lumiranges[0])==1 and lumiranges[0][0]<0 ):
            lslist = [lumiranges[0][0]]
        else:
            lslist = rangelist_to_plainlist( lumiranges )
        tuplelist.append( (int(runnb), lslist) )
    return tuplelist




def get_lcs( jsonlist ):
    ### return a jsondict object that is the largest common subset (LCS) between the jsondict objects in jsonlist
    # input arguments:
    # - jsonlist: a list of dicts in the conventional json format, 
    #   so each element in jsonlist must be e.g. { "294927": [ [ 55,85 ], [ 95,105] ], "294928": [ [1,33 ] ] }
    # remark: this is probably not the most efficient implementation, open for improvement... 
    
    if( len(jsonlist)==1 ): return jsonlist[0]
    lcs = {}
    # loop over run numbers present in first jsondict
    for runnb in jsonlist[0].keys():
        # get the range of lumis for this run number in first json dict
        lumiranges = jsonlist[0][runnb]
        commonls = rangelist_to_plainlist(lumiranges)
        # loop over other json dicts and check overlap for this run number and lumi ranges
        hascommon = True
        #print(runnb)
        #print(commonls)
        for jsondict in jsonlist[1:]:
            if runnb not in jsondict.keys(): 
                hascommon = False; break
            lumiranges_other = jsondict[runnb]
            commonls = list(set(commonls) & set(rangelist_to_plainlist(lumiranges_other)))
            if len(commonls)==0:
                hascommon = False; break
        if not hascommon: continue
        lcs[runnb] = plainlist_to_rangelist( commonls )
    return lcs





