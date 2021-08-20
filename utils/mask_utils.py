#!/usr/bin/env python
# coding: utf-8

# **Utilities for working with HistStruct masks**
# 
# Mostly meant for internal use.



### imports

# local modules




def get_combined_name( masklist ):
    ### concatenate all the masknames in masklist to a combined name
    # input arguments:
    # - masklist: list of strings
    # output:
    # string with contatenated name
    name = masklist[0]
    for mask in masklist[1:]:
        name = name+'_'+mask
    return name





