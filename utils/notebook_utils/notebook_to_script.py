#!/usr/bin/env python
# coding: utf-8



import os
import re




def save_notebook_as_script( notebook_name ):
    
    # make sure notebook_name has correct extension:
    # take extension from notebook_name argument,
    # but in case it has no extension, append '.ipynb' by default
    notebook_name_split = os.path.splitext(notebook_name)
    if len(notebook_name_split[1])==0:
        notebook_name = notebook_name_split[0]+'.ipynb'
        
    # call jupyter command line functionality
    os.system('jupyter nbconvert --to script --PythonExporter.exclude_markdown=True {}'.format(notebook_name) )
    script_name = os.path.splitext(notebook_name)[0]+'.py'
    
    # re-read the created script and remove some unwanted lines
    with open(script_name, 'r') as f:
        lines = f.readlines()
    with open(script_name, 'w') as f:
        for line in lines:
            # avoid writing some lines to the python script
            if 'save_notebook_as_script' in line and not 'def' in line: continue
            if re.match('# In\[.+\]',line): continue
            f.write(line)





