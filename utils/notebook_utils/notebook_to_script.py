#!/usr/bin/env python
# coding: utf-8

# **Functionality for automatic conversion between notebooks (.ipynb) and scripts (.py)**
# 
# Motivation:   
# Code is being developed in python notebooks (.ipynb), but this format does not support full import flexibility.  
# One way to import functionality from a notebook into another notebook is using the so-called 'IPython magic', by using the '%run' command.  
# 
# But this has the disadvantage that one cannot resolve potential ambiguity as is usually done using 'import ... (as ...)'.  
# The function save_notebook_as_script allows to easily save each notebook as an python (.py) file that can be imported in other notebooks or python files, avoiding the need for shady IPython magic commands.
# 
# Use: 
# - Import this script in a notebook using 'from notebook_to_script import save_notebook_as_script' or something equivalent.
# - At the end of your notebook, add the following statement: 'save_notebook_as_script( "your notebook name" )'.
# - When running this cell, a file will be created with the same name as your notebook, but with extension .py instead of .ipynb.   
# 
# Notes:
# - The lines containing the import of notebook_to_script and the call to save_notebook_as_script  will not be written to the .py file!  
# - Markdown cells are written to the .py file as comments! Furthermore, comments in code cells are retained just as they are in the .ipynb file.  
# - Keep in mind that you need to explicitly save the notebook before running the 'save_notebook_as_script' function. Else, recent changes in modified cells will not be written to the resulting .py file!  



import os
import re




def save_notebook_as_script( notebook_name ):
    ### save a .ipynb notebook as a .py script
    # input arguments:
    # - notebook_name: name of the notebook to save, usually the current one.
    # notes:
    # - if notebook_name is provided without extension, '.ipynb' will be appended by default
    # - see the readme header for more explanation on how to use this function!
    notebook_name_split = os.path.splitext(notebook_name)
    if len(notebook_name_split[1])==0:
        notebook_name = notebook_name_split[0]+'.ipynb'
        
    # call jupyter command line functionality
    os.system('jupyter nbconvert --to script --PythonExporter.exclude_markdown=False {}'.format(notebook_name) )
    script_name = os.path.splitext(notebook_name)[0]+'.py'
    
    # re-read the created script and remove some unwanted lines
    with open(script_name, 'r') as f:
        lines = f.readlines()
    with open(script_name, 'w') as f:
        for line in lines:
            if line.strip(' ')[0]=='#': 
                # case of comments: keep all comments regardless of further content,
                # except for the cell numbers
                if re.match('# In\[.+\]',line): continue
                f.write(line)
                continue
            # case of code: avoid writing some meta-lines to the python script
            if 'save_notebook_as_script' in line and not 'def' in line: continue
            f.write(line)





