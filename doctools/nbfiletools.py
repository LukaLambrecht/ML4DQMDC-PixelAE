#########################################################
# tools for converting ipynb notebook files to markdown #
#########################################################

import sys
import os
import json

def ipynb_to_md( ipynbfilename, ipynbfiledir, mdfilename, mdfiledir ):
    ### legacy (manual) conversion from ipython notebook file to markdown
    # does not include output, only markdown and source code.
    
    ipynbfilepath = os.path.join(ipynbfiledir, ipynbfilename)
    # check extensions and existence
    if( os.path.splitext(ipynbfilename)[-1]!='.ipynb' ):
        print('WARNING in ipynb_to_md: ipython notebook filename {}'.format(ipynbfilename)
                +' does not seem to have proper extension, skipping it...')
        return
    if os.path.splitext(mdfilename)[-1]!='.md':
        mdfilename = os.path.splitext(mdfilename)[0]+'.md'
    if not os.path.exists(ipynbfilepath):
        print('WARNING in ipynb_to_md: ipython notebook file {}'.format(ipynbfilepath))
    if not os.path.exists(mdfiledir):
        os.makedirs(mdfiledir)
    mdfilepath = os.path.join(mdfiledir,mdfilename)

    # open ipynb file as json dict
    with open(ipynbfilepath, 'r') as f:
        ipynbjson = json.load(f)

    lines = []
    # loop over all cells
    for cell in ipynbjson['cells']:
        # case of code cell
        if cell['cell_type']=='code':
            lines.append('```python\n')
            for line in cell['source']:
                lines.append(line)
            lines.append('\n')
            lines.append('```\n')
            lines.append('Output:\n')
            lines.append('```text\n')
            #for line in cell['outputs']:
            #    lines.append(line)
            # (does not work since 'outputs' has a more involved structure)
            lines.append('\n')
            lines.append('```\n')
        # case of markdown cell
        if cell['cell_type']=='markdown':
            for line in cell['source']:
                lines.append(line)
            lines.append('\n')

    # write output file
    with open(mdfilepath,'w') as f:
        # write a title for the markdown file
        title = '# '+mdfilename.replace('.md','').replace('_',' ')+'  \n  \n'
        f.write(title)
        for line in lines:
            f.write(line)


def ipynb_to_md_nbconvert( ipynbfilename, ipynbfiledir, mdfilename, mdfiledir ):
    ### new conversion from ipython notebook to markdown with jupyter nbconvert tool

    ipynbfilepath = os.path.join(ipynbfiledir, ipynbfilename)
    # check extensions and existence
    if( os.path.splitext(ipynbfilename)[-1]!='.ipynb' ):
        print('WARNING in ipynb_to_md: ipython notebook filename {}'.format(ipynbfilename)
                +' does not seem to have proper extension, skipping it...')
        return
    if os.path.splitext(mdfilename)[-1]!='.md':
        mdfilename = os.path.splitext(mdfilename)[0]+'.md'
    if not os.path.exists(ipynbfilepath):
        print('WARNING in ipynb_to_md: ipython notebook file {}'.format(ipynbfilepath))
    if not os.path.exists(mdfiledir):
        os.makedirs(mdfiledir)
    mdfilepath = os.path.join(mdfiledir,mdfilename)

    # run the jupyter nbconvert tool
    cmd = 'jupyter nbconvert --to markdown {}'.format(ipynbfilepath)
    os.system(cmd)

    # move the markdown file and possible figure directories
    cmd = 'mv {} {}'.format(ipynbfilepath.replace('.ipynb','.md'), mdfilepath)
    os.system(cmd)
    figdir = ipynbfilepath.replace('.ipynb','')+'_files'
    if os.path.exists(figdir):
        cmd = 'mv {} {}'.format(figdir, mdfiledir)
        os.system(cmd)

if __name__=='__main__':

    ipynbfile = os.path.abspath(sys.argv[1])
    ipynbfiledir,ipynbfilename = os.path.split(ipynbfile)
    #ipynb_to_md(ipynbfilename,ipynbfiledir,'test.md','.')
    ipynb_to_md_nbconvert(ipynbfilename,ipynbfiledir,'test.md','.')
