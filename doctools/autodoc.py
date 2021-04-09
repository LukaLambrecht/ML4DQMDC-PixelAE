########################################################################
# script for generating documentation automatically from code comments #
########################################################################

import sys
import os
import mdfiletools

# define the top level directory of the project
topdir = os.path.abspath('..')
# define top directory for the documentation, relative to project top level
docdir = 'docs'
# define which code directories to take into account, relative to project top level
codedirs = sorted(['utils','src','src/cloudfitters','src/classifiers'])
# define title for site
sitetitle = 'Documentation for the ML4DQM/DC notebooks'

# check if docs directory exists
if not os.path.exists(os.path.join(topdir,docdir)):
    raise Exception('ERROR: documentation directory does not yet exist'
            + ' please create it manually first.')

# initialize yml text
ymltext = 'site_name: {}\n'.format(sitetitle)
ymltext += 'nav:\n'

# copy readme to index
if os.path.exists(os.path.join(topdir,'README.md')):
    os.system('cp {} {}'.format(os.path.join(topdir,'README.md'),
                os.path.join(topdir,docdir,'index.md')))
    ymltext += '    - Home: \'index.md\'\n'

# loop over code directories
for codedir in codedirs:
    # check validity of code dir and clean/make doc dir
    thiscodedir = os.path.join(topdir,codedir)
    if not os.path.exists(thiscodedir):
        print('WARNING: {} is not a valid directory,'.format(thiscodedir)
                +' it will be ignored.')
        continue
    thisdocdir = os.path.join(topdir,docdir,codedir)
    #if not os.path.exists(thisdocdir):
    #    os.makedirs(thisdocdir)
    if os.path.exists(thisdocdir):
        os.system('rm -r {}'.format(thisdocdir))
    os.makedirs(thisdocdir)
    pyfiles = sorted([f for f in os.listdir(thiscodedir) 
                if os.path.splitext(f)[1]=='.py'])
    if len(pyfiles)<1: continue
    # update yml
    level = codedir.count('/')+1
    ymltext += level*4*' '+'- '+codedir.split('/')[-1]+':\n'
    # loop over python files and make the doc files
    for pyfile in pyfiles:
        mdfiletools.py_to_md( pyfile, thiscodedir, pyfile.replace('.py','.md'), thisdocdir)
        # update yml
        ymltext += (level+1)*4*' '+'- '+pyfile.replace('.py','')+': \'{}\'\n'.format(
                os.path.join(codedir,pyfile.replace('.py','.md')))

# write the yml file
ymlfile = os.path.join(topdir,'mkdocs.yml')
with open(ymlfile,'w') as f:
    f.write(ymltext)

# print done
print('done')
