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
codedirs = sorted(['src','src/cloudfitters','src/classifiers','utils','omsapi'])
# define which other directories to take into account 
# (copy markdown files but do not convert python files)
otherdirs = sorted(['run', 'runswan', 'dqmio', 'dqmio/copydastolocal', 'dqmio/utils', 'tutorials'])
# define title for site
sitetitle = 'Documentation for the ML4DQM/DC code'

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

# loop over code and other directories
alldirs = sorted(codedirs+otherdirs)
for codedir in alldirs:
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
    # get python files
    pyfiles = sorted([f for f in os.listdir(thiscodedir) 
                if os.path.splitext(f)[1]=='.py'])
    # get markdown files
    mdfiles = sorted([f for f in os.listdir(thiscodedir)
                if os.path.splitext(f)[1]=='.md'])
    if len(pyfiles)==0 and len(mdfiles)==0: continue
    # update yml
    level = codedir.count('/')+1
    ymltext += level*4*' '+'- '+codedir.split('/')[-1]+':\n'
    # loop over markdown files and copy them directly to doc dir
    for mdfile in mdfiles:
        os.system('cp {} {}'.format(os.path.join(thiscodedir,mdfile),
                                    os.path.join(thisdocdir,mdfile)))
        ymltext += (level+1)*4*' '+'- '+mdfile.replace('.md','')+': \'{}\'\n'.format(
                os.path.join(codedir,mdfile))
    # loop over python files and make the doc files
    if codedir in codedirs:
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
