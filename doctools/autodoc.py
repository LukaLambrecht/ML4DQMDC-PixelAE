########################################################################
# script for generating documentation automatically from code comments #
########################################################################

import sys
import os
import commenttools

# define the top level directory of the project
topdir = os.path.abspath('..')
# define top directory for the documentation, relative to project top level
docdir = 'docs'
# define which code directories to take into account, relative to project top level
codedirs = ['utils']

# check if docs directory exists
if not os.path.exists(os.path.join(topdir,docdir)):
    raise Exception('ERROR: documentation directory does not yet exist'
            + ' please create it manually first.')

# loop over .py files in code directories
for codedir in codedirs:
    thiscodedir = os.path.join(topdir,codedir)
    if not os.path.exists(thiscodedir):
        print('WARNING: {} is not a valid directory,'.format(thiscodedir)
                +' it will be ignored.')
        continue
    thisdocdir = os.path.join(topdir,docdir,codedir)
    if not os.path.exists(thisdocdir):
        os.makedirs(thisdocdir)
    pyfiles = ([f for f in os.listdir(thiscodedir) 
                if os.path.splitext(f)[1]=='.py'])
    for pyfile in pyfiles:
        comments = commenttools.CommentCollection()
        comments.read_defcomments_from_file(os.path.join(thiscodedir,pyfile))
        mdfile = os.path.join(thisdocdir,pyfile.replace('.py','.md'))
        with open(mdfile,'w') as f:
            for c in comments.get_comments():
                f.write(str(c)+'  \n  \n')
        
