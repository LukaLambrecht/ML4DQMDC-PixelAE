###########################################
# some tools for fomatting markdown files #
###########################################

import sys
import os
import commenttools

def py_to_md( pyfilename, pyfiledir, mdfilename, mdfiledir ):
    ### convert a .py file to a corresponding .md file based on the comments in it

    pyfilepath = os.path.join(pyfiledir,pyfilename)
    # check extensions and existence
    if( os.path.splitext(pyfilename)[-1]!='.py' ):
        print('WARNING in py_to_md: python filename {}'.format(pyfilename)
                +' does not seem to have proper extension, skipping it...')
        return
    if os.path.splitext(mdfilename)[-1]!='.md':
        mdfilename = os.path.splitext(mdfilename)[0]+'.md'
    if not os.path.exists(pyfilepath):
        print('WARNING in py_to_md: python file {}'.format(pyfilepath))
    if not os.path.exists(mdfiledir):
        os.makedirs(mdfiledir)
    mdfilepath = os.path.join(mdfiledir,mdfilename)

    # read the comments
    comments = commenttools.CommentCollection()
    comments.read_defcomments_from_file(pyfilepath)
    with open(mdfilepath,'w') as f:
        # write a title for the markdown file
        title = '# '+mdfilename.replace('.md','').replace('_',' ')+'  \n  \n'
        f.write(title)
        for c in comments.get_comments():
            f.write(str(c)+'  \n  \n')

if __name__=='__main__':

    pyfile = os.path.abspath(sys.argv[1])
    pyfiledir,pyfilename = os.path.split(pyfile)
    py_to_md(pyfilename,pyfiledir,'test.md','.')
