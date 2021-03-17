###################################################################
# some tools to convert (python) code comments into documentation #
###################################################################

import sys
import os

class Comment(object):
    ### a general class for comment text and its formatting to markdown

    def __init__( self, rawtext ):
        self.rawtext = rawtext
        self.mdtext = rawtext

    def get_raw( self ):
        return self.rawtext

    def get_md( self ):
        return self.mdtext

    def format( self ):
        ### abstract method that must be overloaded by child classes
        raise NotImplementedError('ERROR: the class {}'.format(type(self))
                +' does not seem to implement the "format" method')

    def __str__( self ):
        return self.mdtext

    def strip_whitespace( self ):
        ### strip all indentation whitespace
        lines = self.mdtext.split('\n')
        lines = [line.strip(' ').strip('\t') for line in lines]
        self.mdtext = '\n'.join(lines)
        self.mdtext = self.mdtext.strip('\n')

    def get_lines( self ):
        return self.mdtext.split('\n')


class DefComment(Comment):
    ### comment class for function definitions

    def __init__( self, defname, rawtext ):
        ### initializer
        # defname: name of the definition (e.g. class name or function signature)
        # rawtext: raw comment text
        super( DefComment, self ).__init__( rawtext )
        self.defname = defname

    def format( self, level=2 ):
        ### implement total formatting specific for function or class definition comments
        self.mdtext = self.rawtext
        self.strip_whitespace()
        lines = self.get_lines()
        for i,line in enumerate(lines):
            if line[:3]=='###':
                line = line.strip('# ')
                line = '**'+line+'**'
            elif line[:1]=='#':
                line = line.strip('# ')
            lines[i] = line
        self.mdtext = '\n'.join(lines)
        self.mdtext = '#'*level+' '+self.defname+'\n'+self.mdtext
        if level==2: self.mdtext = '- - -  \n' + self.mdtext
        self.mdtext = self.mdtext.replace('\n','  \n')



class CommentCollection(object):

    def __init__( self ):
        self.comments = []

    def get_comments( self ):
        return self.comments[:]

    def read_defcomments_from_file( self, filename ):
        f = open(filename,'r')
        lines = f.readlines()
        f.close()
        # strip tabs and spaces but store some indentation info
        levels = [2]*len(lines)
        for i,line in enumerate(lines):
            if( line[0]==' ' or line[0]=='\t' ): levels[i] = 3
            lines[i] = line.strip(' ').strip('\t')
        # scan for 'def' or 'class' keywords at the beginning of a line
        for i,line in enumerate(lines):
            if( line[:3]=='def' or line[:5]=='class' ):
                j = i
                # special treatment for definitions spread over multiple lines
                while (line.count('(')!=line.count(')')):
                    j += 1
                    line = line.rstrip('\n') + ' ' + lines[j]
                # format the definition name
                defname = line.replace('def ','').replace('class ','').strip(':\n')
                # add all consecutive lines that start with a '#' character
                rawtext = ''
                while lines[j+1].strip(' ').strip('\t')[0]=='#':
                    rawtext += lines[j+1]
                    j += 1
                if rawtext=='': rawtext = '(no valid documentation found)'
                defcomment = DefComment( defname, rawtext )
                defcomment.format( level=levels[i] )
                self.comments.append( defcomment ) 


if __name__=='__main__':
    ### testing section

    testfile = os.path.abspath(sys.argv[1])
    
    ccol = CommentCollection()
    ccol.read_defcomments_from_file( testfile )
    for c in ccol.get_comments(): 
        print(c)
