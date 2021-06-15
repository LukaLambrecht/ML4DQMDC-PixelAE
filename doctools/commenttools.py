###################################################################
# some tools to convert (python) code comments into documentation #
###################################################################

import sys
import os

def strip_whitespace( text ):
    ### strip all indentation whitespace (both tabs and spaces)
    lines = text.split('\n')
    lines = [line.strip(' ').strip('\t') for line in lines]
    text = '\n'.join(lines).strip('\n')
    return text

def escape_underscores( text ):
    ### replace uncerscores by their escaped version
    text = text.replace('_','\_')
    return text

def make_code_block( text, lang='text' ):
    ### format a piece of text as a code block
    text = text.strip('\n')
    return '```{}\n'.format(lang)+text+'\n```\n'

def make_title( text, level=2 ):
    ### format a piece of text as a title
    text = text.strip('\n')
    return '#'*level+' '+text+'\n'

def make_newline_markdown( text ):
    ### replace regular newline character by markdown newline characters
    return text.replace('\n','  \n')


class Comment(object):
    ### a general class for comment text and its formatting to markdown

    def __init__( self, rawtext ):
        self.rawtext = rawtext

    def get_raw( self ):
        return self.rawtext

    def format( self ):
        ### abstract method that must be overloaded by child classes
        raise NotImplementedError('ERROR: the class {}'.format(type(self))
                +' does not seem to implement the "format" method')


class FuncComment(Comment):
    ### comment class for function definitions

    def __init__( self, funcname, rawtext, level=2 ):
        ### initializer
        # funcname: name of the definition (e.g. function signature)
        # rawtext: raw comment text
        # level: level of funcname in title structure
        super( FuncComment, self ).__init__( rawtext )
        self.rawdefname = funcname
        self.level = level

    def format( self ):
        ### implement total formatting specific for function definition comments
        lines = self.rawtext.split('\n')
        for i,line in enumerate(lines):
            # remove hashtag characters and one space
            lines[i] = line.replace('# ','').strip('#')
        mdtext = '\n'.join(lines)
        mdtext = make_code_block(mdtext)
        title = escape_underscores(self.rawdefname)
        title = make_title( title, level=self.level )
        res = title+mdtext
        res = make_newline_markdown( res )
        return res

    def __str__( self ):
        ### alias for format for easy printing
        return self.format()


class ClassComment(Comment):
    
    def __init__( self, classname, rawtext, funccomments=[], level=1 ):
        ### initializer
        # classname: name of the definition (e.g. class name)
        # rawtext: raw comment text
        # funccomments: list of FuncComments belonging to this class
        # level: level of funcname in title structure
        super( ClassComment, self ).__init__( rawtext )
        self.rawdefname = classname
        self.level = level
        self.funccomments = funccomments

    def add_func_comment( self, funccomment ):
        self.funccomments.append( funccomment )

    def format( self ):
        ### implement total formatting specific for class definition comments
        res = '- - -\n'
        res += FuncComment( self.rawdefname, self.rawtext, level=self.level ).format()
        for funccomment in self.funccomments:
            thisfc = funccomment.format()
            res += thisfc.split(' ',1)[0] + ' &#10551; ' + thisfc.split(' ',1)[1]
        res += '- - -'
        return res

    def __str__( self ):
        ### alias for format for easy printing
        return self.format()


class CommentCollection(object):

    def __init__( self ):
        self.comments = []

    def get_comments( self ):
        return self.comments[:]

    def read_defcomments_from_file_legacy( self, filename ):
        # older but stable version
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
                defcomment = FuncComment( defname, rawtext, level=levels[i] )
                self.comments.append( defcomment )

    def read_defcomments_from_file( self, filename ):
        # experimental version with ClassComments
        f = open(filename,'r')
        lines = f.readlines()
        f.close()
        currentclass = None
        # remove tabs and spaces
        for i,line in enumerate(lines): lines[i] = line.strip(' \t')
        # loop over the lines
        for i,line in enumerate(lines):

            # scan for 'class' keyword
            if( line.strip(' \t')[:5]=='class' ):
                j = i
                classname = '[class] '+ line.strip(' \t:\n').replace('class ','')
                rawtext = ''
                while lines[j+1].strip(' ').strip('\t')[0]=='#':
                    rawtext += lines[j+1]
                    j += 1
                if rawtext=='': rawtext = '(no valid documentation found)'
                currentclass = ClassComment( classname, rawtext, funccomments=[], level=2)
                self.comments.append(currentclass)
            
            # scan for 'def' keyword
            if( line.strip(' \t')[:3]=='def' ):
                j = i
                # special treatment for definitions spread over multiple lines
                while (line.count('(')!=line.count(')')):
                    j += 1
                    line = line.rstrip('\n') + ' ' + lines[j]
                # add all consecutive lines that start with a '#' character
                rawtext = ''
                while lines[j+1].strip(' ').strip('\t')[0]=='#':
                    rawtext += lines[j+1]
                    j += 1
                if rawtext=='': rawtext = '(no valid documentation found)'
                # format the definition name
                defname = line.strip(' \t:\n').replace('def ','')
                if( currentclass is not None 
                    and ('(self,' in line.replace(' ','') 
                         or '(self)' in line.replace(' ','') ) ):
                    prefix = ''
                    defname = prefix + defname
                    currentclass.add_func_comment( FuncComment( defname, rawtext, level=3 ) )
                else:
                    self.comments.append( FuncComment( defname, rawtext, level=3 ) )



if __name__=='__main__':
    ### testing section

    testfile = os.path.abspath(sys.argv[1])
    
    ccol = CommentCollection()
    ccol.read_defcomments_from_file_2( testfile )
    for c in ccol.get_comments(): 
        print(c)
