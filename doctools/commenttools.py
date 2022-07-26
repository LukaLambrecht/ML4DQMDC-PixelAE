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

    def __init__( self, funcname, funcsignature=None, rawtext=None, level=2 ):
        ### initializer
        # funcname: short name of the function
        # funcsignature: full signature of the function
        # rawtext: raw comment text
        # level: level of funcname in title structure
        super( FuncComment, self ).__init__( rawtext )
        self.rawdefname = funcname
        self.rawdefsignature = funcsignature
        self.level = level

    def format( self ):
        ### implement total formatting specific for function definition comments

        # format the name
        title = escape_underscores(self.rawdefname)
        title = make_title( title, level=self.level )

        # format the signature as a block of code
        subtitle = ''
        if self.rawdefsignature is not None: 
            subtitle = 'full signature:\n'
            subtitle += make_code_block(self.rawdefsignature)

        mdtext = ''
        if self.rawtext is not None:
            # split the text by newline characters, 
            # remove some characacters,
            # and rejoin the lines
            lines = self.rawtext.split('\n')
            for i,line in enumerate(lines):
                # remove hashtag characters and one space
                lines[i] = line.replace('# ','').strip('#')
            mdtext = '\n'.join(lines)
            # format the text as a block of code
            mdtext = 'comments:\n'+make_code_block(mdtext)

        # join titles and text
        res = title+subtitle+mdtext
        # replace newlines by markdown newlines
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
        res += FuncComment( self.rawdefname, funcsignature=None, 
                            rawtext=self.rawtext, level=self.level ).format()
        for funccomment in self.funccomments:
            thisfc = funccomment.format()
            res += thisfc.split(' ',1)[0] + ' &#10551; ' + thisfc.split(' ',1)[1]
        res += '- - -'
        return res

    def __str__( self ):
        ### alias for format for easy printing
        return self.format()

class MarkDownComment( Comment ):
    ### comment class for markdown cells in .ipynb, converted to comments in .py
    # the content of the comments already has markdown syntax, 
    # just need to remove the leading '# '

    def __init__( self, rawtext ):
        super( MarkDownComment, self ).__init__( rawtext )

    def format( self ):
        lines = self.rawtext.split('\n')
        for i,line in enumerate(lines): lines[i] = line[2:]
        return '\n'.join(lines)+'- - -\n'

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
            if( line[:4]=='def ' or line[:6]=='class ' ):
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
        ### read comments from a python file accompanying a class or function definition
        # experimental version with ClassComments
        # input arguments:
        # - filename: path to a python file
        # output:
        #   the current CommentCollection is appended with all class/function definitions 
        #   in the python file, if they are commented correctly.
        f = open(filename,'r')
        lines = f.readlines()
        f.close()
        currentclass = None
        # remove tabs and spaces
        for i,line in enumerate(lines): lines[i] = line.strip(' \t')
        # loop over the lines
        for i,line in enumerate(lines):

            # scan for 'class' keyword
            if( line[:6]=='class ' ):
                j = i
                # get the pure class name from the current line
                classname = line.split('(')[0]
                classname = classname.strip(' \t:\n').replace('class ','')
                # format how to display the classname
                # to do: remove this here and move to mdfiletools?
                classname = '[class] {}'.format(classname)
                # read all consecutive lines that start with a '#' character
                rawtext = ''
                while lines[j+1].strip(' ').strip('\t')[0]=='#':
                    rawtext += lines[j+1]
                    j += 1
                if rawtext=='': rawtext = '(no valid documentation found)'
                # make the comment with given class name and accompanying text
                currentclass = ClassComment( classname, rawtext, funccomments=[], level=2)
                self.comments.append(currentclass)
            
            # scan for 'def' keyword
            if( line[:3]=='def' ):
                j = i
                # get the pure function name from the current line
                funcname = line.split('(')[0]
                funcname = funcname.strip(' \t:\n').replace('def ','')
                # get the full signature of the function from the current line(s)
                # (need special treatment for definitions spread over multiple lines)
                while (line.count('(')!=line.count(')')):
                    j += 1
                    line = line.rstrip('\n') + ' ' + lines[j]
                funcsignature = line.strip(' \t:\n')
                # read all consecutive lines that start with a '#' character
                rawtext = ''
                while lines[j+1].strip(' ').strip('\t')[0]=='#':
                    rawtext += lines[j+1]
                    j += 1
                if rawtext=='': rawtext = '(no valid documentation found)'
                # format the definition name
                if( currentclass is not None 
                    and ('(self,' in funcsignature.replace(' ','') 
                         or '(self)' in funcsignature.replace(' ','') ) ):
                    prefix = ''
                    funcsignature = prefix + funcsignature
                    currentclass.add_func_comment( FuncComment( funcname, funcsignature, 
                                                    rawtext, level=3 ) )
                else:
                    self.comments.append( FuncComment( funcname, funcsignature,
                                                    rawtext, level=3 ) )

    def read_markdowncomments_from_file( self, filename ):
        ### read comments in a python file that originate from markdown cells in .ipynb format
        f = open(filename,'r')
        lines = f.readlines()
        f.close()
        i = 0
        while i<len(lines):
            line = lines[i]
            if line[0]=='#':
                # skip some standard lines
                if( line=='#!/usr/bin/env python\n' 
                        or line=='# coding: utf-8\n'
                        or line=='### imports\n'
                        or line=='# external modules\n'
                        or line=='# local modules\n'): 
                    i += 1
                    continue
                # also require that the line has bold syntax
                if not '**' in line: 
                    i+= 1
                    continue
                # else it is probably a valid comment, start looking for next lines
                j = i
                rawtext = line
                while lines[j+1][0]=='#':
                    rawtext += lines[j+1]
                    j += 1
                i = j+1
                self.comments.append( MarkDownComment( rawtext ) )
                break # for now allow only one markdown comment, as header
            else: i += 1


if __name__=='__main__':
    ### testing section

    testfile = os.path.abspath(sys.argv[1])
    
    ccol = CommentCollection()
    ccol.read_markdowncomments_from_file( testfile )
    #ccol.read_defcomments_from_file( testfile )
    for c in ccol.get_comments(): 
        print(c)
