# to do:
# - small bug: when a frame is made inactive and then active again,
#   non-editable Comboboxes become editable, probably because the state is set to 'normal'.
# - small bug: when the 'new histstruct' button is pressed, the histstruct is set to a new empty one,
#   even if the new histstruct window is closed without actually creating the histstruct.

# external modules

print('importing external modules...')
print('  import os'); import os
os.environ['BROWSER'] = '/usr/bin/firefox'
print('  import sys'); import sys
print('  import math'); import math
print('  import pandas as pd'); import pandas as pd
print('  import numpy as np'); import numpy as np
print('  import json'); import json
print('  import matplotlib.pyplot as plt'); import matplotlib.pyplot as plt
print('  import pickle'); import pickle
print('  import functools'); import functools
print('  import webbrowser'); import webbrowser
print('  import inspect'); import inspect
try: 
    print('importing tkinter for python3')
    import tkinter as tk
    import tkinter.filedialog as fldlg
    from tkinter import ttk
except: 
    print('importing tkinter for python2')
    import Tkinter as tk
    import tkFileDialog as fldlg
    from Tkinter import ttk

# local modules

print('importing utils...')
sys.path.append(os.path.abspath('../utils'))
print('  import csv_utils as csvu'); import csv_utils as csvu
print('  import json_utils as jsonu'); import json_utils as jsonu
print('  import dataframe_utils as dfu'); import dataframe_utils as dfu
print('  import hist_utils as hu'); import hist_utils as hu
print('  import autoencoder_utils as aeu'); import autoencoder_utils as aeu
print('  import plot_utils as pu'); import plot_utils as pu
print('  import generate_data_utils as gdu'); import generate_data_utils as gdu
print('  import generate_data_2d_utils as gd2u'); import generate_data_2d_utils as gd2u
print('  import refruns_utils as rru'); import refruns_utils as rru
print('  import mask_utils as mu'); import mask_utils as mu

print('importing src...')
sys.path.append(os.path.abspath('../src'))
sys.path.append(os.path.abspath('../src/classifiers'))
sys.path.append(os.path.abspath('../src/cloudfitters'))
print('  import HistStruct'); import HistStruct
print('  import PlotStyleParser'); import PlotStyleParser
print('  import HistogramClassifier'); import HistogramClassifier
print('  import AutoEncoder'); import AutoEncoder
print('  import MaxPullClassifier'); import MaxPullClassifier
print('  import NMFClassifier'); import NMFClassifier
print('  import PCAClassifier'); import PCAClassifier
print('  import TemplateBasedClassifier'); import TemplateBasedClassifier
print('  import SeminormalFitter'); import SeminormalFitter
print('  import GaussianKdeFitter'); import GaussianKdeFitter
print('  import HyperRectangleFitter'); import HyperRectangleFitter
print('  import IdentityFitter'); import IdentityFitter

print('done')


### styling

def set_frame_default_style( frame, expandcolumn=-1, expandrow=-1 ):
    ### apply some default stylings to a tk.Frame
    # input arguments:
    # - expandcolumn: index of column in this frame that will be expanded to match parent
    # - expandrow: index of row in this frame that will be expanded to match parent
    frame['padx'] = 10
    frame['pady'] = 10
    frame['borderwidth'] = 2
    frame['relief'] = 'groove'
    if expandrow>=0: frame.grid_rowconfigure(expandrow, weight=1)
    if expandcolumn>=0: frame.grid_columnconfigure(expandcolumn, weight=1)

def change_frame_state( frame, state ):
    ### disable or enable all widgets in a frame
    for widget in frame.winfo_children():
        if isinstance(widget,tk.Frame): change_frame_state(widget,state)
        else: widget.configure(state=state)

def disable_frame( frame ):
    change_frame_state( frame, 'disable' )

def enable_frame( frame ):
    change_frame_state( frame, 'normal' )

### mappings of names to functions, classes, etc.

def get_args_dict( function ):
    ### get a dict of keyword arguments to a function
    sig = inspect.signature(function)
    args = {}
    for argname in sig.parameters:
        arg = sig.parameters[argname]
        if arg.default!=arg.empty: args[argname]=arg.default
    return args

def get_resampling_function( key=None ):
    ### get a resampling function from its name (or other key)
    # note: a dict of keyword arguments with default values is returned as well!
    # input arguments:
    # - key: string representing name or key of resampling function

    # return all valid keys
    allowed = (['None',
                'upsample_hist_set',
                'fourier_noise_nd',
                'white_noise_nd',
                'resample_lico_nd'])
    if key is None: return allowed

    # map key to function
    key = key.strip(' \t\n')
    if key=='None': return (None,{})
    if key=='upsample_hist_set': f = gdu.upsample_hist_set
    elif key=='fourier_noise_nd': f = gd2u.fourier_noise_nd
    elif key=='white_noise_nd': f = gd2u.white_noise_nd
    elif key=='resample_lico_nd': f = gd2u.resample_lico_nd
    else:
        raise Exception('ERROR: resampling function {} not recognized.'.format(key))
    return (f,get_args_dict(f))

def get_fitter_class( key=None ):
    ### get a fitter class from its name (or other key)
    # note: a dict of keyword initialization arguments with default values is returned as well!
    # input arguments:
    # - key: string representing name or key of fitter class
    
    allowed = ['GaussianKdeFitter', 'SeminormalFitter', 'IdentityFitter']
    if key is None: return allowed

    key = key.strip(' \t\n')
    if key=='GaussianKdeFitter': c = GaussianKdeFitter.GaussianKdeFitter
    elif key=='SeminormalFitter': c = SeminormalFitter.SeminormalFitter
    elif key=='IdentityFitter': c = IdentityFitter.IdentityFitter
    else:
        raise Exception('ERROR: fitter class {} not recognized'.format(key))
    return (c,get_args_dict(c))

def get_classifier_class( key=None ):
    ### get a classifier class from its name (or other key)
    # note: a dict of keyword initialization arguments with default values is returned as well!
    # input arguments:
    # - key: string representing name or key of classifier class

    allowed = (['AutoEncoder', 'MaxPullClassifier', 'NMFClassifier', 
                'PCAClassifier', 'TemplateBasedClassifier'])
    if key is None: return allowed

    key = key.strip(' \t\n')
    if key=='AutoEncoder': c = AutoEncoder.AutoEncoder
    elif key=='MaxPullClassifier': c = MaxPullClassifier.MaxPullClassifier
    elif key=='NMFClassifier': c = NMFClassifier.NMFClassifier
    elif key=='PCAClassifier': c = PCAClassifier.PCAClassifier
    elif key=='TemplateBasedClassifier': c = TemplateBasedClassifier.TemplateBasedClassifier
    else:
        raise Exception('ERROR: classifier class {} not recognized'.format(key))
    return (c,get_args_dict(c))

def get_training_options( histstruct, histname=None ):
    ### get options for training classifiers
    # note: the classifiers are assumed to be already present in the histstruct
    # if histname is specified, the classifier options for that specific classifer are returned;
    # if not, it is checked that all classifiers are of the same type 
    # and a single set of options is returned belonging to that type.
    # to do: deal with special cases of non-trival arguments, e.g.
    #        - reference histogram for MaxPullClassifier
    #           (can work with usual mask but not practical for one histogram, 
    #            maybe add option in classifier to average 'training' histograms)
    ctype = None
    classifier = None
    if histname is None: histnames = histstruct.histnames
    else: histnames = [histname]
    for histname in histnames:
        if histname not in histstruct.classifiers.keys():
            print('WARNING: the histstruct seems not to contain a classifier'
                    +' for {}'.format(histname))
            continue
        classifier = histstruct.classifiers[histname]
        if( ctype is None ): ctype = type(classifier)
        if( type(classifier)!=ctype ):
            print('WARNING: the histstruct seems to contain different types of classifiers'
                    +' for different types of histograms.')
            return None
    if classifier is None:
        print('WARNING: could not retrieve options for classifier training.'
                +' (Have any classifiers been initialized?)')
        return {}
    return (classifier.__class__, get_args_dict(classifier.train))

### get link to documentation for an object

def get_docurl( obj ):
    if obj is None: return None
    try:
        # get the physical path of the file where the object is defined
        try:
            # case of a class name or function name
            physicalpath = os.path.abspath(sys.modules[obj.__module__].__file__)
            objname = obj.__name__
        except:
            # case of a class instance
            physicalpath = os.path.abspath(sys.modules[obj.__class__.__module__].__file__)
            objname = obj.__class__.__name__
        # the above does not work in compiled executable mode, need additional logic
        # note: the file directories must be added under 'ML4DQM-DC/' under the temporary directory 
        #       created by the executable, see the gui.spec file.
        if '/tmp/' in physicalpath:
            tempdir,filename = os.path.split(physicalpath)
            sourcedir = os.path.join(tempdir,'ML4DQM-DC')
            filename = filename.replace('.pyc','.py')
            for root, dirs, files in os.walk(sourcedir):
                for f in files:
                    if f==filename: 
                        physicalpath = os.path.join(root,f)
        # make the path relative to the top of the project and remove extensions
        relpath = physicalpath.split('ML4DQM-DC/',-1)[1]
        reldoc = os.path.splitext(relpath)[0]
        # documentation-specific parts: main webpage and paragraph structure
        # (this could be broken if the documentation structure changes!)
        docweb = 'https://lukalambrecht.github.io/ML4DQMDC-PixelAE/'
        paragraph = '/#'+objname
        paragraph = paragraph.replace('_','95') # not sure how universally valid this is
        docurl = docweb+reldoc+paragraph
        return docurl
    except:
        print('WARNING: could not retrieve doc url for object "{}"'.format(obj))
        return None


### get initial directory for file loaders and savers

def get_initialdir():
    # original: use physical __file__ location
    # does not work well in compiled executable version, points to shady places...
    #initialdir = os.path.abspath(os.path.dirname(__file__))
    # alternative: simply use working directory
    initialdir = os.path.abspath(os.getcwd().rstrip('/'))
    # if run from a 'dist' subfolder, return one directory level above
    if os.path.basename(initialdir)=='dist':
        initialdir = os.path.dirname(initialdir)
    # return one directory level above this one
    initialdir = os.path.dirname(initialdir)
    return initialdir


### other help functions

def is_float(s):
    try: float(s); return True
    except ValueError: return False

def is_int(s):
    try: int(s); return True
    except ValueError: return False

def is_bool(s):
    if( s=='True' or s=='true' or s=='False' or s=='false' ): return True
    else: return False

def to_bool(s):
    # note: do not use builtin bool() since it appears to return True for every not-None variable
    return (s=='True' or s=='true')

### help classes

class StdOutRedirector:
    ### helper class to redirect print output to GUI widget
    # use as follows:
    #   stdout = sys.stdout
    #   sys.stdout = StdOutRedirector(<some widget>)
    #   ... <some code execution containing print statements>
    #   sys.stdout = stdout

    def __init__(self, tk_text_widget, tk_root_object):
        self.text_dump = tk_text_widget
        self.root = tk_root_object

    def write(self, text):
        self.text_dump.insert(tk.INSERT, text)
        self.text_dump.see(tk.END)
        self.root.update()

    def flush(self):
        # (empty) flush attribute needed to avoid exception on destroying the window
        pass


class GenericFileLoader:
    ### contains a button to open a file loader dialog and stores the result

    def __init__(self, master, buttontext=None, filetypes=None):
        if buttontext is None: buttontext = '(no file chosen)'
        if filetypes is None: filetypes = (('all files','*.*'),)
        self.filename = None
        self.frame = tk.Frame(master)
        self.load_button = tk.Button(self.frame, text=buttontext, 
                command=functools.partial(self.load_filename,filetypes=filetypes))
        self.load_button.grid(row=0, column=0, sticky='nsew')

    def load_filename(self, filetypes=None):
        if filetypes is None: filetypes = (('all files','*.*'),)
        initialdir = get_initialdir()
        filename = fldlg.askopenfilename(initialdir=initialdir,
                    title='Choose file',
                    filetypes=filetypes)
        # if filename is invalid, print a warning
        if len(filename)==0: print('WARNING: file loading canceled')
        # else set the filename
        else: 
            self.filename = filename
            self.load_button.config(text=os.path.basename(filename))

    def get_filename(self):
        return self.filename

    def grid(self, row=None, column=None):
        self.frame.grid(row=row, column=column, sticky='nsew')

class GenericFileSaver:
    ### same as GenericFileLoader but for saving files

    def __init__(self, master, buttontext=None, filetypes=None):
        if buttontext is None: buttontext = '(no location chosen)'
        if filetypes is None: filetypes = (('all files','*.*'),)
        self.filename = None
        self.frame = tk.Frame(master)
        self.save_button = tk.Button(self.frame, text=buttontext,
                command=functools.partial(self.save_filename,filetypes=filetypes))
        self.save_button.grid(row=0, column=0, sticky='nsew')

    def save_filename(self, filetypes=None):
        if filetypes is None: filetypes = (('all files','*.*'),)
        initialdir = get_initialdir
        filename = fldlg.asksaveasfilename(initialdir=initialdir,
                    title='Save a file',
                    filetypes=filetypes)
        # if filename is invalid, print a warning
        if len(filename)==0: print('WARNING: file saving canceled')
        # else set the filename
        else:
            self.filename = filename
            self.save_button.config(text=os.path.basename(filename))

    def get_filename(self):
        return self.filename

    def grid(self, row=None, column=None):
        self.frame.grid(row=row, column=column, sticky='nsew')


class UrlWidget:
    ### contains a tk.Label with a clickable link

    def __init__(self, master, url, text=None):
        if text is None: text = url
        self.url = url
        self.label = tk.Label(master, text=text, fg='blue', cursor='hand2')
        self.label.bind('<Button-1>', self.openurl)

    def grid(self, **kwargs):
        self.label.grid(**kwargs)

    def openurl(self, event):
        # open a webbrowser on the requested url
        webbrowser.open_new(self.url)


class OptionsFrame:
    ### contains a tk.Frame holding a list of customization options

    def __init__(self, master, labels=None, types=None, values=None,
                        docurls=None, docurl=None, autobool=False):
        # input arguments:
        # - labels: list of strings with the names/labels of the options
        # - types: list of tk types, defaults to tk.Text for each option
        # - values: list of options passed to each widget
        #           (for now only tk.Text, in wich case values is a list of default values,
        #            but can be extended to e.g. combobox, 
        #            where values would be the options to choose from)
        # note: individual elements of types and values can also be None,
        #       in which case these elements will be set to default
        # - docurls: list of urls to documentation per option
        # - docurl: url to documentation for the option collection
        # - autobool: automatically convert boolean arguments to a binary ttk.Combobox 
        #             (instead of Text entry)
        self.frame = tk.Frame(master,width=200)
        self.labels = []
        self.wtypes = []
        self.widgets = []
        self.docwidgets = []
        self.docwidget = None
        self.autobool = autobool
        self.set_options( labels=labels, types=types, values=values, docurls=docurls, docurl=docurl )

    def set_options(self, labels=None, types=None, values=None, docurls=None, docurl=None):
        ### set the options of an option frame
        # serves both as initializer and as resetter

        # check arguments
        if labels is None: 
            raise Exception('ERROR in OptionsFrame initialization:'
                            +' argument "labels" must be specified.')
        if types is None: types = [tk.Text]*len(labels)
        if values is None: values = [None]*len(labels)
        if docurls is None: docurls = [None]*len(labels)
        if( len(types)!=len(labels) 
                or len(values)!=len(labels)
                or len(docurls)!=len(labels) ):
            raise Exception('ERROR in OptionsFrame initialization:'
                            +' argument lists have unequal lengths.')
        labels = list(labels) # explicit conversion from dict_keys or dict_values to list
        types = list(types) # explicit conversion from dict_keys or dict_values to list
        values = list(values) # explicit conversion from dict_keys or dict_values to list

        # additional argument parsing
        if self.autobool:
            for j in range(len(labels)):
                if is_bool(str(values[j])):
                    types[j] = ttk.Combobox
                    values[j] = [values[j], not values[j]]

        # clear current OptionsFrame
        self.labels.clear()
        self.wtypes.clear()
        self.widgets.clear()
        self.docwidgets.clear()
        self.docwidget = None
        for widget in self.frame.winfo_children(): widget.destroy()

        # set widgets
        nrows = len(labels)
        for i, (label, wtype, value, url) in enumerate(zip(labels, types, values, docurls)):
            # make label
            tklabel = tk.Label(self.frame, text=label)
            tklabel.grid(row=i, column=0)
            self.labels.append(tklabel)
            # make widget
            if wtype is None: wtype = tk.Text
            widget = None
            # case 1: simple generic text box
            if wtype is tk.Text:
                widget = tk.Text(self.frame, height=1, width=25)
                if value is not None:
                    widget.insert(tk.INSERT, value)
            # case 2: file loader
            elif wtype is GenericFileLoader:
                widget = GenericFileLoader(self.frame)
            # case 3: file saver
            elif wtype is GenericFileSaver:
                widget = GenericFileSaver(self.frame)
            # case 4: combobox with fixed options
            elif wtype is ttk.Combobox:
                widget = ttk.Combobox(self.frame, values=value, width=25)
                widget['state'] = 'readonly'
                widget.current(0)
            else:
                raise Exception('ERROR in OptionsFrame initialization:'
                                +' widget type {} not recognized'.format(wtype))
            widget.grid(row=i, column=1)
            self.widgets.append(widget)
            self.wtypes.append(wtype)
            # make doc widget
            if url is not None:
                urlwidget = UrlWidget(self.frame, url, text='More info')
                urlwidget.grid(row=i, column=2)
                self.docwidgets.append(urlwidget)

        # set link to documentation
        if docurl is not None:
            self.docwidget = UrlWidget(self.frame, docurl, text='More info')
            self.docwidget.grid(row=nrows, column=0, columnspan=2, sticky='nsew')

    def get_dict(self):
        ### get the options of the current OptionsFrame as a dictionary
        res = {}
        for label, wtype, widget in zip(self.labels, self.wtypes, self.widgets):
            key = label.cget('text')
            value = None
            # case 1: simple generic text box
            if wtype is tk.Text:
                value = widget.get('1.0', tk.END)
            # case 2: file loader
            elif wtype is GenericFileLoader:
                value = widget.get_filename()
                if value is None:
                    msg = 'WARNING: file for option {} is None.'.format(key)
                    print(msg)
            # case 3: file saver
            elif wtype is GenericFileSaver:
                value = widget.get_filename()
                if value is None:
                    msg = 'WARNING: file for option {} is None.'.format(key)
                    print(msg)
            # case 4: combobox with fixed options
            elif wtype is ttk.Combobox:
                value = widget.get()
            else:
                raise Exception('ERROR in OptionsFrame get_dict:'
                               +' no getter method implemented for widget type {}'.format(wtype))
            # basic parsing
            if value is None: value = ''
            value = value.strip(' \t\n')
            if is_int(value): value = int(value)
            elif is_float(value): value = float(value)
            elif is_bool(value): value = to_bool(value)
            elif value=='None': value = None
            elif value=='': value = None
            res[key] = value
        return res


class ScrolledFrame:
    ### contains a tk.Frame holding a widget with vertical and horizontal scrollbars
    # note: it does not seem possible to just pass an arbitrary widget in the constructor,
    #       since the widget must have its master (i.e. this frame) set on creation.
    #       therefore, first create the ScrolledFrame f, then the widget (using the f as master),
    #       and then call set_widget to position the widgt correctly in the frame.
    #       to simplify this, make subclasses (e.g. ScrolledTextFrame below)

    def __init__(self, master, height=50, width=50, 
                    childsize=False, expandable=False, showscrollbars=False):
        # input arguments:
        # - childsize: if True, the Frame will take its size from the child widget
        # - expandable: if True, the frame will be expanded to match the parent
        self.frame = tk.Frame(master, height=height, width=width)
        if not childsize: self.frame.grid_propagate(0)
        if expandable:
            self.frame.grid_rowconfigure(0, weight=1)
            self.frame.grid_columnconfigure(0, weight=1)
        self.showscrollbars = showscrollbars

    def set_widget(self, widget):
        self.widget = widget
        widget.grid(row=0, column=0, sticky='nsew')
        self.yscrollbar = tk.Scrollbar(self.frame, orient="vertical", command=widget.yview)
        self.xscrollbar = tk.Scrollbar(self.frame, orient="horizontal", command=widget.xview)
        if self.showscrollbars:
            self.yscrollbar.grid(row=0, column=1)
            self.xscrollbar.grid(row=1, column=0)
            self.frame.grid_rowconfigure(0, weight=1)
            self.frame.grid_columnconfigure(0, weight=1)

class ScrolledTextFrame(ScrolledFrame):
    ### specific case of ScrolledFrame, where the widget is tk.Text.
    # note: the advantage of using this specification is that the tk.Text widget 
    #       does not need to be created manually;
    #       it is created internally and accessible via the .widget attribute.

    def __init__(self, master, txtheight=50, txtwidth=50, expandable=False, showscrollbars=False):
        super().__init__(master, childsize=True, expandable=expandable, 
                            showscrollbars=showscrollbars)
        text = tk.Text(self.frame, wrap=tk.NONE, height=txtheight, width=txtwidth)
        self.set_widget(text)

class ScrolledFrameFrame(ScrolledFrame):
    ### specific case of ScrolledFrame, where the widget is a tk.Frame
    # note: need special treatment since no scrollbar can be added to a Frame directly,
    #       needs an intermediate Canvas object.
    # to do: does not work yet (frame does not show), so need to fix...

    def __init__(self, master, height=50, width=50, showscrollbars=False):
        super().__init__(master, height=height, width=width, showscrollbars=showscrollbars)
        canvas = tk.Canvas(self.frame)
        # add the scrollbars to the canvas
        self.set_widget(canvas)
        # create a frame within the canvas
        frame = tk.Frame(canvas)
        frame.grid(row=0, column=0, sticky='nsew')
        # overwrite self.widget attribute
        self.widget = frame


### GUI windows

class NewHistStructWindow(tk.Toplevel):
    ### popup window class for creating a new histstruct

    def __init__(self, master, emptyhiststruct):
        super().__init__(master=master)
        self.title('New HistStruct')
        self.histstruct = emptyhiststruct
        self.histfiles = {}
        self.run_mask_widgets = []
        self.highstat_mask_widgets = []
        self.json_mask_widgets = []

        # create a frame for general options
        self.general_options_frame = tk.Frame(self)
        set_frame_default_style( self.general_options_frame )
        self.general_options_frame.grid(row=0, column=0, sticky='nsew')
        # add widgets for general options
        self.general_options_label = tk.Label(self.general_options_frame, 
                                              text='General options')
        self.general_options_label.grid(row=0, column=0)
        self.training_mode_box = ttk.Combobox(self.general_options_frame, 
                                    values=['global','local'])
        self.training_mode_box.current(0)
        self.training_mode_box['state'] = 'readonly'
        self.training_mode_box.grid(row=1,column=0)
        self.year_box = ttk.Combobox(self.general_options_frame,
                                     values=['2017'])
        self.year_box.current(0)
        self.year_box['state'] = 'readonly'
        self.year_box.grid(row=2,column=0)

        # create a frame for choice of histograms
        self.histnames_frame = tk.Frame(self)
        set_frame_default_style( self.histnames_frame )
        self.histnames_frame.grid(row=1, column=0, sticky='nsew')
        # add widgets for choice of histograms
        self.add_histograms_button = tk.Button(self.histnames_frame, text='Add histograms', 
                                        command=self.add_histnames)
        self.add_histograms_button.grid(row=0, column=0, sticky='nsew')
        self.clear_histograms_button = tk.Button(self.histnames_frame, text='Clear',
                                        command=self.clear_histnames)
        self.clear_histograms_button.grid(row=1, column=0, sticky='nsew')
        self.histnames_listbox = tk.Listbox(self.histnames_frame, 
                                            selectmode='multiple',
                                            exportselection=False)
        self.histnames_listbox.grid(row=2, column=0)

        # create a frame for local options
        self.local_options_frame = tk.Frame(self)
        set_frame_default_style(self.local_options_frame)
        self.local_options_frame.grid(row=2, column=0, sticky='nsew')
        # add widgets for local options
        self.local_options_label = tk.Label(self.local_options_frame, 
                                    text='Options for local training')
        self.local_options_label.grid(row=0, column=0)
        tk.Label(self.local_options_frame, text='target run').grid(row=1, column=0)
        self.target_run_text = tk.Text(self.local_options_frame, height=1, width=8)
        self.target_run_text.grid(row=1, column=1)
        tk.Label(self.local_options_frame, text='ntraining').grid(row=2, column=0)
        self.ntraining_text = tk.Text(self.local_options_frame, height=1, width=3)
        self.ntraining_text.insert(tk.INSERT, '5')
        self.ntraining_text.grid(row=2, column=1)
        tk.Label(self.local_options_frame, text='offset').grid(row=3, column=0)
        self.offset_text = tk.Text(self.local_options_frame, height=1, width=3)
        self.offset_text.insert(tk.INSERT, '0')
        self.offset_text.grid(row=3, column=1)
        self.remove_var = tk.IntVar()
        tk.Checkbutton(self.local_options_frame, text='remove unneeded runs',
                    variable=self.remove_var).grid(row=4, column=0)

        # create a frame for run mask addition
        self.run_mask_frame = tk.Frame(self)
        set_frame_default_style( self.run_mask_frame )
        self.run_mask_frame.grid(row=0, column=1, sticky='nsew', rowspan=3)
        # add widgets for run mask addition
        self.add_run_mask_button = tk.Button(self.run_mask_frame, text='Add a run mask',
                                command=functools.partial(self.add_run_mask, self.run_mask_frame))
        self.add_run_mask_button.grid(row=0, column=0, columnspan=2)
        name_label = tk.Label(self.run_mask_frame, text='Name:')
        name_label.grid(row=1, column=0)
        run_label = tk.Label(self.run_mask_frame, text='Run number:')
        run_label.grid(row=1, column=1)

        # create a frame for highstat mask addition
        self.highstat_mask_frame = tk.Frame(self)
        set_frame_default_style( self.highstat_mask_frame )
        self.highstat_mask_frame.grid(row=0, column=2, sticky='nsew', rowspan=3)
        # add widgets for highstat mask additions
        self.add_highstat_mask_button = tk.Button(self.highstat_mask_frame, 
                        text='Add a statistics mask',
                        command=functools.partial(self.add_highstat_mask,self.highstat_mask_frame))
        self.add_highstat_mask_button.grid(row=0, column=0, columnspan=2)
        name_label = tk.Label(self.highstat_mask_frame, text='Name:')
        name_label.grid(row=1, column=0)
        operator_label = tk.Label(self.highstat_mask_frame, text='Operator:')
        operator_label.grid(row=1, column=1)
        apply_label = tk.Label(self.highstat_mask_frame, text='Apply on:')
        apply_label.grid(row=1, column=2)
        threshold_label = tk.Label(self.highstat_mask_frame, text='Threshold:')
        threshold_label.grid(row=1, column=3)

        # create a frame for json mask addition
        self.json_mask_frame = tk.Frame(self)
        set_frame_default_style( self.json_mask_frame )
        self.json_mask_frame.grid(row=0, column=3, sticky='nsew', rowspan=3)
        # add buttons for json mask additions
        self.add_json_mask_button = tk.Button(self.json_mask_frame,
                        text='Add a json mask',
                        command=functools.partial(self.add_json_mask,self.json_mask_frame))
        self.add_json_mask_button.grid(row=0, column=0, columnspan=2)
        name_label = tk.Label(self.json_mask_frame, text='Name:')
        name_label.grid(row=1, column=0)
        file_label = tk.Label(self.json_mask_frame, text='File:')
        file_label.grid(row=1, column=1)

        # add button to start HistStruct creation
        self.make_histstruct_button = tk.Button(self, text='Make HistStruct',
                                        command=self.make_histstruct)
        self.make_histstruct_button.grid(row=3, column=0)

    def get_target_run(self):
        ### get the target run from the corresponding Text widget
        # return None if the widget is empty
        target_run_text = self.target_run_text.get(1.0, tk.END).strip(' \t\n')
        if len(target_run_text)==0: return None
        return int(target_run_text)

    def get_local_training_runs(self, filename):
        ### get the training runs from the corresponding Text widgets
        # return None if the target run is None
        target_run = self.get_target_run()
        if target_run is None: return None
        ntraining = int(self.ntraining_text.get(1.0, tk.END).strip(' \t\n'))
        offset = int(self.offset_text.get(1.0, tk.END).strip(' \t\n'))
        runs = dfu.get_runs( dfu.select_dcson( csvu.read_csv( filename ) ) )
        runs = dfu.get_runs( csvu.read_csv( filename ) )
        target_run_index = runs.index(target_run)
        training_runs = runs[target_run_index-ntraining-offset:target_run_index-offset]
        return training_runs

    def get_needed_runs(self, is_local=False, filename=None):
        all_runs = []
        # add the run from all run masks
        for run in self.get_run_masks().values():
            all_runs.append(run)
        # for local training: add target and training runs
        # (in case a target run is specified)
        if is_local:
            if filename is None:
                raise Exception('ERROR: a filename must be specified'
                                +' for getting needed runs for local training.')
            if not os.path.exists(filename):
                raise Exception('ERROR: the file {} does not seem to exist.'.format(filename))
            target_run = self.get_target_run()
            if target_run is not None:
                training_runs = self.get_local_training_runs( filename )
                all_runs.append(target_run)
                for run in training_runs: all_runs.append(run)
        return all_runs

    def add_run_mask(self, parent=None):
        if parent is None: parent = self
        row = len(self.run_mask_widgets)+2
        column = 0
        name_text = tk.Text(parent, height=1, width=25)
        name_text.grid(row=row, column=column)
        run_text = tk.Text(parent, height=1, width=8)
        run_text.grid(row=row, column=column+1)
        self.run_mask_widgets.append({'name_text':name_text,'run_text':run_text})

    def add_highstat_mask(self, parent=None):
        if parent is None: parent = self
        row = len(self.highstat_mask_widgets)+2
        column = 0
        name_text = tk.Text(parent, height=1, width=15)
        name_text.grid(row=row, column=column)
        operator_box = ttk.Combobox(parent, width=2, values=['>','<'])
        operator_box.current(0)
        operator_box['state'] = 'readonly'
        operator_box.grid(row=row, column=column+1)
        apply_box = ttk.Combobox(parent, values=['all']+self.get_histnames())
        apply_box.current(0)
        apply_box['state'] = 'readonly'
        apply_box.grid(row=row, column=column+2)
        threshold_text = tk.Text(parent, height=1, width=8)
        threshold_text.grid(row=row, column=column+3)
        self.highstat_mask_widgets.append({'name_text':name_text,
                                            'operator_box': operator_box,
                                            'apply_box': apply_box,
                                            'threshold_text':threshold_text})

    def add_json_mask(self, parent=None):
        if parent is None: parent = self
        row = len(self.json_mask_widgets)+2
        column = 0
        name_text = tk.Text(parent, height=1, width=25)
        name_text.grid(row=row, column=column)
        file_loader = GenericFileLoader(parent, filetypes=(('json','*.json'),('txt','*.txt')))
        file_loader.grid(row=row, column=column+1)
        self.json_mask_widgets.append({'name_text':name_text,'file_loader':file_loader})

    def get_run_masks(self):
        run_masks = {}
        for el in self.run_mask_widgets:
            name = el['name_text'].get(1.0, tk.END).strip(' \t\n')
            run = int(el['run_text'].get(1.0, tk.END).strip(' \t\n'))
            run_masks[name] = run
        return run_masks

    def get_highstat_masks(self):
        highstat_masks = {}
        for el in self.highstat_mask_widgets:
            name = el['name_text'].get(1.0, tk.END).strip(' \t\n')
            operator = el['operator_box'].get()
            applyon = el['apply_box'].get()
            threshold = float(el['threshold_text'].get(1.0, tk.END).strip(' \t\n'))
            highstat_masks[name] = (operator,applyon,threshold)
        return highstat_masks

    def get_json_masks(self):
        json_masks = {}
        for el in self.json_mask_widgets:
            name = el['name_text'].get(1.0, tk.END).strip(' \t\n')
            filename = el['file_loader'].get_filename()
            json_masks[name] = filename
        return json_masks

    def clear_histnames(self):
        self.histfiles = {}
        self.histnames_listbox.delete(0, tk.END)

    def add_histnames(self):
        initialdir = get_initialdir()
        filenames = fldlg.askopenfilenames(initialdir=initialdir,
                    title='Load histograms',
                    filetypes=(('csv files','*.csv'),('all files','*.*')))
        # if filename is invalid, return
        if len(filenames)==0:
            print('Loading of histograms canceled')
            return
        for filename in filenames:
            histname = os.path.basename(filename).replace('.csv','')
            self.histfiles[histname] = filename
            self.histnames_listbox.insert(tk.END, '{} ({})'.format(histname, filename))
            self.histnames_listbox.select_set(tk.END)

    def get_histnames(self):
        histnames = ([self.histnames_listbox.get(idx)
                    for idx in self.histnames_listbox.curselection() ])
        return histnames

    def make_histstruct(self):

        # disable frame for the remainder of the processing time
        disable_frame( self )
        # get general settings
        histnames = self.get_histnames()
        histnames = [h.split('(')[0].strip(' ') for h in histnames]
        year = self.year_box.get()
        training_mode = self.training_mode_box.get()
        print('creating a new HistStruct...')
        print('found following settings:')
        print('  - histogram names: {}'.format(histnames))
        print('  - year: {}'.format(year))
        print('  - training mode: {}'.format(training_mode))
        print('finding all available runs...')
        # get a comprehensive set of all explicitly needed runs (for throwing away the rest)
        firstfilename = self.histfiles[histnames[0]]
        needed_runs = self.get_needed_runs( is_local=(training_mode=='local'), 
                                            filename=firstfilename )
        # loop over the histogram types to take into account
        for histname in histnames:
            print('adding {}...'.format(histname))
            # read the histograms from the csv file
            filename = self.histfiles[histname]
            if not os.path.exists(filename):
                raise Exception('ERROR: the file {} does not seem to exist.'.format(filename))
            df = csvu.read_csv( filename )
            # in case of local training, we can remove most of the histograms
            if( training_mode=='local' and self.remove_var.get()>0 ):
                needed_runsls = {str(run): [[-1]] for run in needed_runs}
                df = dfu.select_runsls( df, needed_runsls )
            # add the histograms to the HistStuct 
            self.histstruct.add_dataframe( df )
        print('added {} lumisections with {} histograms each to the HistStruct.'.format(
                len(self.histstruct.runnbs),len(self.histstruct.histnames)))
    
        # add default masks for DCS-bit on and golden json
        # to do: make this more flexible with user options
        print('adding default DCS-on and golden json masks...')
        try: self.histstruct.add_dcsonjson_mask( 'dcson' )
        except: print('WARNING: could not add a mask for DCS-on data.'
                      +' Check access to DCS-on json file.')
        try: self.histstruct.add_goldenjson_mask( 'golden' )
        except: print('WARNING: could not add a mask for golden data.'
                        +' Check access to golden json file.')

        # add training and target mask for local training
        # to do: make this more flexible (e.g. choosing names)
        if training_mode=='local':
            if self.get_target_run() is not None:
                print('adding mask for target runs...')
                json = {str(self.get_target_run()): [[-1]]}
                self.histstruct.add_json_mask( 'target_run', json )
            if self.get_local_training_runs(firstfilename) is not None:
                print('adding mask for local training runs...')
                json = {str(run): [[-1]] for run in self.get_local_training_runs(firstfilename)}
                self.histstruct.add_json_mask( 'local_training', json )

        # add high statistics mask(s)
        highstat_masks = self.get_highstat_masks()
        for name, (operator,applyon,threshold) in highstat_masks.items():
            print('adding mask "{}"'.format(name))
            # set operator type
            min_entries_to_bins_ratio=-1
            max_entries_to_bins_ratio=-1
            if operator=='>':
                min_entries_to_bins_ratio = threshold
            elif operator=='<':
                max_entries_to_bins_ratio = threshold
            else:
                raise Exception('ERROR: highstat mask operator {} not recognized.'.format(operator))
            # set application histograms
            histnames = None
            if applyon!='all': histnames=[applyon]
            self.histstruct.add_stat_mask( name, histnames=histnames,
                                            min_entries_to_bins_ratio=min_entries_to_bins_ratio,
                                            max_entries_to_bins_ratio=max_entries_to_bins_ratio)

        # add run mask(s)
        run_masks = self.get_run_masks()
        for name, run in run_masks.items():
            print('adding mask "{}"'.format(name))
            json = {str(run):[[-1]]}
            self.histstruct.add_json_mask( name, json )

        # add json mask(s)
        json_masks = self.get_json_masks()
        for name, filename in json_masks.items():
            print('adding mask "{}"'.format(name))
            json = jsonu.loadjson( filename )
            self.histstruct.add_json_mask( name, json )

        # close the window
        self.destroy()
        self.update()
        print('done creating HistStruct.')


class AddRunMasksWindow(tk.Toplevel):
    ### popup window class for adding run masks to a HistStruct
    # functionality already exists in NewHistStructWindow, but here one has the advantage
    # that a list of available run numbers in the (existing) HistStruct can be made.

    def __init__(self, master, histstruct):
        super().__init__(master=master)
        self.title('Add run masks')
        self.histstruct = histstruct
        self.run_mask_widgets = []

        # create a frame for run mask addition
        self.run_mask_frame = tk.Frame(self)
        set_frame_default_style( self.run_mask_frame )
        self.run_mask_frame.grid(row=0, column=1, sticky='nsew', rowspan=3)
        # add widgets for run mask addition
        self.add_run_mask_button = tk.Button(self.run_mask_frame, text='Add...',
                                command=functools.partial(self.add_run_mask, self.run_mask_frame))
        self.add_run_mask_button.grid(row=0, column=0)
        self.apply_button = tk.Button(self.run_mask_frame, text='Apply',
                                command=self.apply)
        self.apply_button.grid(row=0, column=1)
        name_label = tk.Label(self.run_mask_frame, text='Name:')
        name_label.grid(row=1, column=0)
        run_label = tk.Label(self.run_mask_frame, text='Run number:')
        run_label.grid(row=1, column=1)
        # add a single run mask already
        self.add_run_mask(self.run_mask_frame)

    def add_run_mask(self, parent=None):
        if parent is None: parent = self
        row = len(self.run_mask_widgets)+2
        column = 0
        name_text = tk.Text(parent, height=1, width=25)
        name_text.grid(row=row, column=column)
        run_box = ttk.Combobox(parent, values=self.histstruct.get_runnbs_unique(), height=10)
        run_box.grid(row=row, column=column+1)
        self.run_mask_widgets.append({'name_text':name_text,'run_box':run_box})

    def get_run_masks(self):
        run_masks = {}
        for el in self.run_mask_widgets:
            name = el['name_text'].get(1.0, tk.END).strip(' \t\n')
            run = int(el['run_box'].get())
            run_masks[name] = run
        return run_masks

    def apply(self):
        run_masks = self.get_run_masks()
        for name, run in run_masks.items():
            print('adding mask "{}"'.format(name))
            json = {str(run):[[-1]]}
            self.histstruct.add_json_mask( name, json )
        # close the window
        self.destroy()
        self.update()
        print('done')


class AddStatMasksWindow(tk.Toplevel):
    ### popup window class for adding statistics masks to a HistStruct
    # functionality already exists in NewHistStructWindow, but here masks can be added on the fly.

    def __init__(self, master, histstruct):
        super().__init__(master=master)
        self.title('Add statistics masks')
        self.histstruct = histstruct
        self.stat_mask_widgets = []

        # create a frame for stat mask addition
        self.stat_mask_frame = tk.Frame(self)
        set_frame_default_style( self.stat_mask_frame )
        self.stat_mask_frame.grid(row=0, column=1, sticky='nsew', rowspan=3)
        # add widgets for stat mask addition
        self.add_stat_mask_button = tk.Button(self.stat_mask_frame, text='Add...',
                                command=functools.partial(self.add_stat_mask, self.stat_mask_frame))
        self.add_stat_mask_button.grid(row=0, column=0)
        self.apply_button = tk.Button(self.stat_mask_frame, text='Apply',
                                command=self.apply)
        self.apply_button.grid(row=0, column=1)
        name_label = tk.Label(self.stat_mask_frame, text='Name:')
        name_label.grid(row=1, column=0)
        operator_label = tk.Label(self.stat_mask_frame, text='Operator:')
        operator_label.grid(row=1, column=1)
        apply_label = tk.Label(self.stat_mask_frame, text='Apply on:')
        apply_label.grid(row=1, column=2)
        threshold_label = tk.Label(self.stat_mask_frame, text='Threshold:')
        threshold_label.grid(row=1, column=3)

        # add a single stat mask already
        self.add_stat_mask(self.stat_mask_frame)

    def add_stat_mask(self, parent=None):
        if parent is None: parent = self
        row = len(self.stat_mask_widgets)+2
        column = 0
        name_text = tk.Text(parent, height=1, width=15)
        name_text.grid(row=row, column=column)
        operator_box = ttk.Combobox(parent, width=2, values=['>','<'])
        operator_box.current(0)
        operator_box['state'] = 'readonly'
        operator_box.grid(row=row, column=column+1)
        apply_box = ttk.Combobox(parent, values=['all']+self.histstruct.histnames)
        apply_box.current(0)
        apply_box['state'] = 'readonly'
        apply_box.grid(row=row, column=column+2)
        threshold_text = tk.Text(parent, height=1, width=8)
        threshold_text.grid(row=row, column=column+3)
        self.stat_mask_widgets.append({'name_text':name_text,
                                            'operator_box': operator_box,
                                            'apply_box': apply_box,
                                            'threshold_text':threshold_text})
    def get_stat_masks(self):
        stat_masks = {}
        for el in self.stat_mask_widgets:
            name = el['name_text'].get(1.0, tk.END).strip(' \t\n')
            operator = el['operator_box'].get()
            applyon = el['apply_box'].get()
            threshold = float(el['threshold_text'].get(1.0, tk.END).strip(' \t\n'))
            stat_masks[name] = (operator,applyon,threshold)
        return stat_masks

    def apply(self):
        stat_masks = self.get_stat_masks()
        for name, (operator,applyon,threshold) in stat_masks.items():
            print('adding mask "{}"'.format(name))
            # set operator type
            min_entries_to_bins_ratio=-1
            max_entries_to_bins_ratio=-1
            if operator=='>':
                min_entries_to_bins_ratio = threshold
            elif operator=='<':
                max_entries_to_bins_ratio = threshold
            else:
                raise Exception('ERROR: stat mask operator {} not recognized.'.format(operator))
            # set application histograms
            histnames = None
            if applyon!='all': histnames=[applyon]
            self.histstruct.add_stat_mask( name, histnames=histnames,
                                            min_entries_to_bins_ratio=min_entries_to_bins_ratio,
                                            max_entries_to_bins_ratio=max_entries_to_bins_ratio)
        # close the window
        self.destroy()
        self.update()
        print('done')


class AddClassifiersWindow(tk.Toplevel):
    ### popup window class for adding classifiers to a histstruct

    def __init__(self, master, histstruct):
        super().__init__(master=master)
        self.title('Add classifiers')
        self.histstruct = histstruct

        # add widgets for setting the classifier type and options
        #self.containerframe = ScrolledFrameFrame(self, height=500, width=500, showscrollbars=True)
        self.containerframe = tk.Frame(self)
        setattr(self.containerframe, 'widget', self.containerframe) # to quickly switch between Frame and ScrolledFrameFrame
        self.containerframe.widget.grid(row=0, column=0, sticky='nsew')
        self.classifier_widgets = {}
        for i,histname in enumerate(self.histstruct.histnames):
            row = int(i/4)
            column = int(i%4)
            frame = tk.Frame(self.containerframe.widget)
            frame.grid(row=row, column=column)
            set_frame_default_style( frame )
            histname_label = tk.Label(frame, text=histname)
            histname_label.grid(row=0, column=0, columnspan=2)
            classifier_type_label = tk.Label(frame, text='Classifier type')
            classifier_type_label.grid(row=1, column=0)
            classifier_type_box = ttk.Combobox(frame, values=get_classifier_class())
            classifier_type_box.current(0)
            classifier_type_box['state'] = 'readonly'
            classifier_type_box.bind('<<ComboboxSelected>>', functools.partial(
                self.set_classifier_options, histname=histname) )
            classifier_type_box.grid(row=1, column=1)
            key_label = tk.Label(frame, text='Parameters')
            key_label.grid(row=2, column=0)
            value_label = tk.Label(frame, text='Values')
            value_label.grid(row=2, column=1)
            classifier_options_frame = OptionsFrame(frame, labels=[], values=[])
            classifier_options_frame.frame.grid(row=3, column=0, columnspan=2)
            # add option to evaluate the model after adding it
            evaluate_label = tk.Label(frame, text='evaluate')
            evaluate_label.grid(row=4, column=0)
            evaluate_box = ttk.Combobox(frame, values=[False,True])
            evaluate_box['state'] = 'readonly'
            evaluate_box.current(0)
            evaluate_box.grid(row=4, column=1)
            # add everything to a structure 
            self.classifier_widgets[histname] = {'type':classifier_type_box, 
                                                 'options':classifier_options_frame,
                                                 'evaluate':evaluate_box}
            self.set_classifier_options(None, histname)

        # add a button for adding the classifiers
        self.add_button = tk.Button(self, text='Add classifiers', command=self.add_classifiers)
        self.add_button.grid(row=1, column=0, columnspan=2)

    def set_classifier_options(self, event, histname):
        classifier_name = self.classifier_widgets[histname]['type'].get()
        (ctype, coptions) = get_classifier_class(classifier_name)
        # do special overridings if needed
        optiontypes = [None]*len(coptions.keys())
        if ctype is AutoEncoder.AutoEncoder:
            if 'modelpath' in list(coptions.keys()):
                idx = list(coptions.keys()).index('modelpath')
                optiontypes[idx] = GenericFileLoader
            if 'model' in list(coptions.keys()):
                idx = list(coptions.keys()).index('model')
                coptions.pop('model')
                optiontypes.pop(idx)
        # retrieve the docurl
        docurl = get_docurl(ctype)
        # now set the options
        self.classifier_widgets[histname]['options'].set_options(
                labels=coptions.keys(), types=optiontypes, values=coptions.values(),
                docurl=docurl)

    def get_classifier(self, histname):
        classifier_name = self.classifier_widgets[histname]['type'].get()
        (classifier, _) = get_classifier_class(classifier_name)
        classifier_options = self.classifier_widgets[histname]['options'].get_dict()
        return (classifier, classifier_options)

    def add_classifiers(self):
        for histname in self.histstruct.histnames:
            (classifier, classifier_options) = self.get_classifier(histname)
            classifier = classifier( **classifier_options )
            self.histstruct.add_classifier( histname, classifier )
            # check if need to evaluate
            do_evaluate = (self.classifier_widgets[histname]['evaluate'].get()=='True')
            if do_evaluate:
                self.histstruct.evaluate_classifier(histname)
        # close the window
        self.destroy()
        self.update()
        print('done')


class PlotSetsWindow(tk.Toplevel):
    ### popup window class for plotting the histograms in a histstruct

    def __init__(self, master, histstruct, plotstyleparser=None):
        super().__init__(master=master)
        self.title('Plotting')
        self.histstruct = histstruct
        self.plotstyleparser = plotstyleparser
        self.set_optionsframe_list = []
        self.select_set_button_list = []
        self.set_selector_list = []

        # make a frame holding the action buttons
        self.buttons_frame = tk.Frame(self)
        self.buttons_frame.grid(row=0, column=0, sticky='nsew')
        set_frame_default_style(self.buttons_frame)

        # add a button to allow adding more sets of options
        self.more_button = tk.Button(self.buttons_frame, text='Add a set', command=self.add_set)
        self.more_button.grid(row=0, column=0, sticky='nsew')

        # add a button to overwrite the plot style
        self.load_plotstyle_button = tk.Button(self.buttons_frame, text='Load plot style', 
                                        command = self.load_plotstyle)
        self.load_plotstyle_button.grid(row=0, column=1, sticky='nsew')

        # add a button to close this window
        self.close_button = tk.Button(self.buttons_frame, text='Close', command = self.close)
        self.close_button.grid(row=0, column=2, sticky='nsew')

        # add a button to make the plot
        self.plot_button = tk.Button(self.buttons_frame, text='Make plot', command=self.make_plot)
        self.plot_button.grid(row=1, column=0, columnspan=3, sticky='nsew')

        # add one set of options
        self.add_set()

    def load_plotstyle(self):
        initialdir = get_initialdir()
        filename = fldlg.askopenfilename(initialdir=initialdir,
                    title='Load a plot style',
                    filetypes=(('json files','*.json'),('all files','*.*')))
        # if filename is invalid, return
        if len(filename)==0:
            print('Loading of plot style canceled')
            return
        # clear current plotstyleparser and related info before loading new one
        self.plotstyleparser = None
        # load a new histstruct
        self.plotstyleparser = PlotStyleParser.PlotStyleParser( filename )

    def add_set(self):
        ### add widgets for one more histogram set to plot
        column = 0
        idx = len(self.set_selector_list)
        row = 1+len(self.set_selector_list)
        set_frame = tk.Frame(self)
        set_frame.grid(row=row, column=column, sticky='nsew')
        set_frame_default_style(set_frame)
        select_set_button = tk.Button(set_frame, text='Select set',
                                    command=functools.partial(self.open_selection_window,idx),
                                    bg='orange')
        select_set_button.grid(row=0, column=0)
        self.select_set_button_list.append(select_set_button)
        set_default_options = {'label':None, 'color':None}
        set_options_frame = OptionsFrame(set_frame,
                                            labels=set_default_options.keys(),
                                            values=set_default_options.values())
        set_options_frame.frame.grid(row=1, column=0)
        self.set_optionsframe_list.append(set_options_frame)
        self.set_selector_list.append( None )

    def open_selection_window(self, idx):
        self.set_selector_list[idx] = SelectorWindow(self.master, self.histstruct,
                                                        set_selection=True, post_selection=True)
        self.select_set_button_list[idx]['bg'] = 'green'
        # (to do: make check if sets were actually selected more robust, as selection window
        #         could be closed without making a selection, but button would still be green.)

    def check_all_selected(self):
        if None in self.set_selector_list: return False
        else: return True

    def make_plot(self):
        ### make the plot with current settings
        if not self.check_all_selected():
            raise Exception('ERROR: some sets were declared but not initialized.')
        optionsdict = {'histograms':[], 'labellist':[], 'colorlist':[]}
        # get histograms to plot
        for setselector in self.set_selector_list:
            optionsdict['histograms'].append( setselector.get_histograms() )
        # set style options
        for optionsframe in self.set_optionsframe_list:
            setoptions = optionsframe.get_dict()
            optionsdict['labellist'].append( setoptions['label'] )
            optionsdict['colorlist'].append( setoptions['color'] )
        if self.plotstyleparser is not None:
            optionsdict['titledict'] = self.plotstyleparser.get_title()
            optionsdict['titlesize'] = self.plotstyleparser.get_titlesize()
            optionsdict['xaxtitledict'] = self.plotstyleparser.get_xaxtitle()
            optionsdict['xaxtitlesize'] = self.plotstyleparser.get_xaxtitlesize()
            optionsdict['yaxtitledict'] = self.plotstyleparser.get_yaxtitle()
            optionsdict['yaxtitlesize'] = self.plotstyleparser.get_yaxtitlesize()
            optionsdict['ymaxfactor'] = self.plotstyleparser.get_ymaxfactor()
            optionsdict['legendsize'] = self.plotstyleparser.get_legendsize()
        # make the plots
        print('making plot...')
        res = self.histstruct.plot_histograms( **optionsdict )
        fig = res[0]
        axs = res[1]
        res2d = res[2] if len(res)>2 else None
        # post-processing of figure for 1D histograms
        # (might need updates to make it more flexible)
        if( fig is not None and axs is not None ):
            if self.plotstyleparser is not None:
                counter = -1
                for i in range(axs.shape[0]):
                    for j in range(axs.shape[1]):
                        counter += 1
                        histname = self.histstruct.histnames[counter]
                        ax = axs[i,j]
                        pu.add_cms_label( ax, pos=(0.05,0.9), 
                                  extratext=self.plotstyleparser.get_extracmstext(), 
                                  fontsize=self.plotstyleparser.get_cmstextsize() )
                        extratext = self.plotstyleparser.get_extratext(histname=histname)
                        if extratext is not None:
                            pu.add_text( ax, extratext,
                                 (0.5,0.75), fontsize=self.plotstyleparser.get_extratextsize() )
                        condtext = self.plotstyleparser.get_condtext()
                        if condtext is not None: 
                            pu.add_text( ax, condtext, (0.75,1.01), 
                                 fontsize=self.plotstyleparser.get_condtextsize() )
        # post-processing of figures for 2D histograms
        if res2d is not None:
            for (fig,axs) in res2d:
                pass
        # show the figures
        plt.show(block=False)

    def close(self):
        ### close the window
        self.destroy()
        self.update()


class PreProcessingWindow(tk.Toplevel):
    ### popup window for doing preprocessing of histograms in a HistStruct

    def __init__(self, master, histstruct):
        super().__init__(master=master)
        self.title('Preprocessing window')
        self.histstruct = histstruct
        self.set_selector = None

        # add a frame for preprocessing options
        options = []
        options.append( {'name':'cropping', 'val':None, 'type':tk.Text, 
                         'docurl':get_docurl(hu.get_cropslices_from_str)} )
        options.append( {'name':'rebinningfactor', 'val':None, 'type':tk.Text,
                        'docurl':get_docurl(hu.get_rebinningfactor_from_str)} )
        options.append( {'name':'smoothinghalfwindow', 'val':None, 'type':tk.Text,
                        'docurl':get_docurl(hu.get_smoothinghalfwindow_from_str)} )
        options.append( {'name':'donormalize', 'val':[False,True], 'type':ttk.Combobox,
                        'docurl':get_docurl(hu.normalizehists)} )
        labels = [el['name'] for el in options]
        types = [el['type'] for el in options]
        values = [el['val'] for el in options]
        docurls = [el['docurl'] for el in options]
        self.optionsframe = OptionsFrame(self, labels=labels, types=types, values=values, 
                                        docurls=docurls, docurl=get_docurl(histstruct.preprocess))
        self.optionsframe.frame.grid(row=0, column=0, sticky='nsew')

        # add a frame for selecting histograms
        self.selectionframe = tk.Frame(self)
        set_frame_default_style(self.selectionframe)
        self.selectionbutton = tk.Button(self.selectionframe, text='Select histogram set',
                                command=self.open_selection_window)
        self.selectionbutton.grid(row=0, column=0, sticky='nsew')
        self.selectionlabel = tk.Label(self.selectionframe, 
                                text='(Default: apply on all original histograms)')
        self.selectionlabel.grid(row=0, column=1, sticky='nsew')
        self.selectionframe.grid(row=1, column=0, sticky='nsew')

        # add a button to apply the preprocessing
        self.apply_button = tk.Button(self, text='Apply', command=self.apply)
        self.apply_button.grid(row=2, column=0, sticky='nsew')

    def apply(self):
        # get masks
        masknames = None
        if self.set_selector is not None:
            masknames = self.set_selector.masks
        # get options
        options = self.optionsframe.get_dict()
        # do special treatment if needed
        slices = hu.get_cropslices_from_str(options.pop('cropping'))
        options['cropslices'] = slices
        rebinningfactor = hu.get_rebinningfactor_from_str(options.pop('rebinningfactor'))
        options['rebinningfactor'] = rebinningfactor
        smoothinghalfwindow = hu.get_smoothinghalfwindow_from_str(options.pop('smoothinghalfwindow'))
        options['smoothinghalfwindow'] = smoothinghalfwindow
        # disable frame for the remainder of the processing time
        disable_frame( self )
        # do the preprocessing
        self.histstruct.preprocess(masknames=masknames, **options)
        # close the window
        self.destroy()
        self.update()
        print('done')

    def open_selection_window(self):
        self.set_selector = SelectorWindow(self.master, self.histstruct,
                                set_selection=False, post_selection=False)
        return


class DisplayHistStructWindow(tk.Toplevel):
    ### popup window class for displaying full info on a HistStruct

    def __init__(self, master, histstruct):
        super().__init__(master=master)
        self.title('HistStruct display')
        self.histstruct = histstruct

        # add scrollview widget for displaying large text
        self.viewframe = ScrolledTextFrame(self, txtheight=30, txtwidth=90)
        self.viewframe.widget.insert(tk.INSERT, self.get_info_str())
        self.viewframe.frame.grid(row=0, column=0)

    def get_info_str(self):
        return self.histstruct.__str__()

class SelectorWindow(tk.Toplevel):
    ### popup window class for histogram selection
    # (common to several other popup windows)

    def __init__(self, master, histstruct, mask_selection=True,
                                            set_selection=True,
                                            post_selection=True,
                                            allow_multi_mask=True,
                                            allow_multi_set=False):
        super().__init__(master=master)
        self.title('Histogram selection')
        self.histstruct = histstruct
        self.histograms = None
        self.masks = None
        self.sets = None
        self.scores = None
        self.globalscores = None
        self.randoms = -1
        self.first = -1
        self.partitions = -1

        # add widgets for choosing masks
        self.histstruct_masks_frame = tk.Frame(self)
        self.histstruct_masks_label = tk.Label(self.histstruct_masks_frame, text='Choose masks')
        self.histstruct_masks_label.grid(row=0, column=0, sticky='nsew')
        mask_selectmode = 'multiple' if allow_multi_mask else 'single'
        self.histstruct_masks_listbox = tk.Listbox(self.histstruct_masks_frame, 
                                                    selectmode=mask_selectmode,
                                                    exportselection=False)
        for maskname in self.histstruct.get_masknames():
            self.histstruct_masks_listbox.insert(tk.END, maskname)
        if len(self.histstruct.get_masknames())==0:
            self.histstruct_masks_listbox.insert(tk.END, '[no masks available]')
        self.histstruct_masks_listbox.grid(row=1, column=0, sticky='nsew')
        if mask_selection: self.histstruct_masks_frame.grid(row=0, column=0, sticky='nsew')
        
        # add widgets for choosing a (resampled) set directly
        self.histstruct_sets_frame = tk.Frame(self)
        self.histstruct_sets_label = tk.Label(self.histstruct_sets_frame, text='Choose sets')
        self.histstruct_sets_label.grid(row=0, column=0, sticky='nsew')
        set_selectmode = 'multiple' if allow_multi_set else 'single'
        self.histstruct_sets_listbox = tk.Listbox(self.histstruct_sets_frame, 
                                                    selectmode=set_selectmode,
                                                    exportselection=False)
        for extname in self.histstruct.exthistograms.keys():
            self.histstruct_sets_listbox.insert(tk.END, extname)
        if len(self.histstruct.exthistograms.keys())==0:
            self.histstruct_sets_listbox.insert(tk.END, '[no sets available]')
        self.histstruct_sets_listbox.grid(row=1, column=0, sticky='nsew')
        if set_selection: self.histstruct_sets_frame.grid(row=1, column=0, sticky='nsew')

        # add widgets for randoms, first, or averages
        self.other_options_frame = tk.Frame(self)
        self.other_options_label = tk.Label(self.other_options_frame, text='Other options')
        self.other_options_label.grid(row=0, column=0)
        options = {'randoms':-1, 'first':-1, 'partitions':-1}
        self.optionsframe = OptionsFrame(self.other_options_frame, labels=list(options.keys()),
                                            values=list(options.values()))
        self.optionsframe.frame.grid(row=1, column=0)
        if post_selection: self.other_options_frame.grid(row=2, column=0, sticky='nsew')

        # add widget for selection
        self.select_button = tk.Button(self, text='Select', command=self.select_histograms)
        self.select_button.grid(row=3, column=0)

    def get_masks(self):
        ### get currently selected masks
        # warning: do not use after selection window has been closed,
        #          use self.masks for that!
        masks = ([self.histstruct_masks_listbox.get(idx)
                    for idx in self.histstruct_masks_listbox.curselection() ])
        return masks

    def get_sets(self):
        ### get currently selected sets
        # warning: do not use after selection window has been closed,
        #          use self.sets for that!
        sets = ([self.histstruct_sets_listbox.get(idx)
                    for idx in self.histstruct_sets_listbox.curselection() ])
        return sets

    def get_histograms(self):
        ### get currently selected histograms
        return self.histograms

    def get_scores(self):
        ### get scores of currently selected histograms
        if self.scores is None:
            print('WARNING: the current histogram selection does not contain scores.'
                    +' Did you properly evaluate a model on the selected set?')
        return self.scores

    def get_globalscores(self):
        ### get global scores of currently selected lumisections
        if self.globalscores is None:
            print('WARNING: the current lumisection selection does not contain global scores.'
                    +' Did you properly evaluate a model on the selected set?')
        return self.globalscores

    def select_histograms(self):
        ### set the histograms based on the current user settings

        # get masks and/or sets
        masks = self.get_masks()
        do_masks = bool(len(masks)>0)
        sets = self.get_sets()
        do_sets = bool(len(sets)>0)
        if( not do_masks and not do_sets ):
            raise Exception('ERROR: you must select either at least one mask or a training set.')
        if( do_masks and do_sets ):
            raise Exception('ERROR: you cannot select both masks and sets.')
        
        # get other options
        options = self.optionsframe.get_dict()
        nspecified = len([val for val in list(options.values()) if val>0])
        if nspecified>1:
            raise Exception('ERROR: you can only specifiy maximum one option'
                            +' of the list {}'.format(list(options.keys)))
        randoms = options['randoms']
        first = options['first']
        partitions = options['partitions']
        
        # get all numbers
        extname = None
        if do_sets: extname = sets[0]
        res = self.histstruct.get_histogramsandscores( extname=extname, masknames=masks, 
                                nrandoms=randoms, nfirst=first)
        self.histograms = res['histograms']
        self.scores = res['scores']
        self.globalscores = res['globalscores']
        self.masks = masks
        self.sets = sets
        self.randoms = randoms
        self.first = first
        self.partitions = partitions
        
        # process other options
        if partitions>0:
            # in case of partitions, set scores to None since they become senseless
            self.scores = None
            self.globalscores = None
            # average the histograms
            for histname in self.histograms.keys():
                self.histograms[histname] = hu.averagehists( self.histograms[histname], partitions )
        
        # close the window
        self.destroy()
        self.update()
        print('done')

class TrainClassifiersWindow(tk.Toplevel):
    ### popup window class for adding classifiers

    def __init__(self, master, histstruct):
        super().__init__(master=master)
        self.title('Training')
        self.histstruct = histstruct
        self.training_set_selector = None
        self.training_options = {}

        # create frame for options
        self.train_options_frame = tk.Frame(self)
        self.train_options_frame.grid(row=0, column=0)
        set_frame_default_style( self.train_options_frame )
        self.train_options_label = tk.Label(self.train_options_frame, text='Training settings')
        self.train_options_label.grid(row=0, column=0)

        # add widget to select histograms
        self.select_train_button = tk.Button(self.train_options_frame, 
                                            text='Select training set',
                                            command=self.open_training_selection_window,
                                            bg='orange')
        self.select_train_button.grid(row=1, column=0)

        # add widget to expand options for different histograms
        self.expand_options_button = tk.Button(self.train_options_frame,
                                                text='Expand/collapse',
                                                command=self.expandcollapse)
        self.expand_options_button.grid(row=2, column=0)
        # set initial state to single if only one classifier is present, multi otherwise
        self.expandstate = 'single' # set to single since one automatic expansion
        if get_training_options( self.histstruct ) is not None:
            self.expandstate = 'multi' # set to multi since one automatic collapse

        # add widgets for training options
        self.container_frame = tk.Frame(self)
        self.container_frame.grid(row=1, column=0)
        set_frame_default_style( self.container_frame )
        self.expandcollapse()

        # add button to start training
        self.train_button = tk.Button(self, text='Start training', command=self.do_training)
        self.train_button.grid(row=4, column=0, columnspan=2)

    def open_training_selection_window(self):
        self.training_set_selector = SelectorWindow(self.master, self.histstruct)
        self.select_train_button['bg'] = 'green'
        return

    def expandcollapse(self):
        # check whether need to collapse or expand
        if self.expandstate=='multi':
            # check if this is allowed
            if get_training_options( self.histstruct ) is None:
                print('WARNING: collapse not allowed'
                        +' since different types of classifiers are present')
                return
            histnames = ['all histogram types']
            self.expandstate = 'single'
        elif self.expandstate=='single':
            histnames = self.histstruct.histnames
            self.expandstate = 'multi'
        else:
            raise Exception('ERROR: expandstate {} not recognized.'.format(self.expandstate))
        # clear current options and frame
        self.training_options = {}
        for widget in self.container_frame.winfo_children(): widget.destroy()
        # make new options and frame
        for i,histname in enumerate(histnames):
            # make a frame and put some labels
            this_options_frame = tk.Frame(self.container_frame)
            this_options_frame.grid(row=0, column=i)
            set_frame_default_style( this_options_frame )
            hist_label = tk.Label(this_options_frame, text=histname)
            hist_label.grid(row=0, column=0)
            # get the training options
            arghistname = histname
            if histname=='all histogram types': arghistname = None
            (c,options) = get_training_options( self.histstruct, histname=arghistname )
            labels = list(options.keys())
            values = list(options.values())
            wtypes = [None]*len(labels)
            # overwrite or parse training options in special cases
            for j in range(len(labels)):
                if is_bool(str(values[j])):
                    wtypes[j] = ttk.Combobox
                    values[j] = [values[j], not values[j]]
            # get docurl
            docurl = get_docurl(c)
            # make the options frame
            options_frame = OptionsFrame(this_options_frame,
                labels=labels, values=values, types=wtypes,
                docurl=docurl)
            options_frame.frame.grid(row=1, column=0)
            self.training_options[histname] = options_frame

    def do_training(self):
        if self.training_set_selector is None:
            raise Exception('ERROR: please select a training set before starting training.')
        training_histograms = self.training_set_selector.get_histograms()
        for histname in training_histograms.keys():
            # check if a classifier is initialized for this histogram type
            if histname not in self.histstruct.classifiers.keys():
                print('WARNING: no classifier was found in the HistStruct'
                        +' for histogram type {}; skipping.'.format(histname))
                continue
            # get the options for this histogram type
            arghistname = histname
            if self.expandstate=='single': arghistname = 'all histogram types'
            training_options = self.training_options[arghistname].get_dict()
            # get the training histograms
            hists = training_histograms[histname]
            print('training a classifier for {}'.format(histname))
            print('size of training set: {}'.format(hists.shape))
            # do training
            self.histstruct.classifiers[histname].train( hists, **training_options )
            # do evaluation
            print('evaluating model for '+histname)
            self.histstruct.evaluate_classifier( histname )
        # close the window
        self.destroy()
        self.update()
        print('done')


class FitWindow(tk.Toplevel):
    ### popup window class for fitting classifier outputs

    def __init__(self, master, histstruct, plotstyleparser=None):
        super().__init__(master=master)
        self.title('Fitting')
        self.histstruct = histstruct
        self.plotstyleparser = plotstyleparser
        self.fitting_set_selector = None
        self.plotfunction = None
        self.plotdim = None
        if len(self.histstruct.histnames)==1:
            self.plotfunction = pu.plot_fit_1d
            self.plotdim = 1
        else:
            self.plotfunction = pu.plot_fit_2d
            self.plotdim = 2

        # create frame for options
        self.fit_options_frame = tk.Frame(self)
        self.fit_options_frame.grid(row=0, column=0, sticky='nsew')
        set_frame_default_style( self.fit_options_frame )
        self.fit_options_label = tk.Label(self.fit_options_frame, text='Fit settings')
        self.fit_options_label.grid(row=0, column=0, columnspan=2)

        # add widgets for choosing fitting set
        self.select_fitting_set_button = tk.Button(self.fit_options_frame, 
                                                text='Select fitting set',
                                                command=self.open_fitting_set_selection_window,
                                                bg='orange')
        self.select_fitting_set_button.grid(row=1, column=0, columnspan=2)

        # add widgets to select fitting parameters
        self.fitter_label = tk.Label(self.fit_options_frame, text='Fitter type')
        self.fitter_label.grid(row=2,column=0)
        self.fitter_box = ttk.Combobox(self.fit_options_frame, values=get_fitter_class())
        self.fitter_box.current(0)
        self.fitter_box['state'] = 'readonly'
        self.fitter_box.bind('<<ComboboxSelected>>', self.set_fitter_options)
        self.fitter_box.grid(row=2,column=1)
        self.key_label = tk.Label(self.fit_options_frame, text='Parameters')
        self.key_label.grid(row=3, column=0)
        self.value_label = tk.Label(self.fit_options_frame, text='Values')
        self.value_label.grid(row=3, column=1)
        self.fitter_options_frame = OptionsFrame(self.fit_options_frame, 
                labels=[], values=[])
        self.fitter_options_frame.frame.grid(row=4, column=0, columnspan=2)
        self.set_fitter_options(None)

        # create frame for plotting options
        self.plot_options_frame = tk.Frame(self)
        self.plot_options_frame.grid(row=1, column=0, sticky='nsew')
        set_frame_default_style( self.plot_options_frame )
        self.plot_options_label = tk.Label(self.plot_options_frame, text='Plot settings')
        self.plot_options_label.grid(row=0, column=0)
    
        # add widgets for plotting options
        plot_options_dict = get_args_dict(self.plotfunction)
        plot_docurl = get_docurl(self.plotfunction)
        # remove some keys that are not user input
        for key in ['fitfunc','xaxtitle','yaxtitle']:
            if key in list(plot_options_dict.keys()):
                plot_options_dict.pop(key)
        # set default values with plotstyleparser
        if self.plotstyleparser is not None:
            for key in list(plot_options_dict.keys()):
                if key=='xaxtitlesize': plot_options_dict[key] = self.plotstyleparser.get_xaxtitlesize()
                elif key=='yaxtitlesize': plot_options_dict[key] = self.plotstyleparser.get_yaxtitlesize()
        # set other default arguments
        for key in list(plot_options_dict.keys()):
            if key=='caxtitle': plot_options_dict[key] = 'Probability density'
            if key=='caxtitlesize': plot_options_dict[key] = 12
        # add meta arguments
        meta_args = {'do_plot':True}
        plot_options_dict = {**meta_args, **plot_options_dict}
        # set the widget types
        labels = list(plot_options_dict.keys())
        values = list(plot_options_dict.values())
        wtypes = [tk.Text]*len(labels)
        for i, value in enumerate(values):
            if isinstance(value, bool):
                wtypes[i] = ttk.Combobox
                values[i] = [value, not value]
        # make the OptionsFrame
        self.plot_options = OptionsFrame(self.plot_options_frame,
                                            labels=labels, types=wtypes, values=values,
                                            docurl=plot_docurl)
        self.plot_options.frame.grid(row=1, column=0)

        # add a frame for some buttons
        self.action_buttons_frame = tk.Frame(self)
        set_frame_default_style( self.action_buttons_frame )
        self.action_buttons_frame.grid(row=2, column=0, sticky='nsew')

        # add a button to start the fit
        self.fit_button = tk.Button(self.action_buttons_frame, text='Start fit', command=self.do_fit)
        self.fit_button.grid(row=0, column=0, sticky='nsew')

        # add a button to close the window
        self.close_button = tk.Button(self.action_buttons_frame, text='Close', command=self.close)
        self.close_button.grid(row=0, column=1, sticky='nsew')

    def set_fitter_options(self, event):
        fitter_name = self.fitter_box.get()
        (c, coptions) = get_fitter_class(fitter_name)
        docurl = get_docurl(c)
        self.fitter_options_frame.set_options(labels=coptions.keys(), values=coptions.values(),
                                                docurl=docurl)

    def open_fitting_set_selection_window(self):
        self.fitting_set_selector = SelectorWindow(self.master, self.histstruct)
        self.select_fitting_set_button['bg'] = 'green'
        return
    
    def get_fitting_scores(self):
        if self.fitting_set_selector is None:
            raise Exception('ERROR: please select a set to fit to before doing the fit.')
        scores_fit_dict = self.fitting_set_selector.get_scores()
        if scores_fit_dict is None:
            raise Exception('ERROR: no valid scores could be found in the HistStruct '
                            +'for the specified fitting set.')
        scores_fit = []
        for histname in self.histstruct.histnames:
            thisscore = scores_fit_dict[histname]
            scores_fit.append( thisscore )
        # transform to arrays with correct shape
        scores_fit = np.array(scores_fit)
        scores_fit = np.transpose(scores_fit)
        print('found score array for fitting set of following shape: {}'.format(scores_fit.shape))
        return scores_fit

    def get_fitter(self):
        fitter_name = self.fitter_box.get()
        (fitter, _) = get_fitter_class(fitter_name)
        fitter_options = self.fitter_options_frame.get_dict()
        return (fitter,fitter_options)

    def do_fit(self):
        # make frame disabled for the rest of the processing time
        disable_frame(self)
        # get fitter and plotting options
        fitting_scores = self.get_fitting_scores()
        (fitter,fitter_options) = self.get_fitter()
        plot_options_dict = self.plot_options.get_dict()
        do_plot = plot_options_dict.pop('do_plot')
        # determine all combinations of dimensions
        dimslist = []
        fitfunclist = []
        nhisttypes = len(self.histstruct.histnames)
        if self.plotdim == 1:
            dimslist = list(range(nhisttypes))
            # (note: for now self.plotdim is only 1 in case nhisttypes is 1, 
            #  but might be added as a user input argument later, so keep as general as possible)
        elif self.plotdim == 2:
            for i in range(0,nhisttypes-1):
                for j in range(i+1,nhisttypes):
                    dimslist.append((i,j))
        plt.close('all')
        # loop over all combinations of dimensions
        for dims in dimslist:
            # make the partial fit and store it
            thismse = fitting_scores[:,dims]
            if len(thismse.shape)==1: thismse = np.expand_dims(thismse, 1)
            fitfunc = fitter( thismse, **fitter_options )
            fitfunclist.append(fitfunc)
            # make the plot if requested
            if do_plot:
                if self.plotdim == 1:
                    xaxtitle = pu.make_text_latex_safe(self.histstruct.histnames[dims])
                    yaxtitle = 'Probability density'
                elif self.plotdim == 2:
                    xaxtitle = pu.make_text_latex_safe(self.histstruct.histnames[dims[0]])
                    yaxtitle = pu.make_text_latex_safe(self.histstruct.histnames[dims[1]])
                (fig,ax) = self.plotfunction(thismse, fitfunc=fitfunc,
                                    xaxtitle=xaxtitle,
                                    yaxtitle=yaxtitle,
                                    **plot_options_dict)
                # add extra text to the axes
                # (might need updates to make it more flexible)
                pu.add_text( ax, 'Density fit of lumisection scores', 
                            (0.05,0.8), fontsize=12, background_alpha=0.75 )
                if self.plotstyleparser is not None:
                    pu.add_cms_label( ax, pos=(0.05,0.9),
                                      extratext=self.plotstyleparser.get_extracmstext(),
                                      fontsize=self.plotstyleparser.get_cmstextsize(),
                                      background_alpha=0.75 )
                    condtext = self.plotstyleparser.get_condtext()
                    if condtext is not None:
                        pu.add_text( ax, condtext, (0.75,1.01),
                                    fontsize=self.plotstyleparser.get_condtextsize() )
                plt.show(block=False)
        self.histstruct.fitfunclist = fitfunclist
        self.histstruct.dimslist = dimslist
        self.histstruct.fitting_scores = fitting_scores
        # to do: same comment as below
        self.histstruct.fitfunc = fitter( fitting_scores, **fitter_options )
        # to do: extend HistStruct class to contain the fitfunc in a cleaner way!
        #        (or decide on another way to make this ad-hod attribute assignment more clean)
        # evaluate the fitted function on the non-extended histstruct
        scores_all = []
        for histname in self.histstruct.histnames:
            thisscore = self.histstruct.get_scores( histname=histname )
            scores_all.append( thisscore )
        scores_all = np.array(scores_all)
        scores_all = np.transpose(scores_all)
        self.histstruct.add_globalscores( np.log(self.histstruct.fitfunc.pdf(scores_all)) )
        # enable the frame again
        enable_frame(self)

    def close(self):
        # close the window
        self.destroy()
        self.update()


class ResampleWindow(tk.Toplevel):
    ### popup window class for resampling testing and training sets

    def __init__(self, master, histstruct):
        super().__init__(master=master)
        self.title('Resampling window')
        self.histstruct = histstruct
        self.resample_set_selectors = {}
        self.resample_set_selector_buttons = {}
        self.resample_functions = {}
        self.resample_options = {}
        self.allhistostr = 'All histogram types'
        self.noresamplestr = '[No resampling]'
        self.nonestr = 'None' # not free to choose, 
                              # should correspond to the one in the global function
                              # get_resampling_function

        # create a frame for the buttons
        self.buttons_frame = tk.Frame(self)
        self.buttons_frame.grid(row=0, column=0, sticky='nsew')
        set_frame_default_style( self.buttons_frame )
        
        # add widget to add a set
        # new approach: do not allow to add multiple sets at once 
        # (can simply open this window multiple times sequentially for that),
        # but instead allow splitting between different histogram types
        # add widget to expand options for different histograms
        self.expand_options_button = tk.Button(self.buttons_frame,
                                                text='Expand/collapse',
                                                command=self.expandcollapse)
        self.expand_options_button.grid(row=0, column=0)
        # set initial state to single
        self.expandstate = 'multi' # set multi since automatic expandcollapse call

        # add widgets to start resampling
        self.resample_button = tk.Button(self.buttons_frame, text='Start resampling',
                                        command=self.do_resampling)
        self.resample_button.grid(row=1,column=0)

        # add widgets to view current resampled sets
        self.view_frame = tk.Frame(self)
        self.view_frame.grid(row=0, column=1)
        set_frame_default_style( self.view_frame )
        self.sets_label = tk.Label(self.view_frame, text='Existing sets')
        self.sets_label.grid(row=0, column=0)
        self.sets_listbox = tk.Listbox(self.view_frame, selectmode='multiple',
                                    exportselection=False)
        self.update_sets_list()
        self.sets_listbox.grid(row=1,column=0)

        # add widgets for training options
        self.container_frame = tk.Frame(self)
        self.container_frame.grid(row=0, column=2)
        set_frame_default_style( self.container_frame )
        self.expandcollapse()

    def update_sets_list(self):
        self.sets_listbox.delete(0, tk.END)
        extnames = self.histstruct.exthistograms.keys()
        for extname in extnames:
            self.sets_listbox.insert(tk.END, extname)
        if len(extnames)==0:
            self.sets_listbox.insert(tk.END, '[no sets available]')

    def expandcollapse(self):
        # check whether need to collapse or expand
        if self.expandstate=='multi':
            histnames = [self.allhistostr]
            self.expandstate = 'single'
        elif self.expandstate=='single':
            histnames = self.histstruct.histnames
            self.expandstate = 'multi'
        else:
            raise Exception('ERROR: expandstate {} not recognized.'.format(self.expandstate))
        # clear current options and frame
        self.resample_set_selectors = {}
        self.resample_set_selector_buttons = {}
        self.resample_functions = {}
        self.resample_options = {}
        for widget in self.container_frame.winfo_children(): widget.destroy()
        # make new name label and text entry
        self.name_label = tk.Label(self.container_frame, text='Set name')
        self.name_label.grid(row=0, column=0)
        self.name_text = tk.Text(self.container_frame, height=1, width=15)
        self.name_text.grid(row=0, column=1)
        # make new options and frame
        for i,histname in enumerate(histnames):
            # create a frame to hold the widgets and put some labels
            this_set_frame = tk.Frame(self.container_frame)
            this_set_frame.grid(row=1, column=2*i, columnspan=2)
            set_frame_default_style( this_set_frame )
            hist_label = tk.Label(this_set_frame, text=histname)
            hist_label.grid(row=0, column=0, columnspan=2)
            # add widgets for choosing resampling basis set
            select_button = tk.Button(this_set_frame, text='Select set',
                                    command=functools.partial(self.open_select_window, histname),
                                    bg='orange')
            select_button.grid(row=1, column=0, columnspan=2)
            # add label and text entry for dataset name -> to take out of the loop!
            #name_label = tk.Label(set_frame, text='Name')
            #name_label.grid(row=1, column=0)
            #name_text = tk.Text(set_frame, height=1, width=15)
            #name_text.grid(row=1, column=1)
            # add label and box entry for resampling function
            function_label = tk.Label(this_set_frame, text='Function')
            function_label.grid(row=2, column=0)
            allowed_functions = get_resampling_function()
            for i,f in enumerate(allowed_functions): 
                if f==self.nonestr: allowed_functions[i] = self.noresamplestr
            function_box = ttk.Combobox(this_set_frame, values=allowed_functions)
            function_box.current(0)
            function_box['state'] = 'readonly'
            function_box.bind('<<ComboboxSelected>>', functools.partial(
                self.set_function_options, histname=histname))
            function_box.grid(row=2, column=1)
            # make initial (empty) OptionsFrame
            key_label = tk.Label(this_set_frame, text='Parameters')
            key_label.grid(row=3, column=0)
            value_label = tk.Label(this_set_frame, text='Values')
            value_label.grid(row=3, column=1)
            function_options_frame = OptionsFrame(this_set_frame, labels=[], values=[], autobool=True)
            function_options_frame.frame.grid(row=4, column=0, columnspan=2)
            # add objects to the dicts
            self.resample_set_selectors[histname] = None
            self.resample_set_selector_buttons[histname] = select_button
            self.resample_functions[histname] = function_box
            self.resample_options[histname] = function_options_frame

    def set_function_options(self, event, histname):
        fname = self.get_function_name(histname)
        (f, foptions) = get_resampling_function(key=fname)
        fdocurl = get_docurl(f)
        self.resample_options[histname].set_options(
            labels=foptions.keys(), values=foptions.values(), docurl=fdocurl)

    def open_select_window(self, histname):
        self.resample_set_selectors[histname] = SelectorWindow(self.master, self.histstruct)
        self.resample_set_selector_buttons[histname]['bg'] = 'green'
        return

    def check_all_selected(self):
        if None in list(self.resample_set_selectors.values()): return False
        else: return True

    def get_name(self):
        return self.name_text.get(1.0,tk.END).strip(' \t\n') 

    def get_function_name(self, histname):
        fname = self.resample_functions[histname].get()
        if fname==self.noresamplestr: fname = self.nonestr
        return fname

    def get_function(self, histname):
        fname = self.get_function_name(histname)
        (function, _) = get_resampling_function(fname)
        function_options = self.resample_options[histname].get_dict()
        return (function, function_options)

    def do_resampling(self):

        # check whether the same set, function and options can be used for all histogram types
        split = None
        if self.expandstate == 'single': split = False
        elif self.expandstate == 'multi': split = True
        else:
            raise Exception('ERROR: expandstate {} not recognized.'.format(self.expandstate))
        # check whether all required sets have been selected
        if not self.check_all_selected():
            for histname, selector in self.resample_set_selectors.items():
                # get the function (allow None selector if no resampling required)
                histkey = histname if split else self.allhistostr
                (function,_) = self.get_function(histkey)
                if( selector is None and function is not None ):
                    raise Exception('ERROR: requested to resample histogram type {}'.format(histname)
                            +' but selector was not set.')
        # get the name for the extended set
        extname = self.get_name()
        if len(extname)==0:
            raise Exception('ERROR: name "{}" is not valid.'.format(extname))
        # loop over histogram types
        for histname in self.histstruct.histnames:
            print('  now processing histogram type {}'.format(histname))
            histkey = histname if split else self.allhistostr
            # get resampling function
            (function, function_options) = self.get_function(histkey)
            if function is None:
                print('WARNING: resampling function for histogram type {}'.format(histname)
                        +' is None, it will not be present in the resampled set.')
                continue
            # get histograms
            hists = self.resample_set_selectors[histkey].get_histograms()[histname]
            exthists = function( hists, **function_options )
            # add extended set to histstruct
            self.histstruct.add_exthistograms( extname, histname, exthists )
            print('  -> generated {} histograms'.format(len(exthists)))
        self.update_sets_list()
        plt.show(block=False)
        # close the window
        self.destroy()
        self.update()
        print('done')


class EvaluateWindow(tk.Toplevel):
    ### popup window class for evaluating a given model

    def __init__(self, master, histstruct, plotstyleparser=None):
        super().__init__(master=master)
        self.title('Evaluation window')
        self.histstruct = histstruct
        self.plotstyleparser = plotstyleparser
        self.test_set_widgets = []

        # create a frame for the addition of test sets
        self.test_set_frame = tk.Frame(self)
        self.test_set_frame.grid(row=0, column=0, sticky='nsew')
        set_frame_default_style( self.test_set_frame )
        self.test_set_label = tk.Label(self.test_set_frame, text='Select test set')
        self.test_set_label.grid(row=0, column=0)

        # create a frame for the test sets
        self.test_set_container_frame = tk.Frame(self)
        self.test_set_container_frame.grid(row=0, column=1, sticky='nsew')
        set_frame_default_style( self.test_set_container_frame )
        self.test_set_container_label = tk.Label(self.test_set_container_frame, text='Test set:')
        self.test_set_container_label.grid(row=0, column=0)
        self.test_sets_type_label = tk.Label(self.test_set_container_frame, text='Type:')
        self.test_sets_type_label.grid(row=1, column=0)
        self.test_sets_label_label = tk.Label(self.test_set_container_frame, text='Label:')
        self.test_sets_label_label.grid(row=2, column=0)

        # add a button to add more test sets
        self.add_button = tk.Button(self.test_set_frame, text='Add test set', 
                                    command=functools.partial(self.add_set, 
                                                              self.test_set_container_frame))
        self.add_button.grid(row=1, column=0)

        # add one test set for type good and one for type bad
        self.add_set( parent=self.test_set_container_frame, default_type='Good' )
        self.add_set( parent=self.test_set_container_frame, default_type='Bad' )

        # create frame for other options
        self.evaluation_options_frame = tk.Frame(self)
        self.evaluation_options_frame.grid(row=1, column=0, columnspan=2, sticky='nsew')
        set_frame_default_style( self.evaluation_options_frame )

        # add widgets for score distribution
        self.score_dist_options_label = tk.Label(self.evaluation_options_frame, 
                                                 text='Options for score plot')
        self.score_dist_options_label.grid(row=0, column=0)
        # get available options
        score_dist_options_dict = get_args_dict(pu.plot_score_dist)
        score_dist_docurl = get_docurl(pu.plot_score_dist)
        # remove some keys that are not user input
        for key in ['fig','ax','doshow']:
            if key in list(score_dist_options_dict.keys()):
                score_dist_options_dict.pop(key)
        # set some default arguments
        for key in list(score_dist_options_dict.keys()):
            if key=='siglabel': score_dist_options_dict[key] = 'Anomalous'
            elif key=='sigcolor': score_dist_options_dict[key] = 'r'
            elif key=='bcklabel': score_dist_options_dict[key] = 'Good'
            elif key=='bckcolor': score_dist_options_dict[key] = 'g'
            elif key=='nbins': score_dist_options_dict[key] = 200
            elif key=='normalize': score_dist_options_dict[key] = True
            elif key=='xaxtitle': 
                score_dist_options_dict[key] = 'Model score'
            elif key=='yaxtitle': 
                score_dist_options_dict[key] = 'Normalized number of lumisections'
        # add meta arguments
        meta_args = {'make score distribution':True}
        score_dist_options_dict = {**meta_args, **score_dist_options_dict}
        # set the widget types
        labels = list(score_dist_options_dict.keys())
        values = list(score_dist_options_dict.values())
        wtypes = [tk.Text]*len(labels)
        for i, value in enumerate(values):
            if isinstance(value, bool):
                wtypes[i] = ttk.Combobox
                values[i] = [value, not value]
        # make the actual OptionsFrame
        self.score_dist_options_frame = OptionsFrame(self.evaluation_options_frame, 
                                            labels=labels, types=wtypes, values=values,
                                            docurl=score_dist_docurl)
        self.score_dist_options_frame.frame.grid(row=1, column=0)

        # add widgets for roc curve
        self.roc_options_label = tk.Label(self.evaluation_options_frame, 
                                          text='Options for ROC curve')
        self.roc_options_label.grid(row=0, column=1)
        # get available options
        roc_options_dict = get_args_dict(aeu.get_roc)
        roc_docurl = get_docurl(aeu.get_roc)
        # remove some keys that are not user input
        for key in ['doplot','doshow']:
            if key in list(roc_options_dict.keys()):
                roc_options_dict.pop(key)
        # set some default arguments
        for key in list(roc_options_dict.keys()):
            if key=='mode': roc_options_dict[key] = 'geom'
        # add meta arguments
        meta_args = {'make ROC curve': True}
        roc_options_dict = {**meta_args, **roc_options_dict}
        # set the widget types
        labels = list(roc_options_dict.keys())
        values = list(roc_options_dict.values())
        wtypes = [tk.Text]*len(labels)
        for i, value in enumerate(values):
            if isinstance(value, bool):
                wtypes[i] = ttk.Combobox
                values[i] = [value, not value]
        self.roc_options_frame = OptionsFrame(self.evaluation_options_frame,
                                            labels=labels, types=wtypes, values=values,
                                            docurl=roc_docurl)
        self.roc_options_frame.frame.grid(row=1, column=1)

        # add widgets for confusion matrix
        self.cm_options_label = tk.Label(self.evaluation_options_frame, 
                                         text='Options for confusion matrix')
        self.cm_options_label.grid(row=0, column=2)
        # get available options
        cm_options_dict = get_args_dict(aeu.get_confusion_matrix)
        cm_docurl = get_docurl(aeu.get_confusion_matrix)
        # add meta arguments
        meta_args = {'make confusion matrix': True}
        cm_options_dict = {**meta_args, **cm_options_dict}
        # set the widget types
        labels = list(cm_options_dict.keys())
        values = list(cm_options_dict.values())
        wtypes = [tk.Text]*len(labels)
        for i, value in enumerate(values):
            if isinstance(value, bool):
                wtypes[i] = ttk.Combobox
                values[i] = [value, not value]
        self.cm_options_frame = OptionsFrame(self.evaluation_options_frame,
                                            labels=labels, types=wtypes, values=values,
                                            docurl=cm_docurl)
        self.cm_options_frame.frame.grid(row=1, column=2)

        # add widgets for output json file
        self.json_label = tk.Label(self.evaluation_options_frame, 
                                    text='Options for output json file')
        self.json_label.grid(row=0, column=3)
        json_options_dict = {'make json file': False,
                             'json filename': ''}
        # set the widget types
        labels = list(json_options_dict.keys())
        values = list(json_options_dict.values())
        wtypes = [tk.Text]*len(labels)
        for i, value in enumerate(values):
            if isinstance(value, bool):
                wtypes[i] = ttk.Combobox
                values[i] = [value, not value]
        for i, label in enumerate(labels):
            if label=='json filename': wtypes[i] = GenericFileSaver
        self.json_options_frame = OptionsFrame(self.evaluation_options_frame,
                                            labels=labels, types=wtypes, values=values)
        self.json_options_frame.frame.grid(row=1, column=3)
        
        # add widgets for 2D contour plots
        self.contour_options_label = tk.Label(self.evaluation_options_frame, 
                                              text='Options for fit plots')
        self.contour_options_label.grid(row=0, column=4)
        # add widgets for plotting options
        self.contourfunction = None
        self.contourdim = None
        if len(self.histstruct.histnames)==1:
            self.contourfunction = pu.plot_fit_1d
            self.contourdim = 1
        else:
            self.contourfunction = pu.plot_fit_2d
            self.contourdim = 2
        contour_options_dict = get_args_dict(self.contourfunction)
        contour_docurl = get_docurl(self.contourfunction)
        # remove some keys that are not user input
        for key in ['fitfunc','xaxtitle','yaxtitle','onlycontour']:
            if key in list(contour_options_dict.keys()):
                contour_options_dict.pop(key)
        # set default values with plotstyleparser
        if self.plotstyleparser is not None:
            for key in list(contour_options_dict.keys()):
                if key=='xaxtitlesize': contour_options_dict[key] = self.plotstyleparser.get_xaxtitlesize()
                elif key=='yaxtitlesize': contour_options_dict[key] = self.plotstyleparser.get_yaxtitlesize()
        # set other default arguments
        for key in list(contour_options_dict.keys()):
            if key=='caxtitle': contour_options_dict[key] = 'Probability density'
            if key=='caxtitlesize': contour_options_dict[key] = 12
        # add meta arguments
        meta_args = {'make fit plots':False}
        contour_options_dict = {**meta_args, **contour_options_dict}
        # set the widget types
        labels = list(contour_options_dict.keys())
        values = list(contour_options_dict.values())
        wtypes = [tk.Text]*len(labels)
        for i, value in enumerate(values):
            if isinstance(value, bool):
                wtypes[i] = ttk.Combobox
                values[i] = [value, not value]
        # make the actual OptionsFrame
        self.contour_options_frame = OptionsFrame(self.evaluation_options_frame,
                                            labels=labels, types=wtypes, values=values,
                                            docurl=contour_docurl)
        self.contour_options_frame.frame.grid(row=1, column=4)

        # add a button to start the evaluation
        self.evaluate_button = tk.Button(self, text='Evaluate', command=self.evaluate)
        self.evaluate_button.grid(row=2, column=0)

        # add a button to close the window
        self.close_button = tk.Button(self, text='Close', command=self.close)
        self.close_button.grid(row=2, column=1)

    def add_set(self, parent=None, default_type='Good'):
        ### add one test set
        # initializations
        if parent is None: parent = self
        row = 0
        column = len(self.test_set_widgets)+1
        idx = len(self.test_set_widgets)
        # add button for set selection
        select_button = tk.Button(parent, text='Select test set', 
                                    command=functools.partial(self.open_select_window,idx),
                                    bg='orange')
        select_button.grid(row=row, column=column)
        # add combobox for type
        type_box = ttk.Combobox(parent, values=['Good','Bad'])
        if default_type=='Bad': type_box.current(1)
        else: type_box.current(0)
        type_box['state'] = 'readonly'
        type_box.grid(row=row+1,column=column)
        # add text box for label
        label_text = tk.Text(parent, height=1, width=15)
        label_text.grid(row=row+2, column=column)
        # store the widgets
        self.test_set_widgets.append( {'button': select_button, 
                                        'selector': None,
                                        'type_box': type_box, 
                                        'label_text': label_text} )

    def open_select_window(self, idx):
        self.test_set_widgets[idx]['selector'] = SelectorWindow(self.master, self.histstruct)
        self.test_set_widgets[idx]['button']['bg'] = 'green'
        return

    def check_all_selected(self):
        if None in [el['selector'] for el in self.test_set_widgets]: return False
        else: return True

    def get_scores(self, test_set_type):
        scores = []
        for el in self.test_set_widgets:
            if el['type_box'].get()!=test_set_type: continue
            scores.append(el['selector'].get_scores())
        if len(scores)==0:
            print('WARNING: there are no test sets with label {}'.format(test_set_type))
        return scores

    def get_globalscores(self, test_set_type):
        globalscores = []
        for el in self.test_set_widgets:
            if el['type_box'].get()!=test_set_type: continue
            globalscores.append(el['selector'].get_globalscores())
        if len(globalscores)==0:
            print('WARNING: there are no test sets with label {}'.format(test_set_type))
        return globalscores

    def get_labels(self, test_set_type):
        labels = []
        for el in self.test_set_widgets:
            if el['type_box'].get()!=test_set_type: continue
            labels.append(el['label_text'].get("1.0", tk.END).strip('\n\t '))
        return labels

    def evaluate(self):
        if not self.check_all_selected():
            raise Exception('ERROR: some test sets were declared but not defined')
        # disable window for rest of processing time
        disable_frame(self)
        # load scores for good and bad test set
        scores_good_parts = self.get_scores('Good')
        scores_bad_parts = self.get_scores('Bad')
        labels_good_parts = self.get_labels('Good')
        labels_bad_parts = self.get_labels('Bad')
        globalscores_good_parts = self.get_globalscores('Good')
        globalscores_good = np.concatenate(tuple(globalscores_good_parts))
        globalscores_bad_parts = self.get_globalscores('Bad')
        globalscores_bad = np.concatenate(tuple(globalscores_bad_parts))
        labels_good = np.zeros(len(globalscores_good)) # background: label = 0
        labels_bad = np.ones(len(globalscores_bad)) # signal: label = 1

        labels = np.concatenate(tuple([labels_good,labels_bad]))
        scores = np.concatenate(tuple([-globalscores_good,-globalscores_bad]))
        scores = aeu.clip_scores( scores )

        # score distribution
        score_dist_options = self.score_dist_options_frame.get_dict()
        do_score_dist = score_dist_options.pop('make score distribution')
        if do_score_dist:
            score_dist_options['doshow'] = False # gui blocks if this is True
            pu.plot_score_dist(scores, labels, **score_dist_options)

        # roc curve
        roc_options = self.roc_options_frame.get_dict()
        do_roc = roc_options.pop('make ROC curve')
        if do_roc: 
            roc_options['doshow'] = False # gui blocks if this is True
            auc = aeu.get_roc(scores, labels, **roc_options)
        cm_options = self.cm_options_frame.get_dict()
        do_cm = cm_options.pop('make confusion matrix')

        # confusion matrix
        if do_cm: 
            working_point = aeu.get_confusion_matrix(scores, labels, **cm_options)

        # write output json
        # to do: make more flexible with user options
        json_options = self.json_options_frame.get_dict()
        do_json = json_options.pop('make json file')
        json_filename = json_options.pop('json filename')
        if do_json and len(json_filename)==0:
            print('WARNING: invalid json filename; not writing an output json')
            do_json = False
        if do_json:
            if(os.path.splitext(json_filename)[1] not in ['json','txt']):
                print('WARNING: unrecognized extension in json filename, replacing by .json')
                json_filename = os.path.splitext(json_filename)[0]+'.json'
            with open(json_filename,'w') as f:
                jsonlist = self.histstruct.get_globalscores_jsonformat(working_point=working_point)
                json.dump( jsonlist, f, indent=2 )

        # contour plots
        contour_options = self.contour_options_frame.get_dict()
        do_contour = contour_options.pop('make fit plots')
        if do_contour:
            if( not hasattr(self.histstruct,'fitfunclist')
                or not hasattr(self.histstruct,'dimslist') ):
                raise Exception('ERROR: cannot make contour plots with test data overlaid'
                        +' as they were not initialized for the training set.')
            badcolorlist = (['red','lightcoral','firebrick','chocolate',
                             'fuchsia','orange','purple'])
            goodcolorlist = ['blue']
            if len(badcolorlist)<len(scores_bad_parts):
                print('WARNING: too many bad test sets for available colors, putting all to red')
                badcolorist = ['red']*len(scores_bad_parts)
            if len(goodcolorlist)<len(scores_good_parts):
                print('WARNING: too many good test sets for available colors, putting all to blue')
                goodcolorist = ['blue']*len(scores_good_parts)

            for dims,partialfitfunc in zip(self.histstruct.dimslist,self.histstruct.fitfunclist):
                # settings for 1D plots 
                if self.contourdim == 1:
                    xaxtitle = pu.make_text_latex_safe(self.histstruct.histnames[dims])
                    yaxtitle = 'Probability density'
                    plotfunction = pu.plot_fit_1d_clusters
                # settings for 2D plots
                else:
                    xaxtitle = pu.make_text_latex_safe(self.histstruct.histnames[dims[0]])
                    yaxtitle = pu.make_text_latex_safe(self.histstruct.histnames[dims[1]])
                    plotfunction = pu.plot_fit_2d_clusters
                # define the clusters
                clusters = []
                labels = []
                colors = []
                for scores in scores_good_parts + scores_bad_parts:
                    if self.contourdim==1:
                        cluster = np.expand_dims( scores[self.histstruct.histnames[dims]], 1 )
                    else:
                        scores1 = np.expand_dims( scores[self.histstruct.histnames[dims[0]]], 1 )
                        scores2 = np.expand_dims( scores[self.histstruct.histnames[dims[1]]], 1 )
                        cluster = np.hstack( (scores1,scores2) )
                    clusters.append( cluster )
                # define the colors and labels
                for j in range(len(scores_good_parts)):
                    colors.append(goodcolorlist[j])
                    labels.append(labels_good_parts[j])
                for j in range(len(scores_bad_parts)):
                    colors.append(badcolorlist[j])
                    labels.append(labels_bad_parts[j])
                dolegend = False
                for l in labels:
                    if len(l)>0: dolegend = True
                if not dolegend: labels = None
                # make the plot
                fig,ax = plotfunction( self.histstruct.fitting_scores, 
                            clusters, labels=labels, colors=colors,
                            fitfunc=partialfitfunc, xaxtitle=xaxtitle, yaxtitle=yaxtitle,
                            **contour_options )
                # add extra text to the axes
                # (might need updates to make it more flexible)
                pu.add_text( ax, 'Density fit of lumisection scores',
                            (0.05,0.8), fontsize=12, background_alpha=0.75 )
                if self.plotstyleparser is not None:
                    pu.add_cms_label( ax, pos=(0.05,0.9),
                                      extratext=self.plotstyleparser.get_extracmstext(),
                                      fontsize=self.plotstyleparser.get_cmstextsize(),
                                      background_alpha=0.75 )
                    condtext = self.plotstyleparser.get_condtext()
                    if condtext is not None:
                        pu.add_text( ax, condtext, (0.75,1.01),
                                    fontsize=self.plotstyleparser.get_condtextsize() )
        # show all plots (but in a non-blocking way)
        plt.show(block=False)
        # enable frame again
        enable_frame(self)

    def close(self):
        ### close the window
        self.destroy()
        self.update()


class ApplyClassifiersWindow(tk.Toplevel):
    ### popup window class for evaluating the classifiers 

    def __init__(self, master, histstruct):
        super().__init__(master=master)
        self.title('Classifier evaluation')
        self.histstruct = histstruct
        self.set_selector = None
        
        # create a frame for the buttons
        self.buttons_frame = tk.Frame(self)
        self.buttons_frame.grid(row=0, column=0)
        set_frame_default_style( self.buttons_frame )
        
        # add a button to select the sets
        self.select_set_button = tk.Button(self.buttons_frame, text='Select sets',
                                            command=self.open_selection_window)
        self.select_set_button.grid(row=0, column=0)

        # add a button to start the evaluation
        self.start_evaluation_button = tk.Button(self.buttons_frame, text='Start evaluation',
                                            command=self.evaluate)
        self.start_evaluation_button.grid(row=1, column=0)

    def open_selection_window(self):
        self.set_selector = SelectorWindow(self.master, self.histstruct, 
                                mask_selection=False, post_selection=False,
                                allow_multi_set=True)

    def evaluate(self):
        extnames = self.set_selector.sets
        for extname in extnames:
            print('evaluating classifiers on set {}'.format(extname))
            for histname in self.histstruct.histnames:
                print('  now processing histogram type {}'.format(histname))
                self.histstruct.evaluate_classifier( histname, extname=extname )
        # close the window
        self.destroy()
        self.update()
        print('done')


class ApplyFitWindow(ApplyClassifiersWindow):
    ### popup window class for evaluating the fitter

    def __init__(self, master, histstruct):
        super().__init__(master, histstruct)
        self.title('Fitter evaluation')

    def evaluate(self):
        extnames = self.set_selector.sets
        for extname in extnames:
            print('evaluating fitter on set {}'.format(extname))
            scores_all = []
            for histname in self.histstruct.histnames:
                scores_all.append( self.histstruct.get_extscores( extname, histname=histname ) )
            scores_all = np.array(scores_all)
            scores_all = np.transpose(scores_all)
            self.histstruct.add_extglobalscores( extname, 
                            np.log(self.histstruct.fitfunc.pdf(scores_all)) )
        # close the window
        self.destroy()
        self.update()
        print('done')


class PlotLumisectionWindow(tk.Toplevel):
    ### popup window class for plotting a run/lumisection

    def __init__(self, master, histstruct, plotstyleparser):
        super().__init__(master)
        self.title('Lumisection plotting')
        self.histstruct = histstruct
        self.plotstyleparser = plotstyleparser
        self.inspect_set_selector = None
        self.refscore_set_selector = None
        self.ref_set_selector = None

        # create a frame for the buttons
        self.buttons_frame = tk.Frame(self)
        self.buttons_frame.grid(row=0, column=0, sticky='nsew')
        set_frame_default_style( self.buttons_frame )

        # add widgets with options
        # define options and default values
        options = {'mode':'ls', 'run': '', 'lumisection':'', 
                    'histlabel': '',
                    'recomode':'auto',
                    'recohistlabel': 'Reconstruction',
                    'refhistslabel': 'Reference histograms',
                    'refhiststransparency': 0.2,
                    'plotscore': False}
        labels = list(options.keys())
        values = list(options.values())
        wtypes = [None]*len(labels)
        # overwrite options in special cases
        for i in range(len(labels)):
            if is_bool(str(values[i])):
                wtypes[i] = ttk.Combobox
                values[i] = [values[i], not values[i]]
            if labels[i]=='mode':
                wtypes[i] = ttk.Combobox
                values[i] = ['ls','run']
            if labels[i]=='run':
                wtypes[i] = ttk.Combobox
                values[i] = self.histstruct.get_runnbs_unique()
            if labels[i]=='lumisection':
                wtypes[i] = ttk.Combobox
                values[i] = [1]
        # make the options frame
        docurl = get_docurl(self.histstruct.plot_ls)
        self.options_frame = OptionsFrame(self.buttons_frame, 
                                labels=labels, values=values, types=wtypes,
                                docurl=docurl)
        self.options_frame.frame.grid(row=0, column=0, sticky='nsew')
        # special case: set available lumisections based on chosen run number
        runidx = labels.index('run')
        lsidx = labels.index('lumisection')
        self.run_box = self.options_frame.widgets[runidx]
        self.lumisection_box = self.options_frame.widgets[lsidx]
        self.run_box.bind('<<ComboboxSelected>>', self.set_lsnbs)
        self.set_lsnbs(None)
        
        # add button to overwrite plotting style
        self.load_plotstyle_button = tk.Button(self.buttons_frame, text='Load plot style', 
                                        command=self.load_plotstyle)
        self.load_plotstyle_button.grid(row=1, column=0, sticky='nsew')

        # add button to close this window
        self.close_button = tk.Button(self.buttons_frame, text='Close', command=self.close)
        self.close_button.grid(row=1, column=1, sticky='nsew')

        # add button to make the plot
        self.plot_button = tk.Button(self.buttons_frame, text='Plot', command=self.plot)
        self.plot_button.grid(row=2, column=0, columnspan=2, sticky='nsew')

        # add widgets for selecting inspect dataset
        self.inspect_set_frame = tk.Frame(self)
        self.inspect_set_frame.grid(row=0, column=1, sticky='nsew')
        set_frame_default_style( self.inspect_set_frame )
        label = 'Select masks for plotting\n(ignored when plotting single lumisection)'
        self.inspect_set_label = tk.Label(self.inspect_set_frame, text=label)
        self.inspect_set_label.grid(row=0, column=0)
        self.inspect_set_button = tk.Button(self.inspect_set_frame, text='Select masks',
                                            command=self.open_select_inspect_set_window,
                                            bg='orange')
        self.inspect_set_button.grid(row=1, column=0)

        # add widgets for selecting reference score dataset
        self.refscore_set_frame = tk.Frame(self)
        self.refscore_set_frame.grid(row=0, column=2, sticky='nsew')
        set_frame_default_style( self.refscore_set_frame )
        label = 'Select masks for reference scores\n(ignored when not plotting score comparison)'
        self.refscore_set_label = tk.Label(self.refscore_set_frame, text=label)
        self.refscore_set_label.grid(row=0, column=0)
        self.refscore_set_button = tk.Button(self.refscore_set_frame, text='Select masks',
                                            command=self.open_select_refscore_set_window,
                                            bg='orange')
        self.refscore_set_button.grid(row=1, column=0)

        # add widgets for selecting reference histogram dataset
        self.ref_set_frame = tk.Frame(self)
        self.ref_set_frame.grid(row=0, column=3, sticky='nsew')
        set_frame_default_style( self.ref_set_frame )
        label = 'Select reference histograms\n(ignored when not plotting reference histograms)'
        self.ref_set_label = tk.Label(self.ref_set_frame, text=label)
        self.ref_set_label.grid(row=0, column=0)
        self.ref_set_button = tk.Button(self.ref_set_frame, text='Select dataset',
                                            command=self.open_select_ref_set_window,
                                            bg='orange')
        self.ref_set_button.grid(row=1, column=0)

    def set_lsnbs(self, event):
        runnb = int(self.run_box.get())
        runnbs = self.histstruct.get_runnbs()
        lsnbs = self.histstruct.get_lsnbs()
        lsnbs = lsnbs[np.nonzero(runnbs==runnb)]
        lsnbslist = [int(el) for el in lsnbs]
        self.lumisection_box.config(values=lsnbslist)

    def load_plotstyle(self):
        initialdir = get_initialdir()
        filename = fldlg.askopenfilename(initialdir=initialdir,
                    title='Load a plot style',
                    filetypes=(('json files','*.json'),('all files','*.*')))
        # if filename is invalid, return
        if len(filename)==0:
            print('Loading of plot style canceled')
            return
        # clear current plotstyleparser and related info before loading new one
        self.plotstyleparser = None
        # load a new histstruct
        self.plotstyleparser = PlotStyleParser.PlotStyleParser( filename )

    def open_select_inspect_set_window(self):
        self.inspect_set_selector = SelectorWindow(self.master, self.histstruct, 
                                                    set_selection=False, post_selection=False)
        self.inspect_set_button['bg'] = 'green'

    def open_select_refscore_set_window(self):
        self.refscore_set_selector = SelectorWindow(self.master, self.histstruct,
                                                    set_selection=False, post_selection=False)
        self.refscore_set_button['bg'] = 'green'

    def open_select_ref_set_window(self):
        self.ref_set_selector = SelectorWindow(self.master, self.histstruct)
        self.ref_set_button['bg'] = 'green'

    def get_reference_histograms(self):
        if self.ref_set_selector is None: return None
        return self.ref_set_selector.get_histograms()

    def get_inspect_masks(self):
        if self.inspect_set_selector is None: return None
        masks = self.inspect_set_selector.masks
        return masks

    def get_refscore_masks(self):
        if self.refscore_set_selector is None: return None
        masks = self.refscore_set_selector.masks
        return masks

    def plot(self):

        # get the correct options depending on the mode
        options = self.options_frame.get_dict()
        mode = options.pop('mode')
        runnb = options.pop('run')
        lsnbs = [options.pop('lumisection')]
        plotscore = options.pop('plotscore')
        options['recohist'] = options.pop('recomode')
        if mode=='ls': pass
        elif mode=='run':
            # check if masks were defined for this case
            if self.get_inspect_masks() is None:
                msg = 'WARNING: no masks were defined,'
                msg += ' will plot all lumisections in run {}...'.format(runnb)
            # get the run and lumisection numbers
            runnbs = self.histstruct.get_runnbs( masknames=self.get_inspect_masks() )
            lsnbs = self.histstruct.get_lsnbs( masknames=self.get_inspect_masks() )
            runsel = np.where(runnbs==runnb)
            lsnbs = lsnbs[runsel]
            # disable plotting scores in this case (maybe enable later)
            plotscore = False
            msg = 'WARNING: plotscore is automatically set to False for mode run!'
            print(msg)
            # print number of lumisections to plot
            print('plotting {} lumisections...'.format(len(lsnbs)))
        else: raise Exception('ERROR: option mode = {} not recognized;'.format(options['mode'])
                                +' should be either "run" or "ls".')

        # get the reference histograms
        refhists = self.get_reference_histograms()

        # get plot style options
        plotstyle_options = {}
        if self.plotstyleparser is not None:
            plotstyle_options['titledict'] = self.plotstyleparser.get_title()
            plotstyle_options['titlesize'] = self.plotstyleparser.get_titlesize()
            plotstyle_options['xaxtitledict'] = self.plotstyleparser.get_xaxtitle()
            plotstyle_options['xaxtitlesize'] = self.plotstyleparser.get_xaxtitlesize()
            plotstyle_options['yaxtitledict'] = self.plotstyleparser.get_yaxtitle()
            plotstyle_options['yaxtitlesize'] = self.plotstyleparser.get_yaxtitlesize()
            plotstyle_options['ymaxfactor'] = self.plotstyleparser.get_ymaxfactor()
            plotstyle_options['legendsize'] = self.plotstyleparser.get_legendsize()

        # run over lumisections to plot
        for i, lsnb in enumerate(lsnbs):
            # only for quick testing:
            #if i>4: 
            #    print('WARNING: plotting loop closed after 5 iterations for testing')
            #    break
            res = self.histstruct.plot_ls(runnb, lsnb, refhists=refhists,
                        **options, opaque_legend=True, **plotstyle_options )
            fig = res[0]
            axs = res[1]
            fig2d = None
            axs2d = None
            if len(res)==4:
                fig2d = res[2]
                axs2d = res[3]
            
            # post-processing of 1D figure
            # (might need updates to make it more flexible)
            if( fig is not None and axs is not None ):
                if self.plotstyleparser is not None:
                    counter = -1
                    for i in range(axs.shape[0]):
                        for j in range(axs.shape[1]):
                            counter += 1
                            histname = self.histstruct.histnames[counter]
                            ax = axs[i,j]
                            pu.add_cms_label( ax, pos=(0.05,0.9),
                                  extratext=self.plotstyleparser.get_extracmstext(),
                                  fontsize=self.plotstyleparser.get_cmstextsize() )
                            extratext = self.plotstyleparser.get_extratext(histname=histname)
                            if extratext is not None:
                                pu.add_text( ax, extratext,
                                    (0.5,0.75), fontsize=self.plotstyleparser.get_extratextsize() )
                            condtext = self.plotstyleparser.get_condtext()
                            if condtext is not None:
                                pu.add_text( ax, condtext, (0.75,1.01),
                                    fontsize=self.plotstyleparser.get_condtextsize() )
            # post-processing of 2D figure
            if( fig2d is not None and axs2d is not None ):
                pass
            plt.show(block=False)

            # get the score
            scorepoint = self.histstruct.get_scores_ls( runnb, lsnb )
            try:
                logprob = self.histstruct.get_globalscore_ls( runnb, lsnb )
            except:
                print('WARNING: could not retrieve the global score'
                        +' for run {}, lumisection {};'.format(runnb, lsnb)
                        +' was it initialized?')
                logprob = None
            print('--- Run: {}, LS: {} ---'.format(runnb, lsnb))
            print('Scores:')
            for histname in self.histstruct.histnames: 
                print('{} : {}'.format(histname,scorepoint[histname]))
            print('Log probability: '+str(logprob))

            if plotscore:
                # check if a reference set was defined
                if self.get_refscore_masks() is None:
                    msg = 'WARNING: requested to plot a reference score distribution,'
                    msg += ' but no reference set for the scores was defined;'
                    msg += ' using all lumisections in the current HistStruct.'
                    print(msg)
                # initialize the figure
                ncols = min(4,len(self.histstruct.histnames))
                nrows = int(math.ceil(len(self.histstruct.histnames)/ncols))
                fig,axs = plt.subplots(nrows,ncols,figsize=(6*ncols,6*nrows),squeeze=False)
                # loop over histogram types
                for dim,histname in enumerate(self.histstruct.histnames):
                    thisscore = scorepoint[histname]
                    refscores = self.histstruct.get_scores( histname=histname, 
                                masknames=self.get_refscore_masks() )
                    _ = pu.plot_score_ls( thisscore, refscores, fig=fig, ax=axs[int(dim/ncols),dim%ncols],
                            thislabel='This LS', 
                            reflabel='Reference LS',
                            title=pu.make_text_latex_safe(histname),
                            xaxtitle='Model output score',
                            yaxtitle='Arbitrary units',
                            doshow=False,
                            nbins=200, normalize=True )
                plt.show(block=False)

    def close(self):
        ### close the window
        self.destroy()
        self.update()


class ML4DQMGUI:

    def __init__(self, master):
        
        self.master = master
        master.title('ML4DQM GUI')

        # initializations
        self.histstruct = None
        self.histstruct_filename = None
        self.plotstyleparser = PlotStyleParser.PlotStyleParser()
        self.plotstyle_filename = None
        self.button_frames = []
        self.all_frames = []
        
        # add a frame for all action buttons
        self.action_button_frame = tk.Frame(master)
        self.action_button_frame.grid(row=0, column=0, sticky='nsew')

        # add widgets for creating a new histstruct
        self.newhs_frame = tk.Frame(self.action_button_frame)
        self.newhs_label = tk.Label(self.newhs_frame, text='HistStruct creation')
        self.newhs_label.grid(row=0, column=0, sticky='ew')
        self.newhs_button = tk.Button(self.newhs_frame, text='New', 
                                        command=self.open_new_histstruct_window)
        self.newhs_button.grid(row=1, column=0, sticky='ew')
        self.addrunmasks_button = tk.Button(self.newhs_frame, text='Add run masks',
                                        command=self.open_add_runmasks_window)
        self.addrunmasks_button.grid(row=2, column=0, sticky='ew')
        self.addstatmasks_button = tk.Button(self.newhs_frame, text='Add stat masks',
                                        command=self.open_add_statmasks_window)
        self.addstatmasks_button.grid(row=3, column=0, sticky='ew')
        self.addclassifiers_button = tk.Button(self.newhs_frame, text='Add classifiers',
                                        command=self.open_add_classifiers_window)
        self.addclassifiers_button.grid(row=4, column=0, sticky='ew')
        # add the frame to the window
        self.newhs_frame.grid(row=0, column=0, sticky='nsew')
        self.button_frames.append(self.newhs_frame)
        self.all_frames.append(self.newhs_frame)

        # add widgets for loading and saving a HistStruct
        self.iobutton_frame = tk.Frame(self.action_button_frame)
        self.iobutton_label = tk.Label(self.iobutton_frame, text='HistStruct I/O')
        self.iobutton_label.grid(row=0, column=0, sticky='ew')
        self.load_button = tk.Button(self.iobutton_frame, text='Load',
                                     command=self.load_histstruct)
        self.load_button.grid(row=1, column=0, sticky='ew')
        self.save_button = tk.Button(self.iobutton_frame, text='Save',
                                     command=self.save_histstruct)
        self.save_button.grid(row=2, column=0, sticky='ew')
        self.display_histstruct_button = tk.Button(self.iobutton_frame, 
                                                    text='Display',
                                                    command=self.open_display_histstruct_window)
        self.display_histstruct_button.grid(row=3, column=0, sticky='ew')
        self.update_histstruct_info_button = tk.Button(self.iobutton_frame, text='Refresh',
                                                command=self.update_histstruct_info)
        self.update_histstruct_info_button.grid(row=4, column=0, sticky='ew')
        # add the frame to the window
        self.iobutton_frame.grid(row=1, column=0, sticky='nsew')
        self.button_frames.append(self.iobutton_frame)
        self.all_frames.append(self.iobutton_frame)

        # add widgets for preprocessing
        self.preprocessing_frame = tk.Frame(self.action_button_frame)
        self.preprocessing_label = tk.Label(self.preprocessing_frame, text='Preprocessing')
        self.preprocessing_label.grid(row=0, column=0, sticky='ew')
        self.preprocessing_button = tk.Button(self.preprocessing_frame, text='Preprocessing',
                                            command=self.open_preprocessing_window)
        self.preprocessing_button.grid(row=1, column=0, sticky='ew')
        # add the frame to the window
        self.preprocessing_frame.grid(row=2, column=0, sticky='nsew')
        self.button_frames.append(self.preprocessing_frame)
        self.all_frames.append(self.preprocessing_frame)


        # add widgets for plotting
        self.plotbutton_frame = tk.Frame(self.action_button_frame)
        self.plotbutton_label = tk.Label(self.plotbutton_frame, text='Plotting')
        self.plotbutton_label.grid(row=0, column=0, sticky='ew')
        self.load_plotstyle_button = tk.Button(self.plotbutton_frame, 
                                            text='Load plot style',
                                            command=self.load_plotstyle)
        self.load_plotstyle_button.grid(row=1, column=0, sticky='ew')
        self.plot_sets_button = tk.Button(self.plotbutton_frame, text='Plot',
                                          command=self.open_plot_sets_window)
        self.plot_sets_button.grid(row=2, column=0, sticky='ew')
        # add the frame to the window
        self.plotbutton_frame.grid(row=3, column=0, sticky='nsew')
        self.button_frames.append(self.plotbutton_frame)
        self.all_frames.append(self.plotbutton_frame)

        # add widgets for resampling
        self.resampling_frame = tk.Frame(self.action_button_frame)
        self.resampling_label = tk.Label(self.resampling_frame, text='Resampling')
        self.resampling_label.grid(row=0, column=0, sticky='ew')
        self.resample_button = tk.Button(self.resampling_frame, text='Resample',
                                    command=self.open_resample_window)
        self.resample_button.grid(row=1, column=0, sticky='ew')
        # add the frame to the window
        self.resampling_frame.grid(row=4, column=0, sticky='nsew')
        self.button_frames.append(self.resampling_frame)
        self.all_frames.append(self.resampling_frame)

        # add widgets for classifier training, fitting and evaluation
        self.model_frame = tk.Frame(self.action_button_frame)
        self.model_label = tk.Label(self.model_frame, text='Model building')
        self.model_label.grid(row=0, column=0, sticky='ew')
        self.train_button = tk.Button(self.model_frame, text='Train classifiers',
                                      command=self.open_train_window)
        self.train_button.grid(row=1, column=0, sticky='ew')
        self.apply_classifiers_button = tk.Button(self.model_frame, 
                                        text='Evaluate classifiers',
                                        command=self.open_apply_classifiers_window)
        self.apply_classifiers_button.grid(row=2, column=0, sticky='ew')
        self.fit_button = tk.Button(self.model_frame, text='Fit', 
                                    command=self.open_fit_window)
        self.fit_button.grid(row=3, column=0, sticky='ew')
        self.apply_fit_button = tk.Button(self.model_frame,
                                            text='Evaluate fit',
                                            command=self.open_apply_fit_window)
        self.apply_fit_button.grid(row=4, column=0, sticky='ew')
        self.evaluate_button = tk.Button(self.model_frame, text='Evaluate model',
                                                command=self.open_evaluate_window)
        self.evaluate_button.grid(row=5, column=0, sticky='ew')
        self.plot_lumisection_button = tk.Button(self.model_frame, text='Plot lumisection',
                                                command=self.open_plot_lumisection_window)
        self.plot_lumisection_button.grid(row=6, column=0, sticky='ew')
        # add the frame to the window
        self.model_frame.grid(row=5, column=0, sticky='nsew')
        self.button_frames.append(self.model_frame)
        self.all_frames.append(self.model_frame)

        # add widgets for displaying text
        self.stdout_frame = tk.Frame(master)
        self.stdout_frame.grid(row=0, column=1, sticky='nsew')
        self.all_frames.append(self.stdout_frame)
        self.stdout_label = tk.Label(self.stdout_frame, text='Stdout')
        self.stdout_label.grid(row=0, column=0)
        self.messages_text = ScrolledTextFrame(self.stdout_frame, txtwidth=50, txtheight=25, 
                                                expandable=True)
        initstring = 'Welcome to the ML4DQM GUI!\n'
        self.messages_text.widget.insert(tk.INSERT, initstring)
        self.messages_text.frame.grid(row=1, column=0, sticky='nsew')

        # redirect stdout (and stderr) to text widget
        stdout = sys.stdout
        stderr = sys.stderr
        sys.stdout = StdOutRedirector( self.messages_text.widget, self.master )
        #sys.stderr = StdOutRedirector( self.messages_text.widget, self.master )

        # add widgets for displaying HistStruct info
        self.histstruct_info_frame = tk.Frame(master)
        self.histstruct_info_frame.grid(row=0, column=2, sticky='nsew')
        self.all_frames.append(self.histstruct_info_frame)
        self.histstruct_info_label = tk.Label(self.histstruct_info_frame, 
                                            text='HistStruct info')
        self.histstruct_info_label.grid(row=0, column=0, columnspan=2)
        self.histstruct_filename_label = tk.Label(self.histstruct_info_frame, text='File')
        self.histstruct_filename_label.grid(row=1, column=0, columnspan=2)
        self.histstruct_filename_text = ScrolledTextFrame(self.histstruct_info_frame,
                                            txtwidth=45, txtheight=1)
        self.histstruct_filename_text.frame.grid(row=2, column=0, columnspan=2)
        initstring = '[no histstruct loaded]'
        self.histstruct_filename_text.widget.insert(tk.INSERT, initstring)
        self.histstruct_masknames_label = tk.Label(self.histstruct_info_frame, text='Masks')
        self.histstruct_masknames_label.grid(row=3, column=0)
        self.histstruct_masknames_text = ScrolledTextFrame(self.histstruct_info_frame, 
                                            txtwidth=25, txtheight=15, expandable=True)
        self.histstruct_masknames_text.frame.grid(row=4, column=0, sticky='nsew')
        initstring = '[no histstruct loaded]'
        self.histstruct_masknames_text.widget.insert(tk.INSERT, initstring)
        self.histstruct_extnames_label = tk.Label(self.histstruct_info_frame, 
                                            text='Extended sets')
        self.histstruct_extnames_label.grid(row=3, column=1)
        self.histstruct_extnames_text = ScrolledTextFrame(self.histstruct_info_frame, 
                                            txtwidth=25, txtheight=15, expandable=True)
        self.histstruct_extnames_text.frame.grid(row=4, column=1, sticky='nsew')
        initstring = '[no histstruct loaded]'
        self.histstruct_extnames_text.widget.insert(tk.INSERT, initstring)

        # add widgets to point to documentation
        self.doc_frame = tk.Frame(master)
        self.doc_frame.grid(row=1, column=0, columnspan=3, sticky='nsew')
        self.all_frames.append(self.doc_frame)
        docurl = 'https://lukalambrecht.github.io/ML4DQM-DC/run/'
        self.docurlwidget = UrlWidget(self.doc_frame, docurl, text='Go to documentation')
        self.docurlwidget.grid(row=0, column=0, sticky='nsew')

        # apply default stylings to button frames
        for frame in self.button_frames:
            set_frame_default_style( frame, expandcolumn=0 )

        # apply default stylings to other frames
        set_frame_default_style( self.stdout_frame, expandrow=1 )
        set_frame_default_style( self.histstruct_info_frame, expandrow=4 )

    def clear_histstruct_info(self):
        ### clear all widgets displaying histstruct info
        self.histstruct_filename_text.widget.delete(1.0, tk.END)
        self.histstruct_masknames_text.widget.delete(1.0, tk.END)
        self.histstruct_extnames_text.widget.delete(1.0, tk.END)

    def update_histstruct_info(self):
        ### update all widgets displaying histstruct info
        self.clear_histstruct_info()
        filename = self.histstruct_filename if self.histstruct_filename is not None else '[None]'
        self.histstruct_filename_text.widget.insert(tk.INSERT, filename)
        masknames = self.histstruct.get_masknames()
        if len(masknames)>0: 
            self.histstruct_masknames_text.widget.insert(tk.INSERT, '\n'.join(masknames))
        else: self.histstruct_masknames_text.widget.insert(tk.INSERT, '[no masks available]')
        extnames = self.histstruct.exthistograms.keys()
        if len(extnames)>0: 
            self.histstruct_extnames_text.widget.insert(tk.INSERT, '\n'.join(extnames))
        else: self.histstruct_extnames_text.widget.insert(tk.INSERT, '[no sets available]')

    def load_histstruct(self):
        ### load a histstruct from a stored zip file
        initialdir = get_initialdir()
        filename = fldlg.askopenfilename(initialdir=initialdir,
                    title='Load a HistStruct',
                    filetypes=(('zip files','*.zip'),('all files','*.*')))
        # if filename is invalid, return
        if len(filename)==0: 
            print('Loading of HistStruct canceled')
            return
        # clear current histstruct and related info before loading new one
        self.histstruct = None
        self.clear_histstruct_info()
        # load a new histstruct
        self.histstruct = HistStruct.HistStruct.load( filename, verbose=True )
        self.histstruct_filename = filename
        # fill related widgets
        self.update_histstruct_info()

    def save_histstruct(self):
        ### save a histstruct to a zip file
        if not self.histstruct:
            print('ERROR: need to load a HistStruct first')
            return
        initialdir = get_initialdir()
        filename = fldlg.asksaveasfilename(initialdir=initialdir,
                    title='Save a HistStruct',
                    filetypes=(('zip files','*.zip'),('all files','*.*')))
        # if filename is invalid, return
        if len(filename)==0:
            print('Saving of HistStruct canceled')
            return
        # save the histstruct
        self.histstruct.save( filename )

    def load_plotstyle(self):
        ### load a plot style from a json file
        initialdir = get_initialdir()
        filename = fldlg.askopenfilename(initialdir=initialdir,
                    title='Load a plot style',
                    filetypes=(('json files','*.json'),('all files','*.*')))
        # if filename is invalid, return
        if len(filename)==0:
            print('Loading of plot style canceled')
            return
        # clear current plotstyleparser and related info before loading new one
        self.plotstyleparser = None
        # load a new histstruct
        self.plotstyleparser = PlotStyleParser.PlotStyleParser( filename )
        self.plotstyle_filename = filename

    def open_new_histstruct_window(self):
        self.histstruct = HistStruct.HistStruct()
        _ = NewHistStructWindow(self.master, self.histstruct)
        return

    def open_add_runmasks_window(self):
        if not self.histstruct:
            print('ERROR: need to load a HistStruct first')
            return
        _ = AddRunMasksWindow(self.master, self.histstruct)
        return

    def open_add_statmasks_window(self):
        if not self.histstruct:
            print('ERROR: need to load a HistStruct first')
            return
        _ = AddStatMasksWindow(self.master, self.histstruct)
        return

    def open_add_classifiers_window(self):
        if not self.histstruct:
            print('ERROR: need to load a HistStruct first')
            return
        _ = AddClassifiersWindow(self.master, self.histstruct)
        return

    def open_preprocessing_window(self):
        if not self.histstruct:
            print('ERROR: need to load a HistStruct first')
            return
        _ = PreProcessingWindow(self.master, self.histstruct)
        return

    def open_plot_sets_window(self):
        if not self.histstruct:
            print('ERROR: need to load a HistStruct first')
            return
        _ = PlotSetsWindow(self.master, self.histstruct, self.plotstyleparser)
        return

    def open_train_window(self):
        if not self.histstruct:
            print('ERROR: need to load a HistStruct first')
            return
        _ = TrainClassifiersWindow(self.master, self.histstruct)
        return

    def open_fit_window(self):
        if not self.histstruct:
            print('ERROR: need to load a HistStruct first')
            return
        _ = FitWindow(self.master, self.histstruct, self.plotstyleparser)
        return

    def open_resample_window(self):
        if not self.histstruct:
            print('ERROR: need to load a HistStruct first')
            return
        _ = ResampleWindow(self.master, self.histstruct)
        return

    def open_evaluate_window(self):
        if not self.histstruct:
            print('ERROR: need to load a HistStruct first')
            return
        _ = EvaluateWindow(self.master, self.histstruct, self.plotstyleparser)
        return

    def open_display_histstruct_window(self):
        if not self.histstruct:
            print('ERROR: need to load a HistStruct first')
            return
        _ = DisplayHistStructWindow(self.master, self.histstruct)
        return

    def open_apply_classifiers_window(self):
        if not self.histstruct:
            print('ERROR: need to load a HistStruct first')
            return
        _ = ApplyClassifiersWindow(self.master, self.histstruct)
        return

    def open_apply_fit_window(self):
        if not self.histstruct:
            print('ERROR: need to load a HistStruct first')
            return
        _ = ApplyFitWindow(self.master, self.histstruct)
        return

    def open_plot_lumisection_window(self):
        if not self.histstruct:
            print('ERROR: need to load a HistStruct first')
            return
        _ = PlotLumisectionWindow(self.master, self.histstruct, self.plotstyleparser)
        return


if __name__=='__main__':

    window = tk.Tk()
    gui = ML4DQMGUI(window)
    window.mainloop()
