# to do:
# - see to do's in the code
# - continue making styling more uniform (e.g. everything in a Frame)

# external modules

print('importing external modules...')
print('  import os'); import os
print('  import sys'); import sys
print('  import math'); import math
print('  import pandas as pd'); import pandas as pd
print('  import numpy as np'); import numpy as np
print('  import matplotlib.pyplot as plt'); import matplotlib.pyplot as plt
print('  import pickle'); import pickle
print('  import functools'); import functools
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
sys.path.append('../utils')
print('  import csv_utils as csvu'); import csv_utils as csvu
print('  import json_utils as jsonu'); import json_utils as jsonu
print('  import dataframe_utils as dfu'); import dataframe_utils as dfu
print('  import hist_utils as hu'); import hist_utils as hu
print('  import autoencoder_utils as aeu'); import autoencoder_utils as aeu
print('  import plot_utils as pu'); import plot_utils as pu
print('  import generate_data_utils as gdu'); import generate_data_utils as gdu
print('  import refruns_utils as rru'); import refruns_utils as rru
print('  import mask_utils as mu'); import mask_utils as mu

print('importing src...')
sys.path.append('../src')
sys.path.append('../src/classifiers')
sys.path.append('../src/cloudfitters')
print('  import HistStruct'); import HistStruct
print('  import HistogramClassifier'); import HistogramClassifier
print('  import AutoEncoder'); import AutoEncoder
print('  import MaxPullClassifier'); import MaxPullClassifier
print('  import NMFClassifier'); import NMFClassifier
print('  import PCAClassifier'); import PCAClassifier
print('  import TemplateBasedClassifier'); import TemplateBasedClassifier
print('  import SeminormalFitter'); import SeminormalFitter
print('  import GaussianKdeFitter'); import GaussianKdeFitter
print('  import HyperRectangleFitter'); import HyperRectangleFitter

print('done')


### styling

def set_frame_default_style( frame ):
    ### apply some default stylings to a tk.Frame
    frame['padx'] = 10
    frame['pady'] = 10
    frame['borderwidth'] = 2
    frame['relief'] = 'groove'


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
    allowed = ['None','upsample_hist_set']
    if key is None: return allowed

    # map key to function
    key = key.strip(' \t\n')
    if key=='None': return (None,{})
    if key=='upsample_hist_set': f = gdu.upsample_hist_set
    elif key=='dummy1': pass
    elif key=='dummy2': pass
    else:
        raise Exception('ERROR: resampling function {} not recognized.'.format(key))
    return (f,get_args_dict(f))

def get_fitter_class( key=None ):
    ### get a fitter class from its name (or other key)
    # note: a dict of keyword initialization arguments with default values is returned as well!
    # input arguments:
    # - key: string representing name or key of fitter class
    
    allowed = ['GaussianKdeFitter', 'SeminormalFitter']
    if key is None: return allowed

    key = key.strip(' \t\n')
    if key=='GaussianKdeFitter': c = GaussianKdeFitter.GaussianKdeFitter
    elif key=='SeminormalFitter': c = SeminormalFitter.SeminormalFitter
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

def get_training_options( histstruct ):
    ### get options for training classifiers
    # note: the classifiers are assumed to be already present in the histstruct
    # note: different types of classifiers for different types of histograms
    #       are not supported for now... need to do.
    # to do: deal with special cases of non-trival arguments, e.g.
    #        - reference histogram for MaxPullClassifier
    #           (can work with usual mask but not practical for one histogram, 
    #            maybe add option in classifier to average 'training' histograms)
    ctype = None
    classifier = None
    for histname in histstruct.histnames:
        if histname not in histstruct.classifiers.keys():
            print('WARNING: the histstruct seems not to contain a classifier'
                    +' for {}'.format(histname))
            continue
        classifier = histstruct.classifiers[histname]
        if( ctype is None ): ctype = type(classifier)
        if( type(classifier)!=ctype ):
            raise Exception('ERROR: the histstruct seems to contain different types of classifiers'
                    +' for different types of histograms.'
                    +' This is currently not yet supported in the GUI.')
    if classifier is None:
        print('WARNING: could not retrieve options for classifier training.'
                +' (Have any classifiers been initialized?)')
        return {}
    return get_args_dict(classifier.train)

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
        initialdir = os.path.abspath(os.path.dirname(__file__))
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

class OptionsFrame:
    ### contains a tk.Frame holding a list of customization options

    def __init__(self, master, labels=None, types=None, values=None):
        # input arguments:
        # - labels: list of strings with the names/labels of the options
        # - types: list of tk types, defaults to tk.Text for each option
        # - values: list of options passed to each widget
        #           (for now only tk.Text, in wich case values is a list of default values,
        #            but can be extended to e.g. combobox, 
        #            where values would be the options to choose from)
        # note: individual elements of types and values can also be None,
        #       in which case these elements will be set to default
        # to do: extend to other widget types than tk.Text (especially combobox)
        self.frame = tk.Frame(master,width=200)
        self.labels = []
        self.wtypes = []
        self.widgets = []
        self.set_options( labels=labels, types=types, values=values)

    def set_options(self, labels=None, types=None, values=None):
        ### set the options of an option frame
        # serves both as initializer and as resetter (? to be tested!)

        # check arguments
        if labels is None: 
            raise Exception('ERROR in OptionsFrame initialization:'
                            +' argument "labels" must be specified.')
        if types is None: types = [tk.Text]*len(labels)
        if values is None: values = [None]*len(labels)
        if( len(types)!=len(labels) or len(values)!=len(labels) ):
            raise Exception('ERROR in OptionsFrame initialization:'
                            +' argument lists have unequal lengths.')
            # (to extend error checking)

        # clear current OptionsFrame
        self.labels.clear()
        self.wtypes.clear()
        self.widgets.clear()
        for widget in self.frame.winfo_children(): widget.destroy()

        # set widgets
        for i, (label, wtype, value) in enumerate(zip(labels, types, values)):
            tklabel = tk.Label(self.frame, text=label)
            tklabel.grid(row=i, column=0)
            self.labels.append(tklabel)
            if wtype is None: wtype = tk.Text
            widget = None
            if wtype is tk.Text:
                widget = tk.Text(self.frame, height=1, width=25)
                if value is not None:
                    widget.insert(tk.INSERT, value)
            elif wtype is GenericFileLoader:
                widget = GenericFileLoader(self.frame)
            else:
                raise Exception('ERROR in OptionsFrame initialization:'
                                +' widget type {} not recognized'.format(wtype))
            widget.grid(row=i, column=1)
            self.widgets.append(widget)
            self.wtypes.append(wtype)

    def get_dict(self):
        ### get the options of the current OptionsFrame as a dictionary
        res = {}
        for label, wtype, widget in zip(self.labels, self.wtypes, self.widgets):
            key = label.cget('text')
            value = None
            if wtype is tk.Text:
                value = widget.get('1.0', tk.END)
            elif wtype is GenericFileLoader:
                value = widget.get_filename()
            else:
                raise Exception('ERROR in OptionsFrame get_dict:'
                               +' no getter method implemented for widget type {}'.format(wtype))
            value = value.strip(' \t\n')
            if is_int(value): value = int(value)
            elif is_float(value): value = float(value)
            elif is_bool(value): value = to_bool(value)
            elif value=='None': value = None
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
                    adaptable_size=False, showscrollbars=False):
        # note: if adaptable_size is True, the Frame will take its size from the child widget
        self.frame = tk.Frame(master, height=height, width=width)
        self.showscrollbars = showscrollbars
        if not adaptable_size: self.frame.grid_propagate(0)

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

    def __init__(self, master, txtheight=50, txtwidth=50, showscrollbars=False):
        super().__init__(master, adaptable_size=True, showscrollbars=showscrollbars)
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
        self.run_mask_widgets = []
        self.highstat_mask_widgets = []
        self.json_mask_widgets = []

        # create a frame for general options
        self.general_options_frame = tk.Frame(self)
        set_frame_default_style( self.general_options_frame )
        self.general_options_frame.grid(row=0, column=0, sticky='nsew')
        self.general_options_label = tk.Label(self.general_options_frame, 
                                              text='General options')
        self.general_options_label.grid(row=0, column=0)

        # add general options
        self.training_mode_box = ttk.Combobox(self.general_options_frame, 
                                    values=['global','local'])
        self.training_mode_box.current(0)
        self.training_mode_box.grid(row=1,column=0)
        self.year_box = ttk.Combobox(self.general_options_frame,
                                     values=['2017'])
        self.year_box.current(0)
        self.year_box.grid(row=2,column=0)
        self.histnames_listbox = tk.Listbox(self.general_options_frame, 
                                            selectmode='multiple',
                                            exportselection=False)
        for histname in (['chargeInner_PXLayer_2',
                          'chargeInner_PXLayer_3',
                          'charge_PXDisk_+1','charge_PXDisk_+2','charge_PXDisk_+3',
                          'size_PXLayer_1','size_PXLayer_2',
                          'size_PXLayer_3']):
            # to do: extract the options automatically (e.g. from the available files)
            self.histnames_listbox.insert(tk.END, histname)
        self.histnames_listbox.grid(row=3, column=0)

        # create a frame for local options
        self.local_options_frame = tk.Frame(self)
        set_frame_default_style(self.local_options_frame)
        self.local_options_frame.grid(row=1, column=0)
        self.local_options_label = tk.Label(self.local_options_frame, 
                                    text='Options for local training')
        self.local_options_label.grid(row=0, column=0)

        # add local options
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
        self.run_mask_frame.grid(row=0, column=1, sticky='nsew')

        # add buttons for run mask addition
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
        self.highstat_mask_frame.grid(row=0, column=2, sticky='nsew')

        # add buttons for highstat mask additions
        self.add_highstat_mask_button = tk.Button(self.highstat_mask_frame, 
                        text='Add a statistics mask',
                        command=functools.partial(self.add_highstat_mask,self.highstat_mask_frame))
        self.add_highstat_mask_button.grid(row=0, column=0, columnspan=2)
        name_label = tk.Label(self.highstat_mask_frame, text='Name:')
        name_label.grid(row=1, column=0)
        threshold_label = tk.Label(self.highstat_mask_frame, text='Threshold:')
        threshold_label.grid(row=1, column=1)

        # create a frame for json mask addition
        self.json_mask_frame = tk.Frame(self)
        set_frame_default_style( self.json_mask_frame )
        self.json_mask_frame.grid(row=0, column=3, sticky='nsew')

        # add buttons for highstat mask additions
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
        self.make_histstruct_button.grid(row=2, column=0)

    def get_target_run(self):
        target_run_text = self.target_run_text.get(1.0, tk.END).strip(' \t\n')
        if len(target_run_text)==0: return None
        return int(target_run_text)

    def get_local_training_runs(self, filename):
        target_run = self.get_target_run()
        ntraining = int(self.ntraining_text.get(1.0, tk.END).strip(' \t\n'))
        offset = int(self.offset_text.get(1.0, tk.END).strip(' \t\n'))
        #try:
        runs = dfu.get_runs( dfu.select_dcson( csvu.read_csv( filename ) ) )
        #except:
        #    print('WARNING in get_local_training_runs: could not filter runs by DCS-on.'
        #            +' Check access to the DCS-on json file.')
        #    runs = dfu.get_runs( csvu.read_csv( filename ) )
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
        if is_local:
            if filename is None:
                raise Exception('ERROR: a filename must be specified'
                                +' for getting needed runs for local training.')
            if not os.path.exists(filename):
                raise Exception('ERROR: the file {} does not seem to exist.'.format(filename))
            target_run = self.get_target_run()
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
        name_text = tk.Text(parent, height=1, width=25)
        name_text.grid(row=row, column=column)
        threshold_text = tk.Text(parent, height=1, width=8)
        threshold_text.grid(row=row, column=column+1)
        self.highstat_mask_widgets.append({'name_text':name_text,'threshold_text':threshold_text})

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
            threshold = float(el['threshold_text'].get(1.0, tk.END).strip(' \t\n'))
            highstat_masks[name] = threshold
        return highstat_masks

    def get_json_masks(self):
        json_masks = {}
        for el in self.json_mask_widgets:
            name = el['name_text'].get(1.0, tk.END).strip(' \t\n')
            filename = el['file_loader'].get_filename()
            json_masks[name] = filename
        return json_masks

    def get_histnames(self):
        histnames = ([self.histnames_listbox.get(idx)
                    for idx in self.histnames_listbox.curselection() ])
        return histnames

    def make_histstruct(self):

        # get general settings
        histnames = self.get_histnames()
        year = self.year_box.get()
        training_mode = self.training_mode_box.get()
        print('creating a new HistStruct...')
        print('found following settings:')
        print('  - histogram names: {}'.format(histnames))
        print('  - year: {}'.format(year))
        print('  - training mode: {}'.format(training_mode))
        print('finding all available runs...')
        # get a comprehensive set of all explicitly needed runs (for throwing away the rest)
        firstfilename = '../data/DF'+year+'_'+histnames[0]+'.csv'
        needed_runs = self.get_needed_runs( is_local=(training_mode=='local'), 
                                            filename=firstfilename )
        # loop over the histogram types to take into account
        for histname in histnames:
            print('adding {}...'.format(histname))
            # read the histograms from the csv file
            filename = '../data/DF'+year+'_'+histname+'.csv'
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
            print('adding masks for target run and local training runs...')
            json = {str(self.get_target_run()): [[-1]]}
            self.histstruct.add_json_mask( 'target_run', json )
            json = {str(run): [[-1]] for run in self.get_local_training_runs(firstfilename)}
            self.histstruct.add_json_mask( 'local_training', json )

        # add high statistics mask(s)
        highstat_masks = self.get_highstat_masks()
        for name, threshold in highstat_masks.items():
            print('adding mask "{}"'.format(name))
            self.histstruct.add_highstat_mask( name, entries_to_bins_ratio=threshold )

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
            classifier_type_box.bind('<<ComboboxSelected>>', functools.partial(
                self.set_classifier_options, histname=histname) )
            classifier_type_box.grid(row=1, column=1)
            key_label = tk.Label(frame, text='Parameters')
            key_label.grid(row=2, column=0)
            value_label = tk.Label(frame, text='Values')
            value_label.grid(row=2, column=1)
            classifier_options_frame = OptionsFrame(frame, labels=[], values=[])
            classifier_options_frame.frame.grid(row=3, column=0, columnspan=2)

            self.classifier_widgets[histname] = {'type':classifier_type_box, 
                                                 'options':classifier_options_frame}
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
        # now set the options
        self.classifier_widgets[histname]['options'].set_options(
                labels=coptions.keys(), types=optiontypes, values=coptions.values())

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
        # close the window
        self.destroy()
        self.update()
        print('done')


# old version (do not use anymore but keep for reference):
'''class AddClassifiersWindow(tk.Toplevel):
    ### popup window class for adding classifiers to a histstruct
    # note: preliminary version, where the user just inputs python code in a text widget.
    #       probably not 'safe', to check later.
    
    def __init__(self, master, histstruct):
        super().__init__(master=master)
        self.title('Add classifiers')
        self.histstruct = histstruct
        
        # create a frame for the code
        self.code_frame = tk.Frame(self)
        set_frame_default_style( self.code_frame )
        self.code_frame.grid(row=0, column=0, sticky='nsew')
        label_text = 'Write code for adding classifiers to the HistStruct'
        tk.Label(self.code_frame, text=label_text).grid(row=0, column=0)
        info_text = 'Tips:\n'
        info_text += '\t- The current HistStruct object is available as "histstruct".\n'
        info_text += '\t- See the documentation for src/HistStruct on how to add classifiers.\n'
        info_text += '\t- See the documentation for src/classifiers on the available classifers.\n'
        tk.Label(self.code_frame, text=info_text, justify=tk.LEFT).grid(row=1, column=0)
        tk.Label(self.code_frame, text='Example:').grid(row=2, column=0)
        example_text = 'for histname in histstruct.histnames\n:'
        example_text += '\tinput_size = histstruct.get_histograms( histname=histname ).shape[1]\n'
        example_text += '\tarch = [int(input_size/2.)]\n'
        example_text += '\tmodel = aeu.getautoencoder(input_size, arch)\n'
        example_text += '\tclassifier = AutoEncoder.AutoEncoder( model=model )\n'
        example_text += '\thiststruct.add_classifier(histname, classifier)\n'
        tk.Label(self.code_frame, text=example_text, justify=tk.LEFT).grid(row=3, column=0)

        # add a text widget for the code
        self.code_text = ScrolledTextFrame(self.code_frame, txtheight=15, txtwidth=50)
        self.code_text.frame.grid(row=4, column=0)

        # add a button to evaluate the code
        self.run_code_button = tk.Button(self, text='Run code', command=self.run_code)
        self.run_code_button.grid(row=2, column=0)

    def get_code_text(self):
        codestr = self.code_text.widget.get(1.0, tk.END)
        return codestr

    def run_code(self):
        codestr = self.get_code_text()
        codestr = codestr.replace('histstruct','self.histstruct')
        if( '__' in codestr
            or 'system' in codestr):
            print('WARNING: code not executed as it contains suspicious content.')
            return
        exec(codestr)
        # close the window
        self.destroy()
        self.update()
        print('done')'''


class PlotSetsWindow(tk.Toplevel):
    ### popup window class for plotting the histograms in a histstruct

    def __init__(self, master, histstruct):
        super().__init__(master=master)
        self.title('Plotting')
        self.histstruct = histstruct
        self.set_optionsframe_list = []
        self.select_set_button_list = []
        self.set_selector_list = []

        # make a frame holding the action buttons
        self.buttons_frame = tk.Frame(self)
        self.buttons_frame.grid(row=0, column=0, sticky='nsew')
        set_frame_default_style(self.buttons_frame)

        # add a button to allow adding more sets of options
        self.more_button = tk.Button(self.buttons_frame, text='Add...', command=self.add_set)
        self.more_button.grid(row=0, column=0)

        # add a button to make the plot
        self.plot_button = tk.Button(self.buttons_frame, text='Plot', command=self.make_plot)
        self.plot_button.grid(row=0, column=1)

        # add one set of options
        self.add_set()

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
                                                        only_mask_selection=True)
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
        optionsdict = {'masknames':[], 'labellist':[], 'colorlist':[]}
        for optionsframe, setselector in zip(self.set_optionsframe_list,self.set_selector_list):
            masks = setselector.masks
            optionsdict['masknames'].append( masks )
            setoptions = optionsframe.get_dict()
            optionsdict['labellist'].append( setoptions['label'] )
            optionsdict['colorlist'].append( setoptions['color'] )
        print('found following plotting options:')
        print(optionsdict)
        print('making plot...')
        fig,axs = self.histstruct.plot_histograms( **optionsdict )
        # save or plot the figure
        #fig.savefig('test.png')
        plt.show(block=False)
        # close the window
        self.destroy()
        self.update()
        print('done')

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

    def __init__(self, master, histstruct, only_mask_selection=False,
                                            only_set_selection=False,
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

        # add widgets for choosing masks
        self.histstruct_masks_label = tk.Label(self, text='Choose masks')
        if not only_set_selection: self.histstruct_masks_label.grid(row=0,column=0)
        mask_selectmode = 'multiple' if allow_multi_mask else 'single'
        self.histstruct_masks_listbox = tk.Listbox(self, selectmode=mask_selectmode,
                                                            exportselection=False)
        for maskname in self.histstruct.get_masknames():
            self.histstruct_masks_listbox.insert(tk.END, maskname)
        if len(self.histstruct.get_masknames())==0:
            self.histstruct_masks_listbox.insert(tk.END, '[no masks available]')
        if not only_set_selection: self.histstruct_masks_listbox.grid(row=1,column=0)
        
        # add widgets for choosing a (resampled) set directly
        self.histstruct_sets_label = tk.Label(self, text='Choose sets')
        if not only_mask_selection: self.histstruct_sets_label.grid(row=2,column=0)
        set_selectmode = 'multiple' if allow_multi_set else 'single'
        self.histstruct_sets_listbox = tk.Listbox(self, selectmode=set_selectmode,
                                                        exportselection=False)
        for extname in self.histstruct.exthistograms.keys():
            self.histstruct_sets_listbox.insert(tk.END, extname)
        if len(self.histstruct.exthistograms.keys())==0:
            self.histstruct_sets_listbox.insert(tk.END, '[no sets available]')
        if not only_mask_selection: self.histstruct_sets_listbox.grid(row=3,column=0)

        # add widget for selection
        self.select_button = tk.Button(self, text='Select', command=self.select_histograms)
        self.select_button.grid(row=4, column=0)

    def get_masks(self):
        ### get currently selected masks
        # warning: do not use after selection window has been closed.
        masks = ([self.histstruct_masks_listbox.get(idx)
                    for idx in self.histstruct_masks_listbox.curselection() ])
        return masks

    def get_sets(self):
        ### get currently selected sets
        # warning: do not use after selection window has been closed.
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
        masks = self.get_masks()
        do_masks = bool(len(masks)>0)
        sets = self.get_sets()
        do_sets = bool(len(sets)>0)
        if( not do_masks and not do_sets ):
            raise Exception('ERROR: you must select either at least one mask or a training set.')
        if( do_masks and do_sets ):
            raise Exception('ERROR: you cannot select both masks and sets.')
        if do_masks:
            self.histograms = self.histstruct.get_histograms(masknames=masks)
            self.masks = masks
            if len(self.histstruct.scores.keys())>0:
                self.scores = self.histstruct.get_scores(masknames=masks)
            if len(self.histstruct.globalscores)>0:
                self.globalscores = self.histstruct.get_globalscores(masknames=masks)
        else:
            extname = sets[0]
            self.histograms = self.histstruct.get_exthistograms(extname)
            self.sets = sets
            if( extname in self.histstruct.extscores.keys()
                and len(self.histstruct.extscores[extname].keys())>0 ):
                self.scores = self.histstruct.get_extscores(extname)
            if extname in self.histstruct.extglobalscores.keys():
                self.globalscores = self.histstruct.get_extglobalscores(extname)
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

        # create frame for options
        self.train_options_frame = tk.Frame(self)
        self.train_options_frame.grid(row=0, column=0)
        set_frame_default_style( self.train_options_frame )
        self.train_options_label = tk.Label(self.train_options_frame, text='Training settings')
        self.train_options_label.grid(row=0, column=0, columnspan=2)

        # add widget to select histograms
        self.select_train_button = tk.Button(self.train_options_frame, 
                                            text='Select training set',
                                            command=self.open_training_selection_window,
                                            bg='orange')
        self.select_train_button.grid(row=1, column=0, columnspan=2)

        # add widgets for training options
        self.key_label = tk.Label(self.train_options_frame, text='Parameters')
        self.key_label.grid(row=2, column=0)
        self.value_label = tk.Label(self.train_options_frame, text='Values')
        self.value_label.grid(row=2, column=1)
        options = get_training_options( self.histstruct )
        self.options_frame = OptionsFrame(self.train_options_frame, 
                labels=options.keys(),values=options.values())
        self.options_frame.frame.grid(row=3, column=0, columnspan=2)

        # add button to start training
        self.train_button = tk.Button(self, text='Start training', command=self.do_training)
        self.train_button.grid(row=4, column=0, columnspan=2)

    def open_training_selection_window(self):
        self.training_set_selector = SelectorWindow(self.master, self.histstruct)
        self.select_train_button['bg'] = 'green'
        return

    def do_training(self):
        training_options = self.options_frame.get_dict()
        if self.training_set_selector is None:
            raise Exception('ERROR: please select a training set before starting training.')
        training_histograms = self.training_set_selector.get_histograms()
        for histname in training_histograms.keys():
            # check if a classifier is initialized for this histogram type
            if histname not in self.histstruct.classifiers.keys():
                print('WARNING: no classifier was found in the HistStruct'
                        +' for histogram type {}; skipping.'.format(histname))
                continue
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

    def __init__(self, master, histstruct):
        super().__init__(master=master)
        self.title('Fitting')
        self.histstruct = histstruct
        self.fitting_set_selector = None

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
        plot_options_dict = get_args_dict(pu.plot_fit_2d)
        meta_args = {'do_plot':'True'}
        plot_options_dict = {**meta_args, **plot_options_dict}
        # do special overridings
        for key in ['fitfunc','xaxtitle','yaxtitle']:
            if key in list(plot_options_dict.keys()):
                plot_options_dict.pop(key)

        self.plot_options = OptionsFrame(self.plot_options_frame,
                                            labels=plot_options_dict.keys(),
                                            values=plot_options_dict.values())
        self.plot_options.frame.grid(row=1, column=0)

        # add widgets to start the fit
        self.fit_button = tk.Button(self, text='Start fit', command=self.do_fit)
        self.fit_button.grid(row=2, column=0, columnspan=2)

    def set_fitter_options(self, event):
        fitter_name = self.fitter_box.get()
        (_, coptions) = get_fitter_class(fitter_name)
        self.fitter_options_frame.set_options(labels=coptions.keys(), values=coptions.values())

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
        fitting_scores = self.get_fitting_scores()
        (fitter,fitter_options) = self.get_fitter()
        plot_options_dict = self.plot_options.get_dict()
        do_plot = plot_options_dict.pop('do_plot')
        if do_plot:
            dimslist = []
            fitfunclist = []
            nhisttypes = len(self.histstruct.histnames)
            for i in range(0,nhisttypes-1):
                for j in range(i+1,nhisttypes):
                    dimslist.append((i,j))
            plt.close('all')
            for dims in dimslist:
                thismse = fitting_scores[:,dims]
                fitfunc = fitter( thismse, **fitter_options )
                (fig,ax) = pu.plot_fit_2d(thismse, fitfunc=fitfunc,
                                    xaxtitle=self.histstruct.histnames[dims[0]],
                                    yaxtitle=self.histstruct.histnames[dims[1]],
                                    **plot_options_dict)
                fitfunclist.append(fitfunc)
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
        # close the window
        self.destroy()
        self.update()
        print('done')


class ResampleWindow(tk.Toplevel):
    ### popup window class for resampling testing and training sets

    def __init__(self, master, histstruct):
        super().__init__(master=master)
        self.title('Resampling window')
        self.histstruct = histstruct
        self.set_widget_list = []
        self.set_selector_list = []

        # create a frame for the buttons
        self.buttons_frame = tk.Frame(self)
        self.buttons_frame.grid(row=0, column=0, sticky='nsew')
        set_frame_default_style( self.buttons_frame )
        
        # add widget to add a set
        self.add_button = tk.Button(self.buttons_frame, text='Add', command=self.add_set)
        self.add_button.grid(row=1, column=0)

        # add widgets to start resampling
        self.resample_button = tk.Button(self.buttons_frame, text='Start resampling',
                                        command=self.do_resampling)
        self.resample_button.grid(row=2,column=0)

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

        # add one set
        self.add_set()

    def update_sets_list(self):
        self.sets_listbox.delete(0, tk.END)
        extnames = self.histstruct.exthistograms.keys()
        for extname in extnames:
            self.sets_listbox.insert(tk.END, extname)
        if len(extnames)==0:
            self.sets_listbox.insert(tk.END, '[no sets available]')

    def add_set(self):
        ### add widgets for one more histogram set to plot
        column = len(self.set_widget_list)+2
        row = 0
        idx = len(self.set_widget_list)
        
        # create a frame to hold the widgets
        set_frame = tk.Frame(self)
        set_frame.grid(row=row, column=column)
        set_frame_default_style( set_frame )

        # add widgets for choosing resampling basis set
        select_button = tk.Button(set_frame, text='Select set',
                                    command=functools.partial(self.open_select_window,idx),
                                    bg='orange')
        select_button.grid(row=0, column=0, columnspan=2)

        # add widgets for resampling options
        name_label = tk.Label(set_frame, text='Name')
        name_label.grid(row=1, column=0)
        name_text = tk.Text(set_frame, height=1, width=15)
        name_text.grid(row=1, column=1)
        partitions_label = tk.Label(set_frame, text='Partitions')
        partitions_label.grid(row=2, column=0)
        partitions_text = tk.Text(set_frame, height=1, width=15)
        partitions_text.insert(tk.INSERT, '-1')
        partitions_text.grid(row=2, column=1)
        function_label = tk.Label(set_frame, text='Function')
        function_label.grid(row=3, column=0)
        function_box = ttk.Combobox(set_frame, values=get_resampling_function())
        function_box.current(0)
        function_box.bind('<<ComboboxSelected>>', functools.partial(
            self.set_function_options, setindex=idx))
        function_box.grid(row=3, column=1)
        key_label = tk.Label(set_frame, text='Parameters')
        key_label.grid(row=4, column=0)
        value_label = tk.Label(set_frame, text='Values')
        value_label.grid(row=4, column=1)
        function_options_frame = OptionsFrame(set_frame, labels=[], values=[])
        function_options_frame.frame.grid(row=5, column=0, columnspan=2)

        self.set_widget_list.append({'select_button':select_button,
                                    'name':name_text,'partitions':partitions_text,
                                    'function':function_box,
                                    'function_options':function_options_frame})
        self.set_function_options(None, idx)
        self.set_selector_list.append( None )

    def set_function_options(self, event, setindex):
        fname = self.get_function_name(setindex)
        (_, foptions) = get_resampling_function(key=fname)
        self.set_widget_list[setindex]['function_options'].set_options( 
            labels=foptions.keys(), values=foptions.values())

    def open_select_window(self, idx):
        self.set_selector_list[idx] = SelectorWindow(self.master, self.histstruct)
        self.set_widget_list[idx]['select_button']['bg'] = 'green'
        return

    def check_all_selected(self):
        if None in self.test_set_selector_list: return False
        else: return True

    def get_name(self, setindex):
        text = self.set_widget_list[setindex]['name']
        nstr = text.get(1.0,tk.END)
        name = nstr.strip(' \t\n')
        return name 

    def get_partitions(self, setindex):
        ### get currently entered partitions
        text = self.set_widget_list[setindex]['partitions']
        pstr = text.get(1.0,tk.END)
        return int(pstr)

    def get_function_name(self, setindex):
        widgets = self.set_widget_list[setindex]
        function_name = widgets['function'].get()
        return function_name

    def get_function(self, setindex):
        fname = self.get_function_name(setindex)
        (function, _) = get_resampling_function(fname)
        function_options = self.set_widget_list[setindex]['function_options'].get_dict()
        return (function, function_options)

    def do_resampling(self):

        nsets = len(self.set_widget_list)
        for i in range(nsets):
            print('generating data for set {} of {}'.format(i+1,nsets))
            hists = self.set_selector_list[i].get_histograms()
            extname = self.get_name(i)
            partitions = self.get_partitions(i)
            (function, function_options) = self.get_function(i)
            for histname in self.histstruct.histnames:
                print('  now processing histogram type {}'.format(histname))
                thishists = hu.averagehists( hists[histname], partitions )
                exthists = function( thishists, **function_options )
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

    def __init__(self, master, histstruct):
        super().__init__(master=master)
        self.title('Evaluation window')
        self.histstruct = histstruct
        self.select_test_set_button_list = []
        self.test_set_type_box_list = []
        self.test_set_selector_list = []

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
        self.test_set_container_label = tk.Label(self.test_set_container_frame, text='Test sets')
        self.test_set_container_label.grid(row=0, column=0)

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
        score_dist_default_options = ({
                    # options needed at this level
                    'make score distribution':'True',
                    # options passed down
                    'siglabel':'anomalous', 'sigcolor':'r', 
                    'bcklabel':'good', 'bckcolor':'g', 
                    'nbins':'200', 'normalize':'True',
                    'xaxtitle':'negative logarithmic probability',
                    'yaxtitle':'number of lumisections (normalized)'})
        self.score_dist_options_frame = OptionsFrame(self.evaluation_options_frame, 
                                            labels=score_dist_default_options.keys(),
                                            values=score_dist_default_options.values())
        self.score_dist_options_frame.frame.grid(row=1, column=0)

        # add widgets for roc curve
        self.roc_options_label = tk.Label(self.evaluation_options_frame, 
                                          text='Options for ROC curve')
        self.roc_options_label.grid(row=0, column=1)
        roc_default_options = ({
                    # options needed at this level
                    'make ROC curve':'True',
                    # options passed down
                    'mode':'geom', 'doprint':'False'})
        self.roc_options_frame = OptionsFrame(self.evaluation_options_frame,
                                            labels=roc_default_options.keys(),
                                            values=roc_default_options.values())
        self.roc_options_frame.frame.grid(row=1, column=1)

        # add widgets for confusion matrix
        self.cm_options_label = tk.Label(self.evaluation_options_frame, 
                                         text='Options for confusion matrix')
        self.cm_options_label.grid(row=0, column=2)
        cm_default_options = ({
                    # options needed at this level
                    'make confusion matrix': 'True',
                    # options passed down
                    'wp':'50'})
        self.cm_options_frame = OptionsFrame(self.evaluation_options_frame,
                                            labels=cm_default_options.keys(),
                                            values=cm_default_options.values())
        self.cm_options_frame.frame.grid(row=1, column=2)

        # add widgets for 2D contour plots
        self.contour_options_label = tk.Label(self.evaluation_options_frame, 
                                              text='Options for fit plots')
        self.contour_options_label.grid(row=0, column=3)
        contour_default_options = ({
                        # options needed at this level
                        'make fit plots':'False',
                        'logprob':'True', 'clipprob':'True',
                        'xlims':'60', 'ylims':'60',
                        'onlypositive':'True', 'transparency':'0.5',
                        'title':'density fit of lumisection MSE'})
        self.contour_options_frame = OptionsFrame(self.evaluation_options_frame,
                                            labels=contour_default_options.keys(),
                                            values=contour_default_options.values())
        self.contour_options_frame.frame.grid(row=1, column=3)

        # add a button to start the evaluation
        self.evaluate_button = tk.Button(self, text='Evaluate', command=self.evaluate)
        self.evaluate_button.grid(row=2, column=0)

    def add_set(self, parent=None, default_type='Good'):
        ### add one test set
        if parent is None: parent = self
        row = 0
        column = len(self.select_test_set_button_list)+1
        idx = len(self.select_test_set_button_list)
        select_button = tk.Button(parent, text='Select test set', 
                                    command=functools.partial(self.open_select_window,idx),
                                    bg='orange')
        select_button.grid(row=row, column=column)
        self.select_test_set_button_list.append( select_button )
        type_box = ttk.Combobox(parent, values=['Good','Bad'])
        if default_type=='Bad': type_box.current(1)
        else: type_box.current(0)
        type_box.grid(row=row+1,column=column)
        self.test_set_type_box_list.append( type_box )
        self.test_set_selector_list.append( None )

    def open_select_window(self, idx):
        self.test_set_selector_list[idx] = SelectorWindow(self.master, self.histstruct)
        self.select_test_set_button_list[idx]['bg'] = 'green'
        return

    def check_all_selected(self):
        if None in self.test_set_selector_list: return False
        else: return True

    def get_scores(self, test_set_type):
        scores = []
        for i in range(len(self.test_set_selector_list)):
            if self.test_set_type_box_list[i].get()!=test_set_type: continue
            scores.append(self.test_set_selector_list[i].get_scores())
        if len(scores)==0:
            print('WARNING: there are no test sets with label {}'.format(test_set_type))
        return scores

    def get_globalscores(self, test_set_type):
        globalscores = []
        for i in range(len(self.test_set_selector_list)):
            if self.test_set_type_box_list[i].get()!=test_set_type: continue
            globalscores.append(self.test_set_selector_list[i].get_globalscores())
        if len(globalscores)==0:
            print('WARNING: there are no test sets with label {}'.format(test_set_type))
        return globalscores

    def evaluate(self):
        if not self.check_all_selected():
            raise Exception('ERROR: some test sets were declared but not defined')
        scores_good_parts = self.get_scores('Good')
        scores_bad_parts = self.get_scores('Bad')
        globalscores_good_parts = self.get_globalscores('Good')
        globalscores_good = np.concatenate(tuple(globalscores_good_parts))
        globalscores_bad_parts = self.get_globalscores('Bad')
        globalscores_bad = np.concatenate(tuple(globalscores_bad_parts))
        labels_good = np.zeros(len(globalscores_good)) # background: label = 0
        labels_bad = np.ones(len(globalscores_bad)) # signal: label = 1

        labels = np.concatenate(tuple([labels_good,labels_bad]))
        scores = np.concatenate(tuple([-globalscores_good,-globalscores_bad]))
        scores = aeu.clip_scores( scores )

        score_dist_options = self.score_dist_options_frame.get_dict()
        do_score_dist = score_dist_options.pop('make score distribution')
        if do_score_dist:
            score_dist_options['doshow'] = False # gui blocks if this is True
            pu.plot_score_dist(scores, labels, **score_dist_options)
        roc_options = self.roc_options_frame.get_dict()
        do_roc = roc_options.pop('make ROC curve')
        if do_roc: 
            roc_options['doshow'] = False # gui blocks if this is True
            auc = aeu.get_roc(scores, labels, **roc_options)
        cm_options = self.cm_options_frame.get_dict()
        do_cm = cm_options.pop('make confusion matrix')
        if do_cm: aeu.get_confusion_matrix(scores,labels, **cm_options)
        plt.show(block=False)
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
                fig,ax = pu.plot_fit_2d(self.histstruct.fitting_scores, fitfunc=partialfitfunc, 
                    xaxtitle=self.histstruct.histnames[dims[0]],
                    yaxtitle=self.histstruct.histnames[dims[1]],
                    onlycontour=True,
                    **contour_options)
                for j in range(len(scores_bad_parts)):
                    ax.plot(    scores_bad_parts[j][self.histstruct.histnames[dims[0]]],
                                scores_bad_parts[j][self.histstruct.histnames[dims[1]]],
                                '.',color=badcolorlist[j],markersize=4)
                for j in range(len(scores_good_parts)): 
                    ax.plot(    scores_good_parts[j][self.histstruct.histnames[dims[0]]],
                                scores_good_parts[j][self.histstruct.histnames[dims[1]]],
                                '.',color=goodcolorlist[j],markersize=4)
            plt.show(block=False)

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
                                only_set_selection=True,
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

    def __init__(self, master, histstruct):
        super().__init__(master)
        self.title('Lumisection plotting')
        self.histstruct = histstruct
        self.inspect_set_selector = None
        self.ref_set_selector = None

        # create a frame for the buttons
        self.buttons_frame = tk.Frame(self)
        self.buttons_frame.grid(row=0, column=0, sticky='nsew')
        set_frame_default_style( self.buttons_frame )

        # add widgets with options
        options = {'mode':'ls', 'run': '', 'lumisection':'', 'plot score':'False',
                    'reco mode':'auto'}
        self.options_frame = OptionsFrame(self.buttons_frame, 
                                labels=options.keys(), values=options.values())
        self.options_frame.frame.grid(row=0, column=0, sticky='nsew')
        self.plot_button = tk.Button(self.buttons_frame, text='Plot', command=self.plot)
        self.plot_button.grid(row=1, column=0)

        # add widgets for selecting inspect datasets
        self.inspect_set_frame = tk.Frame(self)
        self.inspect_set_frame.grid(row=0, column=1, sticky='nsew')
        set_frame_default_style( self.inspect_set_frame )
        self.inspect_set_label = tk.Label(self.inspect_set_frame, text='Select masks for plotting')
        self.inspect_set_label.grid(row=0, column=0)
        self.inspect_set_button = tk.Button(self.inspect_set_frame, text='Select masks',
                                            command=self.open_select_inspect_set_window,
                                            bg='orange')
        self.inspect_set_button.grid(row=1, column=0)

        # add widgets for selecting reference datasets
        self.ref_set_frame = tk.Frame(self)
        self.ref_set_frame.grid(row=0, column=2, sticky='nsew')
        set_frame_default_style( self.ref_set_frame )
        self.ref_set_label = tk.Label(self.ref_set_frame, text='Choose reference dataset')
        self.ref_set_label.grid(row=0, column=0)
        self.ref_set_button = tk.Button(self.ref_set_frame, text='Select dataset',
                                            command=self.open_select_ref_set_window,
                                            bg='orange')
        self.ref_set_button.grid(row=1, column=0)
        options = {'partitions':'-1'}
        self.ref_set_options_frame = OptionsFrame(self.ref_set_frame,
                                    labels=options.keys(), values=options.values())
        self.ref_set_options_frame.frame.grid(row=2, column=0)

    def open_select_inspect_set_window(self):
        self.inspect_set_selector = SelectorWindow(self.master, self.histstruct, 
                                                    only_mask_selection=True)
        self.inspect_set_button['bg'] = 'green'

    def open_select_ref_set_window(self):
        self.ref_set_selector = SelectorWindow(self.master, self.histstruct)
        self.ref_set_button['bg'] = 'green'

    def get_reference_histograms(self):
        if self.ref_set_selector is None: return None
        histograms = self.ref_set_selector.histograms
        if histograms is None: return None
        partitions = self.ref_set_options_frame.get_dict()['partitions']
        if partitions<0: return histograms
        for histname in histograms.keys():
            histograms[histname] = hu.averagehists( histograms[histname], partitions )
        return histograms

    def get_inspect_masks(self):
        if self.inspect_set_selector is None: return None
        masks = self.inspect_set_selector.masks
        return masks

    def plot(self):

        # get the correct options depending on the mode
        options = self.options_frame.get_dict()
        runnb = options['run']
        recomode = options['reco mode']
        if recomode != 'auto': recomode=None
        plotscore = options['plot score']
        if options['mode']=='ls':
            lsnbs = [options['lumisection']]
        elif options['mode']=='run':
            runnbs = self.histstruct.get_runnbs( masknames=self.get_inspect_masks() )
            lsnbs = self.histstruct.get_lsnbs( masknames=self.get_inspect_masks() )
            runsel = np.where(runnbs==runnb)
            lsnbs = lsnbs[runsel]
            plotscore = False # disable for now for mode==run,
                              # maybe enable later.
            print('plotting {} lumisections...'.format(len(lsnbs)))
        else: raise Exception('ERROR: option mode = {} not recognized;'.format(options['mode'])
                                +' should be either "run" or "ls".')

        # get the reference histograms
        refhists = self.get_reference_histograms()

        # run over lumisections to plot
        for i, lsnb in enumerate(lsnbs):
            # only for quick testing:
            if i>4: 
                print('WARNING: plotting loop closed after 5 iterations for testing')
                break
            fig,ax = self.histstruct.plot_ls(runnb, lsnb, recohist=recomode, refhists=refhists )
            plt.show(block=False)
            scorepoint = self.histstruct.get_scores_ls( runnb, lsnb )
            logprob = self.histstruct.get_globalscore_ls( runnb, lsnb )
            print('-------------')
            print('scores:')
            for histname in self.histstruct.histnames: 
                print('{} : {}'.format(histname,scorepoint[histname]))
            print('-------------')
            print('logprob: '+str(logprob))

            if plotscore:
                ncols = min(4,len(self.histstruct.histnames))
                nrows = int(math.ceil(len(self.histstruct.histnames)/ncols))
                fig,axs = plt.subplots(nrows,ncols,figsize=(6*ncols,6*nrows),squeeze=False)
                for dim,histname in enumerate(self.histstruct.histnames):
                    scores = self.histstruct.get_scores( histname=histname, 
                                                     masknames=self.get_inspect_masks() )
                    nscores = len(scores)
                    labels = np.zeros(nscores)
                    scores = np.concatenate((scores,np.ones(int(nscores/15))*scorepoint[histname]))
                    labels = np.concatenate((labels,np.ones(int(nscores/15))))
                    pu.plot_score_dist( scores, labels, fig=fig, ax=axs[int(dim/ncols),dim%ncols], 
                            nbins=200, normalize=False,
                            siglabel='this lumisection', bcklabel='all (masked) lumisections',
                            title=histname, doshow=False )
                plt.show(block=False)


class ML4DQMGUI:

    def __init__(self, master):
        
        self.master = master
        master.title('ML4DQM GUI')

        # initializations
        self.histstruct = None
        self.histstruct_filename = None
        self.button_frames = []
        self.all_frames = []

        # add widgets for creating a new histstruct
        self.newhs_frame = tk.Frame(master)
        self.newhs_label = tk.Label(self.newhs_frame, text='HistStruct creation')
        self.newhs_label.grid(row=0, column=0, sticky='ew')
        self.newhs_button = tk.Button(self.newhs_frame, text='New', 
                                        command=self.open_new_histstruct_window)
        self.newhs_button.grid(row=1, column=0, sticky='ew')
        self.addclassifiers_button = tk.Button(self.newhs_frame, text='Add classifiers',
                                        command=self.open_add_classifiers_window)
        self.addclassifiers_button.grid(row=2, column=0, sticky='ew')
        # add the frame to the window
        self.newhs_frame.grid(row=0, column=0, sticky='nsew')
        self.button_frames.append(self.newhs_frame)
        self.all_frames.append(self.newhs_frame)

        # add widgets for loading and saving a HistStruct
        self.iobutton_frame = tk.Frame(master)
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

        # add widgets for plotting
        self.plotbutton_frame = tk.Frame(master)
        self.plotbutton_label = tk.Label(self.plotbutton_frame, text='Plotting')
        self.plotbutton_label.grid(row=0, column=0, sticky='ew')
        self.plot_sets_button = tk.Button(self.plotbutton_frame, text='Plot',
                                          command=self.open_plot_sets_window)
        self.plot_sets_button.grid(row=1, column=0, sticky='ew')
        # add the frame to the window
        self.plotbutton_frame.grid(row=2, column=0, sticky='nsew')
        self.button_frames.append(self.plotbutton_frame)
        self.all_frames.append(self.plotbutton_frame)

        # add widgets for resampling
        self.resampling_frame = tk.Frame(master)
        self.resampling_label = tk.Label(self.resampling_frame, text='Resampling')
        self.resampling_label.grid(row=0, column=0, sticky='ew')
        self.resample_button = tk.Button(self.resampling_frame, text='Resample',
                                    command=self.open_resample_window)
        self.resample_button.grid(row=1, column=0, sticky='ew')
        # add the frame to the window
        self.resampling_frame.grid(row=3, column=0, sticky='nsew')
        self.button_frames.append(self.resampling_frame)
        self.all_frames.append(self.resampling_frame)

        # add widgets for classifier training, fitting and evaluation
        self.model_frame = tk.Frame(master)
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
        self.model_frame.grid(row=4, column=0, sticky='nsew')
        self.button_frames.append(self.model_frame)
        self.all_frames.append(self.model_frame)

        # add widgets for displaying text
        self.stdout_frame = tk.Frame(master)
        self.stdout_frame.grid(row=0, column=1, rowspan=len(self.button_frames),
                                sticky='nsew')
        self.all_frames.append(self.stdout_frame)
        self.stdout_label = tk.Label(self.stdout_frame, text='Stdout')
        self.stdout_label.grid(row=0, column=0)
        self.messages_text = ScrolledTextFrame(self.stdout_frame, txtwidth=50, txtheight=25)
        initstring = 'Welcome to the ML4DQM GUI!\n'
        self.messages_text.widget.insert(tk.INSERT, initstring)
        self.messages_text.frame.grid(row=1, column=0)

        # redirect stdout (and stderr) to text widget
        stdout = sys.stdout
        stderr = sys.stderr
        sys.stdout = StdOutRedirector( self.messages_text.widget, self.master )
        #sys.stderr = StdOutRedirector( self.messages_text.widget, self.master )

        # add widgets for displaying HistStruct info
        self.histstruct_info_frame = tk.Frame(master)
        self.histstruct_info_frame.grid(row=0, column=2, rowspan=len(self.button_frames),
                                        sticky='nsew')
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
                                            txtwidth=25, txtheight=15)
        self.histstruct_masknames_text.frame.grid(row=4, column=0)
        initstring = '[no histstruct loaded]'
        self.histstruct_masknames_text.widget.insert(tk.INSERT, initstring)
        self.histstruct_extnames_label = tk.Label(self.histstruct_info_frame, 
                                            text='Extended sets')
        self.histstruct_extnames_label.grid(row=3, column=1)
        self.histstruct_extnames_text = ScrolledTextFrame(self.histstruct_info_frame, 
                                            txtwidth=25, txtheight=15)
        self.histstruct_extnames_text.frame.grid(row=4, column=1)
        initstring = '[no histstruct loaded]'
        self.histstruct_extnames_text.widget.insert(tk.INSERT, initstring)

        # apply default stylings to all frames
        for frame in self.all_frames:
            set_frame_default_style( frame )

        # apply default stylings to button frames
        for frame in self.button_frames:
            pass

    def clear_histstruct_info(self):
        ### clear all widgets displaying histstruct info
        self.histstruct_filename_text.widget.delete(1.0, tk.END)
        self.histstruct_masknames_text.widget.delete(1.0, tk.END)
        self.histstruct_extnames_text.widget.delete(1.0, tk.END)

    def update_histstruct_info(self):
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
        initialdir = os.path.abspath(os.path.dirname(__file__))
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
        initialdir = os.path.abspath(os.path.dirname(__file__))
        filename = fldlg.asksaveasfilename(initialdir=initialdir,
                    title='Save a HistStruct',
                    filetypes=(('zip files','*.zip'),('all files','*.*')))
        # if filename is invalid, return
        if len(filename)==0:
            print('Saving of HistStruct canceled')
            return
        # save the histstruct
        self.histstruct.save( filename )

    def open_new_histstruct_window(self):
        self.histstruct = HistStruct.HistStruct()
        _ = NewHistStructWindow(self.master, self.histstruct)
        return

    def open_add_classifiers_window(self):
        if not self.histstruct:
            print('ERROR: need to load a HistStruct first')
            return
        _ = AddClassifiersWindow(self.master, self.histstruct)
        return

    def open_plot_sets_window(self):
        if not self.histstruct:
            print('ERROR: need to load a HistStruct first')
            return
        _ = PlotSetsWindow(self.master, self.histstruct)
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
        _ = FitWindow(self.master, self.histstruct)
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
        _ = EvaluateWindow(self.master, self.histstruct)
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
        _ = PlotLumisectionWindow(self.master, self.histstruct)
        return


if __name__=='__main__':

    window = tk.Tk()
    gui = ML4DQMGUI(window)
    window.mainloop()
