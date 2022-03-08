### imports

# ipywidgets and related modules

import ipywidgets as ipw
from IPython.display import display,clear_output

# external modules

print('importing external modules...')
print('  import os'); import os
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

print('importing local graphical elements')
import importlib
import gui
importlib.reload(gui)
from gui import get_docurl, get_classifier_class, get_resampling_function, get_training_options, get_fitter_class, get_args_dict
sys.path.append('nbgui_src')
import importlib
import UrlWidget
importlib.reload(UrlWidget)
from UrlWidget import UrlWidget
import FileBrowser
importlib.reload(FileBrowser)
from FileBrowser import FileBrowser
import AddRunMasksWidget
importlib.reload(AddRunMasksWidget)
from AddRunMasksWidget import AddRunMasksWidget
import AddStatMasksWidget
importlib.reload(AddStatMasksWidget)
from AddStatMasksWidget import AddStatMasksWidget
import AddJsonMasksWidget
importlib.reload(AddJsonMasksWidget)
from AddJsonMasksWidget import AddJsonMasksWidget
import SelectorWidget
importlib.reload(SelectorWidget)
from SelectorWidget import SelectorWidget
import OptionsBox
importlib.reload(OptionsBox)
from OptionsBox import OptionsBox

print('done')


### define layouts
button_layout = ipw.Layout(width='auto', height='40px')
def set_box_default_style( gridbox ):
    gridbox.layout.border = "1px solid black"


class NewHistStructTab:
    
    def __init__(self, emptyhiststruct):
        ### initializer
        
        # initializations
        self.tab = ipw.Output()
        self.title = 'New'
        self.histstruct = emptyhiststruct
        self.histfiles = {}

        # add widgets for general options
        self.general_options_label = ipw.Label(value='General options')
        self.training_mode_box = ipw.Dropdown(description='Training type', options=['global','local'])
        self.year_box = ipw.Dropdown(description='Training year', options=['2017'])
        self.general_options_box = ipw.GridBox(children=[
                                                    self.general_options_label, 
                                                    self.training_mode_box, 
                                                    self.year_box],
                                            layout=ipw.Layout(
                                                    grid_template_rows='auto auto auto',
                                                    )
        )
        set_box_default_style(self.general_options_box)
        
        # add widgets for choice of histograms
        self.add_histograms_button = ipw.Button(description='Add histograms')
        self.add_histograms_button.on_click(self.add_histnames)
        self.clear_histograms_button = ipw.Button(description='Clear')
        self.clear_histograms_button.on_click(self.clear_histnames)
        self.histnames_listbox = ipw.SelectMultiple(
                                    options=[], rows=10 )
        self.add_histograms_box = ipw.GridBox(children=[
                                                    self.add_histograms_button, 
                                                    self.clear_histograms_button,
                                                    self.histnames_listbox],
                                            layout=ipw.Layout(
                                                     grid_template_rows='auto auto auto')
        )
        set_box_default_style(self.add_histograms_box)

        # add widgets for local options
        self.local_options_label = ipw.Label(value='Options for local training')
        self.target_run_text = ipw.Text(description='Target run')
        self.ntraining_text = ipw.Text(description='Number of training runs',value='5')
        self.offset_text = ipw.Text(description='Number of offset runs',value='0')
        self.remove_runs_checkbox = ipw.Checkbox(value=True,
                                                 description='Remove unneeded runs?')
        self.local_options_box = ipw.GridBox(children=[
                                                    self.local_options_label, 
                                                    self.target_run_text,
                                                    self.ntraining_text,
                                                    self.offset_text,
                                                    self.remove_runs_checkbox],
                                            layout=ipw.Layout(
                                                     grid_template_rows='auto auto auto auto auto')
        )
        set_box_default_style(self.local_options_box)

        # add widgets for run mask addition
        self.add_run_mask_obj = AddRunMasksWidget()
        self.add_run_mask_box = self.add_run_mask_obj.get_widget()
        set_box_default_style(self.add_run_mask_box)
                                                  
        # add widgets for stat mask addition
        self.add_stat_mask_obj = AddStatMasksWidget()
        self.add_stat_mask_box = self.add_stat_mask_obj.get_widget()
        set_box_default_style(self.add_stat_mask_box)

        # add widgets for json mask addition
        self.add_json_mask_obj = AddJsonMasksWidget()
        self.add_json_mask_box = self.add_json_mask_obj.get_widget()
        set_box_default_style(self.add_json_mask_box)

        # add button to start HistStruct creation
        self.make_histstruct_button = ipw.Button(description='Make HistStruct')
        self.make_histstruct_button.on_click(self.make_histstruct)
        
        # display all widgets
        self.total_box = ipw.GridBox(children=[
                                                self.general_options_box,
                                                self.add_histograms_box,
                                                self.local_options_box,
                                                self.add_run_mask_box,
                                                self.add_stat_mask_box,
                                                self.add_json_mask_box,
                                                self.make_histstruct_button
                                                ],
                                            layout=ipw.Layout(
                                                    grid_gap='5px 5px',
                                                    grid_template_rows='auto auto auto auto auto auto auto')
        )
        with self.tab:
            display(self.total_box)

    def get_target_run(self):
        ### get the target run from the corresponding Text widget
        # return None if the widget is empty
        target_run_text = self.target_run_text.value
        if( target_run_text is None or len(target_run_text)==0 ): return None
        return int(target_run_text)

    def get_local_training_runs(self, filename):
        ### get the training runs from the corresponding Text widgets
        # return None if the target run is None
        target_run = self.get_target_run()
        if target_run is None: return None
        ntraining = int(self.ntraining_text.value)
        offset = int(self.offset_text.value)
        runs = dfu.get_runs( dfu.select_dcson( csvu.read_csv( filename ) ) )
        runs = dfu.get_runs( csvu.read_csv( filename ) )
        target_run_index = runs.index(target_run)
        training_runs = runs[target_run_index-ntraining-offset:target_run_index-offset]
        return training_runs

    def get_needed_runs(self, is_local=False, filename=None):
        all_runs = []
        # add the runs from all run masks
        for run in self.add_run_mask_obj.get_run_masks().values():
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

    def clear_histnames(self, event):
        self.histfiles = {}
        self.histnames_listbox.options = []

    def add_histnames(self, event):
        self.filebrowser = FileBrowser()
        box = self.filebrowser.get_widget()
        self.filebrowser.overwrite_selectbutton(self.add_histnames_select)
        self.add_histograms_box.children = ([self.add_histograms_button,
                                             box,
                                             self.clear_histograms_button,
                                             self.histnames_listbox])
        
    def add_histnames_select(self, event):
        filenames = self.filebrowser.get_selected_files()
        # if filename is invalid, return
        if len(filenames)==0:
            with self.tab: print('Loading of histograms canceled')
            return
        for filename in filenames:
            histname = os.path.basename(filename).replace('.csv','')
            self.histfiles[histname] = filename
            self.histnames_listbox.options += tuple([histname])
        self.filebrowser.cancel(event)

    def get_histnames(self):
        histnames = list(self.histnames_listbox.value)
        return histnames

    def make_histstruct(self, event):

        # get general settings
        histnames = self.get_histnames()
        histnames = [h.split('(')[0].strip(' ') for h in histnames]
        year = self.year_box.value
        training_mode = self.training_mode_box.value
        with self.tab: 
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
            with self.tab: print('adding {}...'.format(histname))
            # read the histograms from the csv file
            filename = self.histfiles[histname]
            if not os.path.exists(filename):
                raise Exception('ERROR: the file {} does not seem to exist.'.format(filename))
            df = csvu.read_csv( filename )
            # in case of local training, we can remove most of the histograms
            if( training_mode=='local' and self.remove_runs_checkbox.value ):
                needed_runsls = {str(run): [[-1]] for run in needed_runs}
                df = dfu.select_runsls( df, needed_runsls )
            # add the histograms to the HistStuct 
            self.histstruct.add_dataframe( df )
        with self.tab: 
            print('added {} lumisections with {} histograms each to the HistStruct.'.format(
                len(self.histstruct.runnbs),len(self.histstruct.histnames)))
    
        # add default masks for DCS-bit on and golden json
        # to do: make this more flexible with user options
        with self.tab: print('adding default DCS-on and golden json masks...')
        try: self.histstruct.add_dcsonjson_mask( 'dcson' )
        except: 
            with self.tab:
                print('WARNING: could not add a mask for DCS-on data.'
                      +' Check access to DCS-on json file.')
        try: self.histstruct.add_goldenjson_mask( 'golden' )
        except: 
            with self.tab:
                print('WARNING: could not add a mask for golden data.'
                        +' Check access to golden json file.')

        # add training and target mask for local training
        # to do: make this more flexible (e.g. choosing names)
        if training_mode=='local':
            if self.get_target_run() is not None:
                with self.tab: print('adding mask for target runs...')
                json = {str(self.get_target_run()): [[-1]]}
                self.histstruct.add_json_mask( 'target_run', json )
            if self.get_local_training_runs(firstfilename) is not None:
                with self.tab: print('adding mask for local training runs...')
                json = {str(run): [[-1]] for run in self.get_local_training_runs(firstfilename)}
                self.histstruct.add_json_mask( 'local_training', json )

        # add high statistics mask(s)
        highstat_masks = self.add_stat_mask_obj.apply(None)

        # add run mask(s)
        run_masks = self.add_run_mask_obj.apply(None)

        # add json mask(s)
        json_masks = self.add_json_mask_obj.apply(None)
        
        with self.tab: print('done')
            
            
class AddRunMasksTab:
    def __init__(self):
        ### initializer
        self.tab = ipw.Output()
        self.title = 'Add run masks'
        
    def refresh(self, histstruct=None):
        ### initializer
        # input arguments:
        # - histstruct: if specified, a list of available runs is extracted from it;
        #               else the text box is left blank and can take any value.
        self.widget = AddRunMasksWidget(histstruct=histstruct, applybutton=True)
        with self.tab:
            clear_output()
            display(self.widget.get_widget())
            
            
class AddStatMasksTab:
    def __init__(self):
        ### initializer
        self.tab = ipw.Output()
        self.title = 'Add stat masks'
        
    def refresh(self, histstruct=None):
        ### initializer
        # input arguments:
        # - histstruct: if specified, a list of available histogram names is extracted from it;
        #               else the statistics mask can only be applied on all histograms simultaneously.
        self.widget = AddStatMasksWidget(histstruct=histstruct, applybutton=True)
        with self.tab:
            clear_output()
            display(self.widget.get_widget())
            
            
class AddClassifiersTab:
    
    def __init__(self):
        ### initializer
        self.tab = ipw.Output()
        self.title = 'Add classifiers'
        
    def refresh(self, histstruct=None):
        ### initializer
        self.histstruct = histstruct

        # add widgets for setting the classifier type and options
        self.classifier_widgets = {}
        self.classifier_grids = {}
        for i,histname in enumerate(self.histstruct.histnames):
            row = int(i/4)
            column = int(i%4)
            histname_label = ipw.Label(value=histname)
            classifier_type_box = ipw.Dropdown(options=get_classifier_class(), description='Classifier type')
            classifier_type_box.observe( functools.partial(self.set_classifier_options, histname), names='value')
            key_label = ipw.Label(value='Parameters')
            value_label = ipw.Label(value='Values')
            classifier_options_box = OptionsBox(labels=[], values=[])
            # add option to evaluate the model after adding it
            evaluate_box = ipw.Checkbox(value=False, description='Evaluate models after adding')
            # add everything to a structure 
            self.classifier_widgets[histname] = {'type':classifier_type_box, 
                                                 'options':classifier_options_box,
                                                 'evaluate':evaluate_box}
            self.set_classifier_options(histname, None)
            # make the layout
            self.classifier_grids[histname] = ipw.GridBox(children=[histname_label,
                                                      classifier_type_box,
                                                      classifier_options_box.get_widget(),
                                                      evaluate_box],
                                                      layout=ipw.Layout(grid_template_rows='auto auto auto'))

        # add a button for adding the classifiers
        self.add_button = ipw.Button(description='Add classifiers')
        self.add_button.on_click(self.add_classifiers)
        
        # make the layout
        self.grid = ipw.GridBox(children=list(self.classifier_grids.values()),
                            layout=ipw.Layout(grid_template_columns='auto '*len(self.classifier_grids.values())))
        
        with self.tab:
            display(self.grid)
            display(self.add_button)
        

    def set_classifier_options(self, histname, event):
        classifier_name = self.classifier_widgets[histname]['type'].value
        (ctype, coptions) = get_classifier_class(classifier_name)
        # do special overridings if needed
        optiontypes = [None]*len(coptions.keys())
        if ctype is AutoEncoder.AutoEncoder:
            if 'modelpath' in list(coptions.keys()):
                idx = list(coptions.keys()).index('modelpath')
                optiontypes[idx] = FileBrowser
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
        classifier_name = self.classifier_widgets[histname]['type'].value
        (classifier, _) = get_classifier_class(classifier_name)
        classifier_options = self.classifier_widgets[histname]['options'].get_dict()
        return (classifier, classifier_options)

    def add_classifiers(self, event):
        for histname in self.histstruct.histnames:
            (classifier, classifier_options) = self.get_classifier(histname)
            classifier = classifier( **classifier_options )
            self.histstruct.add_classifier( histname, classifier )
            # check if need to evaluate
            do_evaluate = (self.classifier_widgets[histname]['evaluate'].value)
            if do_evaluate:
                self.histstruct.evaluate_classifier(histname)
        with self.tab: print('done')
            
            
class LoadHistStructTab:
    def __init__(self, external_load_function):
        
        # initializations
        self.tab = ipw.Output()
        self.title = 'Load'
        self.external_load_function = external_load_function
        
        # add widgets for loading a histstruct
        self.load_button = ipw.Button(description='Load selected file')
        self.load_button.on_click(self.load)
        self.filebrowser = FileBrowser()
        self.grid = ipw.GridBox(children=[self.load_button,self.filebrowser.get_widget()],
                                layout=ipw.Layout(grid_template_rows='auto auto')) 
        with self.tab:
            clear_output()
            display(self.grid)
        
    def refresh(self, external_load_function):
        ### initializer
        pass
        
    def load(self, event):
        filename = self.filebrowser.get_selected_files()
        if len(filename)!=1:
            with self.tab: print('ERROR: {} files were selected while expecting 1.'.format(len(filename)))
            return
        filename = filename[0]
        self.external_load_function(filename)
            
            
class SaveHistStructTab:
    def __init__(self):
        ### initializer
        self.tab = ipw.Output()
        self.title = 'Save'
        
    def refresh(self, histstruct=None):
        ### initializer
        if histstruct is None:
            with self.tab: print('[no histstruct found]')
            return
        if len(histstruct.get_runnbs())==0:
            with self.tab: print('[current histstruct is empty]')
            return
        self.histstruct = histstruct
        
        self.filename_text = ipw.Text(description='File name')
        self.save_button = ipw.Button(description='Save')
        self.save_button.on_click(self.save)
        self.grid = ipw.GridBox(children=[self.filename_text,self.save_button],
                                layout=ipw.Layout(grid_template_rows='auto auto'))
        with self.tab:
            clear_output()
            display(self.grid)
            
    def filename_is_valid(self, filename):
        valid = True
        msg = ''
        if( len(filename)==0 ): 
            valid = False
            msg = 'ERROR: empty file name.'
        if( os.path.exists(filename) ):
            valid = False
            msg = 'ERROR: a file with this name already exists.'
        return (valid, msg)
            
    def save(self, event):
        filename = self.filename_text.value
        valid = self.filename_is_valid(filename)
        if not valid[0]:
            with self.tab: print(valid[1])
            return
        self.histstruct.save(filename)
        with self.tab: print('done')
        
            
class DisplayHistStructTab:
    def __init__(self):
        ### initializer
        self.tab = ipw.Output()
        self.title = 'Display'
        
    def refresh(self, histstruct=None):
        ### initializer
        info = '[no histstruct found]'
        if histstruct is not None: info = histstruct.__str__() 
        self.info_text = ipw.Textarea(value=info, disabled=True,
                                      layout=ipw.Layout(width='100%',height='500px'))
        self.grid = ipw.GridBox(children=[self.info_text], 
                                layout=ipw.Layout(grid_template_rows='auto'))
        with self.tab:
            clear_output()
            display(self.grid)
            
            
class PreprocessingTab:
    def __init__(self):
        ### initializer
        self.tab = ipw.Output()
        self.title = 'Preprocessing'
        
    def refresh(self, histstruct=None):
        ### initializer
        
        # initializations
        if histstruct is None:
            with self.tab: print('[no histstruct found]')
            return
        if len(histstruct.get_runnbs())==0:
            with self.tab: print('[current histstruct is empty]')
            return
        self.histstruct = histstruct
        self.set_selector = None
        
        # add preprocessing options
        options = []
        options.append( {'name':'cropping', 'val':None, 'type':ipw.Text, 
                         'docurl':get_docurl(hu.get_cropslices_from_str)} )
        options.append( {'name':'rebinningfactor', 'val':None, 'type':ipw.Text,
                        'docurl':get_docurl(hu.get_rebinningfactor_from_str)} )
        options.append( {'name':'smoothinghalfwindow', 'val':None, 'type':ipw.Text,
                        'docurl':get_docurl(hu.get_smoothinghalfwindow_from_str)} )
        options.append( {'name':'donormalize', 'val':[False,True], 'type':ipw.Combobox,
                        'docurl':get_docurl(hu.normalizehists)} )
        labels = [el['name'] for el in options]
        types = [el['type'] for el in options]
        values = [el['val'] for el in options]
        docurls = [el['docurl'] for el in options]
        self.optionsbox = OptionsBox(labels=labels, types=types, values=values, 
                                        docurls=docurls, docurl=get_docurl(histstruct.preprocess))

        # add widgets for selecting histograms
        self.select_set_obj = SelectorWidget(self.histstruct, 
                                             title='Select set to process (default: all)',
                                             set_selection=False, post_selection=False)
        self.select_set_box = self.select_set_obj.get_widget()

        # add a button to apply the preprocessing
        self.apply_button = ipw.Button(description='Apply')
        self.apply_button.on_click(self.apply)
        
        # make the layout
        self.grid = ipw.GridBox(children=[self.optionsbox.get_widget(), self.select_set_box, self.apply_button],
                                layout=ipw.Layout(grid_template_rows='auto auto auto'))
        
        with self.tab:
            clear_output()
            display(self.grid)

    def apply(self, event):
        # get masks
        masknames = None
        if self.set_selector.valid:
            masknames = self.select_set_obj.get_masks()
            with self.tab:
                print('Applying preprocessing on the following masks:')
                for mask in masknames: print('  - {}'.format(mask))
        else: 
            with self.tab: print('Applying preprocessing on all histograms.')
        # get options
        options = self.optionsbox.get_dict()
        # do special treatment if needed
        slices = hu.get_cropslices_from_str(options.pop('cropping'))
        options['cropslices'] = slices
        rebinningfactor = hu.get_rebinningfactor_from_str(options.pop('rebinningfactor'))
        options['rebinningfactor'] = rebinningfactor
        smoothinghalfwindow = hu.get_smoothinghalfwindow_from_str(options.pop('smoothinghalfwindow'))
        options['smoothinghalfwindow'] = smoothinghalfwindow
        # do the preprocessing
        self.histstruct.preprocess(masknames=masknames, **options)
        with self.tab: print('done')
            
            
class LoadPlotStyleTab:
    
    def __init__(self):
        ### initializer
        self.tab = ipw.Output()
        self.title = 'Load plot style'
        
    def refresh(self, plotstyleparser=None):
        
        # check arguments
        if plotstyleparser is None: return
        self.plotstyleparser = plotstyleparser
        
        # add file selector
        self.filebrowser = FileBrowser()
        
        # add load button
        self.load_button = ipw.Button(description='Load selected file')
        self.load_button.on_click( self.load )
        self.grid = ipw.GridBox(children=[self.load_button,self.filebrowser.get_widget()],
                                layout=ipw.Layout(grid_template_rows='auto auto')) 
        with self.tab:
            clear_output()
            display(self.grid)
        
    def load(self, event):
        filename = self.filebrowser.get_selected_files()
        if len(filename)!=1:
            with self.tab: print('ERROR: {} files were selected while expecting 1.'.format(len(filename)))
            return
        filename = filename[0]
        self.plotstyleparser.load(filename)
            
            
class PlotSetsTab:
    
    def __init__(self):
        ### initializer
        self.tab = ipw.Output()
        self.title = 'Plotting'
        
    def refresh(self, histstruct=None, plotstyleparser=None):
        ### initializer
        
        # check arguments
        if histstruct is None:
            with self.tab: print('[no histstruct found]')
            return
        if len(histstruct.get_runnbs())==0:
            with self.tab: print('[current histstruct is empty]')
            return
        
        # initializations
        self.histstruct = histstruct
        self.plotstyleparser = plotstyleparser
        self.set_selector_list = []
        self.set_options_list = []

        # add a button to allow adding more sets of options
        self.more_button = ipw.Button(description='Add a set')
        self.more_button.on_click(self.add_set)

        # add a button to make the plot
        self.plot_button = ipw.Button(description='Make plot')
        self.plot_button.on_click(self.make_plot)
        
        # make the layout
        self.button_box = ipw.GridBox(children=([self.more_button,
                                                self.plot_button]),
                                      layout=ipw.Layout(
                                          grid_template_rows='auto auto'
                                      )
        )
        self.grid = ipw.GridBox(children=[self.button_box],
                                layout=ipw.Layout(grid_template_columns='auto'))

        # add one set of options
        self.add_set(None)
        
        with self.tab:
            clear_output()
            display(self.grid)

    def add_set(self, event):
        ### add widgets for one more histogram set to plot
        # make widgets
        newidx = len(self.set_selector_list)
        select_set_obj = SelectorWidget(self.histstruct,
                                       title='Select set to plot')
        select_set_box = select_set_obj.get_widget()
        set_default_options = {'label':None, 'color':None}
        set_options_obj = OptionsBox( labels=set_default_options.keys(),
                                      values=set_default_options.values())
        set_options_box = set_options_obj.get_widget()
        self.set_selector_list.append(select_set_obj)
        self.set_options_list.append(set_options_obj)
        # make layout
        grid = ipw.GridBox(children=[select_set_box,set_options_box],layout=ipw.Layout())
        self.grid.children += tuple([grid])
        self.grid.layout.grid_template_columns = 'auto ' +'auto '*(len(self.set_selector_list))

    def check_all_selected(self):
        ### check if the 'select' button was pushed for all selectors
        for sel in self.set_selector_list:
            if not sel.valid: return False
        return True

    def make_plot(self, event):
        ### make the plot with current settings
        if not self.check_all_selected():
            raise Exception('ERROR: some sets were declared but not initialized.')
        optionsdict = {'histograms':[], 'labellist':[], 'colorlist':[]}
        # get histograms to plot
        for setselector in self.set_selector_list:
            optionsdict['histograms'].append( setselector.get_histograms() )
        # set style options
        for optionsbox in self.set_options_list:
            setoptions = optionsbox.get_dict()
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
        with self.tab: print('making plot...')
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

            
class ResamplingTab:
    
    def __init__(self):
        ### initializer
        self.tab = ipw.Output()
        self.title = 'Resampling'

    def refresh(self, histstruct=None):
        ### initializer
        
        # check arguments
        if histstruct is None:
            with self.tab: print('[no histstruct found]')
            return
        if len(histstruct.get_runnbs())==0:
            with self.tab: print('[current histstruct is empty]')
            return
            
        # initializations
        self.histstruct = histstruct
        self.set_selectors = {}
        self.resample_functions = {}
        self.resample_options = {}
        self.allhistostr = 'All histogram types'
        self.noresamplestr = '[No resampling]'
        self.nonestr = 'None' # not free to choose, 
                              # should correspond to the one in the global function
                              # get_resampling_function

        # add widget to expand options for different histograms
        self.expand_options_button = ipw.Button(description='Expand/collapse')
        self.expand_options_button.on_click(self.expandcollapse)
        
        # set initial state to single
        self.expandstate = 'multi' # set multi since automatic expandcollapse call

        # add widgets to start resampling
        self.resample_button = ipw.Button(description='Start resampling')
        self.resample_button.on_click(self.do_resampling)
        
        # make new name label and text entry
        self.name_text = ipw.Text(description='Set name:')

        # add widgets to view current resampled sets
        self.sets_list = ipw.Textarea(value='', description='Currently existing sets:')
        self.update_sets_list()
        
        # make the layout
        self.boxes = ipw.GridBox(children=[], layout=ipw.Layout())

        self.expandcollapse(None)
        
        with self.tab:
            clear_output()
            display(self.expand_options_button)
            display(self.resample_button)
            display(self.name_text)
            display(self.sets_list)
            display(self.boxes)

    def update_sets_list(self):
        self.sets_list.value=''
        extnames = self.histstruct.exthistograms.keys()
        if len(extnames)==0:
            self.sets_list.value = '[no sets available]'
        else:
            self.sets_list.value = '\n'.join(extnames)

    def expandcollapse(self, event):
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
        self.set_selectors = {}
        self.resample_functions = {}
        self.resample_options = {}
        # make new options and frame
        newboxes = []
        for i,histname in enumerate(histnames):
            # create a frame to hold the widgets and put some labels
            hist_label = ipw.Label(value=histname)
            # add widgets for choosing resampling basis set
            select_set_obj = SelectorWidget(self.histstruct,
                                            title='Select basis set for resampling',
                                            set_selection=False)
            select_set_box = select_set_obj.get_widget()
            # add dropdown entry for resampling function
            allowed_functions = get_resampling_function()
            for i,f in enumerate(allowed_functions): 
                if f==self.nonestr: allowed_functions[i] = self.noresamplestr
            function_box = ipw.Dropdown(options=allowed_functions, description='Function:')
            function_box.observe(functools.partial(self.set_function_options, histname=histname), names='value')
            # make initial (empty) OptionsFrame
            function_options_box = OptionsBox(labels=[], values=[], autobool=True)
            # add objects to the dicts
            self.set_selectors[histname] = select_set_obj
            self.resample_functions[histname] = function_box
            self.resample_options[histname] = function_options_box
            # make the layout
            box = ipw.GridBox(children=[hist_label,select_set_box,function_box,function_options_box.get_widget()])
            newboxes.append(box)
        self.boxes.children = newboxes
        self.boxes.layout = ipw.Layout(grid_template_columns='auto '*len(newboxes))

    def set_function_options(self, event, histname):
        fname = self.get_function_name(histname)
        (f, foptions) = get_resampling_function(key=fname)
        fdocurl = get_docurl(f)
        self.resample_options[histname].set_options(
            labels=foptions.keys(), values=foptions.values(), docurl=fdocurl)

    def check_all_selected(self):
        ### check if the 'select' button was pushed for all selectors
        for sel in self.set_selectors.values():
            if not sel.valid: return False
        return True 

    def get_function_name(self, histname):
        fname = self.resample_functions[histname].value
        if fname==self.noresamplestr: fname = self.nonestr
        return fname

    def get_function(self, histname):
        fname = self.get_function_name(histname)
        (function, _) = get_resampling_function(fname)
        function_options = self.resample_options[histname].get_dict()
        return (function, function_options)

    def do_resampling(self, event):

        # check whether the same set, function and options can be used for all histogram types
        split = None
        if self.expandstate == 'single': split = False
        elif self.expandstate == 'multi': split = True
        else:
            raise Exception('ERROR: expandstate {} not recognized.'.format(self.expandstate))
        # check whether all required sets have been selected
        if not self.check_all_selected():
            for histname in self.set_selectors.keys():
                selector = self.set_selectors[histname]
                valid = self.set_selectors[histname].valid
                # get the function (allow invalid selector if no resampling required)
                histkey = histname if split else self.allhistostr
                (function,_) = self.get_function(histkey)
                if( not valid and function is not None ):
                    raise Exception('ERROR: requested to resample histogram type {}'.format(histname)
                                     +' but selector was not set.')
        # get the name for the extended set
        extname = self.name_text.value
        if len(extname)==0:
            raise Exception('ERROR: name "{}" is not valid.'.format(extname))
        # loop over histogram types
        for histname in self.histstruct.histnames:
            with self.tab: print('  now processing histogram type {}'.format(histname))
            histkey = histname if split else self.allhistostr
            # get resampling function
            (function, function_options) = self.get_function(histkey)
            if function is None:
                with self.tab: print('WARNING: resampling function for histogram type {}'.format(histname)
                                     +' is None, it will not be present in the resampled set.')
                continue
            # get histograms
            hists = self.set_selectors[histkey].get_histograms()[histname]
            exthists = function( hists, **function_options )[0]
            # add extended set to histstruct
            self.histstruct.add_exthistograms( extname, histname, exthists )
            with self.tab: print('  -> generated {} histograms'.format(len(exthists)))
        self.update_sets_list()
        plt.show(block=False)
        with self.tab: print('done')
            
            
class TrainClassifiersTab:
    
    def __init__(self):
        ### initializer
        self.tab = ipw.Output()
        self.title = 'Training'
        
    def refresh(self, histstruct=None):
        
        # check arguments
        if histstruct is None:
            with self.tab: print('[no histstruct found]')
            return
        if len(histstruct.get_runnbs())==0:
            with self.tab: print('[current histstruct is empty]')
            return
        if len(histstruct.classifiers.keys())==0:
            with self.tab: print('[current histstruct does not contain classifiers]')
            return
        self.histstruct = histstruct
        self.training_options = {}

        # add widgets for choosing resampling basis set
        self.select_set_obj = SelectorWidget(self.histstruct,
                                             title='Select training set')
        self.select_set_box = self.select_set_obj.get_widget()

        # add widget to expand options for different histograms
        self.expand_options_button = ipw.Button(description='Expand/collapse')
        self.expand_options_button.on_click(self.expandcollapse)
        # set initial state to single if only one classifier is present, multi otherwise
        self.expandstate = 'single' # set to single since one automatic expansion
        if get_training_options( self.histstruct ) is not None:
            self.expandstate = 'multi' # set to multi since one automatic collapse
        self.boxes = ipw.GridBox(children=[], layout=ipw.Layout())

        self.expandcollapse()

        # add button to start training
        self.train_button = ipw.Button(description='Start training')
        self.train_button.on_click(self.do_training)
        
        # make the layout
        with self.tab:
            clear_output()
            display(self.select_set_box)
            display(self.expand_options_button)
            display(self.boxes)
            display(self.train_button)

    def expandcollapse(self):
        # check whether need to collapse or expand
        if self.expandstate=='multi':
            # check if this is allowed
            if get_training_options( self.histstruct ) is None:
                with self.tab: print('WARNING: collapse not allowed'
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
        # make new options and frame
        newboxes = []
        for i,histname in enumerate(histnames):
            hist_label = ipw.Label(value=histname)
            # get the training options
            arghistname = histname
            if histname=='all histogram types': arghistname = None
            (c,options) = get_training_options( self.histstruct, histname=arghistname )
            labels = list(options.keys())
            values = list(options.values())
            wtypes = [None]*len(labels)
            # get docurl
            docurl = get_docurl(c)
            # make the options frame
            options_frame = OptionsBox(labels=labels, values=values, types=wtypes,
                docurl=docurl, autobool=True)
            self.training_options[histname] = options_frame
            # make the layout
            box = ipw.GridBox(children=[hist_label,options_frame.get_widget()])
            newboxes.append(box)
        self.boxes.children = newboxes
        self.boxes.layout = ipw.Layout(grid_template_columns='auto '*len(newboxes))

    def do_training(self, event):
        if not self.select_set_obj.valid:
            with self.tab: print('ERROR: please select a training set before starting training.')
            return
        training_histograms = self.select_set_obj.get_histograms()
        for histname in training_histograms.keys():
            # check if a classifier is initialized for this histogram type
            if histname not in self.histstruct.classifiers.keys():
                with self.tab: print('WARNING: no classifier was found in the HistStruct'
                        +' for histogram type {}; skipping.'.format(histname))
                continue
            # get the options for this histogram type
            arghistname = histname
            if self.expandstate=='single': arghistname = 'all histogram types'
            training_options = self.training_options[arghistname].get_dict()
            # get the training histograms
            hists = training_histograms[histname]
            with self.tab:
                print('training a classifier for {}'.format(histname))
                print('size of training set: {}'.format(hists.shape))
            # do training
            self.histstruct.classifiers[histname].train( hists, **training_options )
            # do evaluation
            with self.tab: print('evaluating model for '+histname)
            self.histstruct.evaluate_classifier( histname )
        with self.tab: print('done')
            
            
class ApplyClassifiersTab:
    
    def __init__(self):
        ### initializer
        self.tab = ipw.Output()
        self.title = 'Apply classifiers'
        
    def refresh(self, histstruct=None):
        
        # check arguments
        if histstruct is None:
            with self.tab: print('[no histstruct found]')
            return
        if len(histstruct.get_runnbs())==0:
            with self.tab: print('[current histstruct is empty]')
            return
        self.histstruct = histstruct
        
        # add widgets for choosing resampling basis set
        self.select_set_obj = SelectorWidget(self.histstruct, 
                                             title='Select application set',
                                             mask_selection=False, post_selection=False,
                                             allow_multi_set=True)
        self.select_set_box = self.select_set_obj.get_widget()

        # add a button to start the evaluation
        self.start_evaluation_button = ipw.Button(description='Start evaluation')
        self.start_evaluation_button.on_click(self.evaluate)
        
        # make the layout
        self.grid = ipw.GridBox(children=[self.select_set_box,self.start_evaluation_button],
                                layout=ipw.Layout(grid_template_rows='auto auto'))
        with self.tab:
            clear_output()
            display(self.grid)

    def evaluate(self, event):
        if self.select_set_obj.valid:
            extnames = self.select_set_obj.get_sets()
        else:
            extnames = list(self.histstruct.exthistograms.keys())
        for extname in extnames:
            with self.tab: print('evaluating classifiers on set {}'.format(extname))
            for histname in self.histstruct.histnames:
                with self.tab: print('  now processing histogram type {}'.format(histname))
                self.histstruct.evaluate_classifier( histname, extname=extname )
        with self.tab: print('done')
            
            
class FitTab:
    
    def __init__(self):
        ### initializer
        self.tab = ipw.Output()
        self.title = 'Fitting'
        
    def refresh(self, histstruct=None, plotstyleparser=None):
        
        # check arguments
        if histstruct is None:
            with self.tab: print('[no histstruct found]')
            return
        if len(histstruct.get_runnbs())==0:
            with self.tab: print('[current histstruct is empty]')
            return
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

        # add widgets for choosing fitting set
        self.fitting_set_selector = SelectorWidget(self.histstruct,
                                                   title='Select fitting set')

        # add widgets to select fitting parameters
        self.fitter_box = ipw.Dropdown(options=get_fitter_class(), description='Fit type:')
        self.fitter_box.observe(self.set_fitter_options, names='value')
        self.fitter_options_frame = OptionsBox(labels=[], values=[])
        self.set_fitter_options(None)
    
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
                if key=='xaxtitlesize': 
                    plot_options_dict[key] = self.plotstyleparser.get_xaxtitlesize()
                elif key=='yaxtitlesize': 
                    plot_options_dict[key] = self.plotstyleparser.get_yaxtitlesize()
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
        wtypes = None
        # make the OptionsFrame
        self.plot_options = OptionsBox(labels=labels, types=wtypes, values=values,
                                            docurl=plot_docurl, autobool=True)

        # add a button to start the fit
        self.fit_button = ipw.Button(description='Start fit')
        self.fit_button.on_click(self.do_fit)
        
        # make the layout
        children = ([self.fitting_set_selector.get_widget(),
                     self.fitter_box,
                     self.fitter_options_frame.get_widget(),
                     self.plot_options.get_widget(),
                     self.fit_button])
        self.grid = ipw.GridBox(children=children,
                                layout=ipw.Layout(grid_template_rows='auto '*len(children)))
        with self.tab:
            display(self.grid)

    def set_fitter_options(self, event):
        fitter_name = self.fitter_box.value
        (c, coptions) = get_fitter_class(fitter_name)
        docurl = get_docurl(c)
        self.fitter_options_frame.set_options(labels=coptions.keys(), values=coptions.values(),
                                                docurl=docurl)
    
    def get_fitting_scores(self):
        if not self.fitting_set_selector.valid:
            with self.tab: print('ERROR: please select a set to fit to before doing the fit.')
            return
        scores_fit_dict = self.fitting_set_selector.get_scores()
        if scores_fit_dict is None:
            with self.tab: print('ERROR: no valid scores could be found in the HistStruct '
                            +'for the specified fitting set.')
            return
        scores_fit = []
        for histname in self.histstruct.histnames:
            thisscore = scores_fit_dict[histname]
            scores_fit.append( thisscore )
        # transform to arrays with correct shape
        scores_fit = np.array(scores_fit)
        scores_fit = np.transpose(scores_fit)
        with self.tab: print('found score array for fitting set of following shape: {}'.format(scores_fit.shape))
        return scores_fit

    def get_fitter(self):
        fitter_name = self.fitter_box.value
        (fitter, _) = get_fitter_class(fitter_name)
        fitter_options = self.fitter_options_frame.get_dict()
        return (fitter,fitter_options)

    def do_fit(self, event):
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
            
            
class ApplyFitTab:
    
    def __init__(self):
        ### initializer
        self.tab = ipw.Output()
        self.title = 'Apply fit'
        
    def refresh(self, histstruct=None):
        
        # check arguments
        if histstruct is None:
            with self.tab: print('[no histstruct found]')
            return
        if len(histstruct.get_runnbs())==0:
            with self.tab: print('[current histstruct is empty]')
            return
        self.histstruct = histstruct
        
        # add widgets for choosing resampling basis set
        self.select_set_obj = SelectorWidget(self.histstruct, 
                                             title='Select fit application set',
                                             mask_selection=False, post_selection=False,
                                             allow_multi_set=True)
        self.select_set_box = self.select_set_obj.get_widget()

        # add a button to start the evaluation
        self.start_evaluation_button = ipw.Button(description='Start evaluation')
        self.start_evaluation_button.on_click(self.evaluate)
        
        # make the layout
        self.grid = ipw.GridBox(children=[self.select_set_box,self.start_evaluation_button],
                                layout=ipw.Layout(grid_template_rows='auto auto'))
        with self.tab:
            clear_output()
            display(self.grid)

    def evaluate(self):
        if self.select_set_obj.valid:
            extnames = self.select_set_obj.get_sets()
        else:
            extnames = list(self.histstruct.exthistograms.keys())
        for extname in extnames:
            with self.tab: print('evaluating fitter on set {}'.format(extname))
            scores_all = []
            for histname in self.histstruct.histnames:
                scores_all.append( self.histstruct.get_extscores( extname, histname=histname ) )
            scores_all = np.array(scores_all)
            scores_all = np.transpose(scores_all)
            self.histstruct.add_extglobalscores( extname, 
                            np.log(self.histstruct.fitfunc.pdf(scores_all)) )
        with self.tab: print('done')
            
            
class EvaluateTab:
    def __init__(self):
        ### initializer
        self.tab = ipw.Output()
        self.title = 'Evaluation'

    def refresh(self, histstruct=None, plotstyleparser=None):

        # check arguments
        if histstruct is None:
            with self.tab: print('[no histstruct found]')
            return
        if len(histstruct.get_runnbs())==0:
            with self.tab: print('[current histstruct is empty]')
            return
        self.histstruct = histstruct
        self.plotstyleparser = plotstyleparser
        self.test_set_widgets = []

        # add a button to add more test sets
        self.add_button = ipw.Button(description='Add test set')
        self.add_button.on_click(self.add_set)
        # make the layout for the test sets
        self.test_set_grid = ipw.GridBox(children=[self.add_button],
                                         layout=ipw.Layout(grid_template_columns='auto'))
        # add one test set for type good and one for type bad
        self.add_set( None, default_type='Good' )
        self.add_set( None, default_type='Bad' )

        # add widgets for score distribution
        self.score_dist_options_label = ipw.Label(value='Options for score plot')
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
        wtypes = None
        # make the actual OptionsFrame
        self.score_dist_options_frame = OptionsBox(labels=labels, types=wtypes, values=values,
                                            docurl=score_dist_docurl, autobool=True)

        # add widgets for roc curve
        self.roc_options_label = ipw.Label(value='Options for ROC curve')
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
        wtypes = None
        self.roc_options_frame = OptionsBox(labels=labels, types=wtypes, values=values,
                                            docurl=roc_docurl, autobool=True)

        # add widgets for confusion matrix
        self.cm_options_label = ipw.Label(value='Options for confusion matrix')
        # get available options
        cm_options_dict = get_args_dict(aeu.get_confusion_matrix)
        cm_docurl = get_docurl(aeu.get_confusion_matrix)
        # add meta arguments
        meta_args = {'make confusion matrix': True}
        cm_options_dict = {**meta_args, **cm_options_dict}
        # set the widget types
        labels = list(cm_options_dict.keys())
        values = list(cm_options_dict.values())
        wtypes = None
        self.cm_options_frame = OptionsBox(labels=labels, types=wtypes, values=values,
                                            docurl=cm_docurl, autobool=True)

        # add widgets for output json file
        self.json_label = ipw.Label(value='Options for output json file')
        json_options_dict = {'make json file': False,
                             'json filename': ''}
        # set the widget types
        labels = list(json_options_dict.keys())
        values = list(json_options_dict.values())
        wtypes = None
        self.json_options_frame = OptionsBox(labels=labels, types=wtypes, values=values, 
                                             docurl='no documentation available at the moment', autobool=True)
        
        # add widgets for 2D contour plots
        self.contour_options_label = ipw.Label(text='Options for fit plots')
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
                if key=='xaxtitlesize': 
                    contour_options_dict[key] = self.plotstyleparser.get_xaxtitlesize()
                elif key=='yaxtitlesize': 
                    contour_options_dict[key] = self.plotstyleparser.get_yaxtitlesize()
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
        wtypes = None
        # make the actual OptionsFrame
        self.contour_options_frame = OptionsBox(labels=labels, types=wtypes, values=values,
                                            docurl=contour_docurl, autobool=True)

        # add a button to start the evaluation
        self.evaluate_button = ipw.Button(description='Evaluate')
        self.evaluate_button.on_click(self.evaluate)
        
        # make the layout
        all_options_children = ([self.score_dist_options_frame.get_widget(),
                                 self.roc_options_frame.get_widget(),
                                 self.cm_options_frame.get_widget(),
                                 self.json_options_frame.get_widget(),
                                 self.contour_options_frame.get_widget()])
        self.all_options_grid = ipw.GridBox(children=all_options_children,
                           layout=ipw.Layout(grid_template_columns='auto '*len(all_options_children)))
        self.grid = ipw.GridBox(children=[self.test_set_grid,
                                          self.all_options_grid,
                                          self.evaluate_button],
                                layout=ipw.Layout(grid_template_row='auto auto auto'))
        with self.tab:
            clear_output()
            display(self.grid)

    def add_set(self, event, default_type='Good'):
        ### add one test set
        # initializations
        row = 0
        column = len(self.test_set_widgets)+1
        idx = len(self.test_set_widgets)
        # add widget for set selection
        selector = SelectorWidget(self.histstruct,
                                  title='Select evaluation set')
        # add dropdown for type
        type_box = ipw.Dropdown(options=['Good','Bad'], description='Type:')
        if default_type=='Bad': type_box.selected_index = 1
        # add text box for label
        label_text = ipw.Text(description='Label:')
        # add to the layout
        grid = ipw.GridBox(children=[selector.get_widget(),type_box,label_text],
                           layout=ipw.Layout(grid_template_rows='auto auto auto'))
        self.test_set_grid.children += tuple([grid])
        self.test_set_grid.layout.grid_template_columns += ' auto'
        # store the widgets
        self.test_set_widgets.append( { 'selector': selector,
                                        'type_box': type_box, 
                                        'label_text': label_text} )

    def check_all_selected(self):
        for el in self.test_set_widgets:
            if not el['selector'].valid: return False
        else: return True

    def get_scores(self, test_set_type):
        scores = []
        for el in self.test_set_widgets:
            if el['type_box'].value != test_set_type: continue
            scores.append(el['selector'].get_scores())
        if len(scores)==0:
            with self.tab: print('WARNING: there are no test sets with label {}'.format(test_set_type))
        return scores

    def get_globalscores(self, test_set_type):
        globalscores = []
        for el in self.test_set_widgets:
            if el['type_box'].value != test_set_type: continue
            globalscores.append(el['selector'].get_globalscores())
        if len(globalscores)==0:
            with self.tab: print('WARNING: there are no test sets with label {}'.format(test_set_type))
        return globalscores

    def get_labels(self, test_set_type):
        labels = []
        for el in self.test_set_widgets:
            if el['type_box'].value != test_set_type: continue
            labels.append(el['label_text'].value)
        return labels

    def evaluate(self, event):
        if not self.check_all_selected():
            with self.tab: print('ERROR: some test sets were declared but not defined')
            return
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
            with self.tab: print('WARNING: invalid json filename; not writing an output json.')
            do_json = False
        if do_json and not do_cm:
            with self.tab: print('WARNING: no working point was set.')
            working_point = None
        if do_json:
            json_fileext = os.path.splitext(json_filename)[1]
            if(json_fileext not in ['.json','.txt']):
                print('WARNING: unrecognized extension in json filename ({}),'.format(json_fileext)
                      +' replacing by .json')
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
                with self.tab: print('WARNING: too many bad test sets for available colors, putting all to red')
                badcolorist = ['red']*len(scores_bad_parts)
            if len(goodcolorlist)<len(scores_good_parts):
                with self.tab: print('WARNING: too many good test sets for available colors, putting all to blue')
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
            
            
class PlotLumisectionTab:
    
    def __init__(self):
        ### initializer
        self.tab = ipw.Output()
        self.title = 'Plot lumisection'
                    
    def refresh(self, histstruct=None, plotstyleparser=None):
        
        # check arguments
        if histstruct is None:
            with self.tab: print('[no histstruct found]')
            return
        if len(histstruct.get_runnbs())==0:
            with self.tab: print('[current histstruct is empty]')
            return
        self.histstruct = histstruct
        self.plotstyleparser = plotstyleparser
        self.inspect_set_selector = None
        self.refscore_set_selector = None
        self.ref_set_selector = None

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
            if labels[i]=='mode':
                wtypes[i] = ipw.Dropdown
                values[i] = ['ls','run']
            if labels[i]=='run':
                wtypes[i] = ipw.Dropdown
                values[i] = self.histstruct.get_runnbs_unique()
            if labels[i]=='lumisection':
                wtypes[i] = ipw.Dropdown
                values[i] = [1]
        # make the options frame
        docurl = get_docurl(self.histstruct.plot_ls)
        self.options_frame = OptionsBox(labels=labels, values=values, types=wtypes,
                                docurl=docurl, autobool=True)
        # special case: set available lumisections based on chosen run number
        runidx = labels.index('run')
        lsidx = labels.index('lumisection')
        self.run_box = self.options_frame.widgets[runidx]
        self.lumisection_box = self.options_frame.widgets[lsidx]
        self.run_box.observe(self.set_lsnbs, names='value')
        self.set_lsnbs(None)

        # add button to make the plot
        self.plot_button = ipw.Button(description='Plot')
        self.plot_button.on_click(self.plot)

        # add widgets for selecting inspect dataset
        label = 'Select masks for plotting\n(ignored when plotting single lumisection)'
        self.inspect_set_label = ipw.Label(value=label)
        self.inspect_set_selector = SelectorWidget(self.histstruct, 
                                                   set_selection=False, 
                                                   post_selection=False)

        # add widgets for selecting reference score dataset
        label = 'Select masks for reference scores\n(ignored when not plotting score comparison)'
        self.refscore_set_label = ipw.Label(value=label)
        self.refscore_set_selector = SelectorWidget(self.histstruct, 
                                                   set_selection=False, 
                                                   post_selection=False)

        # add widgets for selecting reference histogram dataset
        label = 'Select reference histograms\n(ignored when not plotting reference histograms)'
        self.ref_set_label = ipw.Label(value=label)
        self.ref_set_selector = SelectorWidget(self.histstruct)
        
        # make the layout
        children = ([self.options_frame.get_widget(), 
                     self.inspect_set_label,
                     self.inspect_set_selector.get_widget(),
                     self.refscore_set_label,
                     self.refscore_set_selector.get_widget(),
                     self.ref_set_label,
                     self.ref_set_selector.get_widget(),
                     self.plot_button])
        self.grid = ipw.GridBox(children=children,
                                   layout = ipw.Layout(grid_template_rows='auto '*len(children)))
        with self.tab:
            clear_output()
            display(self.grid)

    def set_lsnbs(self, event):
        runnb = int(self.run_box.value)
        runnbs = self.histstruct.get_runnbs()
        lsnbs = self.histstruct.get_lsnbs()
        lsnbs = lsnbs[np.nonzero(runnbs==runnb)]
        lsnbslist = [int(el) for el in lsnbs]
        self.lumisection_box.options = lsnbslist

    def get_reference_histograms(self):
        if not self.ref_set_selector.valid: return None
        return self.ref_set_selector.get_histograms()

    def get_inspect_masks(self):
        if not self.inspect_set_selector.valid: return None
        return self.inspect_set_selector.get_masks()

    def get_refscore_masks(self):
        if not self.refscore_set_selector.valid: return None
        return self.refscore_set_selector.get_masks()

    def plot(self, event):

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
            with self.tab: print(msg)
            # print number of lumisections to plot
            with self.tab: print('plotting {} lumisections...'.format(len(lsnbs)))
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
                with self.tab: print('WARNING: could not retrieve the global score'
                        +' for run {}, lumisection {};'.format(runnb, lsnb)
                        +' was it initialized?')
                logprob = None
            with self.tab:
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
                    with self.tab: print(msg)
                # initialize the figure
                ncols = min(4,len(self.histstruct.histnames))
                nrows = int(math.ceil(len(self.histstruct.histnames)/ncols))
                fig,axs = plt.subplots(nrows,ncols,figsize=(6*ncols,6*nrows),squeeze=False)
                # loop over histogram types
                for dim,histname in enumerate(self.histstruct.histnames):
                    thisscore = scorepoint[histname]
                    refscores = self.histstruct.get_scores( histname=histname, 
                                masknames=self.get_refscore_masks() )
                    _ = pu.plot_score_ls( thisscore, refscores, 
                            fig=fig, ax=axs[int(dim/ncols),dim%ncols],
                            thislabel='This LS', 
                            reflabel='Reference LS',
                            title=pu.make_text_latex_safe(histname),
                            xaxtitle='Model output score',
                            yaxtitle='Arbitrary units',
                            doshow=False,
                            nbins=200, normalize=True )
                plt.show(block=False)
        

class WelcomeTab:
    ### tab with welcome message, also used for testing new gui properties
 
    def __init__(self):
        ### initializer
        
        # initializations
        self.tab = ipw.Output()
        self.title = 'Welcome'
        # add widgets with a welcome message
        self.label = ipw.Label(value='Welcome to the ML4DQM GUI!')
        # add widgets to point to documentation
        docurl = 'https://lukalambrecht.github.io/ML4DQMDC-PixelAE/run/'
        self.docurlwidget = UrlWidget(docurl, text='Show link to documentation')
        with self.tab:
            display(self.label)
            display(self.docurlwidget.widget)

    
class ML4DQMGUI:
    ### base class initializing and managing all tabs in the gui

    def __init__(self):
        ### initializer

        # initializations
        self.histstruct = HistStruct.HistStruct()
        self.histstruct_filename = None
        self.plotstyleparser = PlotStyleParser.PlotStyleParser()
        self.plotstyle_filename = None
        self.alltabs = []

        # define tabs for creating a new histstruct
        self.welcome_tab = WelcomeTab()
        self.newhs_tab = NewHistStructTab( self.histstruct )
        self.addrunmasks_tab = AddRunMasksTab()
        self.addstatmasks_tab = AddStatMasksTab()
        self.addclassifiers_tab = AddClassifiersTab()
        self.newhstabs = []
        self.newhstabs.append( self.welcome_tab )
        self.newhstabs.append( self.newhs_tab )
        self.newhstabs.append( self.addrunmasks_tab )
        self.newhstabs.append( self.addstatmasks_tab )
        self.newhstabs.append( self.addclassifiers_tab )
        self.alltabs.append( self.newhstabs )
        self.newhstabwidget = ipw.Tab(children = [tab.tab for tab in self.newhstabs])
        for i,tab in enumerate(self.newhstabs):
            self.newhstabwidget.set_title(i, tab.title)
        self.newhstabwidget.observe(self.refresh, names='selected_index')

        # define tabs for loading and saving a HistStruct
        self.load_tab = LoadHistStructTab( self.load )
        self.save_tab = SaveHistStructTab()
        self.display_tab = DisplayHistStructTab()
        self.hsiotabs = []
        self.hsiotabs.append( self.load_tab )
        self.hsiotabs.append( self.save_tab )
        self.hsiotabs.append( self.display_tab )
        self.alltabs.append( self.hsiotabs )
        self.hsiotabwidget = ipw.Tab(children = [tab.tab for tab in self.hsiotabs])
        for i,tab in enumerate(self.hsiotabs):
            self.hsiotabwidget.set_title(i, tab.title)
        self.hsiotabwidget.observe(self.refresh, names='selected_index')

        # define tabs for preprocessing and resampling
        self.preprocessing_tab = PreprocessingTab()
        self.resampling_tab = ResamplingTab()
        self.procestabs = []
        self.procestabs.append( self.preprocessing_tab )
        self.procestabs.append( self.resampling_tab )
        self.alltabs.append( self.procestabs )
        self.procestabwidget = ipw.Tab(children = [tab.tab for tab in self.procestabs])
        for i,tab in enumerate(self.procestabs):
            self.procestabwidget.set_title(i, tab.title)
        self.procestabwidget.observe(self.refresh, names='selected_index')

        # define tabs for plotting
        self.load_plotstyle_tab = LoadPlotStyleTab()
        self.plotsets_tab = PlotSetsTab()
        self.plot_lumisection_tab = PlotLumisectionTab()
        self.plottabs = []
        self.plottabs.append( self.plotsets_tab )
        self.plottabs.append( self.plot_lumisection_tab )
        self.plottabs.append( self.load_plotstyle_tab )
        self.alltabs.append( self.plottabs )
        self.plottabwidget = ipw.Tab(children = [tab.tab for tab in self.plottabs])
        for i,tab in enumerate(self.plottabs):
            self.plottabwidget.set_title(i, tab.title)
        self.plottabwidget.observe(self.refresh, names='selected_index')

        # define tabs for classifier training, fitting and evaluation
        self.train_tab = TrainClassifiersTab()
        self.apply_classifiers_tab = ApplyClassifiersTab()
        self.fit_tab = FitTab()
        self.apply_fit_tab = ApplyFitTab()
        self.evaluate_tab = EvaluateTab()
        self.modeltabs = []
        self.modeltabs.append( self.train_tab )
        self.modeltabs.append( self.apply_classifiers_tab )
        self.modeltabs.append( self.fit_tab )
        self.modeltabs.append( self.apply_fit_tab )
        self.modeltabs.append( self.evaluate_tab )
        self.alltabs.append( self.modeltabs )
        self.modeltabwidget = ipw.Tab(children = [tab.tab for tab in self.modeltabs])
        for i,tab in enumerate(self.modeltabs):
            self.modeltabwidget.set_title(i, tab.title)
        self.modeltabwidget.observe(self.refresh, names='selected_index')

        # add all tabs
        self.tabwidget = ipw.Tab(children = [self.newhstabwidget,
                                             self.hsiotabwidget,
                                             self.procestabwidget,
                                             self.plottabwidget,
                                             self.modeltabwidget])
        self.tabwidget.set_title(0, 'New HistStruct')
        self.tabwidget.set_title(1, 'HistStruct I/O')
        self.tabwidget.set_title(2, 'Histogram processing')
        self.tabwidget.set_title(3, 'Plotting')
        self.tabwidget.set_title(4, 'Model training and evaluation')
        self.tabwidget.observe(self.refresh, names='selected_index')
        
    def refresh(self, event):
        ### the tab that is clicked
        new_superindex = self.tabwidget.selected_index
        new_index = self.tabwidget.children[new_superindex].selected_index
        new_tab = self.alltabs[new_superindex][new_index]
        # define default arguments
        kwargs = {}
        # list exceptions below
        if not hasattr(new_tab, 'refresh'): 
            return
        if isinstance(new_tab, AddRunMasksTab): kwargs['histstruct'] = self.histstruct
        if isinstance(new_tab, AddStatMasksTab): kwargs['histstruct'] = self.histstruct
        if isinstance(new_tab, AddClassifiersTab): kwargs['histstruct'] = self.histstruct
        if isinstance(new_tab, LoadHistStructTab): kwargs['external_load_function'] = self.load
        if isinstance(new_tab, SaveHistStructTab): kwargs['histstruct'] = self.histstruct
        if isinstance(new_tab, DisplayHistStructTab): kwargs['histstruct'] = self.histstruct
        if isinstance(new_tab, PreprocessingTab): kwargs['histstruct'] = self.histstruct
        if isinstance(new_tab, LoadPlotStyleTab): kwargs['plotstyleparser'] = self.plotstyleparser
        if isinstance(new_tab, PlotSetsTab): 
            kwargs['histstruct'] = self.histstruct
            kwargs['plotstyleparser'] = self.plotstyleparser
        if isinstance(new_tab, ResamplingTab): kwargs['histstruct'] = self.histstruct
        if isinstance(new_tab, TrainClassifiersTab): kwargs['histstruct'] = self.histstruct
        if isinstance(new_tab, FitTab):
            kwargs['histstruct'] = self.histstruct
            kwargs['plotstyleparser'] = self.plotstyleparser
        if isinstance(new_tab, ApplyFitTab): kwargs['histstruct'] = self.histstruct
        if isinstance(new_tab, EvaluateTab):
            kwargs['histstruct'] = self.histstruct
            kwargs['plotstyleparser'] = self.plotstyleparser
        if isinstance(new_tab, PlotLumisectionTab):
            kwargs['histstruct'] = self.histstruct
            kwargs['plotstyleparser'] = self.plotstyleparser
        # call the refresh function
        new_tab.refresh(**kwargs)
        
    def load(self, filename):
        self.histstruct = None
        self.histstruct = HistStruct.HistStruct.load( filename, verbose=True )
        self.histstruct_filename = filename
    
    def display(self):
        ### display the gui
        display(self.tabwidget)