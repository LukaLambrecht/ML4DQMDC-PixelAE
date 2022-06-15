import sys
import os
import numpy as np
import ipywidgets as ipw
import importlib

import OptionsBox
importlib.reload(OptionsBox)
from OptionsBox import OptionsBox

sys.path.append('../../utils')
import hist_utils as hu


class SelectorWidget:

    def __init__(self, histstruct, 
                 title=None,
                 mask_selection=True,
                 set_selection=True,
                 post_selection=True,
                 allow_multi_mask=True,
                 allow_multi_set=False,
                 model_selection=True):
        self.histstruct = histstruct
        self.histograms = None
        self.modelname = None
        self.masks = None
        self.sets = None
        self.scores = None
        self.globalscores = None
        self.randoms = -1
        self.first = -1
        self.partitions = -1
        self.valid = False

        # add widgets for choosing masks
        self.masks_label = ipw.Label(value='Choose masks')
        mask_options = self.histstruct.get_masknames()
        if len(mask_options)==0: mask_options = ['no masks available']
        if allow_multi_mask:
            self.masks_listbox = ipw.SelectMultiple(options=mask_options)
        else:
            self.masks_listbox = ipw.Select(options=mask_options)
            self.masks_listbox.value = None
        
        # add widgets for choosing a (resampled) set directly
        self.sets_label = ipw.Label(value='Choose sets')
        set_options = list(self.histstruct.exthistograms.keys())
        if len(set_options)==0: set_options = ['no sets available']
        if allow_multi_set:
            self.sets_listbox = ipw.SelectMultiple(options=set_options)
        else:
            self.sets_listbox = ipw.Select(options=set_options)
            self.sets_listbox.value = None

        # add widgets for randoms, first, or averages
        self.other_options_label = ipw.Label(value='Other options')
        options = {'randoms':-1, 'first':-1, 'partitions':-1}
        self.other_options_obj = OptionsBox(labels=list(options.keys()),
                                            values=list(options.values()))
        self.other_options_box = self.other_options_obj.get_widget()
        
        # add widgets for choice of model
        self.model_label = ipw.Label(value='Model')
        self.model_box = ipw.Dropdown(description='Model', options=['None']+histstruct.modelnames)

        # add widget for selection
        self.select_button = ipw.Button(description='Select')
        self.select_button.on_click(self.select)
        self.clear_button = ipw.Button(description='Clear')
        self.clear_button.on_click(self.clear)
        buttonbox = ipw.GridBox(children=[self.select_button,self.clear_button],
                                layout=ipw.Layout(grid_template_columns='auto auto'))
        
        # make the layout
        children = []
        if mask_selection: children += [self.masks_label, self.masks_listbox]
        if set_selection: children += [self.sets_label, self.sets_listbox]
        if post_selection: children += [self.other_options_label, self.other_options_box]
        if model_selection: children += [self.model_label, self.model_box]
        children += [buttonbox]
        
        self.grid = ipw.GridBox(children=children,
                                layout=ipw.Layout(
                                          grid_template_rows='auto '*len(children),
                                )
        )
        self.grid.layout.border = "1px solid black"
        
        # wrap the gridbox in an accordion
        self.accordion = ipw.Accordion(children=[self.grid])
        if title is not None: self.accordion.set_title(0, title)
        self.accordion.selected_index = None
        
    def get_widget(self):
        ### return the GridBox widget, e.g. for usage and displaying in the calling code.
        return self.accordion
    
    def get_selectbutton(self):
        ### return the select button, e.g. for overwriting its on_click function in the calling code
        return self.selectbutton
    
    def overwrite_selectbutton(self, func, extend=True):
        ### overwrite the on_click function for the select button
        # if extend, the default behaviour is extended with the new custom function, else it is replaced.
        if not extend: self.select_button._click_handlers.callbacks = []
        self.select_button.on_click(func)

    def get_masks(self):
        ### get currently selected masks
        # warning: do not use after selection window has been closed,
        #          use self.masks for that!
        masks = self.masks_listbox.value
        if isinstance(masks,str): masks = [sets]
        return masks

    def get_sets(self):
        ### get currently selected sets
        # warning: do not use after selection window has been closed,
        #          use self.sets for that!
        sets = self.sets_listbox.value
        if isinstance(sets,str): sets = [sets]
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
    
    def get_scores_array(self):
        ### get scores of currently selected histograms in array format
        if self.scores is None:
            print('WARNING: the current histogram selection does not contain scores.'
                    +' Did you properly evaluate a model on the selected set?')
            return None
        scores_array = []
        for histname in self.histstruct.histnames:
            scores_array.append(self.scores[histname])
        scores_array = np.transpose(np.array(scores_array))
        return scores_array

    def get_globalscores(self):
        ### get global scores of currently selected lumisections
        if self.globalscores is None:
            print('WARNING: the current lumisection selection does not contain global scores.'
                    +' Did you properly evaluate a model on the selected set?')
        return self.globalscores

    def select(self, event):
        ### default selection behaviour
        # set the histograms based on the current user settings

        # get masks and/or sets
        masks = self.get_masks()
        do_masks = (masks is not None and bool(len(masks)>0))
        sets = self.get_sets()
        do_sets = (sets is not None and bool(len(sets)>0))
        if( not do_masks and not do_sets ):
            print('ERROR: you must select either at least one mask or a training set.')
            return
        if( do_masks and do_sets ):
            print('ERROR: you cannot select both masks and sets.')
            return
        
        # get other options
        options = self.other_options_obj.get_dict()
        nspecified = len([val for val in list(options.values()) if val>0])
        if nspecified>1:
            raise Exception('ERROR: you can only specifiy maximum one option'
                            +' of the list {}'.format(list(options.keys)))
        randoms = options['randoms']
        first = options['first']
        partitions = options['partitions']
        
        # get model name
        modelname = self.model_box.value
        if modelname=='None': modelname = None
        
        # get all numbers
        extname = None
        if do_sets: extname = sets[0]
        
        res = self.histstruct.get_histogramsandscores( modelname=modelname,
                                                       extname=extname, 
                                                       masknames=masks, 
                                                       nrandoms=randoms, 
                                                       nfirst=first )
        self.histograms = res['histograms']
        self.scores = res['scores']
        self.globalscores = res['globalscores']
        self.modelname = modelname
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
                
        # set the selection to valid and close the accordion
        self.valid = True
        self.accordion.selected_index = None
                
    def clear(self, event):
        ### clear current selection (both internally and in widgets)
        self.histograms = None
        self.masks = None
        self.sets = None
        self.scores = None
        self.globalscores = None
        self.randoms = -1
        self.first = -1
        self.partitions = -1
        self.masks_listbox.value = () if isinstance(self.masks_listbox,ipw.SelectMultiple) else None
        self.sets_listbox.value = () if isinstance(self.sets_listbox,ipw.SelectMultiple) else None
        options = {'randoms':-1, 'first':-1, 'partitions':-1}
        self.other_options_obj.set_options(labels=list(options.keys()),
                                           values=list(options.values()))