import ipywidgets as ipw
import os
import sys


class FileBrowser(object):
    # copied and modified from here: https://gist.github.com/DrDub/6efba6e522302e43d055
    
    def __init__(self, initialdir=None, filetypes=None):
        # initialization of variables
        self.initialdir = os.getcwd() if initialdir is None else initialdir
        self.path = self.initialdir
        self.filetypes = filetypes
        self.selected_files = []
        # initialization of widgets
        self.selectedfiles_text = ipw.Textarea(description='Selected file(s):',
                                               layout=ipw.Layout(width='80%'),
                                               style= {'description_width': 'initial'})
        self.button_label = ipw.Label('Actions:')
        self.selectbutton = ipw.Button(description='Select')
        self.selectbutton.on_click(self.select)
        self.clearbutton = ipw.Button(description='Clear')
        self.clearbutton.on_click(self.clear)
        self.cancelbutton = ipw.Button(description='Cancel')
        self.cancelbutton.on_click(self.cancel)
        self.button_box = ipw.GridBox(children=[self.button_label,self.selectbutton,self.clearbutton,self.cancelbutton],
                                      layout=ipw.Layout(grid_template_columns='100px 200px 200px 200px'))
        self.accordion = ipw.Accordion(children=[], titles=(''))
        self.box = ipw.GridBox(children=[self.selectedfiles_text,self.button_box,self.accordion],
                                layout=ipw.Layout(
                                          width='50%',
                                          grid_template_rows='auto auto auto',
                                )
        )
        self.box.layout.border = "1px solid black"
        # update widgets for current (initial) directory
        self._update()
        
    def get_widget(self):
        ### return the GridBox widget, e.g. for usage and displaying in the calling code.
        return self.box
    
    def get_selectbutton(self):
        ### return the select button, e.g. for overwriting its on_click function in the calling code
        return self.selectbutton
    
    def overwrite_selectbutton(self, func, extend=True):
        ### overwrite the on_click function for the select button
        # if extend, the default behaviour is extended with the new custom function, else it is replaced.
        if not extend: self.selectbutton._click_handlers.callbacks = []
        self.selectbutton.on_click(func)
    
    def get_cancelbutton(self):
        ### return the cancel button, e.g. for overwriting its on_click function in the calling code
        return self.cancelbutton
    
    def overwrite_cancelbutton(self, func, extend=True):
        ### overwrite the on_click function for the cancel button
        # if extend, the default behaviour is extended with the new custom function, else it is replaced.
        if not extend: self.cancelbutton._click_handlers.callbacks = []
        self.cancelbutton.on_click(func)
    
    def get_current_path(self):
        ### return the current path (the content of which is currently displayed)
        return self.path
    
    def get_selected_files(self):
        ### return currently selected files and/or directories
        return self.selected_files
    
    def get_currentselection(self):
        ### return current files indicated in the selector
        selected = []
        for cb, bt in zip(self.checkboxes, self.buttons):
            if( cb.value ): selected.append(os.path.join(self.path,bt.description))
        return selected
    
    def select(self, event):
        ### default behaviour of select button
        self.selected_files = self.get_currentselection()
        selstr = '\n'.join(self.selected_files)
        self.selectedfiles_text.value = selstr
        self.accordion.selected_index = None
        
    def clear(self, event):
        ### default behaviour of clear button
        # clear current selection but do not close the widget
        self.selected_files = []
        self.selectedfiles_text.value = ''
        
    def cancel(self, event):
        ### default behaviour of cancel button
        self.checkboxes = []
        self.buttons = []
        self.box.close()
        
    def _update_files(self):
        ### make a list of files and directories in current directory
        # note: for internal use only
        # returns:
        #   the result is stored in the self.files and self.dirs attributes,
        #   which are then used e.g. in _update()
        #   the widgets are not affected by this function, use _update() instead.
        self.files = list()
        self.dirs = list()
        if(os.path.isdir(self.path)):
            for f in os.listdir(self.path):
                ff = os.path.join(self.path,f)
                if os.path.isdir(ff):
                    self.dirs.append(f)
                else:
                    if self.filetypes is not None:
                        if os.path.splitext(f)[1] not in self.filetypes: continue
                    self.files.append(f)
    
    def _update(self):
        ### update the widgets
        # note: for internal use only
        # returns:
        #   the view is updated to the content of the current path (as stored in self.path)
        
        def on_click(b):
            if b.description == '..':
                self.path = os.path.split(self.path)[0]
            else:
                self.path = os.path.join(self.path,b.description)
            self._update()
        
        self._update_files()
        self.checkboxes = []
        self.buttons = []
        entries = []
        # define button layout
        blayout = ipw.Layout(width='100%')
        # first make an entry to go one step back
        checkbox = ipw.Checkbox(value=False, disabled=True)
        button = ipw.Button(description='..', background_color='#d0d0ff',
                            layout=blayout)
        button.on_click(on_click)
        self.checkboxes.append(checkbox)
        self.buttons.append(button)
        entries.append(checkbox)
        entries.append(button)
        # make entries for all directories
        for f in self.dirs:
            checkbox = ipw.Checkbox(value=False)
            button = ipw.Button(description=f, background_color='#d0d0ff',
                                layout=blayout)
            button.on_click(on_click)
            self.checkboxes.append(checkbox)
            self.buttons.append(button)
            entries.append(checkbox)
            entries.append(button)
        # make entries for all files
        for f in self.files:
            checkbox = ipw.Checkbox(value=False)
            button = ipw.Button(description=f, disabled=True,
                                layout=blayout)
            self.checkboxes.append(checkbox)
            self.buttons.append(button)
            entries.append(checkbox)
            entries.append(button)
        box = ipw.GridBox(children=entries,
                          layout=ipw.Layout(grid_template_columns='200px auto'))
        self.accordion.children = [box]