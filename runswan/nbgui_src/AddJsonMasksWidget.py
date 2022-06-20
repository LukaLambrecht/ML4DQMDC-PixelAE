import ipywidgets as ipw
import os
import sys
sys.path.append('../../utils')
import json_utils as jsonu


class AddJsonMasksWidget:
    
    def __init__(self, histstruct=None, applybutton=False):
        ### initializer
        # input arguments:
        # - histstruct: must be provided if applybutton is true.
        # - applybutton: boolean whether to add an apply button.
        
        # initializations
        self.histstruct = histstruct
        self.json_mask_widgets = []
        
        # add widgets for json mask addition
        self.add_json_mask_button = ipw.Button(description='Add a json mask')
        self.add_json_mask_button.on_click(self.add_json_mask)
        name_label = ipw.Label(value='Name:')
        file_label = ipw.Label(value='File:')
        self.add_json_mask_columns_box = ipw.GridBox(children=[
                                                    name_label, 
                                                    file_label],
                                            layout=ipw.Layout(
                                                    width='30%',
                                                    grid_template_columns='auto auto',
                                                    )
        )
        self.apply_button = ipw.Button(description='Apply')
        self.apply_button.on_click(self.apply)
        if applybutton:
            self.add_json_mask_box = ipw.GridBox(children=[
                                                    self.add_json_mask_button, 
                                                    self.add_json_mask_columns_box,
                                                    self.apply_button],
                                            layout=ipw.Layout(
                                                     grid_template_rows='auto auto auto')
            )
        else:
            self.add_json_mask_box = ipw.GridBox(children=[
                                                    self.add_json_mask_button, 
                                                    self.add_json_mask_columns_box],
                                            layout=ipw.Layout(
                                                     grid_template_rows='auto auto')
            )
           
    def get_widget(self):
        ### return the GridBox widget, e.g. for usage and displaying in calling code
        return self.add_json_mask_box
    
    def add_json_mask(self, event):
        name_text = ipw.Text()
        file_loader = ipw.FileUpload(
                        accept='.json',
                        multiple=False
                        )
        self.add_json_mask_columns_box.children += (name_text,file_loader)
        self.json_mask_widgets.append({'name_text':name_text,'file_loader':file_loader})

    def get_json_masks(self):
        json_masks = {}
        for el in self.json_mask_widgets:
            name = el['name_text'].value
            filename = el['file_loader'].value[0]['name']
            json_masks[name] = filename
        return json_masks

    def apply(self, event):
        json_masks = self.get_json_masks()
        for name, filename in json_masks.items():
            json = jsonu.loadjson( filename )
            self.histstruct.add_json_mask( name, json )