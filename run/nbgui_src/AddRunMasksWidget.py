import ipywidgets as ipw


class AddRunMasksWidget:
    
    def __init__(self, histstruct=None, applybutton=False):
        ### initializer
        # input arguments:
        # - histstruct: if specified, a list of available runs is extracted from it;
        #               else the text box is left blank and can take any value.
        #               must be provided if applybutton is true.
        # - applybutton: boolean whether to add an apply button.
        
        # initializations
        self.histstruct = histstruct
        self.run_mask_widgets = []
        
        # add widgets for run mask addition
        self.add_run_mask_button = ipw.Button(description='Add a run mask')
        self.add_run_mask_button.on_click(self.add_run_mask)
        name_label = ipw.Label(value='Name:')
        run_label = ipw.Label(value='Run number:')
        self.add_run_mask_columns_box = ipw.GridBox(children=[
                                                    name_label, 
                                                    run_label],
                                            layout=ipw.Layout(
                                                    width='30%',
                                                    grid_template_columns='auto auto',
                                                    )
        )
        self.apply_button = ipw.Button(description='Apply')
        self.apply_button.on_click(self.apply)
        if applybutton:
            self.add_run_mask_box = ipw.GridBox(children=[
                                                    self.add_run_mask_button, 
                                                    self.add_run_mask_columns_box,
                                                    self.apply_button],
                                            layout=ipw.Layout(
                                                    grid_template_rows='auto auto auto',
                                                    )
            )
        else:
            self.add_run_mask_box = ipw.GridBox(children=[
                                                    self.add_run_mask_button, 
                                                    self.add_run_mask_columns_box],
                                            layout=ipw.Layout(
                                                    grid_template_rows='auto auto',
                                                    )
            )
        
    def get_widget(self):
        ### return the GridBox widget, e.g. for usage and displaying in calling code
        return self.add_run_mask_box
            
    def add_run_mask(self, event):
        name_text = ipw.Text()
        if self.histstruct is None:
            run_text = ipw.Text()
        else:
            run_text = ipw.Dropdown(options=[str(r) for r in self.histstruct.get_runnbs_unique()])
        self.add_run_mask_columns_box.children += (name_text,run_text)
        self.run_mask_widgets.append({'name_text':name_text,'run_text':run_text})

    def get_run_masks(self):
        run_masks = {}
        for el in self.run_mask_widgets:
            name = el['name_text'].value            
            run = int(el['run_text'].value)
            run_masks[name] = run
        return run_masks
        
    def apply(self, event):
        run_masks = self.get_run_masks()
        for name, run in run_masks.items():
            json = {str(run):[[-1]]}
            self.histstruct.add_json_mask( name, json )