import ipywidgets as ipw


class AddStatMasksWidget:
    
    def __init__(self, histstruct=None, applybutton=False):
        ### initializer
        # input arguments:
        # - histstruct: if specified, a list of available histogram names is extracted from it;
        #               else the statistics mask can only be applied on all histograms simultaneously.
        #               must be provided if applybutton is true.
        # - applybutton: boolean whether to add an apply button.
        
        # initializations
        self.histstruct = histstruct
        self.stat_mask_widgets = []
        
        # add widgets for stat mask additions
        self.add_stat_mask_button = ipw.Button(description='Add a statistics mask')
        self.add_stat_mask_button.on_click(self.add_stat_mask)
        name_label = ipw.Label(value='Name:')
        operator_label = ipw.Label(value='Operator:')
        apply_label = ipw.Label(value='Apply on:')
        threshold_label = ipw.Label(value='Threshold:')
        self.add_stat_mask_columns_box = ipw.GridBox(children=[
                                                    name_label,
                                                    operator_label,
                                                    apply_label,
                                                    threshold_label],
                                            layout=ipw.Layout(
                                                    width='60%',
                                                    grid_template_columns='auto auto auto auto',
                                                    )
        )
        self.apply_button = ipw.Button(description='Apply')
        self.apply_button.on_click(self.apply)
        if applybutton:
            self.add_stat_mask_box = ipw.GridBox(children=[
                                                    self.add_stat_mask_button, 
                                                    self.add_stat_mask_columns_box,
                                                    self.apply_button],
                                            layout=ipw.Layout(
                                                    grid_template_rows='auto auto auto',
                                                    )
            )
        else:
            self.add_stat_mask_box = ipw.GridBox(children=[
                                                    self.add_stat_mask_button, 
                                                    self.add_stat_mask_columns_box],
                                            layout=ipw.Layout(
                                                    grid_template_rows='auto auto',
                                                    )
            )
            
    def get_widget(self):
        ### return the GridBox widget, e.g. for usage and displaying in calling code
        return self.add_stat_mask_box
            
    def add_stat_mask(self, event):
        name_text = ipw.Text()
        operator_box = ipw.Dropdown(options=['>','<'])
        options = ['all']
        if self.histstruct is not None: options += self.histstruct.histnames
        apply_box = ipw.Dropdown(options=options)
        threshold_text = ipw.Text()
        self.add_stat_mask_columns_box.children += (name_text,operator_box,apply_box,threshold_text)
        self.stat_mask_widgets.append({'name_text':name_text,
                                            'operator_box': operator_box,
                                            'apply_box': apply_box,
                                            'threshold_text':threshold_text})

    def get_stat_masks(self):
        stat_masks = {}
        for el in self.stat_mask_widgets:
            name = el['name_text'].value
            operator = el['operator_box'].value
            applyon = el['apply_box'].value
            threshold = float(el['threshold_text'].value)
            stat_masks[name] = (operator,applyon,threshold)
        return stat_masks
    
    def apply(self, event):
        stat_masks = self.get_stat_masks()
        for name, (operator,applyon,threshold) in stat_masks.items():
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