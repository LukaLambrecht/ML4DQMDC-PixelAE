import ipywidgets as ipw
from UrlWidget import UrlWidget
from FileBrowser import FileBrowser


### help functions

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


class OptionsBox:

    def __init__(self, labels=None, types=None, values=None,
                        docurls=None, docurl=None, autobool=False):
        # input arguments:
        # - labels: list of strings with the names/labels of the options
        # - types: list of ipywidget types, defaults to ipywidgets.Text for each option
        # - values: list of options passed to each widget
        # note: individual elements of types and values can also be None,
        #       in which case these elements will be set to default
        # - docurls: list of urls to documentation per option
        # - docurl: url to documentation for the option collection
        # - autobool: automatically convert boolean arguments to a binary ipw.Combobox 
        #             (instead of Text entry)
        self.labels = []
        self.wtypes = []
        self.widgets = []
        self.docwidgets = []
        self.docwidget = None
        self.autobool = autobool
        self.grid = ipw.GridBox(children=[], layout=ipw.Layout(grid_template_columns='auto auto auto'))
        self.grid.layout.border = "1px dashed black"
        self.set_options( labels=labels, types=types, values=values, docurls=docurls, docurl=docurl )
        
    def get_widget(self):
        ### return the GridBox widget, e.g. for usage and displaying in the calling code.
        return self.grid

    def set_options(self, labels=None, types=None, values=None, docurls=None, docurl=None):
        ### set the options of an option frame
        # serves both as initializer and as resetter

        # check arguments
        if labels is None: 
            raise Exception('ERROR in OptionsBox initialization:'
                            +' argument "labels" must be specified.')
        if types is None: types = [ipw.Text]*len(labels)
        if values is None: values = [None]*len(labels)
        if docurls is None: docurls = [None]*len(labels)
        if( len(types)!=len(labels) 
                or len(values)!=len(labels)
                or len(docurls)!=len(labels) ):
            raise Exception('ERROR in OptionsBox initialization:'
                            +' argument lists have unequal lengths.')
        labels = list(labels) # explicit conversion from dict_keys or dict_values to list
        types = list(types) # explicit conversion from dict_keys or dict_values to list
        values = list(values) # explicit conversion from dict_keys or dict_values to list

        # additional argument parsing
        if self.autobool:
            for j in range(len(labels)):
                if is_bool(str(values[j])):
                    types[j] = ipw.Dropdown
                    values[j] = [bool(values[j]), bool(not values[j])]

        # clear current OptionsBox
        self.labels.clear()
        self.wtypes.clear()
        self.widgets.clear()
        self.docwidgets.clear()
        self.docwidget = None
        self.grid.children = []
        newchildren = []

        # set widgets
        nrows = len(labels)
        for i, (label, wtype, value, url) in enumerate(zip(labels, types, values, docurls)):
            # make label
            tklabel = ipw.Label(value=label)
            newchildren.append(tklabel)
            self.labels.append(tklabel)
            # make widget
            if wtype is None: wtype = ipw.Text
            widget = None
            # case 1: simple generic text box
            if wtype is ipw.Text:
                widget = ipw.Text()
                if value is not None:
                    widget.value = str(value)
                newchildren.append(widget)
            # case 2: file browser
            elif wtype is FileBrowser:
                widget = FileBrowser()
                newchildren.append(widget.get_widget())
            # case 3: dropdown with fixed options
            elif wtype is ipw.Dropdown:
                widget = ipw.Dropdown(options=[str(v) for v in value])
                newchildren.append(widget)
            # case 4: combobox with fixed options
            elif wtype is ipw.Combobox:
                widget = ipw.Combobox(options=[str(v) for v in value])
                newchildren.append(widget)
            else:
                raise Exception('ERROR in OptionsBox initialization:'
                                +' widget type {} not recognized'.format(wtype))
            self.widgets.append(widget)
            self.wtypes.append(wtype)
            # make doc widget
            if url is not None: urlwidget = UrlWidget(url, text='More info').get_widget()
            else: urlwidget = ipw.Label(value='')
            self.docwidgets.append(urlwidget)
            newchildren.append(urlwidget)

        # set link to documentation
        if docurl is not None:
            self.docwidget = UrlWidget(docurl, text='More info')
            newchildren.append(self.docwidget.get_widget())
            
        # make the layout
        self.grid.children = newchildren

    def get_dict(self):
        ### get the options of the current OptionsBox as a dictionary
        res = {}
        for label, wtype, widget in zip(self.labels, self.wtypes, self.widgets):
            key = label.value
            value = None
            # case 1: simple generic text box
            if wtype is ipw.Text:
                value = widget.value
            # case 2: file browser
            elif wtype is FileBrowser:
                value = widget.get_selected_files()
                if len(value)!=1:
                    msg = 'ERROR in OptionsBox: the number of files selected'
                    msg += ' for option {}'.format(label)
                    msg += ' is {}, while only 1 is supported,'
                    msg += ' returning empty file name.'.format(len(value))
                    print(msg)
                    value = ''
                else: value = value[0]
            # case 3: dropdown with fixed options
            elif wtype is ipw.Dropdown:
                value = widget.value
            # case 4: combobox with fixed options
            elif wtype is ipw.Combobox:
                value = widget.value
            else:
                raise Exception('ERROR in OptionsBox get_dict:'
                               +' no getter method implemented for widget type {}'.format(wtype))
            # basic parsing
            if value is None: value = ''
            if isinstance(value, str):
                value = value.strip(' \t\n')
                if is_int(value): value = int(value)
                elif is_float(value): value = float(value)
                elif is_bool(value): value = to_bool(value)
                elif value=='None': value = None
                elif value=='': value = None
            res[key] = value
        return res