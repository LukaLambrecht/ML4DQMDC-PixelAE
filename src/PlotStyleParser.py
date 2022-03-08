#########################################################
# class for reading a json file with plot style options #
#########################################################

# import external modules
import sys
import os
import json

# import local modules
# (none so far)


class PlotStyleParser:

    def __init__(self, jsonfile=None):
        self.plotstyledict = {}
        if jsonfile is not None: self.load(jsonfile)
        
    def load(self, jsonfile):
        if( jsonfile is None ): return
        # read json file
        if not os.path.exists(jsonfile):
            print('WARNING in PlotStyleParser.load:'
                    +' plot style json file {} does not seem to exist;'.format(jsonfile)
                    +' initializing empty plot style parser.')
            return
        with open(jsonfile,'r') as f:
            plotstyledict = json.load(f)
        # do some checks on the dict
        pass
        # set the plotstyledict for this object
        self.plotstyledict = plotstyledict

    def get_general_plot_options(self):
        if( 'general_plot_options' not in self.plotstyledict.keys() ): return None
        return self.plotstyledict['general_plot_options']

    def get_general_plot_option(self, attribute, histname=None):
        attrs = self.get_general_plot_options()
        if( attrs is None ): return None
        if( attribute not in attrs.keys() ): return None
        attr = attrs[attribute]
        if( not isinstance(attr,dict) or histname is None ): return attr
        if( histname not in attr.keys() ): return None
        return attr[histname]

    def get_title(self, histname=None):
        return self.get_general_plot_option('titledict', histname=histname)

    def get_titlesize(self):
        return self.get_general_plot_option('titlesize')

    def get_xaxtitle(self, histname=None):
        return self.get_general_plot_option('xaxtitledict', histname=histname)

    def get_xaxtitlesize(self):
        return self.get_general_plot_option('xaxtitlesize')

    def get_physicalxax(self):
        return self.get_general_plot_option('physicalxax')

    def get_yaxtitle(self, histname=None):
        return self.get_general_plot_option('yaxtitledict', histname=histname)

    def get_yaxtitlesize(self):
        return self.get_general_plot_option('yaxtitlesize')
    
    def get_ymaxfactor(self):
        return self.get_general_plot_option('ymaxfactor')

    def get_extratext(self, histname=None):
        return self.get_general_plot_option('extratextdict', histname=histname)

    def get_extratextsize(self):
        return self.get_general_plot_option('extratextsize')

    def get_legendsize(self):
        return self.get_general_plot_option('legendsize')

    def get_extracmstext(self):
        return self.get_general_plot_option('extracmstext')

    def get_cmstextsize(self):
        return self.get_general_plot_option('cmstextsize')

    def get_condtext(self):
        return self.get_general_plot_option('conditionstext')

    def get_condtextsize(self):
        return self.get_general_plot_option('conditionstextsize')