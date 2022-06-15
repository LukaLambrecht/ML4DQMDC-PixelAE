import ipywidgets as ipw
import webbrowser


class UrlWidget:
    ### contains a label with a clickable link

    def __init__(self, url, text=None):
        if text is None: text = url
        self.url = url
        self.widget = ipw.Button(description=text)
        self.widget.on_click(self.openurl)
        
    def get_widget(self):
        ### return the GridBox widget, e.g. for usage and displaying in the calling code.
        return self.widget

    def openurl(self, event):
        # open a webbrowser on the requested url
        # original version (does not work on remote server)
        #webbrowser.open_new(self.url)
        # new version: print clickable link
        print(self.url)