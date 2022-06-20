import ipywidgets as ipw
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from copy import copy


class HistStructViewer(object):
    ### a class for viewing the contents of a HistStruct;
    # simple text-based for now, extend to more graphical view later
    
    def __init__(self, histstruct=None):
        self.histstruct = histstruct
        self.box = ipw.GridBox(children=[])
        self.box.layout.border = "1px solid black"
        # update widgets for current (initial) directory
        if histstruct is not None: self.refresh(histstruct)
        
    def get_widget(self):
        ### return the GridBox widget, e.g. for usage and displaying in the calling code.
        return self.box      
    
    def refresh(self, histstruct):
        ### update the info
        
        # make the row titles (run and lumisection numbers)
        self.runnbs = histstruct.get_runnbs()
        self.lsnbs = histstruct.get_lsnbs()
        self.rows = ['Run {}, LS {}'.format(self.runnbs[i],self.lsnbs[i]) for i in range(len(self.runnbs))]
        
        # make the column titles (masks, maybe extend later)
        masks = histstruct.get_masknames()
        self.columns = masks[:]
        
        # get the info
        passmasks = []
        for c in self.columns: passmasks.append(histstruct.pass_masks([c]))
        self.passmasks = np.transpose(np.array(passmasks))
        
        # update the view (option 1):
        #self.refresh_gridbox()
        
        # make a plot (option 2):
        self.make_plot()
        
    def refresh_gridbox(self):
        ### update the view
        # works but runs rather slow... 
        
        # make the header
        labelgrid = []
        labelgrid.append(ipw.Label(value='Mask:'))
        for j,c in enumerate(self.columns): 
            text = '{}\n({}/{})'.format(c,np.sum(self.passmasks[:,j]),len(self.runnbs))
            labelgrid.append(ipw.Label(value=text))
            
        # fill the grid
        if len(self.rows)>1000:
            pass
        else:
            for i,row in enumerate(self.rows):
                labelgrid.append(ipw.Label(value=row))
                for j,c in enumerate(self.columns): labelgrid.append(ipw.Label(value=str(self.passmasks[i,j])))
       
        # modify the layout
        self.box.children = labelgrid
        self.box.layout = ipw.Layout(grid_template_columns='auto '*(len(self.columns)+1))
        
    def make_plot(self):
        ### make a plot (visual representation of the grid)
        # note: does not work yet, gives error that figures are too large...

        # determine aspect ratio and figure size
        nrows,ncols = self.passmasks.shape
        aspect_ratio = nrows/ncols
        figsize = (12,12*aspect_ratio/100.)
        # initialize the figure
        fig,ax = plt.subplots(figsize=figsize)    
        # make the color scale
        vmin = np.amin(0)
        vmax = np.amax(1)
        my_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
        # old:
        #my_cmap = copy(mpl.cm.get_cmap('cool'))
        #my_cmap.set_under(color='white')
        # new:
        my_cmap = mpl.colors.LinearSegmentedColormap.from_list('rg',["r", "limegreen"], N=2) 
        # make the plot
        ax.imshow(self.passmasks, interpolation='none', cmap=my_cmap, norm=my_norm, aspect='auto')
        # write the bin values
        # works but slow...
        #for i in range(ncols):
        #    for j in range(nrows):
        #        txt = str(self.passmasks[j,i])
        #        ax.text(i, j, txt, horizontalalignment='center',
        #                verticalalignment='center', fontsize=8)
        # find indices where run changes
        run_change_inds = [idx for idx in range(len(self.runnbs)) 
                           if (idx>0 and idx<len(self.runnbs)-1 and self.runnbs[idx]!=self.runnbs[idx-1])]
        # draw vertical lines
        xcoords = [cnum+0.5 for cnum in range(len(self.columns)-1)]
        ax.vlines(xcoords, -0.5, len(self.rows)-0.5, color='k', linestyles='dashed')
        # draw horizontal lines
        ycoords = [rnum-0.5 for rnum in run_change_inds]
        ax.hlines(ycoords,-0.5, len(self.columns)-0.5, color='k', linestyle='dashed')
        # set the x-axis tick marks
        xtickpos = np.linspace(0,len(self.columns)-1,num=len(self.columns))
        xtickvalues = [el.replace('_',' ') for el in self.columns]
        ax.set_xticks(xtickpos)
        ax.set_xticklabels(xtickvalues)
        # set the y-axis tick marks
        row_inds = [0]
        row_inds += run_change_inds
        row_inds += ([i for i in range(len(self.rows)) 
                      if (i>0 and i<len(self.rows)-1 and self.lsnbs[i]%100==0) ])
        row_inds += [len(self.rows)-1]
        row_inds.sort()
        ytickvalues = [self.rows[indx].replace('_',' ') for indx in row_inds]
        ytickpos = row_inds
        #ax.set_xlabel(xdimtitle)
        #ax.set_ylabel(ydimtitle)
        ax.set_yticks(ytickpos)
        ax.set_yticklabels(ytickvalues)
        # save the figure
        #thisoutputfile = os.path.splitext(outputfile)[0]+'_{}.png'.format(name)
        #fig.savefig(thisoutputfile)
        plt.show(block=False)