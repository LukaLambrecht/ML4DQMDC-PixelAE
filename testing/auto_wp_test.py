######################################################################
# scripts to test the methods for automatic working point extraction #
######################################################################

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../utils')
import plot_utils as pu
import autoencoder_utils as aeu

if __name__=='__main__':

    # make an artificial score distribution for signal and background
    # new numpy syntax:
    #rng = np.default_rng()
    # old:
    rng = np.random
    size = 1000
    signal_scores = rng.normal(loc=1, scale=0.5, size=size)
    background_scores = rng.normal(loc=0, scale=0.5, size=size)
    scores = np.concatenate((signal_scores, background_scores))
    labels = np.concatenate((np.ones(len(signal_scores)), np.zeros(len(background_scores))))
    pu.plot_score_dist( scores, labels, nbins=100, normalize=True )
    plt.show(block=False)

    # get the optimal working point automatically
    aeu.get_wp( scores, labels, method='maxauc', doplot=True )
    plt.show(block=True)
