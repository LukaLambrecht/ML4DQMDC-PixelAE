#############################
# some tests for hist_utils #
#############################

# external modules
import sys

# local modules
sys.path.append('../utils')
import hist_utils
from hist_utils import *

# test for smoothhists

if True:
    # check correctness of output on simple 1D array
    hists = np.array([[1.,3.,3.,4.,2.,1.]])
    halfwindow = 1
    weights = np.array([1,1,0])
    smhists = smoothhists( hists, halfwindow=halfwindow, weights=weights )
    print(hists)
    print(smhists)

if True:
    # check correctness of output on a simple 2D array
    hists = np.array([  [[1.,3.,3.,4.,2.,1.],
                        [2.,1.,4.,5.,1.,2.],
                        [0.,3.,0.,1.,2.,1.]] ])
    halfwindow = (1,1)
    smhists = smoothhists( hists, halfwindow=halfwindow )
    print(hists)
    print(smhists)
