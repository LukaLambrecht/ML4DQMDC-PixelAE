# fitfunction  
**abstract base class for all fit functions**  
all other fit functions inherit from fitfunction and overload its functions  
no concrete fitting procedure is implemented,  
but some basic checks on dimensionality are performed  
  
## __init__(self,points)  
(no valid documentation found)  
  
## pdf(self,points)  
**get the pdf (probability density function) value at given points**  
points is a 2D numpy array of shape (npoints,ndims)  
the output is a 1D array of shape (npoints)  
  
## pdfgrid(self,grid)  
**get the pdf (probability density function) value at a given grid**  
(only applicable to 2D case!)  
grid is a np array of shape (nx,ny,2)  
containing the x- and y-values in its first and second depth-wise dimension respectively.  
the grid is typically (but not necessarily) created via:  
x,y = np.mgrid[<xrange>,<yrange>]  
grid = np.dstack(x,y)  
  
# lognormal(fitfunction)  
(no valid documentation found)  
  
## __init__(self,points)  
points is a np array of shape (npoints,ndims)  
  
## pdf(self,points)  
(no valid documentation found)  
  
## pdfgrid(self,grid)  
(no valid documentation found)  
  
# exponential(fitfunction)  
(no valid documentation found)  
  
## __init__(self,points)  
points is a np array of shape (npoints,ndims)  
  
## pdf(self,points)  
(no valid documentation found)  
  
## pdfgrid(self,grid)  
(no valid documentation found)  
  
# seminormal(fitfunction)  
this is not strictly speaking a probability distribution,  
only the first quadrant of the result of fitting a normal distribution  
to the data + its mirror image wrt the origin  
  
## __init__(self,points=[])  
(no valid documentation found)  
  
## pdf(self,points)  
(no valid documentation found)  
  
## pdfgrid(self,grid)  
(no valid documentation found)  
  
## save(self,path)  
(no valid documentation found)  
  
## load(self,path)  
(no valid documentation found)  
  
# gaussiankde(fitfunction)  
wrapper for scipy.stats.gaussian_kde (gaussian kernel density estimation)  
  
## __init__(self,points=[],bw='default')  
(no valid documentation found)  
  
## pdf(self,points)  
(no valid documentation found)  
  
## pdfgrid(self,grid)  
(no valid documentation found)  
  
# vecdist(moments,index)  
does not work well if there are outliers which dominate the distance  
  
# costhetadist(moments,index)  
works more or less but not all bad points have small values,  
allows to identify problematic regions but not individual LS  
  
# avgnndist(moments,index,nn)  
seems to work well for the runs tested!  
  
# getavgnndist(hists,nmoments,xmin,xmax,nbins,nneighbours)  
(no valid documentation found)  
  
# filteranomalous(df,nmoments=3,rmouterflow=True,rmlargest=0.,doplot=True,)  
(no valid documentation found)  
  
