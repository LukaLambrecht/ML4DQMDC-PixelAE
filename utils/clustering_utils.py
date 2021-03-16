#!/usr/bin/env python
# coding: utf-8



### imports 

# external modules
from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde
import numpy as np

# local modules




### functionality for fitting a function to point multidimensional point clouds

class fitfunction:
    ### abstract base class for all fit functions
    # all other fit functions inherit from fitfunction and overload its functions
    # no concrete fitting procedure is implemented,
    # but some basic checks on dimensionality are performed
    
    # constructor
    def __init__(self,points):
        self.npoints = points.shape[0]
        self.ndims = points.shape[1]
        
    # get pdf at points
    def pdf(self,points):
        ### get the pdf (probability density function) value at given points
        # points is a 2D numpy array of shape (npoints,ndims)
        # the output is a 1D array of shape (npoints)
        pshape = points.shape
        if not len(pshape)==2:
            print('wrong input shape')
            return False
        return True
    
    def pdfgrid(self,grid):
        ### get the pdf (probability density function) value at a given grid
        # (only applicable to 2D case!)
        # grid is a np array of shape (nx,ny,2)
        # containing the x- and y-values in its first and second depth-wise dimension respectively.
        # the grid is typically (but not necessarily) created via:
        # x,y = np.mgrid[<xrange>,<yrange>]
        # grid = np.dstack(x,y)
        gshape = grid.shape
        if not (self.ndims==2 and len(gshape)==3 and gshape[2]==2):
            print('wrong input shape')
            return False
        return True
        
class lognormal(fitfunction):
    
    # parameters
    # mean: multidim mean of underlying normal
    # cov: multidim covariance matrix of underlying normal
    # mvn: scipy.stats multivariate_normal object
    
    # constructor
    def __init__(self,points):
        # points is a np array of shape (npoints,ndims)
        super().__init__(points)
        # transform the data from assumed log-normal to normal
        points_log = np.log(points)
        # fit a total multivariate normal distribution
        self.mean = np.mean(points_log,axis=0)
        self.cov = np.cov(points_log,rowvar=False)
        self.mvn = multivariate_normal(self.mean,self.cov)
        
    # get pdf at points
    def pdf(self,points):
        if not super().pdf(points): return None
        return self.mvn.pdf(np.log(points))
    
    def pdfgrid(self,grid):
        if not super().pdfgrid(points): return None
        return self.mvn.pdf(np.log(grid))
    
class exponential(fitfunction):
    
    # parameters
    # l: multidim lambda parameter of exponential
    
    # constructor
    def __init__(self,points):
        # points is a np array of shape (npoints,ndims)
        super().__init__(points)
        # for now use mean for beta, maybe change later!
        self.l = np.reciprocal(np.mean(points,axis=0))
        
    # get pdf at points
    def pdf(self,points):
        if not super().pdf(points): return None
        return np.prod(self.l)*np.exp(-np.multiply(np.repeat(self.l,len(points),axis=0),points))
        
    def pdfgrid(self,grid):
        if not super().pdfgrid(grid): return None
        return np.prod(self.l)*np.exp(-np.sum(np.multiply(self.l,grid),axis=2))

class seminormal(fitfunction):
    # this is not strictly speaking a probability distribution,
    # only the first quadrant of the result of fitting a normal distribution
    # to the data + its mirror image wrt the origin
    
    # parameters
    # cov: multidim covariance matrix of normal distribution
    # mvn: scipy.stats multivariate_normal object
        
    # constructor from point cloud
    def __init__(self,points=[]):
        if len(points)==0: return
        super().__init__(points)
        points = np.vstack((points,-points))
        self.cov = np.cov(points,rowvar=False)
        self.mvn = multivariate_normal(np.zeros(self.ndims),self.cov)
        
    # get pdf at points
    def pdf(self,points):
        if not super().pdf(points): return None
        return self.mvn.pdf(points)
    
    def pdfgrid(self,grid):
        if not super().pdfgrid(grid): return None
        return self.mvn.pdf(grid)
    
    def save(self,path):
        np.save(path,self.cov)
        
    def load(self,path):
        self.cov = np.load(path)
        self.ndims = len(self.cov)
        self.mvn = multivariate_normal(np.zeros(self.ndims),self.cov)
        
class gaussiankde(fitfunction):
    # wrapper for scipy.stats.gaussian_kde (gaussian kernel density estimation)
    
    # parameters
    # kernel: scipy.stats.gaussian_kde object
    # cov: covariance matrix 
    # (use same definition as for seminormal, maybe later replace by internal kernel.covariance)
    
    # constructor from point cloud
    def __init__(self,points=[],bw='default'):
        if len(points)==0: return
        super().__init__(points)
        self.cov = np.cov(points,rowvar=False)
        if bw=='default': bw = 20*np.power(self.npoints,-1/(self.ndims+4))
        elif bw=='scott': bw = np.power(self.npoints,-1/(self.ndims+4))
        self.kernel = gaussian_kde(np.transpose(points),bw_method=bw)
        
    # get pdf at points
    def pdf(self,points):
        if not super().pdf(points): return None
        return self.kernel.pdf(np.transpose(points))
    
    # get pdf at grid
    def pdfgrid(self,grid):
        if not super().pdfgrid(grid): return None
        # implementation seems to be different from scipy.mvn, explicit conversion to point array and back is needed
        X = grid[:,:,0]
        Y = grid[:,:,1]
        pos = np.vstack((np.ravel(X),np.ravel(Y)))
        Z = self.kernel.pdf(pos)
        return np.reshape(Z,X.shape)




def vecdist(moments,index):
    # does not work well if there are outliers which dominate the distance
    relmoments = moments-np.tile([moments[index,:]],(len(moments),1))
    sumsm = np.sum(relmoments[:index,:],axis=0)
    sumgr = np.sum(relmoments[index+1:,:],axis=0)
    distofsum = np.sqrt(np.sum(np.power(sumsm-sumgr,2)))
    sumofdist = np.sum(np.sqrt(np.sum(np.power(relmoments,2),axis=1)))
    return distofsum/sumofdist

def costhetadist(moments,index):
    # works more or less but not all bad points have small values, 
    # allows to identify problematic regions but not individual LS
    if(index==0 or index==len(moments)-1): return 0
    inprod = np.sum(np.multiply(moments[index-1,:],moments[index+1,:]))
    norms = np.sqrt(np.sum(np.power(moments[index-1,:],2)))*np.sqrt(np.sum(np.power(moments[index+1,:],2)))
    return inprod/norms

def avgnndist(moments,index,nn):
    # seems to work well for the runs tested!
    rng = np.array(range(index-nn,index+nn+1))
    rng = rng[(rng>=0) & (rng<len(moments))]
    moments_sec = moments[rng,:]-np.tile(moments[index,:],(len(rng),1))
    return np.sum(np.sqrt(np.sum(np.power(moments_sec,2),axis=1)))/len(rng)




def getavgnndist(hists,nmoments,xmin,xmax,nbins,nneighbours):
    dists = np.zeros(len(hists))
    moments = np.zeros((len(hists),nmoments))
    binwidth = (xmax-xmin)/nbins
    bins = np.tile(np.linspace(xmin+binwidth/2,xmax-binwidth/2,num=nbins,endpoint=True),(len(hists),1))
    for i in range(1,nmoments+1):
        moments[:,i-1] = moment(bins,hists,i)
    for i in range(len(hists)):
        dists[i] = avgnndist(moments,i,nneighbours)
    return dists




def filteranomalous(df,nmoments=3,rmouterflow=True,rmlargest=0.,doplot=True,):
    
    # do preliminary filtering (no DCS-bit OFF)
    # (implicitly re-index)
    print('total number of LS: '+str(len(df)))
    df = select_golden_and_bad(df)
    print('filtered number of LS (DCS-bit ON): '+str(len(df)))
    runs = get_runs(df)
    #print('found following runs: '+str(runs))
    
    # initializations
    nlumi = 0
    dists = []
    xmin = 0.
    xmax = 1.
    
    # loop over runs and calculate distances
    for run in runs:
        dfr = select_runs(df,[run])
        (hists,_,_) = get_hist_values(dfr)
        nlumi += len(hists)
        if rmouterflow: hists = hists[:,1:-1]
        rdists = getavgnndist(hists,nmoments,xmin,xmax,len(hists[0]),2)
        for d in rdists: dists.append(d)
            
    # concatenate all runs
    dists = np.array(dists)
    ind = np.linspace(0,nlumi,num=nlumi,endpoint=False)
    if doplot: plotdistance(dists,ind,rmlargest=rmlargest)
    (gmean,gstd) = getmeanstd(dists,ind,rmlargest=rmlargest)
        
    # add a columns to the original df
    df['dist'] = dists
    df['passmomentmethod'] = np.where(dists<gmean+3*gstd,1,0)
    
    # select separate df's
    dfpass = df[df['dist']<gmean+3*gstd]
    dfpass.reset_index(drop=True,inplace=True)
    dffail = df[df['dist']>gmean+3*gstd]
    dffail.reset_index(drop=True,inplace=True)
    
    return (df,dfpass,dffail,gmean,gstd)










