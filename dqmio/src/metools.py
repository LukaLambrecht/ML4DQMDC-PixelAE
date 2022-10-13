##########################################
# Tools for manipulating MonitorElements #
##########################################

import sys
import os
import numpy as np

def th1_to_nparray( th1, integer_values=False ):
  ### convert a TH1 to a numpy array
  nxbins = th1.GetNbinsX()
  dtype = float
  if integer_values: dtype = int
  res = np.zeros(nxbins+2, dtype=dtype)
  for i in range(nxbins+2):
    val = th1.GetBinContent(i)
    if integer_values: val = int(val)
    res[i] = val
  return res

def th2_to_nparray( th2, integer_values=False ):
  ### convert a TH2 to a numpy array
  nxbins = th2.GetNbinsX()
  nybins = th2.GetNbinsY()
  if integer_values: dtype = int
  res = np.zeros((nxbins+2,nybins+2), dtype=dtype)
  for i in range(nxbins+2):
    for j in range(nybins+2):
      val = th2.GetBinContent(i,j)
      if integer_values: val = int(val)
      res[i,j] = val
  return res

def th2_to_1dnparray( th2, integer_values=False ):
  ### convert a TH2 to a flattened numpy array
  nxbins = th2.GetNbinsX()
  nybins = th2.GetNbinsY()
  if integer_values: dtype = int
  res = np.zeros((nxbins+2)*(nybins+2), dtype=dtype)
  for i in range(nybins+2):
    for j in range(nxbins+2):
      val = th2.GetBinContent(j,i)
      if integer_values: val = int(val)
      res[i*(nxbins+2)+j] = val
  return res

def th3_to_nparray( th3, integer_values=False ):
  ### convert a TH3 to a numpy array
  nxbins = th3.GetNbinsX()
  nybins = th3.GetNbinsY()
  nzbins = th3.GetNbinsZ()
  if integer_values: dtype = int
  res = np.zeros((nxbins+2,nybins+2,nzbins+2), dtype=dtype)
  for i in range(nxbins+2):
    for j in range(nybins+2):
      for k in range(nzbins+2):
        val = th3.GetBinContent(i,j,k)
        if integer_values: val = int(val)
        res[i,j,k] = val
  return res

def tprofile_to_nparray( tprofile ):
  ### convert a TProfile to a numpy array
  # returns:
  # dict of arrays 'values', 'errors', 'xax'
  nxbins = tprofile.GetNbinsX()
  values = np.zeros(nxbins+2)
  errors = np.zeros(nxbins+2)
  xax = np.zeros(nxbins+2)
  for i in range(nxbins+2):
    values[i] = tprofile.GetBinContent(i)
    errors[i] = tprofile.GetBinError(i)
    xax[i] = int(round(tprofile.GetBinLowEdge(i)))
  return {'values':values, 'errors':errors, 'xax':xax}

def me_to_nparray( me, integer_values=False ):
  ### convert a MonitorElement to a numpy array
  metype = me.type
  if metype in [3,4,5]:
    return th1_to_nparray( me.data, integer_values=integer_values )
  elif metype in [6,7,8]:
    return th2_to_nparray( me.data, integer_values=integer_values )
  elif metype in [9]:
    return th3_to_nparray( me.data, integer_values=integer_values )
  elif metype in [10]:
    return tprofile_to_nparray( me.data )
  else:
    raise Exception('ERROR in metools.me_to_nparray:'
                    +' unrecognized me.type {}'.format(metype)
                    +' for me.name {}.'.format(me.name))
