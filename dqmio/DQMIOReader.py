# **Class for reading (nano)DQMIO files and extracting histograms**  
# 
# Originally copied from here: https://github.com/cms-DQM/ML4DQM-DC_SharedTools/blob/master/dqmio/moredqmiodata.ipynb


### imports

import ROOT
import numpy as np
#import root_numpy
# disable temporary since not available on SWAN, yet to find best solution...
# note: root_numpy provides an efficient interface between ROOT and numpy
from fnmatch import fnmatch 
# note: fnmatch provides support for unix shell-style wildcards, which are not the same as regular expressions in python
from collections import namedtuple
# note: a namedtuple is a pseudo-class consisting of a tuple with named fields
from collections import defaultdict
# note: a defaultdict is like a regular python dictionary but providing default values for missing keys 
#       instead of throwing execptions
from multiprocessing.pool import ThreadPool
# note: ThreadPool is used for parallel processing, calling the same function on parallel inputs 
#       and collecting the results in a list


### static definitions

IndexEntry = namedtuple('IndexEntry', ['run', 'lumi', 'type', 'file', 'firstidx', 'lastidx'])
MonitorElement = namedtuple('MonitorElement', ['run', 'lumi', 'name', 'type', 'data'])
NTHREADS=4 # not sure how high this can be put...

def extractdatafromROOT(x, hist2array=False):
    ### extract ROOT-type data into useful formats, depending on the data type
    # input arguments:
    # - x: a ROOT object
    # - hist2array: boolean whether to convert ROOT histograms to numpy arrays
    #               (default: keep as ROOT histogram objects)
    #               note: option True is not yet supported (need to fix root_numpy import in SWAN)
    if isinstance(x, ROOT.string): return unicode(x.data())
    if isinstance(x, int): return x
    if isinstance(x, float): return x
    else: 
        if hist2array:
            raise NotImplementedError('ERROR in extractdatafromROOT: option hist2array is not yet supported.')
            #return root_numpy.hist2array(x)
        else: return x.Clone()
    

### DQMIOReader definition

class DQMIOReader:
    ### class for reading (nano)DQMIO input files and extracting histograms
    # class attributes:
    # - rootfiles: a list of root files (DQMIO format), opened in read mode
    # - index: defaultdict matching tuples of the form (run number, lumisection number) to lists of IndexEntries
    # - melist: dict containing all available monitor element names matched to their type
    
    @staticmethod # (needed in python 2, not in python 3)
    def getMEType(metype):
        ### convert integer monitoring element type to string representation
        # note: the string representation must correspond to the directory structure in a DQMIO file!
        # note: this is a static function and does not require an instance to be called
        treenames = { 
          0: "Ints",
          1: "Floats",
          2: "Strings",
          3: "TH1Fs",
          4: "TH1Ss",
          5: "TH1Ds",
          6: "TH2Fs",
          7: "TH2Ss",
          8: "TH2Ds",
          9: "TH3Fs",
          10: "TProfiles",
          11: "TProfile2Ds",
        }
        return treenames[metype]
    
    def __init__(self, *files):
        ### initializer
        # open the passed in files and read their index data.
        # input arguments:
        # - files: a filename (or multiple filenames) to open
        #          if stored locally, the filenames should contain the full path.
        #          if stored on the grid, prefix the file path with "root://cms-xrd-global.cern.ch/" (not yet tested)
        self.rootfiles = [ROOT.TFile.Open(f) for f in files]
        self.readindex()
        self.makelist()

    def readindex(self):
        ### read index tables
        # note: for internal use in initializer only, do not call.
        self.index = defaultdict(list)
        def readfileidx(f):
            ### read file index
            # note: for internal use in initializer only, do not call.
            idxtree = getattr(f, "Indices")
            # release GIL in long operations. Disable if it causes trouble.
            #idxtree.GetEntry._threaded = True
            knownlumis = set()
            # loop over all "entries" in the current file
            for i in range(idxtree.GetEntries()):
                idxtree.GetEntry(i)
                # get run number, lumi number, and type of monitoring element for this entry
                run, lumi, metype = idxtree.Run, idxtree.Lumi, idxtree.Type
                if lumi == 0:
                    # read only per-lumisection monitoring elements for now.
                    continue
                # inclusive range -- for 0 entries, row is left out
                firstidx, lastidx = idxtree.FirstIndex, idxtree.LastIndex
                e = IndexEntry(run, lumi, metype, f, firstidx, lastidx)
                self.index[(run, lumi)].append(e)
        p = ThreadPool(NTHREADS)
        p.map(readfileidx, self.rootfiles)
        p.close()
                
    def makelist(self):
        ### make a cached list for monitoring elements
        # note: for internal use in initializer only, do not call.
        # note: this function reads one lumisection and assumes all lumisection contains the same monitoring elements!
        runlumi = next(iter(self.index.keys()))
        mes = self.getMEsForLumi(runlumi, "*")
        self.melist = dict()
        for me in mes:
            self.melist[me.name] = me.type
    
    def listMEs(self):
        ### returns an iterable with the names of the monitoring elements available per lumisection.
        return self.melist.keys()
    
    def listLumis(self):
        ### returns an iterable of (run number, lumisection number) pairs for the lumis available in the files.
        return self.index.keys()
    
    def getMEsForLumi(self, runlumi, *namepatterns):
        ### get selected monitoring elements for a given lumisection
        # input arguments:
        # - runlumi: a tuple of the form (run number, lumisection number)
        # - namepatterns: a wildcard pattern (or multiple) to select monitoring elements
        # returns:
        # a list of named tuples of type MonitorElement
        
        def check_interesting(mename):
            ### check if a monitoring element name matches required selections
            # note: for internal use in getMEsForLumi only, do not call!
            for pattern in namepatterns:
                if fnmatch(mename,pattern):
                    return True
                return False
 
        # get the data for the requested lumisection
        entries = self.index[runlumi]
        if not entries: 
            raise IndexError("ERROR in DQMIOReader.getMEsForLumi:"
                             +" requested to read data for lumisection {},".format(runlumi)
                             +" but no data was found for this lumisection in the current DQMIOReader.")
        
        # loop over all entries for this lumisection
        result = []
        for e in entries:
            # read the tree and disable all branches except "FullName"
            metree = getattr(e.file, DQMIOReader.getMEType(e.type))
            metree.GetEntry(0)
            metree.SetBranchStatus("*",0)
            metree.SetBranchStatus("FullName",1)
            # release GIL in long operations. Disable if it causes trouble.
            #metree.GetEntry._threaded = True
            # loop over entries for this tree
            for x in range(e.firstidx, e.lastidx+1):
                metree.GetEntry(x)
                # extract the monitoring element name and check if it is needed
                mename = str(metree.FullName)
                if not check_interesting(mename): continue
                metree.GetEntry(x, 1)
                value = metree.Value
                value = extractdatafromROOT(value)
                me = MonitorElement(runlumi[0], runlumi[1], mename, e.type, value)
                result.append(me)
        return result

    def getSingleMEForLumi(self, runlumi, name):
        ### get selected monitoring element for a given lumisection
        # input arguments:
        # - runlumi: a tuple of the form (run number, lumisection number)
        # - name: the name of a monitoring element to extract
        # returns:
        # a named tuple of type MonitorElement
        # note: this can be much faster than getMEsForLumi when only few MEs are read per lumi.
        
        def binsearch(a, key, lower, upper):
            ### binary search algorithm
            # note: for internal use in getSingleMEForLumi only, do not call.
            # input arguments:
            # - a: a callable that takes an integer and returns an object
            # - key: an instance of the same type of object as returned by a
            # - lower, upper: lower and upper integers to perform the search
            # returns:
            # the integer res where a(res)==key
            # note: what happens if no such res exists?
            n = upper - lower
            if n <= 1: return lower
            mid = int(n / 2) + lower
            if a(mid) <= key: return binsearch(a, key, mid, upper)
            else: return binsearch(a, key, lower, mid)
       
        def linsearch(a, key, lower, upper):
            ### linear search algorithm
            # note: for internal use in getSingleMEForLumi only, do not call.
            # input arguments:
            # - a: a callable that takes an integer and returns an object
            # - key: an instance of the same type of object as returned by a
            # - lower, upper: lower and upper integers to perform the search
            # returns:
            # the integer res where a(res)==key, or 0 if no such res exists.
            for k in range(lower, upper):
                if a(k) == key: return k
            return 0
        
        # get all entries for the given lumisection and monitoring element name
        entries = [e for e in self.index[runlumi] if e.type == self.melist[name]]
        if len(entries)!=1:
            raise IndexError("ERROR in DQMIOReader.getSingleMEForLumi:"
                             +" requested to read data for lumisection {}".format(runlumi)
                             +" and monitoring element {}".format(name)
                             +" but {} entries were found, while expecting 1.".format(len(entries)))
        
        # loop over all entries for this lumisection and monitoring element (should be only 1)
        for e in entries:
            # read the tree and disable all branches except "FullName"
            metree = getattr(e.file, DQMIOReader.getMEType(e.type))
            metree.GetEntry(0)
            metree.SetBranchStatus("*",0)
            metree.SetBranchStatus("FullName",1)
            # release GIL in long operations. Disable if it causes trouble.
            #metree.GetEntry._threaded = True
            
            def searchkey(fullname):
                # split into dir and name, since that is how the DQMStore orders things.
                return ("/".join(fullname.split("/")[:-1]), fullname.split("/")[-1])
            def getentry(idx):
                metree.GetEntry(idx)
                return searchkey(str(metree.FullName))
                
            pos = binsearch(getentry, searchkey(name), e.firstidx, e.lastidx+1)
            metree.GetEntry(pos, 1) # read full row
            if str(metree.FullName) != name:
                return None
            value = metree.Value
            value = extractdatafromROOT(value)
            return MonitorElement(runlumi[0], runlumi[1], name, e.type, value)
        
    def getMEs(self, *namepatterns):
        ### read monitoring elements matching the given wildcard patterns from all lumis.
        # input arguments:
        # - namepatterns: a wildcard patterns (or multiple) to select monitoring elements
        # returns:
        # a list of named tuples of type MonitorElement
        return sum((self.getMEsForLumi(lumi, *namepatterns) for lumi in self.listLumis()), [])
        # (note: sum is list concat here)
    
    def getSingleMEs(self, name, callback=None):
        ### read a single monitoring element with the given name from all lumis.
        # input arguments:
        # - name: the name of a monitoring element to extract
        # - callback: ?
        # returns:
        # a list of named tuples of type MonitorElement
        # note: this can be much faster than getMEsForLumi when only few MEs are read per lumi.
        files = defaultdict(list)
        ctr = [0]
        # make a dict storing which lumisections are stored in which file
        for lumi in self.listLumis():
            files[self.index[lumi][0].file.GetName()].append(lumi)
                             
        def readlumi(lumi):
            ### read a single lumisection
            # note: for internal use in getSingleMEs only, do not call.
            l = self.getSingleMEForLumi(lumi, name)
            if callback:
                ctr[0] += 1
                if ctr[0] % 10 == 0:
                    callback(ctr[0])
            return l
                             
        def readfile(f):
            ### read a single file
            # note: for internal use in getSingleMEs only, do not call.
            return [readlumi(lumi) for lumi in files[f]]
                             
        p = ThreadPool(NTHREADS)
        result = p.map(readfile, files.keys())
        p.close()
        return sum(result, [])