# **Class for reading (nano)DQMIO files and extracting histograms**  
# 
# Originally copied from here: https://github.com/cms-DQM/ML4DQM-DC_SharedTools/blob/master/dqmio/moredqmiodata.ipynb


### imports

import sys
import ROOT
import numpy as np
#import root_numpy
# disable temporary since not available on SWAN, for now just define manual conversion function from root to numpy array
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

from timeit import default_timer
# note: only used for callback method to print the progress of getSingleMes

import pandas as pd
# note: only used for conversion into dataframe


### static definitions

IndexEntry = namedtuple('IndexEntry', ['run', 'lumi', 'type', 'file', 'firstidx', 'lastidx'])
# an instance of IndexEntry represents one "entry" in a DQMIO file.
# this "entry" corresponds to a single lumisection (characterized by run and lumi)
# and a single type (e.g. TH1F, TH2F, etc.).
# so all monitoring elements for this lumisection and for this type are in the same IndexEntry,
# numbered from firstidx to lastidx.
# note: the firstidx to lastidx range runs in parallel for multiple types (as they are stored in different trees);
#       so multiple IndexEntries for the same lumisection (but different type) can have overlapping indices,
#       but multiple IndexEntries for the same type and file but different lumisections have disjoint indices!
MonitorElement = namedtuple('MonitorElement', ['run', 'lumi', 'name', 'type', 'data'])
# an instance of MonitorElement represents one monitor element, with all associated information:
# - the run and lumisection number
# - the full name of the monitoring element
# - the type (e.g. TH1F, TH2F, etc., see function getMEType below for all allowed types)
# - the actual data

def extractdatafromROOT(x, hist2array=False):
    ### extract ROOT-type data into useful formats, depending on the data type
    # input arguments:
    # - x: a ROOT object
    # - hist2array: boolean whether to convert ROOT histograms to numpy arrays
    #               (default: keep as ROOT histogram objects)
    #               note: option True is not yet supported (need to fix root_numpy import in SWAN)
    
    # first check for clear-cut data types such as ROOT strings, python ints and floats
    if isinstance(x, ROOT.string): return unicode(x.data())
    if isinstance(x, int): return x
    if isinstance(x, float): return x
    # additional check for python long, which is only defined in python 2!
    # (gives error in python 3, so need to check version explicitly)
    if sys.version_info[0]<3:
        if isinstance(x, long): return x
    # at this point, if the function reaches to this stage,
    # the type of x is probably some kind of ROOT histogram
    # (more exceptions to be added above when encountered).
    if hist2array:
        raise NotImplementedError('ERROR in DQMIOReader.extractdatafromROOT:'
                                  +' option hist2array is not yet supported.')
        #return root_numpy.hist2array(x)
    else: return x.Clone()
    # throw error if the function did not return in any of the above cases
    raise Exception('ERROR in DQMIOReader.extractdatafromROOT:'
                    +' type {} not recognized.'.format(type(x)))
    

### DQMIOReader definition

class DQMIOReader:
    ### class for reading (nano)DQMIO input files and extracting histograms
    # class attributes:
    # - rootfiles: a list of root files (DQMIO format), opened in read mode
    # - index: defaultdict matching tuples of the form (run number, lumisection number) to lists of IndexEntries.
    #          for each key of the form (run number, lumisection number), the value is a list of a few IndexEntries,
    #          one for each monitoring element type (so 12 at maximum).
    #          and empty list is returned from the dict if a given (run number, lumisection number) is not present.
    # - indexlist: separate list of index keys for sortability in python2.
    # - medict: dict containing all available monitor element names matched to their type.
    # - melist: separate list of medict keys for sortabiltiy in python2.
    # - nthreads: number of threads for multithreaded processing.
    
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
    
    def __init__(self, *files, **kwargs):
        ### initializer
        # open the passed in files and read their index data.
        # input arguments:
        # - files: a filename (or multiple filenames) to open
        #          if stored locally, the filenames should contain the full path.
        #          if stored on the grid, prefix the file path with "root://cms-xrd-global.cern.ch/".
        # - kwargs: may contain:
        #   -- sortindex: bool (default False) whether or not to sort the index 
        #                 (by run and lumisection number in ascending order).
        #   -- sortmes: bool (default False) whether or not to sort the ME names
        #               (alphabetically)
        #   -- nthreads: int (default 4) for number of threads

        # check arguments
        sortindex = False
        sortmes = False
        self.nthreads = 4
        for key,val in kwargs.items():
            if( key=='sortindex' and val ): sortindex = True
            elif( key=='sortmes' and val ): sortmes = True
            elif( key=='nthreads' ): self.nthreads = int(val)
            else:
                raise Exception('ERROR in DQMIOReader.__init__:'
                               +' unrecognized keyword argument "{}"'.format(key))
  
        # do the initialization
        print('DQMIOReader.__init__: opening {} files...'.format(len(files)))
        sys.stdout.flush()
        sys.stderr.flush()
        self.rootfiles = [ROOT.TFile.Open(f) for f in files]
        print('all files opened, now making index')
        sys.stdout.flush()
        sys.stderr.flush()
        self.readIndex( sort=sortindex )
        print('index made, now making list of monitoring elements')
        sys.stdout.flush()
        sys.stderr.flush()
        self.makeMEList( sort=sortmes )

    def readIndex(self, sort=False):
        ### read index tables
        # note: for internal use in initializer only, do not call.
        self.index = defaultdict(list)
        self.indexlist = [] # separate list of index keys for sortability
        def readfileidx(f):
            ### read file index from one file
            # note: for internal use in initializer only, do not call.
            idxtree = getattr(f, "Indices")
            # release GIL in long operations. Disable if it causes trouble.
            #idxtree.GetEntry._threaded = True
            knownlumis = set()
            # loop over all "entries" in the current file
            for i in range(idxtree.GetEntries()):
                idxtree.GetEntry(i)
                # get run number, lumi number, and type of monitoring element for this entry.
                # note: apparently idxtree contains one "entry" per run, lumisection and type;
                #       that is: all monitoring elements of the same type (e.g. TH1F) and for the same lumisection
                #       are in the same "entry"; this is what FirstIndex and LastIndex are for (see below).
                # note: apparently idxtree.Lumi gives 0 for per-run monitoring elements,
                #       but for now we ignore those and only read per-ls monitoring elements.
                run, lumi, metype = idxtree.Run, idxtree.Lumi, idxtree.Type
                if lumi == 0: continue
                firstidx, lastidx = idxtree.FirstIndex, idxtree.LastIndex
                e = IndexEntry(run, lumi, metype, f, firstidx, lastidx)
                self.index[(run, lumi)].append(e)
        p = ThreadPool(self.nthreads)
        p.map(readfileidx, self.rootfiles)
        p.close()
        # convert the defaultdict to a regular dict
        # (else unwanted behaviour when trying to retrieve lumisections that are not present;
        #  in case of defaultdict they are added to the index as empty lists of IndexEntries)
        self.index = dict(self.index)
        # store the keys (i.e. run and lumisection numbers) in a separate list;
        # needed for sortability in python2.
        self.indexlist = list(self.index.keys())
        # sort the index
        if sort: self.sortIndex()
        
    def sortIndex(self):
        ### sort the index by run and lumisection number
        # note: only sort the indexlist; the index itself remains unordered.
        #       this is because the index is a dict, that has no order in python2.
        #       if you want to get the runs/lumisections in an ordered way,
        #       loop over indexlist, not index.keys()!
        self.indexlist = sorted(self.indexlist)
                
    def makeMEList(self, sort=False):
        ### make a cached list for monitoring elements
        # note: for internal use in initializer only, do not call.
        # note: this function reads one lumisection and assumes all lumisection contains the same monitoring elements!
        runlumi = self.indexlist[0]
        mes = self.getMEsForLumi(runlumi, "*")
        self.medict = dict()
        self.melist = [] # separate list of ME names for sortability
        for me in mes:
            self.medict[me.name] = me.type
            self.melist.append(me.name)
        if sort: self.sortMEList()

    def sortMEList(self):
        ### sort the list of MEs alphabetically
        self.melist = sorted(self.melist)   
 
    def listMEs(self):
        ### returns a list with the names of the monitoring elements available per lumisection.
        # warning: copying the list is avoided to for speed and memory;
        #          only meant for reading; if you want to modify the result, make a copy first!
        return self.melist
    
    def listLumis(self):
        ### returns a list of (run number, lumisection number) pairs for the lumis available in the files.
        # warning: copying the list is avoided to for speed and memory;
        #          only meant for reading; if you want to modify the result, make a copy first!
        return self.indexlist
    
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
        entries = self.index.get(runlumi, None)
        if entries is None: 
            raise IndexError("ERROR in DQMIOReader.getMEsForLumi:"
                             +" requested to read data for lumisection {},".format(runlumi)
                             +" but no data was found for this lumisection in the current DQMIOReader.")
        
        # loop over all entries for this lumisection;
        # this corresponds to looping over all types of monitoring elements
        # (see the documentation of IndexEntry for more info).
        result = []
        for e in entries:
            # read the correct tree from the file corresponding to this type of monitoring element
            metree = getattr(e.file, DQMIOReader.getMEType(e.type))
            metree.GetEntry(0)
            # disable all branches except "FullName"
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
        entries = [e for e in self.index.get(runlumi, []) if e.type == self.medict[name]]
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
 
    @staticmethod
    def showcount(ncurrent, ntot):
        ### default callback method showing the progress of getSingleMEs
        # input arguments:
        # - ncurrent: current number of instance being processed
        # - ntot: total number of instances to process
        global start, lasttime, lastcount
        try:
            assert(lastcount < ncurrent)
            # this fails if things are not initialized (e.g. in the first call)
            # or the ctr was reset.
        except:
            # (re-)initialize
            start = default_timer()
            lasttime = default_timer()
            lastcount = 0
        tottime = default_timer() - start
        deltatime = default_timer() - lasttime
        lasttime = default_timer()
        deltacount = ncurrent - lastcount
        lastcount = ncurrent
        msg = "Processed {} out of {} lumis in {:.2f} s ({:.2f}%, {:.2f}/s, avg {:.2f}/s)".format(
               ncurrent, ntot, tottime, 100.0*ncurrent/ntot, deltacount/deltatime, ncurrent/tottime)
        print(msg)
        sys.stdout.flush()
        sys.stderr.flush()
    
    def getSingleMEs(self, name, callback='default'):
        ### read a single monitoring element with the given name from all lumis.
        # input arguments:
        # - name: the name of a monitoring element to extract
        # - callback: can be used for progress printing.
        #             can be None (no callback), a custom function, or 'default',
        #             in which case the default callback showcount will be called.
        # returns:
        # a list of named tuples of type MonitorElement
        # note: this can be much faster than getMEs when only few MEs are read per lumi.
        files = defaultdict(list)
        ctr = [0]
        # make a dict storing which lumisections are stored in which file
        for lumi in self.listLumis():
            files[self.index[lumi][0].file.GetName()].append(lumi)
        # set the callback function as an instancde attribute
        self.callback = callback
        if self.callback=='default': self.callback = DQMIOReader.showcount
                             
        def readlumi(lumi):
            ### read a single lumisection
            # note: for internal use in getSingleMEs only, do not call.
            l = self.getSingleMEForLumi(lumi, name)
            if self.callback is not None:
                ctr[0] += 1
                if ctr[0] % 10 == 0:
                    self.callback(ctr[0], len(self.listLumis()))
            return l
                             
        def readfile(f):
            ### read a single file
            # note: for internal use in getSingleMEs only, do not call.
            return [readlumi(lumi) for lumi in files[f]]
                             
        p = ThreadPool(self.nthreads)
        result = p.map(readfile, files.keys())
        p.close()
        return sum(result, [])
    
    def getSingleMEsToDataFrame(self, name):
        ### return a pandas dataframe for a given monitoring element
        # note: the same naming convention is used as in the 2017/2018 csv input!
        
        # get the monitoring elements
        mes = self.getSingleMEs(name)
        # initialize a dict with all info
        dfdict = dict()
        dfdict['fromrun'] = []
        dfdict['fromlumi'] = []
        dfdict['hname'] = []
        dfdict['metype'] = []
        dfdict['histo'] = []
        dfdict['entries'] = []
        dfdict['Xmax'] = []
        dfdict['Xmin'] = []
        dfdict['Xbins'] = []
        dfdict['Ymax'] = []
        dfdict['Ymin'] = []
        dfdict['Ybins'] = []
        # extract bin edges (assume the same for all monitoring elements!)
        metype = mes[0].type
        if metype in [3,4,5]:
            nxbins = mes[0].data.GetNbinsX()
            xmin = mes[0].data.GetBinLowEdge(1)
            xmax = mes[0].data.GetBinLowEdge(nxbins)+mes[0].data.GetBinWidth(nxbins)
            nybins = 1
            ymin = 0
            ymax = 1
        elif metype in [6,7,8]:
            nxbins = mes[0].data.GetNbinsX()
            xmin = mes[0].data.GetXaxis().GetBinLowEdge(1)
            xmax = (mes[0].data.GetXaxis().GetBinLowEdge(nxbins)
                    +mes[0].data.GetXaxis().GetBinWidth(nxbins))
            nybins = mes[0].data.GetNbinsY()
            ymin = mes[0].data.GetYaxis().GetBinLowEdge(1)
            ymax = (mes[0].data.GetYaxis().GetBinLowEdge(nybins)
                    +mes[0].data.GetYaxis().GetBinWidth(nybins))
        else:
            raise Exception('ERROR in DQMIOReader.getSingleMEsToDataFrame:'
                            +' monitoring element type not recognized: {}'.format(metype))
        # loop over monitoring elements
        for idx,me in enumerate(mes):
            # extract the histogram
            if metype in [3,4,5]:
                histo = np.zeros(nxbins+2, dtype=int)
                for i in range(nxbins+2):
                    histo[i] = int(me.data.GetBinContent(i))
            elif metype in [6,7,8]:
                histo = np.zeros((nxbins+2)*(nybins+2), dtype=int)
                for i in range(nybins+2):
                    for j in range(nxbins+2):
                        histo[i*(nxbins+2)+j] = int(me.data.GetBinContent(j,i))
            # append all info
            dfdict['fromrun'].append(me.run)
            dfdict['fromlumi'].append(me.lumi)
            dfdict['hname'].append(me.name)
            dfdict['metype'].append(me.type)
            dfdict['histo'].append(list(histo))
            dfdict['entries'].append(int(np.sum(histo)))
            dfdict['Xmax'].append(xmax)
            dfdict['Xmin'].append(xmin)
            dfdict['Xbins'].append(nxbins)
            dfdict['Ymax'].append(ymax)
            dfdict['Ymin'].append(ymin)
            dfdict['Ybins'].append(nybins)
        # make a dataframe
        df = pd.DataFrame(dfdict)
        return df
