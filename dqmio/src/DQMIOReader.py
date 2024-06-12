# **Class for reading (nano)DQMIO files and extracting histograms**  
# 
# Originally copied from here: https://github.com/cms-DQM/ML4DQM-DC_SharedTools/blob/master/dqmio/moredqmiodata.ipynb
# Note: this code relies on ROOT via the PyROOT bindings;
#       attempts to remove the ROOT dependency by using uproot are unsuccessful so far,
#       as DQMIO files seem to have a particular feature not covered by uproot,
#       see here: https://github.com/scikit-hep/uproot5/issues/38
#       and also here: https://github.com/scikit-hep/uproot5/issues/1190


### imports

## Python standard library

import sys
from fnmatch import fnmatch 
# note: fnmatch provides support for unix shell-style wildcards,
#       which are not the same as regular expressions in python
from collections import namedtuple
# note: a namedtuple is a pseudo-class consisting of a tuple with named fields
from collections import defaultdict
# note: a defaultdict is like a regular python dictionary,
#       but providing default values for missing keys instead of throwing execptions
from multiprocessing.pool import ThreadPool
# note: ThreadPool is used for parallel processing, calling the same function on parallel inputs 
#       and collecting the results in a list
from timeit import default_timer
# note: only used for callback method to print the progress of getSingleMes

## third-party

import ROOT
import numpy as np

pandas_import_error = None
try:
    import pandas as pd
except ImportError as e:
    pandas_import_error = e
    pd = None
# note: only used for conversion into dataframe,
#       now turned into an optional import


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
# - the type (e.g. TH1F, TH2F, etc., see function getMETreeName below for all allowed types)
# - the actual data


### DQMIOReader definition

class DQMIOReader:
    ### class for reading (nano)DQMIO input files and extracting histograms
    # class attributes:
    # - rootfiles: a list of root files (DQMIO format), opened in read mode
    # - index: dict matching tuples of the form (run number, lumisection number) to lists of IndexEntries.
    #          for each key of the form (run number, lumisection number), the value is a list of a few IndexEntries,
    #          one for each monitoring element type (so 12 at maximum).
    # - indexlist: separate list of index keys for sortability in python2.
    # - medict: dict containing all available monitor element names matched to their type.
    # - melist: separate list of medict keys for sortabiltiy in python2.
    # - nthreads: number of threads for multithreaded processing.


    ### static helper function definitions

    @staticmethod
    def extractDataFromROOT(x):
        ### extract ROOT-type data into useful formats, depending on the data type
        # input arguments:
        # - x: a ROOT object

        # first check for clear-cut data types such as ROOT strings, python ints and floats
        if isinstance(x, ROOT.string):
            if sys.version_info[0]<3: return unicode(x.data())
            else: return str(x.data())
        if isinstance(x, int): return x
        if isinstance(x, float): return x
        # additional check for python long, which is only defined in python 2!
        # (gives error in python 3, so need to check version explicitly)
        if sys.version_info[0]<3:
            if isinstance(x, long): return x
        # at this point, if the function reaches to this stage,
        # the type of x is probably some kind of ROOT histogram
        # (more exceptions to be added above when encountered).
        else: return x.Clone()
        # throw error if the function did not return in any of the above cases
        raise Exception('ERROR in DQMIOReader.extractDataFromROOT:'
                        +' type {} not recognized.'.format(type(x)))

    @staticmethod
    def keepMEName(mename, namepatterns):
        if isinstance(namepatterns, str): namepatterns = [namepatterns]
        for namepattern in namepatterns:
            if fnmatch(mename, namepattern): return True
        return False

    @staticmethod
    def filterMENames(oglist, namepatterns):
        ### filter a list of monitoring element names
        # input arguments:
        # - oglist: original list of names
        # - namepatterns: string (may contain unix-style wildcards) of pattern to keep,
        #   or a list of such strings.
        res = []
        if isinstance(namepatterns, str): namepatterns = [namepatterns]
        for mename in oglist:
            if DQMIOReader.keepMEName(mename, namepatterns): res.append(mename)
        return res    
    
    @staticmethod
    def getMETreeName(metype):
        ### convert integer monitoring element type to the name of the corresponding Tree.
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
        if metype not in treenames.keys():
            msg = 'ERROR: provided metype {} not recognized;'.format(metype)
            msg += ' options are {}.'.format(treenames)
            raise Exception(msg)
        return treenames[metype]

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
  
 
    ### initializer and auxiliary functions
 
    def __init__(self,
        *files,
        dummy=True,
        sortindex=False,
        sortmes=False,
        nthreads=1,
        verbose=True ):

        # note: the syntax above does not work in python2, only in python3...
        #       might change this later on, but the best is probably to get rid of *files
        #       and just make it a regular list argument;
        #       this implies a change in syntax for all calling code though...

        ### initializer
        # open the passed in files and read their index data.
        # input arguments:
        # - files: one or multiple file names to open.
        #   if stored locally, the filenames should contain the full path.
        #   if stored on the grid, prefix the file path with "root://cms-xrd-global.cern.ch/"
        #   (see https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookXrootdService).
        # - sortindex: bool whether or not to sort the index 
        #   (by run and lumisection number in ascending order).
        # - sortmes: bool whether or not to sort the ME names (alphabetically).
        # - nthreads: int for number of threads to use.
        #   (more threads will run faster, but might lead to instabilities in some cases).
        # - verbose: bool whether to do printouts.

        # set nthreads argument as attribute
        self.nthreads = int(nthreads)

        # do the initialization
        if verbose: print('Initializing a DQMIOReader with {} files...'.format(len(files)))
        sys.stdout.flush()
        sys.stderr.flush()
        self.rootfiles = [ROOT.TFile.Open(f) for f in files]
        if verbose: print('Making index...')
        sys.stdout.flush()
        sys.stderr.flush()
        self.readIndex( sort=sortindex )
        if verbose: print('Making list of monitoring elements...')
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
            # loop over all "entries" in the current file
            for i in range(idxtree.GetEntries()):
                idxtree.GetEntry(i)
                # get run number, lumi number, and type of monitoring element for this entry.
                # note: apparently idxtree contains one "entry" per run, lumisection and type;
                #       that is: all monitoring elements of the same type (e.g. TH1F) and for the same lumisection
                #       are in the same "entry"; this is what FirstIndex and LastIndex are for (see below).
                run, lumi, metype = idxtree.Run, idxtree.Lumi, idxtree.Type
                # note: apparently idxtree.Lumi gives 0 for per-run monitoring elements,
                #       but for now we ignore those and only read per-ls monitoring elements.
                if lumi == 0: continue
                # note: the type should usually be an integer between 0 and 11
                #       (see getMETreeName), but in rare cases, type 1000 is observed;
                #       not clear what it is exactly, but skip it for now.
                #       also automatically skip other cases where the type is not recognized.
                try:
                    treename = DQMIOReader.getMETreeName(metype)
                except:
                    msg = 'WARNING: found index entry of type {}'.format(metype)
                    msg += ' which is not recognized; will skip this index entry.'
                    #print(msg)
                    continue
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

    def printIndex(self):
        ### print index diagnostics
        # note: mostly for debugging purposes
        # note: basically same structure as readIndex,
        #       but print more info and do not store results
        def printfileidx(f):
            idxtree = getattr(f, "Indices")
            print('Reading "Indices" tree of file {}'.format(f.GetName()))
            print('Found {} index entries:'.format(idxtree.GetEntries()))
            for i in range(idxtree.GetEntries()):
                idxtree.GetEntry(i)
                print('  - Run {}, lumi {}, type {}, first index {}, last index {}'.format(
                      idxtree.Run, idxtree.Lumi, idxtree.Type, idxtree.FirstIndex, idxtree.LastIndex))
        for f in self.rootfiles: printfileidx(f)
                
    def makeMEList(self, sort=False):
        ### make a cached list for monitoring elements
        # note: for internal use in initializer only, do not call.
        # note: this function reads one lumisection and assumes all lumisection contains the same monitoring elements!
        self.medict = {}
        self.melist = []
        if len(self.indexlist)==0:
            msg = 'WARNING: current index has 0 entries, cannot make list of monitoring elements.'
            print(msg)
            return
        runlumi = self.indexlist[0]
        self.medict = self.getMENamesForLumi(runlumi)
        self.melist = list(self.medict.keys()) # separate list of ME names for sortability
        if sort: self.sortMEList()

    def sortMEList(self):
        ### sort the list of MEs alphabetically
        self.melist = sorted(self.melist)   


    ### callable member functions
 
    def listMEs(self, namepatterns=None):
        ### returns a list with the names of the monitoring elements available per lumisection.
        # warning: copying the list is avoided to for speed and memory;
        #          only meant for reading; if you want to modify the result, make a copy first!
        # input arguments:
        # - namepatterns: a strings (can contain unix-style wildcards) for filtering the results,
        #   or a list of such strings.
        if namepatterns is None: return self.melist
        if isinstance(namepatterns, str): namepatterns = [namepatterns]
        return DQMIOReader.filterMENames(self.melist, namepatterns)
    
    def listLumis(self):
        ### returns a list of (run number, lumisection number) pairs for the lumis available in the files.
        # warning: copying the list is avoided to for speed and memory;
        #          only meant for reading; if you want to modify the result, make a copy first!
        return self.indexlist

    def getEntries(self, runlumis=None, namepatterns=None, expect=None):
        ### get entries for provided lumisections and/or monitoring element names
        # input arguments:
        # - runlumis: lumisection specifier.
        #   can be None (to use all available lumisections),
        #   a tuple of the form (run number, lumisection number),
        #   or a list of such tuples.
        # - namepatterns: monitoring element name specifier.
        #   can be None (to use all available monitoring elments),
        #   a string (can contain unix-style wildcards),
        #   or a list of such strings.
        # - expect: if specified, should be the expected number of entries
        #   (for raising an Exception if a different number is found).
        # note: mostly for internal use, not recommended to be called.
        
        # format runlumis and check validity
        if isinstance(runlumis, tuple): runlumis = [runlumis]
        if runlumis is not None:
            for runlumi in runlumis:
                if runlumi not in self.indexlist:
                    msg = "ERROR in DQMIOReader.getEntries:"
                    msg += " requested lumisection {}".format(runlumi)
                    msg += " not found in list of available lumisections."
                    raise Exception(msg)

        # format name patterns, check validity and expand wildcards
        names = None
        if isinstance(namepatterns, str): namepatterns = [namepatterns]
        if namepatterns is not None:
            names = []
            for namepattern in namepatterns:
                # in case of wildcards: expand
                # todo: this part could potentially be sped up by grouping
                #       the names with wildcards first and making a single listMEs call,
                #       rather than making a separate call for each pattern.
                if( '*' in namepattern or '?' in namepattern ):
                    thisnames = self.listMEs(namepatterns=[namepattern])
                    names += thisnames
                # else: check validity
                else:
                    if namepattern not in self.melist:
                        msg = "ERROR in DQMIOReader.getEntries:"
                        msg += " requested monitoring element {}".format(name)
                        msg += " not found in list of available monitoring elements."
                        raise Exception(msg)
                    names.append(namepattern)

        # find all monitoring element types that should be kept
        metypes = None
        if names is not None:
            metypes = [self.medict[name] for name in names]
            metypes = list(set(metypes))

        # find all entries for requested lumisections and monitoring element types
        entries = []
        if runlumis is None: runlumis = self.listLumis()
        for runlumi in runlumis:
            thisentries = self.index[runlumi]
            if metypes is not None: thisentries = [e for e in thisentries if e.type in metypes]
            for entry in thisentries: entries.append(entry)

        # check if number of found entries matches expectation
        if( expect is not None and len(entries)!=expect ):
            msg = "ERROR in DQMIOReader.getEntries:"
            msg += " requested to read data for lumisections {}".format(runlumis)
            msg += " and monitoring elements {}".format(names)
            msg += " but {} entries were found,".format(len(entries))
            msg += " while expecting {}.".format(expect)
            raise Exception(msg)

        # return the resulting entries
        return entries

    def loopEntries(self, entries):
        ### utility function to loop over entries
        # note: for internal use only, do not call.
        for entry in entries:
            # read the correct tree from the file corresponding to this type of monitoring element
            metree = getattr(entry.file, DQMIOReader.getMETreeName(entry.type))
            metree.GetEntry(0)
            # disable all branches except "FullName"
            metree.SetBranchStatus("*",0)
            metree.SetBranchStatus("FullName",1)
            # loop over entries for this tree
            for idx in range(entry.firstidx, entry.lastidx+1):
                metree.GetEntry(idx)
                # extract the monitoring element name and yield
                mename = str(metree.FullName)
                yield (entry, metree, idx, mename)

    def getMENamesForLumi(self, runlumi, namepatterns=None):
        ### get the names (and types) of available monitoring elements for a given lumisection
        # input arguments:
        # - runlumi: a tuple of the form (run number, lumisection number)
        # - namepatterns: a string (can contain unix-style wildcards) to filter monitoring element names,
        #   or a list of such strings.
        # returns:
        # - a dict matching monitoring element names to their types
        # note: mostly for internal usage (i.e. initializing the medict and melist);
        #       after initialization of the DQMIOReader, it is much faster to simply call listMEs()
        #       instead of re-reading the monitoring element names from scratch.
        # note: this function is much faster than reading the actual histogram data
        #       and then keeping only their names.
        
        # get all entries for the requested lumisection
        entries = self.getEntries(runlumis=[runlumi], namepatterns=namepatterns)

        # loop over entries and filter monitoring elements
        medict = {}
        for entry, metree, idx, mename in self.loopEntries(entries):
            if( namepatterns is not None and not DQMIOReader.keepMEName(mename, namepatterns) ): continue
            medict[mename] = entry.type
        return medict

    def getMEsForLumi(self, runlumi, namepatterns=None):
        ### get selected monitoring elements for a given lumisection
        # input arguments:
        # - runlumi: a tuple of the form (run number, lumisection number)
        # - namepatternss: a string (may contain unix-style wildcards) to select monitoring elements,
        #   or a list of such strings.
        # returns:
        # - a list of named tuples of type MonitorElement
        return self.getMEs(runlumis=runlumi, namepatterns=namepatterns)

    def getSingleMEForLumi(self, runlumi, name):
        ### get selected monitoring element for a given lumisection
        # input arguments:
        # - runlumi: a tuple of the form (run number, lumisection number)
        # - name: the name of a monitoring element to extract
        # returns:
        # - a single named tuple of type MonitorElement
        # note: alternatively, one could use getMEsForLumi with namepatterns = name
        #       as optional argument, but this dedicated implementation can be much faster.
        
        def binsearch(a, key, lower, upper):
            ### binary search algorithm
            # note: for internal use in getSingleMEForLumi only, do not call.
            # input arguments:
            # - a: a callable that takes an integer and returns an object
            # - key: an instance of the same type of object as returned by a
            # - lower, upper: lower and upper integers to perform the search
            # returns:
            # - the integer res where a(res)==key
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
            # - the integer res where a(res)==key, or 0 if no such res exists.
            for k in range(lower, upper):
                if a(k) == key: return k
            return 0
        
        # get all entries for the given lumisection and monitoring element name
        # (if everything goes well, this should be only one entry)
        entries = self.getEntries(runlumis=[runlumi], namepatterns=[name], expect=1)
        
        # loop over all entries for this lumisection and monitoring element
        # (should be only one however, see check above)
        for entry in entries:
            # read the tree and disable all branches except "FullName"
            metree = getattr(entry.file, DQMIOReader.getMETreeName(entry.type))
            metree.GetEntry(0)
            metree.SetBranchStatus("*",0)
            metree.SetBranchStatus("FullName",1)
            
            def searchkey(fullname):
                # split into dir and name, since that is how the DQMStore orders things.
                return ("/".join(fullname.split("/")[:-1]), fullname.split("/")[-1])
            def getentry(idx):
                metree.GetEntry(idx)
                return searchkey(str(metree.FullName))
            
            # find the entry index of the requested monitoring element
            pos = binsearch(getentry, searchkey(name), entry.firstidx, entry.lastidx+1)
            # read the correct entry with all branches activated
            metree.GetEntry(pos, 1)
            # extra check if entry was correctly found
            if str(metree.FullName) != name:
                msg = 'ERROR in getSingleMEForLumi: could not find correc position'
                msg += ' for monitoring element {}'.format(name)
                raise Exception(msg)
            value = metree.Value
            value = DQMIOReader.extractDataFromROOT(value)
            return MonitorElement(runlumi[0], runlumi[1], name, entry.type, value)
        
    def getMEs(self, runlumis=None, namepatterns=None):
        ### read monitoring elements matching the given wildcard pattern from all lumis.
        # input arguments:
        # - runlumis: a tuple of the form (run number, lumisection number),
        #   or a list of such tuples;
        #   if None, all available lumisections will be used.
        # - namepatterns: a string (may contain unix-style wildcards) to select monitoring elements,
        #   or a list of such strings;
        #   if None, all available monitoring elements will be used.
        # returns:
        # - a list of named tuples of type MonitorElement

        # get all entries for the requested lumisections
        entries = self.getEntries(runlumis=runlumis, namepatterns=namepatterns)

        # loop over entries and filter monitoring elements
        mes = []
        for entry, metree, idx, mename in self.loopEntries(entries):
            if( namepatterns is not None and not DQMIOReader.keepMEName(mename, namepatterns) ): continue
            # reload the entry with all branches activated
            metree.GetEntry(idx, 1)
            # read the actual histogram data
            value = metree.Value
            value = DQMIOReader.extractDataFromROOT(value)
            # make the MonitorElement and append to output array
            me = MonitorElement(entry.run, entry.lumi, mename, entry.type, value)
            mes.append(me)
        return mes
 
    def getSingleMEs(self, name, runlumis=None, callback='default'):
        ### read a single monitoring element with the given name from all lumis.
        # input arguments:
        # - name: the name of a monitoring element to extract
        # - runlumis: a tuple of the form (run number, lumisection number),
        #   or a list of such tuples;
        #   if None, all available lumisections will be used.
        # - callback: can be used for progress printing.
        #             can be None (no callback), a custom function, or 'default',
        #             in which case the default callback showcount will be called.
        # returns:
        # a list of named tuples of type MonitorElement
        # note: this can be much faster than getMEs when only few MEs are read per lumi.

        # format runlumis
        if runlumis is None: runlumis = self.listLumis()
        if isinstance(runlumis, tuple): runlumis = [runlumis]

        # make a dict storing which lumisections are stored in which file
        files = defaultdict(list)
        for runlumi in runlumis:
            files[self.index[runlumi][0].file.GetName()].append(runlumi)

        # set the callback function as an instancde attribute
        self.callback = callback
        if self.callback=='default': self.callback = DQMIOReader.showcount

        # initialize a counter
        ctr = [0]

        def readlumi(lumi):
            ### read a single lumisection
            # note: for internal use in getSingleMEs only, do not call.
            l = self.getSingleMEForLumi(lumi, name)
            if self.callback is not None:
                ctr[0] += 1
                if ctr[0] % 10 == 0:
                    self.callback(ctr[0], len(runlumis))
            return l
                             
        def readfile(f):
            ### read a single file
            # note: for internal use in getSingleMEs only, do not call.
            return [readlumi(lumi) for lumi in files[f]]
                             
        p = ThreadPool(self.nthreads)
        result = p.map(readfile, files.keys())
        p.close()
        return sum(result, [])
    
    def getSingleMEsToDataFrame(self, name, runlumis=None, verbose=False):
        ### return a pandas dataframe for a given monitoring element
        # note: the same naming convention is used as in the 2017/2018 csv input!

        if not pd: raise pandas_import_error

        # get the monitoring elements
        callback = None
        if verbose: callback='default'
        mes = self.getSingleMEs(name, runlumis=runlumis, callback=callback)

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
        if verbose: print('Start conversion to dict...')
        for idx,me in enumerate(mes):
            if verbose:
                if( idx>0 and idx%10==0 ): print('  entry {} of {}'.format(idx,len(mes)))
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
            dfdict['fromrun'].append(int(me.run))
            dfdict['fromlumi'].append(int(me.lumi))
            dfdict['hname'].append(str(me.name))
            dfdict['metype'].append(int(me.type))
            dfdict['histo'].append(list(histo))
            dfdict['entries'].append(int(np.sum(histo)))
            dfdict['Xmax'].append(float(xmax))
            dfdict['Xmin'].append(float(xmin))
            dfdict['Xbins'].append(int(nxbins))
            dfdict['Ymax'].append(float(ymax))
            dfdict['Ymin'].append(float(ymin))
            dfdict['Ybins'].append(int(nybins))
        # make a dataframe
        if verbose: print('Start conversion to DataFrame...')
        df = pd.DataFrame(dfdict)
        if verbose: print('Conversion finished, returning.')
        sys.stdout.flush()
        sys.stderr.flush()
        return df
