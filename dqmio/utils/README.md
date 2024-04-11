**Some tools for per-lumisection DQMIO file diagnostics**

Intended for quick checks of the content of one or more DQMIO files.

### In case of problems
The DQMIO files are read using the DQMIOReader class, which in turn uses ROOT.
As usual with ROOT, strange errors might pop up depending on the details of your environment.
In expectation of a more stable DQMIOReader (hopefully avoiding ROOT altogether), here are a few tips:

- Check that you can run `import ROOT` without errors inside a `python` or `python3` session. If not, something is wrong with your PyROOT setup.
- Run inside a `CMSSW` environment (e.g. after running `cmsenv` in the appropriate `CMSSW_X/src` directory), preferably a `CMSSW` version that matches the version with which the DQMIO file was produced. For example, trying to read a file created with `CMSSW_14_0_4` after sourcing `CMSSW_12_4_6` was observed to give an error, while it worked after sourcing `CMSSW_14_0_4` (even though nothing in the reader code depends on `CMSSW` explicitly, it is probably a ROOT mismatch).
- Try on `lxplus`, `lxplus8` or `lxplus7`.

Making a DQMIOReader using `uproot` for example turns out to be non-trivial because of the complicated file structure of DQMIO files. In case you have ideas on how to make this work, feel free to contribute!
