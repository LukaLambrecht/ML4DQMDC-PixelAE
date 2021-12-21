# -*- mode: python ; coding: utf-8 -*-
# this is a configuration file for pyinstaller
# it can be modified in order to include more paths and/or files in the executable

block_cipher = None

# modify the paths that need to be added to allow local python imports
added_paths = ['../utils', '../src', '../src/classifiers', '../src/cloudfitters']

# modify the files that need to be added (e.g. json files)
added_files = []
# add the default dcs-on and golden json files
added_files.append( ('../jsons','ML4DQM-DC/jsons') )
# add the source and utils files (not needed for importing, but for docurls)
added_files.append( ('../src','ML4DQM-DC/src') )
added_files.append( ('../utils','ML4DQM-DC/utils') )

a = Analysis(['gui.py'],
             pathex=added_paths,
             binaries=[],
             datas=added_files,
             hiddenimports=[],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,  
          [],
          name='gui',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
