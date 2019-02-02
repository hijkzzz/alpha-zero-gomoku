"""
setup.py file for SWIG example
"""

import sys

from distutils.core import setup, Extension


swig_module = Extension('_swig',
                           include_dirs=['.'],
                           sources=['swig_wrap.cxx', 'mcts.cpp', 'gomoku.cpp', 'lib_torch.cpp'],
                           )

setup(name='swig',
      version='0.1',
      author="SWIG Docs",
      description="""Simple swig example from docs""",
      ext_modules=[swig_module],
      py_modules=["swig"],
      )
