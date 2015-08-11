# -*- coding: utf-8 -*-
from setuptools import setup, Extension
from Cython.Distutils import build_ext
#from Cython.Build import cythonize
import sys
import numpy as np
from glob import glob
sys.path.append('./src')


ext_modules = [Extension(
        'pyworld',
        ["src/pyworld.pyx"] + glob("lib/world/*.cpp"),
        include_dirs=['lib/world', np.get_include()],
        language="c++",
    )]

setup(
    name="pyworld",
    description='Python wrapper for World(Speech Analysis and Synthesis System)',
    version="0.01",
    long_description=open('README.md').read(),
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    author='Shunsuke Aihara',
    url='https://github.com/shunsukeaihara/pyworld',
    license="New BSD License",
    include_package_data=True,
    test_suite='nose.collector',
    tests_require=['Nose', 'cython', 'numpy'])
