# -*- coding: utf-8 -*-
from setuptools import setup, Extension, find_packages
from Cython.Distutils import build_ext
#from Cython.Build import cythonize
import sys
import numpy as np
from glob import glob
#sys.path.append('./src')


ext_modules = [Extension(
        'pysas.world',
        ["pysas/world.pyx"] + glob("lib/world/*.cpp"),
        include_dirs=['lib/world', np.get_include()],
        language="c++",
    )]

setup(
    name="pysas",
    description='Speech Analysis and Synthesis for Python',
    version="0.01",
    long_description=open('README.md').read(),
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    author='Shunsuke Aihara',
    url='https://github.com/shunsukeaihara/pysas',
    license="MIT License",
    include_package_data=True,
    test_suite='nose.collector',
    tests_require=['Nose', 'cython', 'numpy'])
