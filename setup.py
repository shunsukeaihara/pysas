# -*- coding: utf-8 -*-
from setuptools import setup, Extension, find_packages
import sys
from glob import glob
sys.path.append('./src')
sys.path.append('./test')

try:
    from Cython.Distutils import build_ext
except ImportError:
    def build_ext(*args, **kwargs):
        from Cython.Distutils import build_ext
        return build_ext(*args, **kwargs)


class lazy_extlist(list):
    def __init__(self, callback):
        self._list, self.callback = None, callback

    def c_list(self):
        if self._list is None:
            self._list = self.callback()
        return self._list

    def __iter__(self):
        for e in self.c_list():
            yield e

    def __getitem__(self, ii):
        return self.c_list()[ii]

    def __len__(self):
        return len(self.c_list())


def extensions():
    __builtins__.__NUMPY_SETUP__ = False
    import numpy as np
    ext_modules = [
        Extension(
            'pysas.world',
            ["pysas/world.pyx"] + glob("lib/world/*.cpp"),
            include_dirs=['lib/world', np.get_include()],
            extra_compile_args=["-O3"],
            language="c++",
        ),
        Extension(
            'pysas.mcep',
            ["pysas/mcep.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3"],
            language="c++",
        ),
        Extension(
            'pysas.excite',
            ["pysas/excite.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3"],
            language="c++",
        ),
        Extension(
            'pysas.synthesis.mlsa',
            ["pysas/synthesis/mlsa.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3"],
            language="c++",
        )
    ]
    return ext_modules


setup(
    name="pysas",
    description='Speech Analysis and Synthesis for Python',
    version="0.3.3",
    long_description=open('README.rst').read(),
    packages=find_packages(),
    install_requires=["numpy", "cython", 'nose'],
    setup_requires=["numpy", "cython", 'nose'],
    ext_modules=lazy_extlist(extensions),
    cmdclass={'build_ext': build_ext},
    author='Shunsuke Aihara',
    author_email="aihara@argmax.jp",
    url='https://github.com/shunsukeaihara/pysas',
    license="MIT License",
    include_package_data=True,
    test_suite='nose.collector',
    tests_require=['nose', 'cython', 'numpy'],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering"
    ])
