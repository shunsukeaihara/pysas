# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: sources = mcep.cpp

import numpy as np
cimport numpy as np


def mcep_from_matrix(np.ndarray[np.float64_t, ndim=1, mode="c"] spmat, int order, double alpha):
    """
    calucurate mel-cepstrum from spectrogram matrix
    """
    cdef int fftsize = (spmat.shape[1] - 1) * 2
    cdef np.ndarray[np.float64_t, ndim=2, mode="c"] logspmat = np.log(spmat)
    cdef np.ndarray[np.float64_t, ndim=2, mode="c"] c = np.real(np.fft.irfft(logspmat, fftsize))
    
def mcep(np.ndarray[np.float64_t, ndim=1, mode="c"] sp, int order, double alpha):
    """
    calucurate mel-cepstrum from spectrogram
    """
    cdef int fftsize = (sp.size - 1) * 2
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] logsp = np.log(sp)
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] c = np.real(np.fft.irfft(logsp, fftsize))
    c[1] /= 2.0
    cdef double *carray = <double *>c.data
    pass

cdef freqt(double *c, int order, double alpha):
    pass


def estimate_alpha():
    """
    estimate alpha parameter for mel-cepstrum from sampling rate.
    """
    pass
