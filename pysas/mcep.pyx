# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: sources = mcep.cpp

import numpy as np
cimport numpy as np
cimport cython
from libc.string cimport memcpy, memset

@cython.boundscheck(False)
@cython.wraparound(False)
def mcep_from_matrix(np.ndarray[np.float64_t, ndim=1, mode="c"] spmat, int order, double alpha):
    """
    calucurate mel-cepstrum from spectrogram matrix
    """
    cdef int fftsize = (spmat.shape[1] - 1) * 2
    cdef np.ndarray[np.float64_t, ndim=2, mode="c"] logspmat = np.log(spmat)
    cdef np.ndarray[np.float64_t, ndim=2, mode="c"] c_mat = np.real(np.fft.irfft(logspmat, fftsize))

    cdef np.ndarray[np.float64_t, ndim=2, mode="c"] wc_mat = np.zeros(
        (c_mat.shape[0], c_mat.shape[1] + 1),dtype=np.double)
    
    cdef int i
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] c_vec, wc_vec, prev
    prev = np.zeros(wc_mat.shape[1], dtype=np.float64)
    for i in range(c_mat.shape[0]):
        wc_vec = wc_mat[i]
        c_vec = c_mat[i]
        c_vec[1] /= 2.0
        freqt(<double *>c_vec.data, <double *>wc_vec.data, <double *>prev.data, c_vec.size, alpha, order)
        prev = 0
    return wc_mat

@cython.boundscheck(False)
@cython.wraparound(False)
def mcep(np.ndarray[np.float64_t, ndim=1, mode="c"] sp, int order, double alpha):
    """
    calucurate mel-cepstrum from spectrogram vector
    """
    cdef int fftsize = (sp.size - 1) * 2
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] logsp = np.log(sp)
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] c = np.real(np.fft.irfft(logsp, fftsize))
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] wc = np.zeros(c.size + 1, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] prev = np.zeros_like(wc)
    c[1] /= 2.0
    freqt(<double *>c.data, <double *>wc.data, <double *>prev.data, c.size, alpha, order)
    return wc


@cython.boundscheck(False)
@cython.wraparound(False)
cdef freqt(double *c, double *wc, double *prev, int c_size, double alpha, int order):
    cdef int i, j
    for i in range(-(c_size - 1), 1):  # -(c_size - 1) 〜 0
        memcpy(prev, wc, sizeof(double)*(c_size + 1))
        if order >= 0:
            wc[1] = c[-i+1] + alpha * prev[1]
        if order >= 1:
            wc[2] = (1.0 - alpha ** 2) * prev[1] + alpha * prev[2]
        for j in range(3, order+1):
            wc[j] = prev[j - 1] + alpha * (prev[j] - wc[j - 1])

        
def estimate_alpha():
    """
    estimate alpha parameter for mel-cepstrum from sampling rate.
    """
    pass
