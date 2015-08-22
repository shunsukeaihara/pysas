# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: sources = mcep.cpp

import numpy as np
cimport numpy as np
cimport cython
from cython.view cimport array as cyarray
from libc.string cimport memcpy, memset
from libc.stdlib cimport malloc, free
from libc.math cimport log, M_PI, sin, cos, atan
from libc.float cimport DBL_MAX


@cython.boundscheck(False)
@cython.wraparound(False)
def spec2mcep_from_matrix(np.ndarray[np.float64_t, ndim=2, mode="c"] spmat, int order, double alpha):
    """
    calucurate mel-cepstrum from spectrogram matrix
    """
    cdef int fftsize = (spmat.shape[1] - 1) * 2
    cdef np.ndarray[np.float64_t, ndim=2, mode="c"] logspmat = np.log(spmat)
    cdef np.ndarray[np.float64_t, ndim=2, mode="c"] c_mat = np.real(np.fft.irfft(logspmat, fftsize))

    cdef np.ndarray[np.float64_t, ndim=2, mode="c"] wc_mat = np.zeros(
        (c_mat.shape[0], order + 1),dtype=np.double)
    
    cdef int i
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] c_vec, wc_vec, prev
    prev = np.zeros(wc_mat.shape[1], dtype=np.float64)
    for i in range(c_mat.shape[0]):
        # ToDo: change ndarray to typed memoryview
        wc_vec = wc_mat[i]
        c_vec = c_mat[i]
        c_vec[0] /= 2.0
        freqt(<double *>c_vec.data, <double *>wc_vec.data, <double *>prev.data, c_vec.size, alpha, order)
    return wc_mat


@cython.boundscheck(False)
@cython.wraparound(False)
def spec2mcep(np.ndarray[np.float64_t, ndim=1, mode="c"] sp, int order, double alpha):
    """
    calucurate mel-cepstrum from spectrogram vector
    """
    cdef int fftsize = (sp.size - 1) * 2
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] logsp = np.log(sp)
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] c = np.real(np.fft.irfft(logsp, fftsize))
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] wc = np.zeros(order + 1, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] prev = np.zeros_like(wc)
    c[0] /= 2.0
    freqt(<double *>c.data, <double *>wc.data, <double *>prev.data, c.size, alpha, order)
    return wc


@cython.boundscheck(False)
@cython.wraparound(False)
def mcep2spec(np.ndarray[np.float64_t, ndim=1, mode="c"] mcep, double alpha, int fftsize):
    cdef int fsize = fftsize >> 1
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] c = np.zeros(fsize + 1, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] prev = np.zeros_like(c)
    freqt(<double *>mcep.data, <double *>c.data, <double *>prev.data, mcep.size, -alpha, fsize)
    c[0] *= 2.0
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] ret = c.copy()
    ret.resize(fftsize, refcheck=False)
    ret = ret
    cdef int i
    for i in range(fsize):
        ret[fftsize - i - 1] = ret[i+1]
    return np.exp(np.real(np.fft.rfft(ret)))


@cython.boundscheck(False)
@cython.wraparound(False)
def mcep2spec_from_matrix(np.ndarray[np.float64_t, ndim=2, mode="c"] mcepmat, double alpha, int fftsize):
    cdef int fsize = fftsize >> 1
    cdef np.ndarray[np.float64_t, ndim=2, mode="c"] c = np.zeros((mcepmat.shape[0], fsize + 1), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] prev = np.zeros(fsize + 1, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2, mode="c"] ret = np.zeros((mcepmat.shape[0], fftsize), dtype=np.float64)
    cdef int i
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] c_vec, mcep_vec, ret_vec
    for i in range(mcepmat.shape[0]):
        # ToDo: change ndarray to typed memoryview
        c_vec = c[i]
        mcep_vec = mcepmat[i]
        ret_vec = ret[i]
        freqt(<double *>mcep_vec.data, <double *>c_vec.data, <double *>prev.data, mcep_vec.size, -alpha, fsize)
        c_vec[0] *= 2.0
        np.copyto(ret_vec[:fsize + 1], c_vec)
        for i in range(fsize):
            ret_vec[fftsize - i - 1] = ret_vec[i+1]
    return np.exp(np.real(np.fft.rfft(ret)))

@cython.boundscheck(False)
@cython.wraparound(False)
def mcep2coef(np.ndarray[np.float64_t, ndim=1, mode="c"] mcep, double alpha):
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] coef = mcep.copy()
    cdef int i
    for i from mcep.size - 2 >= i >= 0:
        coef[i] = coef[i] - alpha * coef[i+1]
    return coef

@cython.boundscheck(False)
@cython.wraparound(False)
def coef2mcep(np.ndarray[np.float64_t, ndim=1, mode="c"] coef, double alpha):
    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] mcep = np.copy(coef)
    cdef size = coef.size
    cdef double d = mcep[size - 1]
    cdef double o = 0
    cdef int i
    for i from size - 2 >= i >= 0:
        o = mcep[i] + alpha * d
        d = mcep[i]
        mcep[i] = o
    return mcep


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void freqt(const double *c, double *wc, double *prev, int c_size, double alpha, int order) nogil:
    cdef int i, j
    for i in range(-(c_size - 1), 1):  # -(c_size - 1) 〜 0
        memcpy(prev, wc, sizeof(double)*(order + 1))
        if order >= 0:
            wc[0] = c[-i] + alpha * prev[0]
        if order >= 1:
            wc[1] = (1.0 - alpha ** 2) * prev[0] + alpha * prev[1]
        for j in range(2, order + 1):  # 2 ~ order -> length(wc) -> order + 1
            wc[j] = prev[j - 1] + alpha * (prev[j] - wc[j - 1])


@cython.boundscheck(False)
@cython.wraparound(False)
def estimate_alpha(int sampfreq, double start=0, double end=1.0, double step=0.001, int size=1000):
    """
    estimate alpha parameter for mel-cepstrum from sampling rate.
    original imprementation https://bitbucket.org/happyalu/mcep_alpha_calc
    """
    cdef double *melscale_vector = create_melscale_vector(sampfreq, size)
    cdef double *warping_vector = <double *>malloc(size * sizeof(double))
    cdef double i, 
    cdef double min_dist, best_alpha, dist
    best_alpha = 0.0
    min_dist = DBL_MAX
    for a from start <= a <= end by step:
        calc_warping_vector(warping_vector, a, size)
        dist = rms_distance_like(melscale_vector, warping_vector, size)
        if dist < min_dist:
            min_dist = dist
            best_alpha = a
    free(melscale_vector)
    free(warping_vector)
    return best_alpha
    

cdef double* create_melscale_vector(int sampfreq, int size) nogil:
    cdef double *vec = <double *>malloc(size * sizeof(double))
    cdef double step = (sampfreq / 2.0) / size
    cdef int i
    for i in range(size):
        vec[i] = (1000.0 / log(2)) * log(1.0+((step * i)/1000.0))
    cdef double last = vec[size - 1]
    for i in range(size):
        vec[i] /= last
    return vec


cdef void calc_warping_vector(double *vec, double alpha, int size) nogil:
    cdef double step = M_PI / size
    cdef int i
    cdef double omega, warpfreq, num, den
    for i in range(size):
        omega = step * i
        num = (1 - alpha * alpha) * sin(omega)
        den = (1 + alpha * alpha) * cos(omega) - 2 * alpha
        warpfreq = atan(num / den)
        if warpfreq < 0:
            warpfreq += M_PI
        vec[i] = warpfreq
    cdef double last = vec[size - 1]
    for i in range(size):
        vec[i] /= last

cdef double rms_distance_like(double *a, double *b, int size) nogil:
    cdef int i
    cdef double s = 0.0
    for i in range(size):
        s += (a[i] - b[i]) ** 2
    return s
