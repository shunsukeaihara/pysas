# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: sources = cysyntesis.cpp

import numpy as np
cimport numpy as np


cdef class Synthesis(object):
    cdef int frameperiod
    cdef object filter
    
    def __init__(self, int frameperiod, object filter):
        self.frameperiod = frameperiod
        self.filter = filter

    def synthesis_frame(self, np.ndarray[np.float64_t, ndim=1, mode="c"] cur_coef,
                        np.ndarray[np.float64_t, ndim=1, mode="c"] prev_coef,
                        np.ndarray[np.float64_t, ndim=1, mode="c"] pulse):
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] ret = np.zeros_like(pulse)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] slope = (cur_coef - prev_coef) / self.frameperiod
        cdef int i
        cdef double scaled
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] coef
        for i in range(self.frameperiod):
            coef = prev_coef + slope * i
            scaled = pulse[i] * np.exp(coef[0])
            ret[i] = self.filter.filter(scaled, coef)
        return ret
        
    def synthesis(self, np.ndarray[np.float64_t, ndim=1, mode="c"] pulse,
                  np.ndarray[np.float64_t, ndim=2, mode="c"] coef_mat):
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] ret = np.zeros_like(pulse)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] prev_coef = np.zeros_like(coef_mat[0])
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] cur_coef
        cdef int i, offset
        for i in range(coef_mat.shape[0]):
            offset = self.frameperiod * i
            cur_coef = coef_mat[i]
            ret[offset:offset+self.frameperiod] = self.synthesis_frame(cur_coef, prev_coef, pulse[offset:offset+self.frameperiod])
            prev_coef = cur_coef
        return ret
