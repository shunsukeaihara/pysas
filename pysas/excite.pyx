# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: sources = excite.cpp

import numpy as np
cimport numpy as np
cimport cython
from cpython cimport bool


cdef class ExcitePulse:
    cdef int gen_samples, frameperiod, samplingrate
    cdef bool gauss
    def __cinit__(self, int samplingrate, int frameperiod, bool gauss):
        self.gen_samples = 0
        self.samplingrate = samplingrate
        self.frameperiod = frameperiod
        self.gauss = gauss

    cdef np.ndarray[np.float64_t, ndim=1, mode="c"] gen_frame(self, double cur_f0, double prev_f0):
        if cur_f0 == 0.0 or prev_f0 == 0.0:
            if self.gauss:
                return np.random.normal(0.0, 1.0, self.frameperiod)
            else:
                return np.random.random(self.frameperiod)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] ret = np.zeros(self.frameperiod)
        cdef double cur_hz = <double>self.samplingrate / cur_f0
        cdef double prev_hz = <double>self.samplingrate / prev_f0
        cdef double slope = (cur_hz - prev_hz) / <double>self.frameperiod
        cdef int i
        cdef double f0
        for i in range(self.frameperiod):
            f0 = prev_hz + slope * i
            if self.gen_samples > <int>f0:
                ret[i] = f0
                self.gen_samples -= <int>f0
            self.gen_samples += 1
        return np.sqrt(ret)
        
    def gen(self, np.ndarray[np.float64_t, ndim=1, mode="c"] f0):
        ret = np.zeros(f0.shape[0] * self.frameperiod)
        self.gen_samples = 0
        cdef int i, offset
        cdef double cur_f0, prev_f0
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] pluses
        prev_f0 = f0[0]
        for i in range(f0.size):
            cur_f0 = f0[i]
            pluses = self.gen_frame(cur_f0, prev_f0)
            prev_f0 = cur_f0
            offset = self.frameperiod * i
            ret[offset:offset+self.frameperiod] = pluses
        return ret
