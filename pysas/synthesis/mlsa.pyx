# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: sources = mlsa.cpp

# porting from https://gist.github.com/r9y9/7735120

import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free
from cpython cimport PyObject, Py_INCREF, Py_DECREF


cdef double[:] pade_coef4 = np.array([1.0, 4.999273e-1, 1.067005e-1, 1.170221e-2, 5.656279e-4])
cdef double[:] pade_coef5 = np.array([1.0, 4.999391e-1, 1.107098e-1, 1.369984e-2, 9.564853e-4, 3.041721e-5])


cdef class Filter:
    cdef double alpha
    cdef int order
    cdef double[:] delay

    def __cinit__(self, int order, double alpha):
        self.order = order
        self.alpha = alpha
        self.delay = np.zeros(order + 1, dtype=np.float64)

    cdef filter(self, double x, np.ndarray[np.float64_t, ndim=1, mode="c"] coefficients):
        cdef double result = 0.0
        self.delay[0] = x
        self.delay[1] = (1.0 - self.alpha ** 2) * self.delay[0] + self.alpha * self.delay[1]
        cdef int i
        for i in range(2, coefficients.size):
            self.delay[i] = self.delay[i] + self.alpha * (self.delay[i + 1] - self.delay[i - 1])
            result += self.delay[i] * coefficients[i]
        if coefficients.shape[0] == 2:
            result += self.delay[1] * coefficients[1]
        for i in range(-(self.delay.size - 1), -1):
            i = -i
            self.delay[i] = self.delay[i - 1]
        return result

cdef class CascadeFilter:
    cdef int pade_order, order, filter_num
    cdef double alpha
    cdef double[:] delay
    cdef double[:] pade_coefficients
    cdef PyObject **filters

    def __cinit__(self, int order, double alpha, int pade_order):
        self.pade_order = pade_order
        self.filter_num = pade_order + 1
        self.order = order
        self.alpha = alpha
        self.delay = np.zeros(self.filter_num, dtype=np.float64)
        self.filters = <PyObject **>malloc(sizeof(PyObject *) * self.filter_num)
        for i in range(self.filter_num):
            filt = Filter(order, alpha)
            Py_INCREF(filt)
            self.filters[i] = <PyObject *>filt
        if pade_order == 4:
            self.pade_coefficients = pade_coef4
        else:
            self.pade_coefficients = pade_coef5

    def __dealloc__(self):
        cdef int i
        for i in range(self.filter_num):
            Py_DECREF(<object>self.filters[i])
        free(self.filters)

    cdef filter(self, double x, np.ndarray[np.float64_t, ndim=1, mode="c"] coefficients):
        cdef double result = 0.0
        cdef double feedback = 0.0
        cdef int i
        for i in range(-(len(self.pade_coefficients) - 1), 0):
            i = -i
            self.delay[i] = (<Filter>self.filters[i]).filter(self.delay[i - 1], coefficients)
            val = self.delay[i] * self.pade_coefficients[i]
            if i % 2 == 1:
                feedback += val
            else:
                feedback -= val
            result += val
        self.delay[0] = feedback + x
        result += self.delay[0]
        return result

cdef class MLSAFilter(object):
    cdef int pade_order, order
    cdef double alpha
    cdef CascadeFilter f1
    cdef CascadeFilter f2

    def __init__(self, int order, double alpha, int pade_order):
        assert pade_order == 4 or pade_order == 5, "order of pade must be 4 or 5."
        self.f1 = CascadeFilter(2, alpha, pade_order)
        self.f2 = CascadeFilter(order + 1, alpha, pade_order)

    def filter(self, x, coefficients):
        coef = np.array([0, coefficients[1]])
        return self.f2.filter(self.f1.filter(x, coef), coefficients)
