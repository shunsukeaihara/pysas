# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: sources = d4c.cpp

cdef extern from "../lib/world/src/world/d4c.h" nogil:
    cdef cppclass D4COption:
        double threshold

    void InitializeD4COption(D4COption *option)
    void D4C(double *x, int x_length, int fs, double *time_axis, double *f0,
             int f0_length, int fft_size, D4COption *option, double **aperiodicity)

