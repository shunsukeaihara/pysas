# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: sources = d4c.cpp

cdef extern from "../../lib/world/d4c.h" nogil:
    void D4C(double *x, int x_length, int fs, double *time_axis, double *f0,
             int f0_length, int fft_size, double **aperiodicity)
