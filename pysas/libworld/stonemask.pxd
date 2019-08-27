# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: sources = stonemask.cpp

cdef extern from "../lib/world/src/world/stonemask.h" nogil:
    void StoneMask(double *x, int x_length, int fs, double *time_axis, double *f0,
                   int f0_length, double *refined_f0);
