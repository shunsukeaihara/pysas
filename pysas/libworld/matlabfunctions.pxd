# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: sources = matlabfunctions.cpp

cdef extern from "../lib/world/src/world/matlabfunctions.h" nogil:
    void interp1(double *x, double *y, int x_length, double *xi, int xi_length, double *yi)
