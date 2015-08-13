# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: sources = cheaptrick.cpp

cdef extern from "../lib/world/cheaptrick.h" nogil:
    void CheapTrick(double *x, int x_length, int fs, double *time_axis, double *f0,
                    int f0_length, double **spectrogram)
    int GetFFTSizeForCheapTrick(int fs)
