# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: sources = cheaptrick.cpp

cdef extern from "../lib/world/src/world/cheaptrick.h" nogil:
    cdef cppclass CheapTrickOption:
        double q1
        double f0_floor
        int fft_size
    void InitializeCheapTrickOption(int fs, CheapTrickOption *option)
    void CheapTrick(double *x, int x_length, int fs, double *time_axis, double *f0,
                    int f0_length, CheapTrickOption *option, double **spectrogram)
    int GetFFTSizeForCheapTrick(int fs, CheapTrickOption *option)
