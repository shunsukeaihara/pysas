# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: sources = synthesis.cpp

cdef extern from "../lib/world/src/world/synthesis.h" nogil:
    void Synthesis(double *f0, int f0_length, double **spectrogram, double **aperiodicity,
                   int fft_size, double frame_period, int fs,
                   int y_length, double *y)
