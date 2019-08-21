# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: sources = dio.cpp

cdef extern from "../lib/world/src/world/dio.h" nogil:
    cdef cppclass DioOption:
        double f0_floor
        double f0_ceil
        double channels_in_octave
        double frame_period  # msec
        int speed  # (1, 2, ..., 12)
        double allowed_range  # // Threshold used for fixing the F0 contour.
    
    void Dio(double *x, int x_length, int fs, const DioOption *option, double *time_axis, double *f0)
    void InitializeDioOption(DioOption *option)
    int GetSamplesForDIO(int fs, int x_length, double frame_period)
