# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: sources = dio.cpp

cdef extern from "../lib/world/dio.h" nogil:
    struct DioOption:
        pass
    
    void Dio(double *x, int x_length, int fs, const DioOption option, double *time_axis, double *f0)
    void InitializeDioOption(DioOption *option)
    int GetSamplesForDIO(int fs, int x_length, double frame_period)
