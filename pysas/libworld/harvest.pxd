# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: sources = harvest.cpp

cdef extern from "../lib/world/src/world/harvest.h" nogil:
    cdef cppclass HarvestOption:
        double f0_floor
        double f0_ceil
        double frame_period  # msec
    
    void Harvest(double *x, int x_length, int fs, const HarvestOption *option, double *time_axis, double *f0)
    void InitializeHarvestOption(HarvestOption *option)
    int GetSamplesForHarvest(int fs, int x_length, double frame_period)
