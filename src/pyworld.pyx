# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: sources = pyworld.cpp
import numpy as np
cimport numpy as np

cimport dio

import wave

def waveread(path):
    """
    read signals from wavefile, only support monoral and 8 or 16bit data
    @return normalized signals between [-1,1], sampling rate, bit rate
    """
    wf = wave.open(path, 'rb')
    datalength = wf.getnframes()
    bits = wf.getsampwidth()  # bit rate
    
    fs = wf.getframerate()  # sampling rate(Hz)
    raw = wf.readframes(datalength)
    if bits == 8:
        signals = np.float64(np.fromstring(raw, dtype=np.uint8) - 127) / 127.0
    elif bits == 16:
        signals = np.float64(np.fromstring(raw, dtype=np.int16)) / 32767.0
    return signals, fs, bits

    
cdef class World:
    """
    """
    def __init__(self, signal, sampling_rate):
        pass
    
