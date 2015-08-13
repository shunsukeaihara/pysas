# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: sources = pyworld.cpp
import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free

cimport dio
cimport d4c
cimport stonemask
cimport cheaptrick

import wave

def waveread(path):
    """
    read signals from wavefile, only support monoral and 8 or 16bit data
    @return normalized signals between [-1,1], sampling rate[Hz], bit rate[bits]
    """
    wf = wave.open(path, 'rb')
    channels = wf.getnchannels()
    assert channels == 1, "wave file should be monoral"
    
    bits = wf.getsampwidth()  # bit rate
    assert bits==8 or bits==16, "bit rate should be 8 or 16bits"
    
    fs = wf.getframerate()  # sampling rate(Hz)
    datalength = wf.getnframes()
    raw = wf.readframes(datalength)

    
    if bits == 8:
        signals = np.float64(np.fromstring(raw, dtype=np.uint8) - 127) / 127.0
    elif bits == 16:
        signals = np.float64(np.fromstring(raw, dtype=np.int16)) / 32767.0
    return signals, fs, bits


cdef double ** malloc_matrix(int size, int dims):
    cdef int i
    cdef double **mat = <double **>malloc(size * sizeof(double *))
    for i in range(size):
        mat[i] = <double *>malloc(dims * sizeof(double))
    return mat


cdef void free_matrix(double ** mat, int size):
    cdef int i
    for i in range(size):
        free(mat[i])
    free(mat)
        
    
cdef class World:
    """
    """

    cdef int samplingrate, bitrate, fft_size, envelope_size
    cdef double freamperiod
    
    def __init__(self, np.ndarray[np.float64_t, ndim=1, mode="c"] signal,
                 int samplingrate, int bitrate, double freamperiod=0.5):

        self.freamperiod = freamperiod
        self.samplingrate = samplingrate
        self.bitrate = bitrate
        self.fft_size = cheaptrick.GetFFTSizeForCheapTrick(self.samplingrate)
        self.envelope_size = self.fft_size / 2 + 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def analyze(self, np.ndarray[np.float64_t, ndim=1, mode="c"] signal):
        f0, time_axis = self.estimate_f0(signal)
        cdef double **c_spectrogram
        cdef double **c_aperiodicity
        
        c_spectrogram = self.estimate_spectral_envelope(signal, f0, time_axis)
        c_aperiodicity = self.estimate_aperiodicity(signal, f0, time_axis)

        cdef np.ndarray[np.float64_t, ndim=2, mode="c"] spectrogram, aperiodicity
        spectrogram = np.zeros((f0.size, self.envelope_size), dtype=np.float64)
        aperiodicity = np.zeros((f0.size, self.envelope_size), dtype=np.float64)
        
        cdef int i, j
        for i in range(f0.size):
            for j in range(self.envelope_size):
                spectrogram[i, j] = c_spectrogram[i][j]
                aperiodicity[i, j] = c_aperiodicity[i][j]
        free_matrix(c_spectrogram, f0.size)
        return f0, spectrogram, aperiodicity

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def synthesis(self,
                 np.ndarray[np.float64_t, ndim=1, mode="c"] f0,
                 np.ndarray[np.float64_t, ndim=2, mode="c"] spectrogram,
                 np.ndarray[np.float64_t, ndim=2, mode="c"] aperiodicity):
        pass
        
    cdef estimate_f0(self, np.ndarray[np.float64_t, ndim=1, mode="c"] signal):
        """
        estimate F0 from speech signal
        @signal speech signal
        @return estimated f0 series
        """
        f0_length = dio.GetSamplesForDIO(self.samplingrate, signal.size, self.freamperiod)
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] f0, refined_f0, time_axis
        f0 = np.zeros(f0_length, dtype=np.float64)
        refined_f0 = np.zeros(f0_length, dtype=np.float64)
        time_axis = np.zeros(f0_length, dtype=np.float64)

        cdef dio.DioOption option
        dio.InitializeDioOption(&option)
        
        # modify parameter
        option.frame_period = self.frameperiod
        option.speed = 1
        option.f0_floor = 71.0
        option.allowed_range = 0.1
  
        dio.Dio(<double *> signal.data, signal.size, self.samplingrate,
                option, <double *> time_axis.data, <double *> f0.data)
        stonemask.StoneMask(<double *> signal.data, signal.size, self.samplingrate,
                            <double *> time_axis.data, <double *> f0.data,
                            f0_length, <double *> refined_f0.data)
        return refined_f0, time_axis

    
    cdef double ** estimate_spectral_envelope(self,
            np.ndarray[np.float64_t, ndim=1, mode="c"] signal,
            np.ndarray[np.float64_t, ndim=1, mode="c"] f0,
            np.ndarray[np.float64_t, ndim=1, mode="c"] time_axis):
        cdef double **spectrogram = malloc_matrix(f0.size, self.envelope_size)

        # estimete spectral envelope
        cheaptrick.CheapTrick(<double *> signal.data, signal.size, self.samplingrate,
                              <double *> time_axis.data, <double *> f0.data,
                              f0.size, spectrogram)
        return spectrogram

    cdef double ** estimate_aperiodicity(self,
            np.ndarray[np.float64_t, ndim=1, mode="c"] signal,
            np.ndarray[np.float64_t, ndim=1, mode="c"] f0,
            np.ndarray[np.float64_t, ndim=1, mode="c"] time_axis):
        cdef double **aperiodicity = malloc_matrix(f0.size, self.envelope_size)

        # estimete aperiodicity
        d4c.D4C(<double *> signal.data, signal.size, self.samplingrate,
                <double *> time_axis.data, <double *> f0.data,
                f0.size, self.fft_size, aperiodicity)
        return aperiodicity
