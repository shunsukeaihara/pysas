# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: sources = world.cpp
import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport log, exp

from libworld cimport dio
from libworld cimport d4c
from libworld cimport stonemask
from libworld cimport cheaptrick
from libworld cimport matlabfunctions
from libworld cimport synthesis

    
cdef class World:
    """
    cython Wrapper For World
    """
    cdef int samplingrate, fft_size, envelope_size
    cdef double frameperiod
    cdef cheaptrick.CheapTrickOption cheaptrickOption
    cdef d4c.D4COption d4cOption
    
    def __init__(self, int samplingrate, double frameperiod=5.0):
        """
        @param samplingrate sampling rate of signal(integaer)
        @param frameperiod frame period(msec, default=5.0msec)
        """
        self.frameperiod = frameperiod  # ms
        self.samplingrate = samplingrate
        cheaptrick.InitializeCheapTrickOption(self.samplingrate, &self.cheaptrickOption)
        d4c.InitializeD4COption(&self.d4cOption)
        self.fft_size = cheaptrick.GetFFTSizeForCheapTrick(self.samplingrate, &self.cheaptrickOption)
        self.envelope_size = self.fft_size / 2 + 1

    def fftsize(self):
        return self.fft_size

    def envelopesize(self):
        return self.envelope_size
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def analyze(self, np.ndarray[np.float64_t, ndim=1, mode="c"] signal):
        """
        @return f0, spectral envelope, aperiodicity
        """
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] f0, time_axis
        f0, time_axis = self.estimate_f0(signal)
        
        cdef np.ndarray[np.float64_t, ndim=2, mode="c"] spectrogram, aperiodicity
        spectrogram = np.zeros((f0.size, self.envelope_size), dtype=np.float64)
        aperiodicity = np.zeros((f0.size, self.envelope_size), dtype=np.float64)

        cdef double **c_spectrogram = <double **>malloc(f0.size * sizeof(double *))
        cdef double **c_aperiodicity = <double **>malloc(f0.size * sizeof(double *))

        cdef int i
        # copy pointer to c array
        for i in range(f0.size):
            c_spectrogram[i] = &(<double *>spectrogram.data)[i * spectrogram.shape[1]]
            c_aperiodicity[i] = &(<double *>aperiodicity.data)[i * aperiodicity.shape[1]]

        cheaptrick.CheapTrick(<double *>signal.data, signal.size, self.samplingrate,
                              <double *>time_axis.data, <double *>f0.data,
                              f0.size, &self.cheaptrickOption, c_spectrogram)
        d4c.D4C(<double *>signal.data, signal.size, self.samplingrate,
                <double *>time_axis.data, <double *>f0.data,
                f0.size, self.fft_size, &self.d4cOption, c_aperiodicity)

        free(c_spectrogram)
        free(c_aperiodicity)
        return f0, spectrogram, aperiodicity

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def spectral_envelope(self, np.ndarray[np.float64_t, ndim=1, mode="c"] signal):
        """
        estimate F0 and spectral envelope from speech signal
        @signal speech signal
        @return f0, spectral envelope
        """
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] f0, time_axis
        f0, time_axis = self.estimate_f0(signal)
        
        cdef np.ndarray[np.float64_t, ndim=2, mode="c"] spectrogram
        spectrogram = np.zeros((f0.size, self.envelope_size), dtype=np.float64)

        cdef double **c_spectrogram = <double **>malloc(f0.size * sizeof(double *))

        cdef int i
        # copy pointer to c array
        for i in range(f0.size):
            c_spectrogram[i] = &(<double *>spectrogram.data)[i * spectrogram.shape[1]]

        cheaptrick.CheapTrick(<double *>signal.data, signal.size, self.samplingrate,
                              <double *>time_axis.data, <double *>f0.data,
                              f0.size, &self.cheaptrickOption, c_spectrogram)

        free(c_spectrogram)
        return f0, spectrogram
    
    def f0_scaling(self, np.ndarray[np.float64_t, ndim=1, mode="c"] f0, double shift):
        if shift <= 0:
            return f0
        return f0 * shift
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def spectral_stretching(self, np.ndarray[np.float64_t, ndim=2, mode="c"] spectrogram,
                            double ratio):
        if ratio <= 0:
            return spectrogram
        cdef int dim = spectrogram.shape[1]
        assert dim == self.envelope_size, "dimension of spectrogram should be same as self.envelope_size"
        cdef double *freq_axis1 = <double *>malloc(dim * sizeof(double))
        cdef double *freq_axis2 = <double *>malloc(dim * sizeof(double))
        cdef double *spectrum1 = <double *>malloc(dim * sizeof(double))
        cdef double *spectrum2 = <double *>malloc(dim * sizeof(double))

        cdef int i, j, k
        k = <int>(self.fft_size / 2.0 * ratio)
        for i in range(dim):
            freq_axis1[i] = ratio * i /self.fft_size * self.samplingrate
            freq_axis2[i] = <double>i / self.fft_size * self.samplingrate

        for i in range(spectrogram.size):
            for j in range(dim):
                spectrum1[j] = log(spectrogram[i, j])
            matlabfunctions.interp1(freq_axis1, spectrum1, dim,
                                    freq_axis2, dim, spectrum2);
            for j in range(dim):
                spectrogram[i, j] = exp(spectrum2[j])
            if (ratio < 1.0 and k > 0):
                for j in range(k, dim):
                    spectrogram[i, j] = spectrogram[i, k - 1]
        free(freq_axis1)
        free(freq_axis2)
        free(spectrum1)
        free(spectrum2)
        return spectrogram

    def synthesis(self,
                  np.ndarray[np.float64_t, ndim=1, mode="c"] f0,
                  np.ndarray[np.float64_t, ndim=2, mode="c"] spectrogram,
                  np.ndarray[np.float64_t, ndim=2, mode="c"] aperiodicity):
        assert f0.size == spectrogram.shape[0] == aperiodicity.shape[0], "all arguments should be same sample size"
        assert spectrogram.shape[1] == aperiodicity.shape[1], "spectrogram and aperiodicity should be same size"
                
        cdef double **c_spectrogram = <double **>malloc(f0.size * sizeof(double *))
        cdef double **c_aperiodicity = <double **>malloc(f0.size * sizeof(double *))

        cdef int result_length = <int>((f0.size - 1) * self.frameperiod / 1000.0 * self.samplingrate) + 1
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] result = np.zeros(result_length, dtype=np.float64)

        cdef int i
        # copy to c array
        for i in range(spectrogram.shape[0]):
            c_spectrogram[i] = &(<double *>spectrogram.data)[i * spectrogram.shape[1]]
            c_aperiodicity[i] = &(<double *>aperiodicity.data)[i * aperiodicity.shape[1]]
        
        synthesis.Synthesis(<double *>f0.data, f0.size, c_spectrogram, c_aperiodicity,
                            self.fft_size, self.frameperiod, self.samplingrate,
                            result_length, <double *>result.data)

        free(c_spectrogram)
        free(c_aperiodicity)
        return result
    
    cdef estimate_f0(self, np.ndarray[np.float64_t, ndim=1, mode="c"] signal):
        """
        estimate F0 from speech signal
        @signal speech signal
        @return estimated f0 series
        """
        f0_length = dio.GetSamplesForDIO(self.samplingrate, signal.size, self.frameperiod)
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
                &option, <double *> time_axis.data, <double *> f0.data)
        stonemask.StoneMask(<double *> signal.data, signal.size, self.samplingrate,
                            <double *> time_axis.data, <double *> f0.data,
                            f0_length, <double *> refined_f0.data)
        return refined_f0, time_axis
    
