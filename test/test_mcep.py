# -*- coding: utf-8 -*-
from  unittest import TestCase
from nose.tools import eq_
import numpy as np

from pysas import waveread
from pysas.mcep import estimate_alpha, spec2mcep, mcep2spec, mcep2coef, coef2mcep
from pysas.mcep import mcep2spec_from_matrix, spec2mcep_from_matrix
from pysas.synthesis_filter.mlsa import MLSAFilter
from pysas import World


class MecpTest(TestCase):
    def setUp(self):
        signal, samplingrate, _ = waveread("test/cmu_arctic/arctic_a0001.wav")
        self.signal = signal
        self.samplingrate = samplingrate
        start = 80*200
        self.windowsize = 1024
        signal = signal[start:start+self.windowsize] * np.blackman(self.windowsize)
        self.pspec = np.absolute(np.fft.fft(signal) ** 2)[:(self.windowsize>>1) + 1]
        self.alpha = 0.41

    def test_spec2mcep(self):
        mcep = spec2mcep(self.pspec, 20, self.alpha)
        spec = mcep2spec(mcep, self.alpha, self.windowsize)
        sqerr = np.sqrt((np.log(self.pspec) - np.log(spec)) ** 2).sum() / self.pspec.size
        assert sqerr < 1.2, sqerr
        
    def test_mcep2coef(self):
        mcep = spec2mcep(self.pspec, 20, self.alpha)
        coef = mcep2coef(mcep, self.alpha)
        mcep2 = coef2mcep(coef, self.alpha)
        sqerr = np.sqrt((mcep - mcep2) ** 2).sum()
        assert sqerr < 0.0001, sqerr

    def test_from_matrix(self):
        world = World(self.samplingrate)
        _, spec, _ = world.analyze(self.signal)
        mcepmat = spec2mcep_from_matrix(spec, 20, self.alpha)
        mcep = spec2mcep(spec[300], 20, self.alpha)
        assert (mcep == mcepmat[300]).all()
        specmat = mcep2spec_from_matrix(mcepmat, self.alpha, self.windowsize)
        spec = mcep2spec(mcep, self.alpha, self.windowsize)
        assert (spec == specmat[300]).all()
        
class EstimateAlphaTest(TestCase):
    def _callFUT(self, sampfreq):
        return estimate_alpha(sampfreq)
    def test_8k(self):
        eq_(round(self._callFUT(8000), 3), 0.312)
    def test_16k(self):
        eq_(round(self._callFUT(16000), 3), 0.41)
    def test_44k(self):
        eq_(round(self._callFUT(44100), 3), 0.544)

