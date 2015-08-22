# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from  unittest import TestCase
from nose.tools import eq_
import numpy as np

from pysas import waveread
from pysas.mcep import estimate_alpha, spec2mcep, mcep2spec, mcep2coef, coef2mcep
from pysas.synthesis_filter.mlsa import MLSAFilter

class SynthesisTest(TestCase):
    def setUp(self):
        signal, _, _ = waveread("test/cmu_arctic/arctic_a0001.wav")
        start = 80*200
        self.windowsize = 1024
        window = signal[start:start+self.windowsize] * np.blackman(self.windowsize)
        self.pspec = np.absolute(np.fft.fft(signal) ** 2)[:(self.windowsize>>1) + 1]
        self.alpha = 0.41

    def test_synthesis_filter(self):
        mcep = spec2mcep(self.pspec, 20, self.alpha)
        coef = mcep2coef(mcep, self.alpha)
        mlsa = MLSAFilter(20, 0.41, 5)
