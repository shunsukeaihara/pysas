# -*- coding: utf-8 -*-
from unittest import TestCase
# from nose.tools import eq_
import numpy as np

from pysas import waveread, World
from pysas.mcep import estimate_alpha, spec2mcep_from_matrix, mcep2coef
from pysas.synthesis.mlsa import MLSAFilter
from pysas.synthesis import Synthesis
from pysas.excite import ExcitePulse


class SynthesisTest(TestCase):
    def setUp(self):
        signal, samplingrate, _ = waveread("test/cmu_arctic/arctic_a0001.wav")
        self.world = World(samplingrate)
        self.alpha = estimate_alpha(samplingrate)
        self.samplingrate = samplingrate
        self.signal = signal
        self.f0, self.spec_mat, _ = self.world.analyze(signal)
        self.ep = ExcitePulse(16000, 80, False)
        self.order = 24

    def test_synthesis_filter(self):
        excite = self.ep.gen(self.f0)
        mcep_mat = spec2mcep_from_matrix(self.spec_mat, self.order, self.alpha)
        coef_mat = []
        for i in range(mcep_mat.shape[0]):
            coef_mat.append(mcep2coef(mcep_mat[i], 0.41))
        coef_mat = np.array(coef_mat)
        mlsa = MLSAFilter(self.order, self.alpha, 5)
        syn = Synthesis(80, mlsa)
        syn.synthesis(excite, coef_mat)
