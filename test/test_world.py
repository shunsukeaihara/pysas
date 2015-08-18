# -*- coding: utf-8 -*-
from  unittest import TestCase
from nose.tools import eq_
import numpy as np

from pysas import World, waveread


class WorldTest(TestCase):
    def setUp(self):
        signal, samplingrate, _ = waveread("test/cmu_arctic/arctic_a0001.wav")
        self.signal = signal
        self.sampfreq = samplingrate
        self.world = World(samplingrate)

    def _callFUT(self, signal):
        return self.world.analyze(signal)

    def test_analyze(self):
        f0, spec, aperiod = self._callFUT(self.signal)
        eq_(f0.shape[0], spec.shape[0], aperiod.shape[0])
        out = self.world.synthesis(f0, spec, aperiod)
        assert len(out) == len(self.signal), "{}, {}".format(len(out), len(self.signal))
        
