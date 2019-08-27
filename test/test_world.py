# -*- coding: utf-8 -*-
from unittest import TestCase
from nose.tools import eq_

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

    def test_f0_estimate(self):
        dio_f0, dio_tx = self.world.dio(self.signal)
        harvest_f0, harvest_tx = self.world.harvest(self.signal)
        assert harvest_f0.shape[0] == dio_f0.shape[0]
        assert harvest_tx.shape[0] == dio_tx.shape[0]

    def test_sbs(self):
        f0, tx = self.world.estimate_f0(self.signal)
        spec = self.world.cheaptrick(self.signal, f0, tx)
        aperiod = self.world.d4c(self.signal, f0, tx)
        out = self.world.synthesis(f0, spec, aperiod)
        assert len(out) == len(self.signal), "{}, {}".format(len(out), len(self.signal))
