# -*- coding: utf-8 -*-

from  unittest import TestCase
from nose.tools import eq_
import numpy as np

from pysas.mcep import estimate_alpha


class MecpTest(TestCase):
    pass

class EstimateAlphaTest(TestCase):
    def _callFUT(self, sampfreq):
        return estimate_alpha(sampfreq)
    def test_8k(self):
        eq_(round(self._callFUT(8000), 3), 0.312)
    def test_16k(self):
        eq_(round(self._callFUT(16000), 3), 0.41)
    def test_44k(self):
        eq_(round(self._callFUT(44100), 3), 0.544)

