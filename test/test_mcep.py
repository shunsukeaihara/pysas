# -*- coding: utf-8 -*-

from  unittest import TestCase
from nose.tools import eq_

from pysas.mcep import estimate_alpha


class EstimateAlphaTest(TestCase):
    def test_8k(self):
        eq_(round(estimate_alpha(8000), 3), 0.312)
    def test_16k(self):
        eq_(round(estimate_alpha(16000), 3), 0.41)
    def test_44k(self):
        eq_(round(estimate_alpha(44100), 3), 0.544)
