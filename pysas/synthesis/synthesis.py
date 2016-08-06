# -*- coding: utf-8 -*-
import numpy as np


class Synthesis(object):
    def __init__(self, frameperiod, filter):
        self.frameperiod = frameperiod
        self.filter = filter

    def synthesis_frame(self, cur_coef, prev_coef, pulse):
        ret = np.zeros_like(pulse)
        slope = (cur_coef - prev_coef) / self.frameperiod
        for i in range(self.frameperiod):
            coef = prev_coef + slope * i
            scaled = pulse[i] * np.exp(coef[0])
            ret[i] = self.filter.filter(scaled, coef)
        return ret

    def synthesis(self, pulse, coef_mat):
        ret = np.zeros_like(pulse)
        prev_coef = np.zeros_like(coef_mat[0])
        for i in range(coef_mat.shape[0]):
            offset = self.frameperiod * i
            cur_coef = coef_mat[i]
            synthesized = self.synthesis_frame(cur_coef, prev_coef, pulse[offset:offset + self.frameperiod])
            if i > 0:
                ret[offset:offset + self.frameperiod] = synthesized
            prev_coef = cur_coef
        return ret
