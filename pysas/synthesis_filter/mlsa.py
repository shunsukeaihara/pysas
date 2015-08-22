# -*- coding: utf-8 -*-
import numpy as np

padeCoef4 = np.array([1.0, 4.999273e-1, 1.067005e-1, 1.170221e-2, 5.656279e-4])
padeCoef5 = np.array([1.0, 4.999391e-1, 1.107098e-1, 1.369984e-2, 9.564853e-4, 3.041721e-5])


class Filter(object):
    def __init__(self, order, alpha):
        self.order = order
        self.alpha = alpha
        self.delay = np.zeros(order + 1)

    def filter(self, x, coefficients):
        result = 0.0
        self.delay[0] = x
        self.delay[1] = (1.0 - self.alpha ** 2) * self.delay[0] + self.alpha * self.delay[1]
        for i in range(2, coefficients.size):
            self.delay[i] = self.delay[i] + self.alpha * (self.delay[i + 1] - self.delay[i - 1])
            result += self.delay[i] * coefficients[i]
        if coefficients.shape[0] == 2:
            result += self.delay[1] * coefficients[1]
        for i in range(-(self.delay.size - 1), -1):
            i = -i
            self.delay[i] = self.delay[i - 1]
        return result


class CascadeFilter(object):
    def __init__(self, order, alpha, pade_order):
        self.pade_order = pade_order
        self.filter_num = pade_order + 1
        self.order = order
        self.alpha = alpha
        self.filters = []
        self.delay = np.zeros(self.filter_num, dtype=np.float64)
        for i in range(self.filter_num):
            self.filters.append(Filter(order, alpha))

        if pade_order == 4:
            self.pade_coefficients = padeCoef4
        else:
            self.pade_coefficients = padeCoef5
            
    def filter(self, x, coefficients):
        result = 0.0
        feedback = 0.0
        for i in range(-(len(self.pade_coefficients) - 1), 0):
            i = -i
            self.delay[i] = self.filters[i].filter(self.delay[i - 1], coefficients)
            val = self.delay[i] * self.pade_coefficients[i]
            if i%2 == 1:
                feedback += val
            else:
                feedback -= val
            result += val
        self.delay[0] = feedback + x
        result += self.delay[0]
        return result

class MLSAFilter(object):
    def __init__(self, order, alpha, pade_order):

        assert pade_order == 4 or pade_order == 5, "order of pade must be 4 or 5."
        self.f1 = CascadeFilter(2, alpha, pade_order)
        self.f2 = CascadeFilter(order + 1, alpha, pade_order)

    def filter(self, x, coefficients):
        coef = np.array([0, coefficients[1]])
        return self.f2.filter(self.f1.filter(x, coef),coefficients)
        
