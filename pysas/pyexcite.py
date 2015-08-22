# -*- coding: utf-8 -*-
import numpy as np
try:
    from numba import jit
except:
    from pysas.decorators import do_nothing as jit

@jit
def gen_frame(cur_f0, prev_f0, frame, samplingrate, gen_samples, gauss):
    
    if cur_f0 == 0.0 or prev_f0 == 0.0:
        if gauss:
            return np.random.normal(0.0, 1.0, frame), gen_samples
        else:
            return np.random.random(frame), gen_samples
    ret = np.zeros(frame)

    ncur = float(samplingrate) / cur_f0
    nprev = float(samplingrate) / prev_f0
    slope = (nprev - ncur) / float(frame)
    for i in range(frame):
        f0 = ncur + slope * i
        if gen_samples > int(f0):
            ret[i] = f0
            gen_samples -= int(f0)
        gen_samples += 1
    return np.sqrt(ret), gen_samples

@jit
def gen_pulse(f0, frame, samplingrate, gauss):
    ret = np.zeros(f0.shape[0] * frame)
    prev_f0 = f0[0]
    gen_samples = 0
    for i in range(f0.size):
        cur_f0 = f0[i]
        pluses, gen_samples = gen_frame(cur_f0, prev_f0, frame, samplingrate, gen_samples, gauss)
        prev_f0 = cur_f0
        start = frame * i
        ret[start:start+frame] = pluses
    return ret
    
