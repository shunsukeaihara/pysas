# -*- coding: utf-8 -*-
import wave
import numpy as np

def waveread(path):
    """
    read signals from wavefile, only support monoral and 8 or 16bit data
    @return normalized signals between [-1,1], sampling rate[Hz], bit rate[bits]
    """
    wf = wave.open(path, 'rb')
    channels = wf.getnchannels()
    assert channels == 1, "wave file should be monoral"
    
    bits = wf.getsampwidth() * 8  # bit rate
    assert bits==8 or bits==16, "bit rate should be 8 or 16bits (input {}bit)".format(bits)
    
    fs = wf.getframerate()  # sampling rate(Hz)
    datalength = wf.getnframes()
    raw = wf.readframes(datalength)

    
    if bits == 8:
        signal = np.float64(np.fromstring(raw, dtype=np.uint8) - 127) / 127.0
    elif bits == 16:
        signal = np.float64(np.fromstring(raw, dtype=np.int16)) / 32767.0
    return signal, fs, bits


def wavewrite(signal, samplingrate, bits, path):
    assert bits==8 or bits==16, "bit rate expect 8 or 16bits but recevied {}bit)".format(bits)
    wf=wave.open(path,'wb')
    if bits == 8:
        sampwidth = 1
        s = np.int8((signal + 1.0) * 127).tostring()
    else:
        sampwidth = 2
        s = np.int16(signal * 32767.0).tostring()
        
    params = (1, sampwidth, samplingrate, len(signal), 'NONE', 'not compressed')
    wf.setparams(params)
    wf.writeframes(s)
