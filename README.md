# pyworld

Python(2.X, 3.X) wrapper for World Speech Analysis and Synthesis System.

This module is based on [World](http://ml.cs.yamanashi.ac.jp/world/english/index.html) C++ library by M. Morise.

## usage

### analyze

```python
from pyworld import World, waveread
signal, samplingrate, bitrate = waveread("path/to/monoral/wave/file")
world = World(samplingrate, bitrate)

f0, spectrogram, aperiodicity = world.analyze(signal)
```

signal, f0 are 1d numpy.ndarray. spectrogram, aperiodicity  are same shape 2d numpy.ndarray.

### F0 scaling and spectral stretching

```python
f0 = world.f0_scaling(f0, 2.0)
spectrogram = world.spectral_stretching(spectrogram, 1.5)
```

### syntesis

```python
outsignal = world.syntesis(f0, spectrogram, aperiodicity)
```

outsignal is 1d numpy.ndarray normalized between [-1,1].
