# (WIP)pysas

Speech Analysis and Synthesis Toolkit for Python(2.X, 3.X).

Inspired by r9y9's ssp products(https://github.com/r9y9).

This module include [World](http://ml.cs.yamanashi.ac.jp/world/english/index.html) C++ library by M. Morise.

# usage

## analyzing by world

```python
from pysas import World, waveread
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


## LICENSE

Licensed under MIT License. Bundled World C++ library is licensed under New BSD license.
