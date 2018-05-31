import numpy as np


def write_wav(data, fname='test.wav', freq=44100):
    from scipy.io.wavfile import write
    scaled = np.int16(data/np.max(np.abs(data)) * 32767)
    write(fname, freq, scaled)


def read_wav(fname='../classical1.wav'):
    import wave
    import struct
    fw = wave.open(fname)
    frames = fw.readframes(fw.getnframes())
    sig = struct.unpack_from('%dh' % fw.getnframes(), frames)
    sig = np.array(sig)/np.abs(sig).max()
    return sig
