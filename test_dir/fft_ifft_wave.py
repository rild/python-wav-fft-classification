# -*- coding: utf-8 -*-
from scipy import arange, hamming, sin, pi
from scipy.fftpack import fft, ifft
from matplotlib import pylab as pl

import wave
import sys
import scipy.io.wavfile
import numpy as np

fs = 8000 # Sampling rate
L = 1024 # Signal length


def load_wav(filename):
    try:
        wavedata=scipy.io.wavfile.read(filename)
        samplerate=int(wavedata[0])
        wavef=wavedata[1]*(1.0/32768.0) # pcm
        if len(wavef.shape)>1: #convert to mono
            wavef=(wavef[:,0]+wavef[:,1])*0.5
        return (samplerate,wavef)
    except:
        print ("Error loading wav: "+filename)
        return None

def save_as_wav(resyn_sig):
    #  2 ** 16 / 2
    #  32768.0
    # wavef=wavedata[1]*(1.0/32768.0) # pcm
    # print (type(resyn_sig[1]))
    resyn_data = (resyn_sig * 32768.0).astype(np.int16)
    scipy.io.wavfile.write("resyn_sig.wav", samplerate, resyn_data)


argv = sys.argv
argc = len(argv)
filename = argv[1] # "test.wav"
(samplerate,waveform) = load_wav(filename)

sig = waveform

# 窓関数
L = len(sig)
win = hamming(L)

# フーリエ変換
spectrum_nw = fft(sig) # 窓関数なし
spectrum = fft(sig * win) # 窓関数あり
half_spectrum_nw = abs(spectrum_nw[: L / 2 + 1])
half_spectrum = abs(spectrum[: L / 2 + 1])

# フーリエ逆変換
resyn_sig = ifft(spectrum)
resyn_sig /= win

# 図を表示
fig = pl.figure()
fig.add_subplot(411)
pl.plot(sig)
pl.xlim([0, L])
pl.title("1. Input signal", fontsize = 20)
fig.add_subplot(412)
pl.plot(half_spectrum_nw)
pl.xlim([0, len(half_spectrum_nw)/10]) # scale down with "/10"
pl.title("2. Spectrum (no window)", fontsize = 20)
fig.add_subplot(413)
pl.plot(half_spectrum)
pl.xlim([0, len(half_spectrum)/10]) # scale down with "/10"
pl.title("3. Spectrum (with window)", fontsize = 20)
fig.add_subplot(414)
pl.plot(resyn_sig)
pl.xlim([0, L])
pl.title("4. Resynthesized signal", fontsize = 20)

pl.show()

# save_as_wav(resyn_sig)
