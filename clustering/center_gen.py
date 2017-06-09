# -*- coding: utf-8 -*-
from scipy import arange, hamming, sin, pi
from scipy.fftpack import fft, ifft
from matplotlib import pylab as plt

from scipy import signal

import scipy.io.wavfile
import numpy as np

import wave

def wavfilename_format(fname):
    import os.path
    file_name = fname
    name, ext = os.path.splitext(fname)
    if ext != '.wav':
        file_name = fname + '.wav'
    return file_name

def save_as_wav(resyn_sig, filename):
    #  2 ** 16 / 2
    #  32768.0
    # wavef=wavedata[1]*(1.0/32768.0) # pcm
    resyn_data = (resyn_sig * 32768.0).astype(np.int16)
    scipy.io.wavfile.write(filename, samplerate, resyn_data)

# it seems to have something wrong 17-05-30 rild
# solved: it was just because of two other instance ... つらい
#     ref: http://introcs.cs.princeton.edu/python/code/stdaudio.py.html
# import array
def save_wav(resyn_sig, filename):
    # resyn_sig = (resyn_sig * 32768)

    resyn_sig = resyn_sig * float(0x7fff) # Why is this necessary? 06-01 rild
    # 0x7fff seems to mean 2 ** 16

    samples = np.array(resyn_sig, np.int16)

    filename = wavfilename_format(filename)
    w = wave.Wave_write(filename)
    w.setnchannels(1)
    w.setsampwidth(2)  # 2 bytes
    w.setframerate(samplerate)
    w.setnframes(len(samples))
    w.setcomptype('NONE', 'descrip')  # No compression

    w.writeframes(samples.tostring())
    w.close()


def save_spec(x, fs, new_filename='non'):
    # matplotlib.imshowではextentを指定して軸を決められます。aspect="auto"で適切なサイズ比になります
    f, t, Sxx = signal.spectrogram(x, fs)
    plt.pcolormesh(t, f, Sxx)
    plt.xlabel("time[s]")
    plt.ylabel("frequency[Hz]")
    plt.colorbar()
    if new_filename != 'non':
        plt.savefig(new_filename)

output_files_path = "out/"
filename = "res/hanekawa_nandemoha01.wav"

tag = "stft"
scale = 'log' # 'normal', 'log'
part = 'all' # or int val, less than t.shape (time steps)
NFFT = 512  # scipy.signal.stft Defaults to 256.
# 512 is likely to draw the best spectrogram
# if making 'nperseg' bigger, Zxx.shape become (freqs:decrease, times:increase)
# Zxx[0], freqs, about half of 'nperseg', rild 06-05

import pickle

def loader(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

fname = 'hanekawa_nandemoha01'

rpath = 'pkls/'
t_filename = rpath + fname + '_t.pickle'
f_filename = rpath + fname + '_f.pickle'
ZxxC_filename = rpath + 'generated' + 'ZxxC.pickle'

def stft_data_loader(ffile, tfile, Zxxfile):
    f = loader(ffile)
    t = loader(tfile)
    Zxx = loader(Zxxfile)
    return f, t, Zxx

f, t, Zxx = stft_data_loader(f_filename,
                             t_filename,
                             ZxxC_filename)
print(f.shape)
print(t.shape)
print(Zxx.shape)

samplerate  = 48000
_, xrec = signal.istft(Zxx, samplerate)


resyn_sig = xrec
new_filename = 'kmeans_center_gen'
save_wav(resyn_sig, output_files_path + new_filename + ".wav")

if part != 'all':
    _t = t[:part]
    _Zxx = Zxx[:, :part]
    plt.pcolormesh(_t, f, np.log(np.abs(_Zxx) ** 2)) # log scale
elif scale == 'log':
    plt.pcolormesh(t, f, np.log(np.abs(Zxx) ** 2)) # log scale
elif scale == 'normal':
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.abs(Zxx).max()) # normal scale

plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.savefig(output_files_path + new_filename + ".png")
# plt.show()

# save_spec(sig, samplerate, new_filename + ".png")
exit(0)

'''
rild's memo 

# Reference
## scipy.signal.stft
https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.signal.stft.html

### code reading
ABST: It seems to be normal fft transform for me. 

dependence 
stft - _spectral_helper - _fft_helper
 
#### _fft_helper memo 
windowed fft: fft.pack (twosided) or np.fft.rfft (onesided) 
return freqs, time, result 
- 'freqs' seems to be difference phase proceduct from 'result'

## scipy.signal.istft
https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.signal.istft.html#scipy.signal.istft


## plt.pcolormesh

https://matplotlib.org/api/pyplot_api.html
https://matplotlib.org/examples/pylab_examples/pcolor_demo.html

## numpy array, get row column 
http://qiita.com/supersaiakujin/items/d63c73bb7b5aac43898a#%E5%88%97%E3%82%92%E6%8A%BD%E5%87%BA%E3%81%99%E3%82%8B 

---
# Unused 
## scipy.signal.spectrogram
https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.signal.spectrogram.html
'''

