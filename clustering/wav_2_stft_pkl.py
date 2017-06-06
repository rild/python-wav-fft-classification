# -*- coding: utf-8 -*-
from scipy import arange, hamming, sin, pi
from matplotlib import pylab as plt

from scipy import signal

import scipy.io.wavfile
import numpy as np

import wave


def load_wav_all_data(filename):
    wavfile = wave.open(filename, 'r')

    nchannels = wavfile.getnchannels()
    sampling_rate = wavfile.getframerate()
    quantization_bits = wavfile.getsampwidth() * 8
    sample_width = wavfile.getsampwidth()
    nsamples = wavfile.getnframes()

    return (wavfile, nchannels, sampling_rate, quantization_bits, sample_width, nsamples)


def load_wav_data(filename):
    wavfile = wave.open(filename, 'r')

    nchannels = wavfile.getnchannels()
    sampling_rate = wavfile.getframerate()
    quantization_bits = wavfile.getsampwidth() * 8
    sample_width = wavfile.getsampwidth()
    nsamples = wavfile.getnframes()

    frames = wavfile.readframes(wavfile.getnframes())  # frameの読み込み :binary data
    npframes = np.frombuffer(frames, dtype="int16")  # numpy.arrayに変換 :integer data
    print(npframes)
    npframes = npframes * (1.0 / 32768.0)  # pcm
    # ref http://nonbiri-tereka.hatenablog.com/entry/2014/06/24/110011

    return (sampling_rate, npframes)


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

    resyn_sig = resyn_sig * float(0x7fff)  # Why is this necessary? 06-01 rild
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


def triangle(length):
    h = int(np.ceil(length / 2.))
    ret = np.zeros(length, dtype=float)
    ret[: h] = (arange(h, dtype=float) / (h - 1))
    if length % 2 == 0:
        ret[- h:] = ret[h - 1:: -1]
    else:
        ret[- h + 1:] = ret[h - 1:: -1]
    return ret


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
scale = 'log'  # 'normal', 'log'
part = 'all'  # or int val, less than t.shape (time steps)

(samplerate, waveform) = load_wav_data(filename)

sig = waveform
NFFT = 512  # scipy.signal.stft Defaults to 256.
# 512 is likely to draw the best spectrogram
# if making 'nperseg' bigger, Zxx.shape become (freqs:decrease, times:increase)
# Zxx[0], freqs, about half of 'nperseg', rild 06-05

# OVERLAP = int(NFFT / 2)  # 窓をずらした時のフレームの重なり具合. half shiftが一般的らしい
# frame_length = sig.shape[0]  # wavファイルの全フレーム数
# time_song = float(frame_length) / samplerate  # 波形長さ(秒)
# time_unit = 1 / float(samplerate)  # 1サンプルの長さ(秒)


f, t, Zxx = signal.stft(sig, samplerate, nperseg=NFFT)

print(Zxx.shape)
import pickle

with open('hanekawa_nandemoha01' + '_f.pickle', 'wb') as file:
    pickle.dump(f, file)

with open('hanekawa_nandemoha01' + '_t.pickle', 'wb') as file:
    pickle.dump(t, file)

with open('hanekawa_nandemoha01' + '_Zxx.pickle', 'wb') as file:
    pickle.dump(Zxx, file)

# _, xrec = signal.istft(Zxx, samplerate)
#
# resyn_sig = xrec
# new_filename = tag + "_" + scale + "_NFFT_" + str(NFFT) + '_part_' + part  # + "_OVERLAP_" + str(OVERLAP)
# save_wav(resyn_sig, output_files_path + new_filename + ".wav")
#
# if part != 'all':
#     _t = t[:part]
#     _Zxx = Zxx[:, :part]
#     plt.pcolormesh(_t, f, np.log(np.abs(_Zxx) ** 2))  # log scale
# elif scale == 'log':
#     plt.pcolormesh(t, f, np.log(np.abs(Zxx) ** 2))  # log scale
# elif scale == 'normal':
#     plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.abs(Zxx).max())  # normal scale
#
# plt.title('STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.savefig(output_files_path + new_filename + ".png")
# plt.show()

# save_spec(sig, samplerate, new_filename + ".png")
exit(0)

'''
rild's memo 

# Reference

---
# Unused 

'''

