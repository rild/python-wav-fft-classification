# -*- coding: utf-8 -*-
from scipy import arange, hamming, sin, pi
from scipy.fftpack import fft, ifft
from matplotlib import pylab as plt

import wave
import sys
import scipy.io.wavfile
import numpy as np


def load_wav(filename):
    try:
        wavedata = scipy.io.wavfile.read(filename)
        samplerate = int(wavedata[0])
        wavef = wavedata[1] * (1.0 / 32768.0)  # pcm
        if len(wavef.shape) > 1:  # convert to mono
            wavef = (wavef[:, 0] + wavef[:, 1]) * 0.5
        return (samplerate, wavef)
    except:
        print("Error loading wav: " + filename)
        return None


def save_as_wav(resyn_sig, filename):
    #  2 ** 16 / 2
    #  32768.0
    # wavef=wavedata[1]*(1.0/32768.0) # pcm
    # print (type(resyn_sig[1]))
    resyn_data = (resyn_sig * 32768.0).astype(np.int16)
    scipy.io.wavfile.write(filename, samplerate, resyn_data)


def triangle(length):
    h = int(np.ceil(length / 2.))
    ret = np.zeros(length, dtype=float)
    ret[: h] = (arange(h, dtype=float) / (h - 1))
    if length % 2 == 0:
        ret[- h:] = ret[h - 1:: -1]
    else:
        ret[- h + 1:] = ret[h - 1:: -1]
    return ret


filename = "res/input.wav"
(samplerate, waveform) = load_wav(filename)

sig = waveform

NFFT = 4096  # フレームの大きさ
OVERLAP = NFFT / 4  # 窓をずらした時のフレームの重なり具合. half shiftが一般的らしい
frame_length = sig.shape[0]  # wavファイルの全フレーム数
time_song = float(frame_length) / samplerate  # 波形長さ(秒)
time_unit = 1 / float(samplerate)  # 1サンプルの長さ(秒)

# 💥 1.
# FFTのフレームの時間を決めていきます
# time_rulerに各フレームの中心時間が入っています
start = (NFFT / 2) * time_unit
stop = time_song
step = (NFFT - OVERLAP) * time_unit
time_ruler = np.arange(start, stop, step)

# 💥 2.
# 窓関数は周波数解像度が高いハミング窓を用います
window = np.hamming(NFFT)

spec = np.zeros([len(time_ruler), 1 + (NFFT / 2)])  # 転置状態で定義初期化
pos = 0

# spec的な 復元したいからデータを残す
freq_vectors = np.zeros([len(time_ruler), NFFT])

for fft_index in range(len(time_ruler)):
    # 💥 1.フレームの切り出します
    frame = sig[pos:pos + NFFT]

    # フレームが信号から切り出せない時はアウトです
    if len(frame) == NFFT:
        # 💥 2.窓関数をかけます
        windowed = window * frame
        # 💥 3.FFTして周波数成分を求めます
        # rfftだと非負の周波数のみが得られます
        fft_result = np.fft.rfft(windowed)
        # 復元したいから複素数も含めたデータを残したい
        freq_vector = fft(windowed)

        # 💥 4.周波数には虚数成分を含むので絶対値をabsで求めてから2乗します
        # グラフで見やすくするために対数をとります
        fft_data = np.log(np.abs(fft_result) ** 2)
        # fft_data = np.log(np.abs(fft_result))
        # fft_data = np.abs(fft_result) ** 2
        # fft_data = np.abs(fft_result)
        # これで求められました。あとはspecに格納するだけです
        for i in range(len(spec[fft_index])):
            spec[fft_index][-i - 1] = fft_data[i]

            # 復元したいから複素数も含めたデータを残したい
            freq_vectors[fft_index][-i - 1] = freq_vector[i]

        # 💥 4. 窓をずらして次のフレームへ
        pos += (NFFT - OVERLAP)

resyn_sig = np.zeros([len(sig)])  # 転置状態で定義初期化
resyn_pos = 0

# cut_num = NFFT
# cross_num = int(OVERLAP / 2)
cross_num = OVERLAP
# shift_num = 1

# win = triangle(OVERLAP * 2)
# win = np.hamming(OVERLAP * 2)
win = np.hanning(OVERLAP * 2)

for i in range(len(freq_vectors)):
    resyn_windowed = ifft(freq_vectors[i])
    resyn_frame = resyn_windowed / window

    resyn_frame[:cross_num] *= win[:cross_num]
    resyn_frame[-cross_num:] *= win[-cross_num:]

    if (resyn_pos + NFFT > len(resyn_sig)): break
    # resyn_sig[resyn_pos:resyn_pos + NFFT] = resyn_frame
    resyn_sig[resyn_pos:resyn_pos + NFFT] += resyn_frame.astype(np.float64)
    resyn_pos += (NFFT - OVERLAP)

print("sig ==============")

new_filename = "resyn" + "_NFFT_" + str(NFFT) + "_OVERLAP_" + str(OVERLAP)
save_as_wav(resyn_sig, new_filename + ".wav")

## プロットします
# matplotlib.imshowではextentを指定して軸を決められます。aspect="auto"で適切なサイズ比になります
plt.imshow(spec.T, extent=[0, time_song, 0, samplerate / 2], aspect="auto")
plt.xlabel("time[s]")
plt.ylabel("frequency[Hz]")
plt.colorbar()
plt.savefig(new_filename + ".png")

# # 窓関数
# L = len(sig)
# win = hamming(L)
#
# # フーリエ変換
# spectrum_nw = fft(sig) # 窓関数なし
# spectrum = fft(sig * win) # 窓関数あり
# half_spectrum_nw = abs(spectrum_nw[: L / 2 + 1])
# half_spectrum = abs(spectrum[: L / 2 + 1])
#
# # フーリエ逆変換
# resyn_sig = ifft(spectrum)
# resyn_sig /= win
#
# # 図を表示
# fig = plt.figure()
# fig.add_subplot(411)
# plt.plot(sig)
# plt.xlim([0, L])
# plt.title("1. Input signal", fontsize = 20)
# fig.add_subplot(412)
# plt.plot(half_spectrum_nw)
# plt.xlim([0, len(half_spectrum_nw)/10]) # scale down with "/10"
# plt.title("2. Spectrum (no window)", fontsize = 20)
# fig.add_subplot(413)
# plt.plot(half_spectrum)
# plt.xlim([0, len(half_spectrum)/10]) # scale down with "/10"
# plt.title("3. Spectrum (with window)", fontsize = 20)
# fig.add_subplot(414)
# plt.plot(resyn_sig)
# plt.xlim([0, L])
# plt.title("4. Resynthesized signal", fontsize = 20)
#
# pl.show()

# save_as_wav(resyn_sig)
