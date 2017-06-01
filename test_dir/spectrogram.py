# -*- coding: utf-8 -*-
from scipy import arange, hamming, sin, pi
from scipy.fftpack import fft, ifft
from matplotlib import pylab as plt

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
    npframes = npframes * (1.0 / 32768.0) # pcm
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
    # print (type(resyn_sig[1]))
    resyn_data = (resyn_sig * 32768.0).astype(np.int16)
    scipy.io.wavfile.write(filename, samplerate, resyn_data)

# it seems to have something wrong 17-05-30 rild
# import array
def save_wav(resyn_sig, filename):
    resyn_sig = (resyn_sig * 32768)
    # resyn_sig = resyn_sig.astype(int)

    # write_wave.setparams(wavfile.getparams())
    resyn_sig = resyn_sig * float(0x7fff) # Why is this necessary? 06-01 rild

    samples = np.array(resyn_sig, np.int16)

    filename = wavfilename_format(filename)
    # w = wave.Wave_write(filename)
    w = wave.Wave_write(filename)
    w.setnchannels(1)
    w.setsampwidth(2)  # 2 bytes
    w.setframerate(samplerate)
    w.setnframes(len(samples))
    w.setcomptype('NONE', 'descrip')  # No compression
    # w.setparams((
    #     1,  # channel, mono: 1
    #     2,  # byte width
    #     samplerate,  # sampling rate
    #     len(samples),  # number of frames
    #     "NONE", "not compressed"  # no compression
    # ))

    # print(w.getnchannels())
    #  write_wave.writeframes(array.array('h', resyn_sig).tostring())
    w.writeframes(samples.tostring())
    w.close()

# _SAMPLES_PER_SECOND = 44100
# _CHANNEL_COUNT = 1           # 1 => mono, 2 => stereo
# _CHECK_RATE = 44100          # How often to check the queue

# def save_np_as_wav(f, npa):
#     """
#     Save all samples in array a to the WAVE file whose name is f.wav.
#     """
#
#     # Saving to a WAV file isn't handled by PyGame, so use the
#     # standard "wave" module instead.
#
#     import wave
#
#     fileName = wavfilename_format(f)
#
#     npa = npa * float(0x7fff)
#
#     samples = np.array(npa, np.int16)
#     file = wave.open(fileName, 'w')
#     file.setnchannels(_CHANNEL_COUNT)
#     file.setsampwidth(2)  # 2 bytes
#     file.setframerate(samplerate)
#     file.setnframes(len(samples))
#     file.setcomptype('NONE', 'descrip')  # No compression
#     file.writeframes(samples.tostring())
#     file.close()


def triangle(length):
    h = int(np.ceil(length / 2.))
    ret = np.zeros(length, dtype=float)
    ret[: h] = (arange(h, dtype=float) / (h - 1))
    if length % 2 == 0:
        ret[- h:] = ret[h - 1:: -1]
    else:
        ret[- h + 1:] = ret[h - 1:: -1]
    return ret

def save_spec_as_img(new_filename='non'):
    # matplotlib.imshowではextentを指定して軸を決められます。aspect="auto"で適切なサイズ比になります
    plt.imshow(spec.T, extent=[0, time_song, 0, samplerate / 2], aspect="auto")
    plt.xlabel("time[s]")
    plt.ylabel("frequency[Hz]")
    plt.colorbar()
    if new_filename != 'non':
        plt.savefig(new_filename)

output_files_path = "out/"
filename = "res/hanekawa_nandemoha01.wav"

(samplerate, waveform) = load_wav_data(filename)

sig = waveform
NFFT = 2048  # フレームの大きさ
OVERLAP = int(NFFT / 2)  # 窓をずらした時のフレームの重なり具合. half shiftが一般的らしい
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

spec = np.zeros([len(time_ruler), int(1 + (NFFT / 2))])  # 転置状態で定義初期化
pos = 0

# spec的な 復元したいからデータを残す
# spectrums = np.zeros([len(time_ruler), NFFT])
# astype(complex) をつけないと spectrum を代入する時に, 複素数成分が消えちゃう
spectrums = np.zeros([len(time_ruler), NFFT]).astype(complex)

# dpends on params: spec, spectrums
def store_data(fft_data, spectrum):
    for i in range(len(spec[fft_index])):
        spec[fft_index][-i - 1] = fft_data[i]

        # 復元したいから複素数も含めたデータを残したい
        spectrums[fft_index][-i - 1] = spectrum[i]

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
        spectrum = fft(windowed)

        # 💥 4.周波数には虚数成分を含むので絶対値をabsで求めてから2乗します
        # グラフで見やすくするために対数をとります
        fft_data = np.log(np.abs(fft_result) ** 2)
        # fft_data = np.log(np.abs(fft_result))
        # fft_data = np.abs(fft_result) ** 2
        # fft_data = np.abs(fft_result)
        # これで求められました。あとはspecに格納するだけです
        store_data(fft_data, spectrum)

        # 💥 4. 窓をずらして次のフレームへ
        pos += (NFFT - OVERLAP)

# resyn_sig = np.zeros([len(sig)])
resyn_sig = np.zeros([len(sig)]).astype(float)  # 転置状態で定義初期化
resyn_pos = 0

# cut_num = NFFT
# cross_num = int(OVERLAP / 2)
cross_num = OVERLAP
# shift_num = 1

# win = triangle(OVERLAP * 2)
# win = np.hamming(OVERLAP * 2)
win = np.hanning(OVERLAP * 2)

for i in range(len(spectrums)):
    # resyn_windowed = ifft(spectrums[i]).astype(float)
    # astype(float) is needed
    # cuz original windowed sig data type is float64

    resyn_windowed = np.real(ifft(spectrums[i])).astype(float)
    # ref: http://org-technology.com/posts/smoother.html

    resyn_frame = resyn_windowed / window

    resyn_frame[:cross_num] *= win[:cross_num]
    resyn_frame[-cross_num:] *= win[-cross_num:]

    if (resyn_pos + NFFT > len(resyn_sig)): break
    # resyn_sig[resyn_pos:resyn_pos + NFFT] = resyn_frame
    resyn_sig[resyn_pos:resyn_pos + NFFT] += resyn_frame.astype(np.float64)
    resyn_pos += (NFFT - OVERLAP)

print("sig ==============")
new_filename = "resyn" + "_NFFT_" + str(NFFT) + "_OVERLAP_" + str(OVERLAP)
# save_as_wav(resyn_sig, output_files_path + new_filename + ".wav")

# save_np_as_wav(output_files_path + 'test' + new_filename, resyn_sig)
# print(len(resyn_sig))
save_wav(resyn_sig, output_files_path + new_filename + ".wav")

## スペクトログラムを保存
save_spec_as_img(output_files_path + new_filename  + ".png")
