#coding:utf-8

# このサイトから
# http://aidiary.hatenablog.com/entry/20111001/1317441171

import wave
from pylab import *

import pickle

if __name__ == "__main__":
    # WAVEファイルから波形データを取得
    wf = wave.open("res/a.wav", "rb")

    data = wf.readframes(wf.getnframes())
    data = frombuffer(data, dtype="int16")
    print(data[:10])
    # if len(wf.getnchannels)>1: #convert to mono
    #     wf=(wf[:,0]+wf[:,1])*0.5
    length = float(wf.getnframes()) / wf.getframerate()  # 波形長さ（秒）

    # FFTのサンプル数
    N = 512

    # FFTで用いるハミング窓
    hammingWindow = np.hamming(N)

    # スペクトログラムを描画
    pxx, freqs, bins, im = specgram(data, NFFT=N, Fs=wf.getframerate(), noverlap=0, window=hammingWindow)

    print(pxx.shape)
    print(pxx)
    print(freqs.shape) # 周波数の刻み
    print(bins.shape) # 時刻の刻み

    axis([0, length, 0, wf.getframerate() / 2])
    xlabel("time [second]")
    ylabel("frequency [Hz]")

    show()
