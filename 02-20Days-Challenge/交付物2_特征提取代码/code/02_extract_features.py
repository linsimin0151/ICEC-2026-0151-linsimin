# -*- coding: utf-8 -*-
"""
02_extract_features.py  时域/频域特征提取
对 data/wav 下 5 种工况的 WAV 录音逐段提取特征，汇总为 features/features.csv。

时域特征: RMS、峰值、波峰因子(Crest Factor)、过零率(ZCR)、峭度
频域特征: Welch 功率谱 -> 主频、频谱质心、频谱熵、频谱平坦度、
          频带能量占比(0-100/100-300/300-800/800-2000/2000-4000/4000-8000 Hz)

用法: python 02_extract_features.py
"""
import os
import glob
import csv
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import welch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAV_ROOT = os.path.join(BASE, "data", "wav")
FEAT_DIR = os.path.join(BASE, "features")
SAMPLE_RATE = 16000

# 频带划分 (Hz)：覆盖风扇旋转基频/叶频谐波(低频带)与电机电磁噪声、湍流宽带噪声(高频带)
BANDS = [(0, 100), (100, 300), (300, 800), (800, 2000), (2000, 4000), (4000, 8000)]
BAND_NAMES = [f"band_{lo}-{hi}Hz" for lo, hi in BANDS]

CONDITIONS = [
    "W1_关闭(背景噪声基准)",
    "W2_1挡正常",
    "W3_3挡正常",
    "W4_扇叶失衡",
    "W5_全遮挡出风口",
]


def extract_features(x, fs):
    """x: float 一维信号; fs: 采样率。返回 (特征字典, Welch频谱freqs, PSD)"""
    x = x - np.mean(x)  # 去直流偏置，避免污染过零率与低频质心
    # ---------- 时域 ----------
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    crest = float(peak / (rms + 1e-12))
    zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))
    kurt = float(np.mean((x - x.mean()) ** 4) / (x.std() ** 4 + 1e-12))

    # ---------- 频域 (Welch 平均周期图) ----------
    nper = min(len(x), 2048)
    freqs, psd = welch(x, fs=fs, nperseg=nper, noverlap=nper // 2)
    psd_norm = psd / (psd.sum() + 1e-12)

    dom_freq = float(freqs[np.argmax(psd)])                                   # 主频
    centroid = float(np.sum(freqs * psd_norm))                                # 频谱质心
    spec_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))       # 频谱熵
    geo_mean = np.exp(np.mean(np.log(psd + 1e-12)))
    flatness = float(geo_mean / (psd.mean() + 1e-12))                         # 频谱平坦度

    # 次峰频率: 平滑谱上按峰高取前 3 个局部极大(间隔>=30Hz), 用于区分"低频嗡声型"
    # 与"中频音调型"噪声——主频单特征在多时段环境噪声下区分度不足
    from scipy.signal import find_peaks
    psd_sm = np.convolve(psd, np.ones(5) / 5, mode="same")
    min_dist = max(1, int(30 / (freqs[1] - freqs[0])))
    pk, _ = find_peaks(psd_sm, distance=min_dist,
                       prominence=np.max(psd_sm) * 0.02)
    top = pk[np.argsort(psd_sm[pk])[::-1]][:3]
    peak2 = float(freqs[top[1]]) if len(top) > 1 else 0.0
    peak3 = float(freqs[top[2]]) if len(top) > 2 else 0.0

    band_energy = {}
    for (lo, hi), name in zip(BANDS, BAND_NAMES):
        m = (freqs >= lo) & (freqs < hi)
        band_energy[name] = float(psd_norm[m].sum())                          # 频带能量占比

    feats = {
        "rms": rms, "peak": peak, "crest_factor": crest, "zcr": zcr,
        "kurtosis": kurt, "dominant_freq": dom_freq, "peak2_freq": peak2,
        "peak3_freq": peak3, "spectral_centroid": centroid,
        "spectral_entropy": spec_entropy, "spectral_flatness": flatness,
        **band_energy,
    }
    return feats, freqs, psd


def main():
    os.makedirs(FEAT_DIR, exist_ok=True)
    rows, spec_cache = [], {}
    for cond in CONDITIONS:
        wavs = sorted(glob.glob(os.path.join(WAV_ROOT, cond, "*.wav")),
                      key=lambda f: int("".join(c for c in os.path.basename(f) if c.isdigit())))
        for wav_path in wavs:
            fs, x = wavfile.read(wav_path)
            x = x.astype(np.float64)
            if np.max(np.abs(x)) > 1:  # int16 数据归一化到 [-1, 1]
                x = x / 32768.0
            feats, freqs, psd = extract_features(x, fs)
            rows.append({"condition": cond, "file": os.path.basename(wav_path), **feats})
            spec_cache[wav_path] = (freqs, psd)
        print(f"{cond}: 提取 {len(wavs)} 组特征完成")

    fieldnames = ["condition", "file"] + list(rows[0].keys())[2:]
    out_csv = os.path.join(FEAT_DIR, "features.csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n特征表已保存: {out_csv} (共 {len(rows)} 行 x {len(fieldnames)} 列)")
    return rows, spec_cache


if __name__ == "__main__":
    main()
