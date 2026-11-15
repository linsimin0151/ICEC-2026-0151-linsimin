# -*- coding: utf-8 -*-
"""
02b_augment_features.py  训练数据增强
手机自动增益(AGC)会导致同一场景不同时段的录音音量相差 10 倍以上，
且安静场景下电源嗡声占比会随噪声本底变化。为此对 75 组原始训练音频做波形级增强：

  副本1: 随机增益 (0.1~10 倍, 对数均匀)      -> 模拟 AGC 音量漂移
  副本2: 随机增益 + 注入白噪声(SNR 10~25dB)  -> 模拟"嗡声被环境噪声稀释"
  副本3: 原音量 + 注入白噪声(SNR 8~20dB)     -> 模拟高噪声环境

输出 features_aug.csv = 原始 75 行 + 增强 225 行, 增加 augmented 标记列。
后续 03/04 脚本优先读取 features_aug.csv。

用法: python 02b_augment_features.py
"""
import os
import glob
import csv
import numpy as np
import scipy.io.wavfile as wavfile
import importlib.util

_here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("feat", os.path.join(_here, "02_extract_features.py"))
feat_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feat_mod)

BASE = os.path.dirname(_here)
WAV_ROOT = os.path.join(BASE, "data", "wav")
FEAT_CSV = os.path.join(BASE, "features", "features.csv")
AUG_CSV = os.path.join(BASE, "features", "features_aug.csv")
RNG = np.random.default_rng(42)


def load_wav(path):
    fs, x = wavfile.read(path)
    x = x.astype(np.float64)
    if np.max(np.abs(x)) > 1:
        x = x / 32768.0
    return fs, x


def add_white_noise(x, snr_db):
    """按指定信噪比向信号注入白噪声"""
    sig_rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    noise_rms = sig_rms / (10 ** (snr_db / 20))
    return x + RNG.normal(0, noise_rms, size=x.shape)


def build_real_noise_pool():
    """从跨环境归档的 W1(纯房间噪声)录音中构建真实噪声片段池。
    正式盲测样本不参与; 若归档不存在则返回空列表(回退白噪声)。"""
    import subprocess
    import imageio_ffmpeg
    pool = []
    cross = os.path.join(BASE, "data", "blind_crossenv")
    if not os.path.isdir(cross):
        return pool
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    for mp3 in glob.glob(os.path.join(cross, "W1_*.mp3")):
        tmp = mp3.rsplit(".", 1)[0] + "_noise_tmp.wav"
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", mp3,
                        "-ac", "1", "-ar", "16000", "-vn", tmp], check=True)
        fs, n = load_wav(tmp)
        os.remove(tmp)
        pool.append(n)          # 8s 左右真实房间噪声
    return pool


def mix_real_noise(x, noise_seg, snr_db):
    """把真实噪声片段随机截取并与信号按 SNR 混合(噪声不足时循环补齐)"""
    if len(noise_seg) < len(x):
        reps = int(np.ceil(len(x) / len(noise_seg)))
        noise_seg = np.tile(noise_seg, reps)
    sig_rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    noise_rms = sig_rms / (10 ** (snr_db / 20))
    seg_rms = np.sqrt(np.mean(noise_seg ** 2)) + 1e-12
    start = RNG.integers(0, max(1, len(noise_seg) - len(x)))
    piece = noise_seg[start:start + len(x)]
    return x + piece * (noise_rms / seg_rms)


def main():
    with open(FEAT_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    feat_names = [k for k in rows[0].keys() if k not in ("condition", "file")]
    out_rows = [{**r, "augmented": 0} for r in rows]

    noise_pool = build_real_noise_pool()
    print(f"真实噪声池: {len(noise_pool)} 段 (来自跨环境归档 W1 录音)")

    for wav_path in sorted(glob.glob(os.path.join(WAV_ROOT, "*", "*.wav"))):
        cond = os.path.basename(os.path.dirname(wav_path))
        name = os.path.basename(wav_path)
        fs, x = load_wav(wav_path)
        if noise_pool:
            # 真实噪声增强: 随机选一段房间噪声按低 SNR 混入, 模拟"信号被环境噪声稀释"
            def pick():
                return noise_pool[int(RNG.integers(0, len(noise_pool)))]
            recipe = {
                "gain": lambda s: s * (10 ** RNG.uniform(-1, 1)),            # 0.1~10 倍
                "gain+noise": lambda s: mix_real_noise(
                    s * (10 ** RNG.uniform(-1, 0)), pick(), RNG.uniform(0, 15)),
                "noise": lambda s: mix_real_noise(
                    s, pick(), RNG.uniform(0, 12)),
            }
        else:
            recipe = {
                "gain": lambda s: s * (10 ** RNG.uniform(-1, 1)),
                "gain+noise": lambda s: add_white_noise(
                    s * (10 ** RNG.uniform(-1, 0)), RNG.uniform(0, 15)),
                "noise": lambda s: add_white_noise(s, RNG.uniform(0, 12)),
            }
        for tag, fn in recipe.items():
            xa = fn(x)
            xa = np.clip(xa, -1, 1)
            feats, _, _ = feat_mod.extract_features(xa, fs)
            out_rows.append({"condition": cond, "file": f"{name}[{tag}]",
                             **feats, "augmented": 1})

    fieldnames = ["condition", "file"] + feat_names + ["augmented"]
    with open(AUG_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    n_aug = sum(r["augmented"] for r in out_rows)
    print(f"增强完成: 原始 {len(out_rows)-n_aug} 行 + 增强 {n_aug} 行 = {len(out_rows)} 行")
    print(f"已保存: {AUG_CSV}")


if __name__ == "__main__":
    main()
