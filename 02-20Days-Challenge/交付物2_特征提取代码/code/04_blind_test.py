# -*- coding: utf-8 -*-
"""
04_blind_test.py  盲测验证
用模型**从未见过**的新录音（data/blind/ 目录）检验分类器的泛化能力。
盲测样本命名规则: {工况编号}_{序号}.mp3，例如 W2_1.mp3 = 1挡正常工况第1组。

用法: python 04_blind_test.py
输出: 终端打印每段盲测音频的预测结果与总体准确率；results/blind_test_report.csv
"""
import os
import glob
import csv
import subprocess
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import importlib.util
_here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("feat", os.path.join(_here, "02_extract_features.py"))
feat_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feat_mod)

import imageio_ffmpeg
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
BASE = os.path.dirname(_here)
BLIND_ROOT = os.path.join(BASE, "data", "blind")
RESULTS_DIR = os.path.join(BASE, "results")

CONDITIONS = feat_mod.CONDITIONS
COND_BY_PREFIX = {"W1": 0, "W2": 1, "W3": 2, "W4": 3, "W5": 4}
SHORT = ["W1 关闭", "W2 1挡", "W3 3挡", "W4 失衡", "W5 遮挡"]


def mp3_to_wav(mp3_path):
    wav_path = mp3_path.rsplit(".", 1)[0] + "_tmp.wav"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", mp3_path,
                    "-ac", "1", "-ar", "16000", "-vn", wav_path], check=True)
    return wav_path


def main():
    # ---------- 1. 用全部训练样本(含增强副本)重新训练 ----------
    import csv as _csv
    aug_csv = os.path.join(BASE, "features", "features_aug.csv")
    feat_csv = aug_csv if os.path.exists(aug_csv) else os.path.join(BASE, "features", "features.csv")
    with open(feat_csv, encoding="utf-8-sig") as f:
        rows = list(_csv.DictReader(f))
    feat_names = [k for k in rows[0].keys() if k not in ("condition", "file", "augmented")]
    X_train = np.array([[float(r[k]) for k in feat_names] for r in rows])
    y_train = np.array([COND_BY_PREFIX[r["condition"][:2]] for r in rows])

    scaler = StandardScaler().fit(X_train)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(scaler.transform(X_train), y_train)
    print(f"模型已用全部 {len(y_train)} 组训练数据重新训练\n")

    # ---------- 2. 对盲测样本逐个预测 ----------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    blind_files = sorted(glob.glob(os.path.join(BLIND_ROOT, "*.mp3"))) + \
                  sorted(glob.glob(os.path.join(BLIND_ROOT, "*.wav")))
    if not blind_files:
        print(f"盲测目录为空: {BLIND_ROOT}")
        print("请先录制盲测音频，命名为 W1_1.mp3 / W2_2.mp3 ... 放入该目录后重跑本脚本。")
        return

    report = []
    n_correct = 0
    print(f"{'文件':<16}{'真实':<10}{'预测':<10}结果")
    print("-" * 46)
    for bf in blind_files:
        stem = os.path.splitext(os.path.basename(bf))[0]
        true_idx = COND_BY_PREFIX.get(stem.split("_")[0].upper(), -1)
        wav = mp3_to_wav(bf) if bf.endswith(".mp3") else bf
        fs, x = wavfile.read(wav)
        x = x.astype(np.float64)
        if np.max(np.abs(x)) > 1:
            x = x / 32768.0
        feats, _, _ = feat_mod.extract_features(x, fs)
        x_feat = np.array([[feats[k] for k in feat_names]])
        pred_idx = int(clf.predict(scaler.transform(x_feat))[0])
        pred_proba = clf.predict_proba(scaler.transform(x_feat))[0].max()
        hit = pred_idx == true_idx
        n_correct += hit
        name = os.path.basename(bf)
        print(f"{name:<16}{SHORT[true_idx]:<10}{SHORT[pred_idx]:<10}{'✓' if hit else '✗'}  (置信度 {pred_proba:.2f})")
        report.append({"file": name, "true": SHORT[true_idx], "predicted": SHORT[pred_idx],
                       "confidence": round(float(pred_proba), 3), "correct": hit})
        if wav.endswith("_tmp.wav"):
            os.remove(wav)

    acc = n_correct / len(blind_files) * 100
    print("-" * 46)
    print(f"盲测结果: {n_correct}/{len(blind_files)} 正确, 准确率 {acc:.1f}%")

    out = os.path.join(RESULTS_DIR, "blind_test_report.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.DictWriter(f, fieldnames=["file", "true", "predicted", "confidence", "correct"])
        w.writeheader()
        w.writerows(report)
    print(f"盲测报告已保存: {out}")


if __name__ == "__main__":
    main()
