# -*- coding: utf-8 -*-
"""
03_train_classify.py  工况分类模型训练与评估
读取 features/features.csv，训练随机森林分类器识别 5 种风扇工况：
  - 分层 5 折交叉验证评估准确率
  - 输出混淆矩阵图、各类平均频谱对比图、特征重要性图到 results/

用法: python 03_train_classify.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_CSV = os.path.join(BASE, "features", "features.csv")
RESULTS_DIR = os.path.join(BASE, "results")

CONDITIONS = [
    "W1_关闭(背景噪声基准)",
    "W2_1挡正常",
    "W3_3挡正常",
    "W4_扇叶失衡",
    "W5_全遮挡出风口",
]
SHORT = ["W1 关闭", "W2 1挡", "W3 3挡", "W4 失衡", "W5 遮挡"]

def load_rows():
    """优先加载增强特征表；交叉验证只用原始行，最终模型用全部行。"""
    aug_csv = os.path.join(BASE, "features", "features_aug.csv")
    src = aug_csv if os.path.exists(aug_csv) else FEAT_CSV
    with open(src, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    original = [r for r in rows if r.get("augmented", "1") == "0"]
    print(f"数据源: {os.path.basename(src)} (原始 {len(original)} 行, 全部 {len(rows)} 行)")
    return rows, original


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows, original = load_rows()
    feat_names = [k for k in rows[0].keys() if k not in ("condition", "file", "augmented")]
    X_all = np.array([[float(r[k]) for k in feat_names] for r in rows])
    y_all = np.array([r["condition"] for r in rows])
    X = np.array([[float(r[k]) for k in feat_names] for r in original])
    y = np.array([r["condition"] for r in original])
    print(f"交叉验证用: {X.shape[0]} 样本 x {X.shape[1]} 特征, {len(set(y))} 类")

    # ---------- 交叉验证 ----------
    clf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=42))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    print(f"分层5折交叉验证准确率: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")
    print("各折:", ", ".join(f"{s*100:.0f}%" for s in scores))

    y_pred = cross_val_predict(clf, X, y, cv=cv)
    print("\n分类报告:")
    print(classification_report(y, y_pred, target_names=SHORT, digits=3))

    # ---------- 全量(含增强)训练出的特征重要性 ----------
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(StandardScaler().fit_transform(X_all), y_all)
    importances = rf.feature_importances_

    # ---------- 图1: 混淆矩阵 ----------
    cm = confusion_matrix(y, y_pred, labels=CONDITIONS)
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(5), SHORT, rotation=30)
    ax.set_yticks(range(5), SHORT)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xlabel("预测工况")
    ax.set_ylabel("真实工况")
    ax.set_title(f"5折交叉验证混淆矩阵 (准确率 {scores.mean()*100:.1f}%)")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print("已保存 results/confusion_matrix.png")

    # ---------- 图2: 特征重要性 Top10 ----------
    order = np.argsort(importances)[::-1][:10]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh([feat_names[i] for i in order][::-1], importances[order][::-1], color="#4878b8")
    ax.set_xlabel("随机森林特征重要性")
    ax.set_title("特征重要性 Top 10")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)
    print("已保存 results/feature_importance.png")

    # ---------- 图3: 各工况平均 Welch 频谱对比 ----------
    import glob
    import scipy.io.wavfile as wavfile
    from scipy.signal import welch
    fig, ax = plt.subplots(figsize=(8.6, 5))
    colors = ["#9e9e9e", "#2e7d32", "#1565c0", "#c62828", "#ef6c00"]
    for cond, short, c in zip(CONDITIONS, SHORT, colors):
        psds, freqs = [], None
        for wv in sorted(glob.glob(os.path.join(BASE, "data", "wav", cond, "*.wav"))):
            fs, x = wavfile.read(wv)
            x = x.astype(np.float64) / 32768.0
            f_, p_ = welch(x, fs=fs, nperseg=2048, noverlap=1024)
            psds.append(10 * np.log10(p_ + 1e-12))
            freqs = f_
        ax.semilogx(freqs[1:], np.mean(psds, axis=0)[1:], label=short, color=c, lw=1.6)
    ax.set_xlabel("频率 (Hz, 对数轴)")
    ax.set_ylabel("功率谱密度 (dB/Hz)")
    ax.set_title("5 种工况平均频谱对比 (各 15 组均值)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "spectra_comparison.png"), dpi=150)
    plt.close(fig)
    print("已保存 results/spectra_comparison.png")

    # ---------- 图4: 特征箱线图 (最区分性的 4 个特征) ----------
    top4 = order[:4]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax_, fi in zip(axes, top4):
        data = [X[y == c, fi] for c in CONDITIONS]
        bp = ax_.boxplot(data, tick_labels=SHORT, showfliers=False, patch_artist=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.65)
        ax_.set_title(feat_names[fi], fontsize=10)
        ax_.tick_params(axis="x", rotation=45, labelsize=8)
    fig.suptitle("区分性最强的 4 个特征分布", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "feature_boxplot.png"), dpi=150)
    plt.close(fig)
    print("已保存 results/feature_boxplot.png")


if __name__ == "__main__":
    main()
