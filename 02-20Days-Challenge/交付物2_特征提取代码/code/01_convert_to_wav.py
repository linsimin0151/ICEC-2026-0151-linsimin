# -*- coding: utf-8 -*-
"""
01_convert_to_wav.py  音频格式统一转换
将 5 个工况文件夹中的原始 mp3 录音解码为 16kHz 单声道 WAV，
作为后续特征提取的标准输入（同时满足考核"原始 WAV 序列"交付要求）。

用法: python 01_convert_to_wav.py
"""
import os
import glob
import subprocess
import imageio_ffmpeg

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SAMPLE_RATE = 16000  # 16kHz 足够覆盖风扇噪声主要能量(<8kHz)，且减小数据量

CONDITIONS = [
    "W1_关闭(背景噪声基准)",
    "W2_1挡正常",
    "W3_3挡正常",
    "W4_扇叶失衡",
    "W5_全遮挡出风口",
]

def main():
    out_root = os.path.join(BASE, "data", "wav")
    os.makedirs(out_root, exist_ok=True)
    total, ok = 0, 0
    for cond in CONDITIONS:
        src_dir = os.path.join(BASE, cond)
        dst_dir = os.path.join(out_root, cond)
        os.makedirs(dst_dir, exist_ok=True)
        mp3s = sorted(glob.glob(os.path.join(src_dir, "*.mp3")))
        for mp3 in mp3s:
            name = os.path.splitext(os.path.basename(mp3))[0]
            wav = os.path.join(dst_dir, name + ".wav")
            # -ac 1 单声道, -ar 16000 重采样, -vn 去掉可能存在的封面图
            cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", mp3,
                   "-ac", "1", "-ar", str(SAMPLE_RATE), "-vn", wav]
            r = subprocess.run(cmd, capture_output=True, text=True)
            total += 1
            if r.returncode == 0 and os.path.getsize(wav) > 0:
                ok += 1
            else:
                print(f"[FAIL] {mp3}: {r.stderr.strip()}")
        print(f"{cond}: {len(mp3s)} 个 mp3 -> WAV 完成")
    print(f"\n共 {total} 个文件, 成功 {ok} 个, 失败 {total - ok} 个")
    print(f"输出目录: {out_root}")

if __name__ == "__main__":
    main()
