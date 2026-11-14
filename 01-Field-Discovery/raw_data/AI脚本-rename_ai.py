# -*- coding: utf-8 -*-
"""
课题10：AI 批量重命名脚本
用法：python rename_ai.py
功能：把 AI 文件夹里的 10 个乱名 Excel 文件统一改成
     "课题10_主题_序号.xlsx" 格式，并记录总耗时。
"""

import os
import time

# 目标文件夹（脚本就在课题10目录下，AI 文件夹与脚本同级）
FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AI")

# 改名映射表：原文件名 -> 新文件名
# （这就是"AI 规则引擎"的核心：解析原文件名 -> 提取主题词 -> 生成规范名）
RENAME_MAP = {
    "新建 Microsoft Excel 工作表(2).xlsx": "课题10_库存表_01.xlsx",
    "数据表-最终版v7-改-真最终版.xlsx":     "课题10_数据表_02.xlsx",
    "~临时统计(1)(3).xlsx":                "课题10_~临时统计_03.xlsx",
    "汇总  11月  -副本2.xlsx":             "课题10_11月汇总_04.xlsx",
    "Sheet1-导出-未整理-.xlsx":            "课题10_导出表_05.xlsx",
    "成绩单final-FINAL-v3(新).xlsx":       "课题10_成绩单_06.xlsx",
    "下载 (5).xlsx":                       "课题10_下载记录_07.xlsx",
    "未命名 23.xlsx":                      "课题10_未命名整理_08.xlsx",
    "库存表_0914old.xlsx":                 "课题10_旧库存表_09.xlsx",
    "统计表-张三-李四-合并用.xlsx":         "课题10_合并统计表_10.xlsx",
}


def batch_rename(folder: str, mapping: dict) -> None:
    print(f"目标文件夹: {folder}")
    print(f"待处理文件: {len(mapping)} 个")
    print("-" * 60)

    success, failed = 0, 0
    start = time.perf_counter()  # 高精度计时开始

    for old_name, new_name in mapping.items():
        old_path = os.path.join(folder, old_name)
        new_path = os.path.join(folder, new_name)
        try:
            if not os.path.exists(old_path):
                print(f"[跳过] 找不到文件: {old_name}")
                failed += 1
                continue
            if os.path.exists(new_path):
                print(f"[跳过] 目标已存在, 防止覆盖: {new_name}")
                failed += 1
                continue
            os.rename(old_path, new_path)
            success += 1
            print(f"[完成] {old_name}  ->  {new_name}")
        except OSError as e:
            failed += 1
            print(f"[失败] {old_name}: {e}")

    elapsed = time.perf_counter() - start  # 计时结束
    print("-" * 60)
    print(f"成功 {success} 个 | 失败/跳过 {failed} 个")
    print(f"AI 脚本总耗时: {elapsed:.4f} 秒（约 {elapsed*1000:.1f} 毫秒）")

    # 把耗时结果追加写入记录文件，方便填对比表
    log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AI耗时记录.txt")
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"成功 {success}/{len(mapping)} | 总耗时 {elapsed:.4f} 秒\n")
    print(f"耗时已记录到: {log}")


if __name__ == "__main__":
    batch_rename(FOLDER, RENAME_MAP)
