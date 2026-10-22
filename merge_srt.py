#!/usr/bin/env python3
"""
SRT文件合并脚本
将SRT字幕文件按照30秒一组进行合并，生成TXT文件
"""

import re
import os
from typing import List, Tuple

def parse_time(time_str: str) -> int:
    """将时间字符串转换为秒数"""
    # 格式: 00:00:05,840
    parts = time_str.replace(',', '.').split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return int(hours * 3600 + minutes * 60 + seconds)

def format_time(seconds: int) -> str:
    """将秒数转换为时间格式字符串"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def parse_srt_file(file_path: str) -> List[Tuple[int, int, str]]:
    """解析SRT文件，返回(开始时间, 结束时间, 文本)的列表"""
    subtitles = []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 用空行分割每个字幕块
    blocks = content.strip().split('\n\n')

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        # 解析时间行
        time_line = lines[1]
        time_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', time_line)

        if time_match:
            start_time = parse_time(time_match.group(1))
            end_time = parse_time(time_match.group(2))

            # 合并文本内容
            text = ' '.join(lines[2:])
            subtitles.append((start_time, end_time, text))

    return subtitles

def merge_subtitles_by_time_groups(subtitles: List[Tuple[int, int, str]], group_size: int = 30) -> List[Tuple[str, List[str]]]:
    """将字幕按时间分组合并"""
    if not subtitles:
        return []

    # 找到最大时间以确定分组数量
    max_time = max(end_time for _, end_time, _ in subtitles)
    num_groups = (max_time // group_size) + 1

    groups = []
    for i in range(num_groups):
        start_seconds = i * group_size
        end_seconds = (i + 1) * group_size

        time_label = f"{format_time(start_seconds)} --> {format_time(end_seconds)}"

        # 收集这个时间组内的所有字幕文本
        group_texts = []
        for start_time, end_time, text in subtitles:
            # 如果字幕与当前时间组有重叠，则包含它
            if start_time < end_seconds and end_time > start_seconds:
                group_texts.append(text)

        groups.append((time_label, group_texts))

    return groups

def write_merged_file(groups: List[Tuple[str, List[str]]], output_path: str):
    """写入合并后的文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (time_label, texts) in enumerate(groups):
            # 写入时间标签
            f.write(f"({time_label})\n")

            # 写入该组的文本内容
            if texts:
                f.write(' '.join(texts) + '\n')

            # 组间添加空行（最后一组除外）
            if i < len(groups) - 1:
                f.write('\n')

def main():
    import sys

    if len(sys.argv) != 2:
        print("用法: python merge_srt.py <srt文件路径>")
        sys.exit(1)

    srt_file = sys.argv[1]

    if not os.path.exists(srt_file):
        print(f"错误: 文件 '{srt_file}' 不存在")
        sys.exit(1)

    # 生成输出文件路径
    base_path = os.path.splitext(srt_file)[0]
    output_file = f"{base_path}_merged.txt"

    try:
        # 解析SRT文件
        subtitles = parse_srt_file(srt_file)

        # 按30秒一组进行合并
        groups = merge_subtitles_by_time_groups(subtitles, 30)

        # 写入合并后的文件
        write_merged_file(groups, output_file)

        print(f"处理完成！输出文件: {output_file}")

    except Exception as e:
        print(f"处理文件时出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()