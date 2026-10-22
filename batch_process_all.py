#!/usr/bin/env python3
"""递归遍历目录及其所有子目录，对每个子目录调用 batch_process_videos.py 进行批量转录。"""
# cd ~/audio && uv run batch_process_all.py "/mnt/e/vm-mnt/class/26高代强化课"

import os
import subprocess
import sys
from pathlib import Path

# ==================== 重要参数 ====================
DEFAULT_MODEL = "large-v3"          # Whisper 模型
RECURSIVE = True                    # 是否递归遍历子目录
INCLUDE_ROOT = True                 # 是否同时处理根目录本身（设为 False 则只处理子目录）
# ==================================================


def collect_directories(root: str) -> list[Path]:
    """收集需要处理的目录列表。"""
    root_path = Path(root).resolve()
    dirs: list[Path] = []

    if INCLUDE_ROOT:
        dirs.append(root_path)

    if RECURSIVE:
        for d in sorted(root_path.rglob("*")):
            if d.is_dir():
                dirs.append(d)

    return dirs


def has_media_files(directory: Path) -> bool:
    """检查目录中是否存在支持的媒体文件。"""
    media_extensions = {
        ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
        ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma",
    }
    for f in directory.iterdir():
        if f.is_file() and f.suffix.lower() in media_extensions:
            return True
    return False


def run_batch_process(directory: Path, model: str) -> bool:
    """调用 batch_process_videos.py 处理单个目录。返回是否成功。"""
    script_path = Path(__file__).parent / "batch_process_videos.py"
    cmd = [sys.executable, str(script_path), str(directory), model]

    print(f"\n{'#' * 60}")
    print(f"[Batch] 处理目录: {directory}")
    print(f"{'#' * 60}")

    result = subprocess.run(cmd)
    return result.returncode == 0


def main() -> None:
    # ---------- 参数解析 ----------
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <根目录路径> [模型名]")
        print(f"  模型默认为: {DEFAULT_MODEL}")
        sys.exit(1)

    root_dir = os.path.abspath(sys.argv[1])
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL

    if not os.path.isdir(root_dir):
        print(f"错误: 目录不存在 -> {root_dir}")
        sys.exit(1)

    # ---------- 收集目录 ----------
    all_dirs = collect_directories(root_dir)

    # 过滤掉没有媒体文件的目录
    target_dirs = [d for d in all_dirs if has_media_files(d)]

    if not target_dirs:
        print(f"在 '{root_dir}' 及其子目录中未找到包含媒体文件的目录。")
        sys.exit(0)

    print(f"根目录: {root_dir}")
    print(f"模型: {model}")
    print(f"找到 {len(target_dirs)} 个包含媒体文件的目录:")
    for i, d in enumerate(target_dirs, 1):
        rel = os.path.relpath(d, root_dir)
        print(f"  {i}. {rel}/")

    # ---------- 逐个目录处理 ----------
    success_list: list[str] = []
    fail_list: list[str] = []

    for idx, directory in enumerate(target_dirs, 1):
        rel = os.path.relpath(directory, root_dir)
        print(f"\n>>> 目录进度: [{idx}/{len(target_dirs)}] {rel}/")

        if run_batch_process(directory, model):
            success_list.append(rel)
        else:
            fail_list.append(rel)

    # ---------- 汇总 ----------
    print(f"\n{'=' * 60}")
    print(f"全部完成！成功 {len(success_list)}/{len(target_dirs)}")
    if fail_list:
        print(f"失败目录:")
        for name in fail_list:
            print(f"  - {name}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
