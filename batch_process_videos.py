#!/usr/bin/env python3
"""批量处理目录中的视频/音频文件，依次执行 Whisper 转录和 SRT 合并。"""

import os
import subprocess
import sys
from pathlib import Path

# ==================== 可配置参数 ====================
# 支持的视频/音频扩展名（小写）
MEDIA_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma",
}
DEFAULT_THREADS = 16 # 在这里定义没用, run_whisper.py 没有使用这个传递的参数
DEFAULT_MODEL = "large-v3"  # Whisper 模型，None 则使用 run_whisper.py 默认值
# ==================================================


def collect_media_files(directory: str) -> list[Path]:
    """收集目录中所有视频/音频文件，按文件名排序。"""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        print(f"错误: 目录不存在 -> {directory}")
        sys.exit(1)

    files = sorted(
        f for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in MEDIA_EXTENSIONS
    )
    return files


def run_whisper(input_path: Path, model: str | None, threads: int | None) -> bool:
    """调用 run_whisper.py 处理单个文件。返回是否成功。"""
    cmd = [sys.executable, str(Path(__file__).parent / "run_whisper.py"), str(input_path)]
    # if model:
    #     cmd.append("--model")
    #     cmd.append(model)
    # if threads:
    #     cmd.append("--threads")
    #     cmd.append(str(threads))

    print(f"\n{'=' * 60}")
    print(f"[Whisper] 处理: {input_path.name}")
    print(f"{'=' * 60}")

    result = subprocess.run(cmd)
    return result.returncode == 0


def run_merge_srt(srt_path: Path) -> bool:
    """调用 merge_srt.py 处理 SRT 文件。返回是否成功。"""
    cmd = [sys.executable, str(Path(__file__).parent / "merge_srt.py"), str(srt_path)]

    print(f"\n[Merge] 处理: {srt_path.name}")

    result = subprocess.run(cmd)
    return result.returncode == 0


def main() -> None:
    # ---------- 参数解析 ----------
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <目录路径> [模型名]")
        print(f"  模型默认为: {DEFAULT_MODEL}")
        sys.exit(1)

    directory = os.path.abspath(sys.argv[1])
    # 获取最后一级目录名
    last_dir_name = os.path.basename(directory)

    # 判断并退出
    if last_dir_name == "double-speed": # TODO: 添加过滤列表,作为默认参数
        print(f"检测到目录名为 '{last_dir_name}'，跳过")
        sys.exit(0)  # 0 表示正常退出（如果是异常拦截，可以填 1）

    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL

    # ---------- 收集文件 ----------
    media_files = collect_media_files(directory)

    if not media_files:
        print(f"在 '{directory}' 中未找到支持的视频/音频文件。")
        print(f"支持的格式: {', '.join(sorted(MEDIA_EXTENSIONS))}")
        sys.exit(0)

    print(f"目录: {directory}")
    print(f"模型: {model}")
    print(f"找到 {len(media_files)} 个待处理文件:")
    for i, f in enumerate(media_files, 1):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {i}. {f.name} ({size_mb:.1f} MB)")

    # ---------- 逐个处理 ----------
    success_list: list[str] = []
    fail_list: list[str] = []

    for idx, media_file in enumerate(media_files, 1):
        print(f"\n>>> 进度: [{idx}/{len(media_files)}] {media_file.name}")

        # Step 1: Whisper 转录
        if not run_whisper(media_file, model, DEFAULT_THREADS):
            print(f"[跳过] Whisper 处理失败: {media_file.name}")
            fail_list.append(media_file.name)
            continue

        # Step 2: 合并 SRT
        srt_path = media_file.with_suffix(".srt")
        if srt_path.exists():
            run_merge_srt(srt_path)
        else:
            print(f"[警告] 未找到生成的 SRT 文件: {srt_path}")

        success_list.append(media_file.name)

    # ---------- 汇总 ----------
    print(f"\n{'=' * 60}")
    print(f"处理完成！成功 {len(success_list)}/{len(media_files)}")
    if fail_list:
        print(f"失败文件:")
        for name in fail_list:
            print(f"  - {name}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
