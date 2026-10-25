
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：
    递归遍历指定目录及其所有子目录。
    如果同一目录下同时存在 b.srt 和 b_merged.txt，
    且 b_merged.txt 的大小严格大于 b.srt，
    则删除这两个文件。

用法：
    1. 预览待删除文件（默认模式，不实际删除）：
       python clean_subtitles.py "D:\\Subtitles"

    2. 确认后执行删除：
       python clean_subtitles.py "D:\\Subtitles" --delete

    3. Linux 示例：
       python3 clean_subtitles.py /data/subtitles --delete

注意：
    - 默认只预览，不删除文件。
    - 只有 TXT 大小严格大于 SRT 时才删除。
    - 文件大小单位为字节（Bytes）。
    - 只匹配同一目录下的对应文件。
    - 请先预览结果，确认无误后再使用 --delete。
"""

import argparse
from pathlib import Path


# ==================== 重要参数 ====================

# 对应文件的命名规则：
# b.srt -> b_merged.txt
SRT_SUFFIX = ".srt"
TXT_SUFFIX = "_merged.txt"

# 是否递归搜索子目录；True 表示递归遍历
RECURSIVE = True


# ==================== 核心逻辑 ====================

def clean_files(root: Path, delete: bool = False) -> None:
    """查找符合条件的文件对，并根据 delete 参数决定是否删除。"""

    # 递归搜索所有 .srt 文件
    srt_files = root.rglob(f"*{SRT_SUFFIX}") if RECURSIVE else root.glob(
        f"*{SRT_SUFFIX}"
    )

    matched_count = 0
    deleted_count = 0

    for srt in srt_files:
        # b.srt -> b_merged.txt
        txt = srt.with_name(f"{srt.stem}{TXT_SUFFIX}")

        # 对应 TXT 不存在时，跳过
        if not txt.is_file():
            continue

        try:
            # 获取文件大小，单位为字节
            srt_size = srt.stat().st_size
            txt_size = txt.stat().st_size

            # 只有 TXT 严格大于 SRT 时才符合条件
            if txt_size <= srt_size:
                continue

            matched_count += 1
            print(
                f"[{'删除' if delete else '预览'}] "
                f"{srt} ({srt_size} bytes)"
            )
            print(f"         {txt} ({txt_size} bytes)")

            if delete:
                # 先删除 SRT，再删除 TXT
                srt.unlink()
                txt.unlink()
                deleted_count += 1

        except OSError as exc:
            # 例如文件被占用、权限不足或文件已不存在
            print(f"[错误] 处理文件失败：{srt}")
            print(f"       原因：{exc}")

    print("\n========== 处理结果 ==========")
    print(f"符合条件的文件对：{matched_count}")

    if delete:
        print(f"成功删除的文件对：{deleted_count}")
    else:
        print("当前为预览模式，没有主动删除任何文件。")
        print("确认结果无误后，添加 --delete 参数执行删除。")


# ==================== 命令行入口 ====================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="递归清理 TXT 大于对应 SRT 的字幕文件对。"
    )

    # 必填参数：要遍历的根目录
    parser.add_argument(
        "directory",
        type=Path,
        help="要遍历的目录路径",
    )

    # 可选参数：只有指定后才真正删除
    parser.add_argument(
        "--delete",
        action="store_true",
        help="实际删除符合条件的文件对；默认只预览",
    )

    args = parser.parse_args()
    root = args.directory

    # 检查目录是否合法
    if not root.exists():
        parser.error(f"目录不存在：{root}")

    if not root.is_dir():
        parser.error(f"指定路径不是目录：{root}")

    clean_files(root.resolve(), delete=args.delete)


if __name__ == "__main__":
    main()
