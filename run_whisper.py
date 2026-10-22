#!/usr/bin/env python3
"""使用 Whisper 将视频/音频文件转换为字幕(SRT)和文本(TXT)。"""

import os
import sys
import subprocess
import shutil

# ==================== 重要参数 ====================
DEFAULT_MODEL = "large-v3"          # 默认 Whisper 模型
DEFAULT_MODEL = "turbo"          # 默认 Whisper 模型
LANGUAGE = "Chinese"                # 识别语言
OUTPUT_FORMAT = "all"                # Whisper 输出格式（all 会生成所有格式）
CLEANUP_EXTS = ["vtt", "tsv", "json"]  # 运行后需要删除的格式
WAV_SAMPLE_RATE = 16000             # 转换后 WAV 采样率
WAV_CHANNELS = 1                    # 转换后 WAV 声道数（单声道）
WHISPER_THREADS = 16                 # Whisper 使用的线程数
TEMPERATURE = 0              # 采样温度，0 表示贪婪解码
BEAM_SIZE = 5                # beam search 大小
BEST_OF = 5                  # 从 beam search 中保留的最佳候选数
PATIENCE = 1                 # beam search 耐心值
LENGTH_PENALTY = 1           # 长度惩罚系数
SUPPRESS_BLANK = True        # 是否抑制空白输出
CONDITION_ON_PREVIOUS_TEXT = False  # 是否基于前文进行条件约束
COMPRESSION_RATIO_THRESHOLD = 2.0   # 压缩比阈值，超过则视为失败
LOGPROB_THRESHOLD = -0.5     # 平均对数概率阈值，低于则视为失败
NO_SPEECH_THRESHOLD = 0.5    # 静音概率阈值，高于则视为静音段
INITIAL_PROMPT = "以下是普通话的句子："  # 初始提示词
# ==================================================


def check_dependencies() -> None:
    """检查 ffmpeg 和 whisper 是否可用。"""
    for cmd in ["ffmpeg", "whisper"]:
        if shutil.which(cmd) is None:
            print(f"错误: 未找到 '{cmd}'，请先安装。")
            sys.exit(1)


def remove_spaces(filename: str) -> str:
    """去除文件名中的空格，保留扩展名。"""
    name, ext = os.path.splitext(filename)
    return name.replace(" ", "") + ext


def run_command(cmd: list[str], description: str) -> None:
    """运行命令并检查返回码。"""
    print(f"[执行] {description}")
    print(f"  命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"错误: {description} 失败 (返回码 {result.returncode})")
        sys.exit(result.returncode)


def main() -> None:
    # ---------- 参数解析 ----------
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <视频/音频文件> [模型名]")
        print(f"  模型默认为: {DEFAULT_MODEL}")
        sys.exit(1)

    input_path = os.path.abspath(sys.argv[1])
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL

    if not os.path.isfile(input_path):
        print(f"错误: 文件不存在 -> {input_path}")
        sys.exit(1)

    # ---------- 依赖检查 ----------
    check_dependencies()

    # ---------- 路径准备 ----------
    dir_path = os.path.dirname(input_path)
    filename_ori = os.path.basename(input_path)
    filename_new = remove_spaces(filename_ori)
    filepath_new = os.path.join(dir_path, filename_new)

    # 新文件名（无扩展名），仅用于生成中间 wav
    name_no_ext_new = os.path.splitext(filename_new)[0]
    wav_file = os.path.join(dir_path, name_no_ext_new + ".wav")

    # 原始文件名（无扩展名），用于生成最终 srt / txt 及跳过检查
    name_no_ext_ori = os.path.splitext(filename_ori)[0]

    # ---------- 跳过已存在的文件 ----------
    srt_file = os.path.join(dir_path, name_no_ext_ori + ".srt")
    if os.path.isfile(srt_file):
        print(f"[跳过] 字幕文件已存在: {srt_file}")
        sys.exit(0)

    print(f"原文件名: {filename_ori}")
    print(f"新文件名: {filename_new}")
    print(f"工作目录: {dir_path}")
    print(f"使用模型: {model}")

    # 如果文件名本身没有空格，也要保证 filepath_new 存在
    if filename_ori != filename_new:
        print(f"[重命名] '{filename_ori}' -> '{filename_new}'")
        os.rename(input_path, filepath_new)
    else:
        print("[重命名] 文件名无需更改")
        filepath_new = input_path

    try:
        # ---------- 转为 WAV ----------
        run_command(
            [
                "ffmpeg", "-y", "-i", filepath_new,
                "-ar", str(WAV_SAMPLE_RATE),
                "-ac", str(WAV_CHANNELS),
                wav_file,
            ],
            "ffmpeg 转换为 WAV",
        )

        # ---------- Whisper 转录 ----------
        output_dir = dir_path
        whisper_cmd = [
            "whisper", wav_file,
            "--model", model,
            "--language", LANGUAGE,
            "--output_dir", output_dir,
            "--output_format", OUTPUT_FORMAT,
            "--threads", str(WHISPER_THREADS),
            "--fp16", "False",
            "--beam_size", str(BEAM_SIZE),
            "--best_of", str(BEST_OF),
            "--temperature", str(TEMPERATURE),
            "--patience", str(PATIENCE),
            "--length_penalty", str(LENGTH_PENALTY),
            # "--suppress_blank", str(SUPPRESS_BLANK),
            # "--without_timestamps",
        ]
        run_command(whisper_cmd, "Whisper 转录")

    finally:
        # ---------- 删除临时 WAV ----------
        if os.path.isfile(wav_file):
            print(f"[清理] 删除临时文件: {wav_file}")
            os.remove(wav_file)

        # ---------- 删除多余格式文件 ----------
        for ext in CLEANUP_EXTS:
            cleanup_file = os.path.join(dir_path, name_no_ext_new + f".{ext}")
            if os.path.isfile(cleanup_file):
                print(f"[清理] 删除多余格式: {cleanup_file}")
                os.remove(cleanup_file)

        # ---------- 恢复原文件名 ----------
        if filename_ori != filename_new and os.path.isfile(filepath_new):
            print(f"[恢复] '{filename_new}' -> '{filename_ori}'")
            os.rename(filepath_new, os.path.join(dir_path, filename_ori))

        # ---------- 将 srt / txt 重命名为原始文件名 ----------
        for ext in ["srt", "txt"]:
            generated = os.path.join(dir_path, name_no_ext_new + f".{ext}")
            target = os.path.join(dir_path, name_no_ext_ori + f".{ext}")
            if os.path.isfile(generated) and generated != target:
                print(f"[重命名] '{os.path.basename(generated)}' -> '{os.path.basename(target)}'")
                os.rename(generated, target)

    print("\n完成！输出文件:")
    for ext in ["srt", "txt"]:
        out_file = os.path.join(dir_path, name_no_ext_ori + f".{ext}")
        if os.path.isfile(out_file):
            size = os.path.getsize(out_file)
            print(f"  {out_file} ({size} bytes)")


if __name__ == "__main__":
    main()
