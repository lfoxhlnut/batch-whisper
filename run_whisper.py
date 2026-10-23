#!/usr/bin/env python3
"""使用 Faster-Whisper 服务把视频/音频文件转换为字幕(SRT)和文本(TXT)。

用法
----
    # 1. 先启动常驻转录服务(模型只加载一次,常驻显存)
    uv run python whisper_server.py

    # 2. 再转译单个文件(模型由服务端配置决定,[模型名] 参数已忽略)
    uv run python run_whisper.py <视频/音频文件>

流程
----
    ffmpeg 转 16kHz 单声道 WAV -> 调用 whisper_server 的 /transcribe 接口
    -> 根据返回的 segments 写出 SRT / TXT -> 清理临时文件。

说明
----
    - 转录参数(batch_size、VAD、beam_size 等)统一在 whisper_server.py 顶部配置;
    - 本脚本只负责 ffmpeg 转换、HTTP 调用和字幕文件生成。
"""

import os
import sys
import subprocess
import shutil

import httpx

# ==================== 重要参数 ====================
DEFAULT_MODEL = "large-v3"          # 仅用于提示;实际模型由 whisper_server.py 的 MODEL_NAME 决定
LANGUAGE = "zh"                     # faster-whisper 语言码,中文为 zh
INITIAL_PROMPT = "以下是普通话的句子:"  # 初始提示词,可设为 None
WAV_SAMPLE_RATE = 16000             # 转换后 WAV 采样率
WAV_CHANNELS = 1                    # 转换后 WAV 声道数(单声道)
CLEANUP_EXTS = ["vtt", "tsv", "json"]  # 运行后需要删除的多余格式

# ---------- whisper_server(FastAPI)服务地址 ----------
SERVER_HOST = "127.0.0.1"           # 服务地址,需与 whisper_server.py 一致
SERVER_PORT = 8000                  # 服务端口,需与 whisper_server.py 一致
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
HEALTH_ENDPOINT = f"{SERVER_URL}/health"
TRANSCRIBE_ENDPOINT = f"{SERVER_URL}/transcribe"
REQUEST_TIMEOUT_SECONDS = 7200      # 单次转录请求超时(秒),长音频建议调大
# ==================================================


def check_dependencies() -> None:
    """检查 ffmpeg 以及 whisper_server 服务是否可用。"""
    if shutil.which("ffmpeg") is None:
        print("错误: 未找到 'ffmpeg',请先安装。")
        sys.exit(1)

    try:
        response = httpx.get(HEALTH_ENDPOINT, timeout=5)
    except Exception:
        print(f"错误: 无法连接 Whisper 服务 {SERVER_URL}")
        print("请先启动服务: uv run python whisper_server.py")
        sys.exit(1)

    if response.status_code != 200:
        print(f"错误: Whisper 服务未就绪 (HTTP {response.status_code}): {response.text}")
        print("模型可能仍在加载,请稍后重试。")
        sys.exit(1)


def remove_spaces(filename: str) -> str:
    """去除文件名中的空格,保留扩展名。"""
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


def transcribe_with_server(wav_path: str) -> dict:
    """调用 whisper_server 的 /transcribe 接口,返回响应 JSON。"""
    payload = {
        "file_path": wav_path,
        "language": LANGUAGE,
        "initial_prompt": INITIAL_PROMPT,
    }
    print(f"[转录] 请求 {TRANSCRIBE_ENDPOINT}")
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = client.post(TRANSCRIBE_ENDPOINT, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        print(f"错误: 服务端转录失败 ({exc.response.status_code}): {exc.response.text}")
        sys.exit(1)
    except Exception as exc:
        print(f"错误: 调用转录服务失败: {exc}")
        sys.exit(1)


def format_srt_time(seconds: float) -> str:
    """把秒数格式化为 SRT 时间戳:HH:MM:SS,mmm。"""
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments: list[dict], output_path: str) -> None:
    """把 segments 写成 SRT 字幕文件。"""
    with open(output_path, "w", encoding="utf-8") as file:
        index = 0
        for segment in segments:
            text = (segment.get("text") or "").strip()
            if not text:
                continue
            index += 1
            file.write(f"{index}\n")
            file.write(
                f"{format_srt_time(segment['start'])} --> "
                f"{format_srt_time(segment['end'])}\n"
            )
            file.write(f"{text}\n\n")


def write_txt(segments: list[dict], output_path: str, fallback_text: str = "") -> None:
    """把 segments 写成纯文本文件(每段一行);没有 segments 时使用整段文本。"""
    lines = [(segment.get("text") or "").strip() for segment in segments]
    lines = [line for line in lines if line]
    text = "\n".join(lines) if lines else fallback_text.strip()
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(text)
        if text:
            file.write("\n")


def main() -> None:
    # ---------- 参数解析 ----------
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <视频/音频文件>")
        print(f"  模型由 whisper_server.py 决定,默认: {DEFAULT_MODEL}")
        sys.exit(1)

    input_path = os.path.abspath(sys.argv[1])
    if len(sys.argv) > 2:
        print(f"提示: 模型参数 '{sys.argv[2]}' 已忽略,模型由 whisper_server.py 的 MODEL_NAME 决定。")

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
    srt_file = os.path.join(dir_path, name_no_ext_ori + ".srt")
    txt_file = os.path.join(dir_path, name_no_ext_ori + ".txt")

    # ---------- 跳过已存在的文件 ----------
    if os.path.isfile(srt_file):
        print(f"[跳过] 字幕文件已存在: {srt_file}")
        sys.exit(0)

    print(f"原文件名: {filename_ori}")
    print(f"新文件名: {filename_new}")
    print(f"工作目录: {dir_path}")
    print(f"使用模型: {DEFAULT_MODEL} (由 whisper_server.py 决定)")

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
                "-hide_banner",       # 隐藏横幅
                "-loglevel", "error", # 只显示错误
                "-ar", str(WAV_SAMPLE_RATE),
                "-ac", str(WAV_CHANNELS),
                wav_file,
            ],
            "ffmpeg 转换为 WAV",
        )

        # ---------- 调用 whisper_server 转录 ----------
        result = transcribe_with_server(wav_file)

        # ---------- 写出 SRT / TXT（直接使用原始文件名） ----------
        segments = result.get("segments", [])
        write_srt(segments, srt_file)
        write_txt(segments, txt_file, fallback_text=result.get("text", ""))
        print(f"[输出] {srt_file}")
        print(f"[输出] {txt_file}")

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

    print("\n完成！输出文件:")
    for out_file in (srt_file, txt_file):
        if os.path.isfile(out_file):
            size = os.path.getsize(out_file)
            print(f"  {out_file} ({size} bytes)")


if __name__ == "__main__":
    main()
