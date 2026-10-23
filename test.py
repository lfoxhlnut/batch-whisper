#!/usr/bin/env python3
"""whisper_server / run_whisper 端到端测试脚本。

用法
----
    # 1. 先在另一个终端启动服务(模型加载完再跑测试)
    uv run python whisper_server.py

    # 2. 运行全部测试
    uv run python test.py

    # 也可以只跑其中一个
    uv run python test.py server    # 测试 whisper_server
    uv run python test.py whisper   # 测试 run_whisper

测试内容
--------
    测试 1(server):用 ./test/input.mp4(文件名不含空格)直接 POST 给
        whisper_server 的 /transcribe 接口,验证返回了 segments/text。
    测试 2(whisper):用 ./test/ff gg.mp4(文件名含空格)调用 run_whisper.py,
        验证 SRT/TXT 正常生成、原文件名被正确恢复、临时重命名文件没有残留。

说明
----
    - 路径全部用 pathlib 拼接,Linux / Windows 都通用;
    - 两个测试都要求 whisper_server 已启动并加载完模型;
    - 测试 2 会先删除旧的 SRT/TXT,避免命中 run_whisper 的"已存在则跳过"逻辑;
    - 生成的 SRT/TXT 默认保留在 test/ 目录,方便查看,可用下方参数关闭。
"""

import subprocess
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    print("错误: 缺少 httpx,请先安装依赖: uv add fastapi uvicorn faster-whisper httpx")
    sys.exit(1)

# ==================== 关键参数 ====================
PROJECT_ROOT = Path(__file__).resolve().parent
TEST_MEDIA_DIR = PROJECT_ROOT / "test"

INPUT_MP4 = TEST_MEDIA_DIR / "input.mp4"       # 测试 1:文件名不含空格
SPACED_MP4 = TEST_MEDIA_DIR / "ff gg.mp4"      # 测试 2:文件名含空格

SERVER_HOST = "127.0.0.1"                      # 需与 whisper_server.py 一致
SERVER_PORT = 8000                             # 需与 whisper_server.py 一致
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
HEALTH_ENDPOINT = f"{SERVER_URL}/health"
TRANSCRIBE_ENDPOINT = f"{SERVER_URL}/transcribe"
LANGUAGE = "zh"                                # 识别语言

HEALTH_TIMEOUT_SECONDS = 5                     # 健康检查超时
TRANSCRIBE_TIMEOUT_SECONDS = 3600              # 单次转录请求超时
RUN_WHISPER_TIMEOUT_SECONDS = 7200             # 子进程 run_whisper 超时
CLEANUP_OUTPUTS_BEFORE_TEST = True             # 测试 2 前删除旧的 srt/txt
CLEANUP_OUTPUTS_AFTER_TEST = False             # 测试 2 后是否删除生成的 srt/txt
# ==================================================


def check_server_ready() -> None:
    """确认 whisper_server 已启动并完成模型加载。"""
    try:
        response = httpx.get(HEALTH_ENDPOINT, timeout=HEALTH_TIMEOUT_SECONDS)
    except Exception as exc:
        raise RuntimeError(
            f"无法连接 {SERVER_URL},请先启动服务: uv run python whisper_server.py"
        ) from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"服务未就绪 (HTTP {response.status_code}): {response.text}"
            "(模型可能仍在加载,请稍后重试)"
        )


def test_server_input_mp4() -> None:
    """测试 1:用不含空格的 ./test/input.mp4 直接调用 whisper_server。"""
    if not INPUT_MP4.is_file():
        raise FileNotFoundError(f"测试文件不存在: {INPUT_MP4}")

    payload = {"file_path": str(INPUT_MP4), "language": LANGUAGE}
    print(f"[测试1] POST {TRANSCRIBE_ENDPOINT}")
    print(f"[测试1] file_path = {INPUT_MP4}")

    with httpx.Client(timeout=TRANSCRIBE_TIMEOUT_SECONDS) as client:
        response = client.post(TRANSCRIBE_ENDPOINT, json=payload)
        response.raise_for_status()
        data = response.json()

    if data.get("status") != "success":
        raise AssertionError(f"服务返回状态异常: {data.get('status')!r}")

    segments = data.get("segments") or []
    text = (data.get("text") or "").strip()
    if not segments and not text:
        raise AssertionError("服务没有返回任何 segments/text")

    print(f"[测试1] 返回 {len(segments)} 个 segment,文本 {len(text)} 字符")


def test_run_whisper_spaced() -> None:
    """测试 2:用含空格的 ./test/ff gg.mp4 调用 run_whisper.py。"""
    if not SPACED_MP4.is_file():
        raise FileNotFoundError(f"测试文件不存在: {SPACED_MP4}")

    srt_path = SPACED_MP4.with_suffix(".srt")
    txt_path = SPACED_MP4.with_suffix(".txt")
    leftover_path = TEST_MEDIA_DIR / "ffgg.mp4"  # run_whisper 处理时的临时文件名

    if CLEANUP_OUTPUTS_BEFORE_TEST:
        for path in (srt_path, txt_path):
            path.unlink(missing_ok=True)

    script_path = PROJECT_ROOT / "run_whisper.py"
    cmd = [sys.executable, str(script_path), str(SPACED_MP4)]
    print(f"[测试2] 执行: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        timeout=RUN_WHISPER_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise AssertionError(f"run_whisper.py 退出码非 0: {result.returncode}")

    if not SPACED_MP4.is_file():
        raise AssertionError(f"原文件没有恢复: {SPACED_MP4}")
    if leftover_path.exists():
        raise AssertionError(f"临时重命名文件有残留: {leftover_path}")

    if not srt_path.is_file() or srt_path.stat().st_size == 0:
        raise AssertionError(f"SRT 未生成或为空: {srt_path}")
    if "-->" not in srt_path.read_text(encoding="utf-8"):
        raise AssertionError(f"SRT 内容格式异常(缺少时间轴): {srt_path}")

    if not txt_path.is_file() or txt_path.stat().st_size == 0:
        raise AssertionError(f"TXT 未生成或为空: {txt_path}")

    print(f"[测试2] 生成: {srt_path.name} / {txt_path.name}")

    if CLEANUP_OUTPUTS_AFTER_TEST:
        for path in (srt_path, txt_path):
            path.unlink(missing_ok=True)


def main() -> int:
    """按顺序执行测试,返回退出码:0 全部通过,1 有失败,2 参数错误。"""
    tests = {
        "server": ("测试 1: whisper_server + input.mp4(无空格)", test_server_input_mp4),
        "whisper": ("测试 2: run_whisper + 'ff gg.mp4'(含空格)", test_run_whisper_spaced),
    }

    selected = sys.argv[1].lower() if len(sys.argv) > 1 else None
    if selected is not None and selected not in tests:
        print(f"用法: {Path(sys.argv[0]).name} [server|whisper]")
        return 2

    try:
        check_server_ready()
    except Exception as exc:
        print(f"✗ 无法开始测试: {exc}")
        return 1

    failures = 0
    for key, (title, test_func) in tests.items():
        if selected is not None and key != selected:
            continue

        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)

        started = time.perf_counter()
        try:
            test_func()
        except Exception as exc:  # 测试脚本统一捕获,最后汇总结果
            failures += 1
            print(f"✗ 失败: {exc}")
        else:
            print(f"✓ 通过 ({time.perf_counter() - started:.1f}s)")

    print("\n" + "=" * 60)
    if failures:
        print(f"结果: {failures} 个测试失败")
        return 1
    print("结果: 全部测试通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
