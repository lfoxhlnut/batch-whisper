#!/usr/bin/env python3
"""Faster-Whisper 转录服务(FastAPI)。

用途
----
把 Whisper 模型加载成一个常驻显存的本地 HTTP 服务,避免每次转译都重新
加载/卸载模型。配合 run_whisper.py 使用:run_whisper.py 负责用 ffmpeg 转
WAV,然后把 WAV 路径 POST 到本服务,拿回带时间戳的 segments 再写出 SRT/TXT。

依赖
----
    uv add fastapi uvicorn faster-whisper

启动
----
    uv run python whisper_server.py
    # 等价于: uv run uvicorn whisper_server:app --host 127.0.0.1 --port 8000
    # 注意: 只用一个 worker,多 worker 会重复加载模型、重复占用显存。

接口
----
    GET  /health      健康检查,返回模型/设备信息
    POST /transcribe  转录单个文件
    GET  /            查看服务信息与调用示例

    POST /transcribe 请求体(JSON):
    {
        "file_path": "D:/audio/input.wav",   # 必填,服务器本机可访问的路径
        "language": "zh",                    # 可选,ISO 语言码;传 null 自动检测
        "batch_size": 8,                     # 可选,默认 BATCH_SIZE
        "beam_size": 5,                      # 可选,默认 BEAM_SIZE
        "temperature": 0.0,                  # 可选,0 表示贪婪解码
        "vad_filter": true,                  # 可选,是否开启 VAD 静音过滤
        "condition_on_previous_text": false, # 可选,长音频建议 false 防幻觉
        "initial_prompt": "以下是普通话的句子:"
    }

    响应(JSON):
    {
        "status": "success",
        "language": "zh",
        "language_probability": 0.99,
        "duration": 123.4,
        "text": "整段文本",
        "segments": [{"id": 1, "start": 0.0, "end": 2.5, "text": "..."}]
    }

注意
----
    - MAX_CONCURRENT_REQUESTS 控制同时占用 GPU 的请求数,默认 1(串行),
      避免多个请求同时进出把显存打爆;显存充足时可以调大。
    - batch_size 与显存直接相关:16G 显存建议 8~12,不够就继续调小。
"""

import os
import sysconfig
import glob
import ctypes

# ==========================================
# 核心修复：自动定位并预加载 NVIDIA CUDA 库
# ==========================================
def preload_nvidia_libs():
    # 获取当前虚拟环境的 site-packages 路径
    site_packages = sysconfig.get_paths()['purelib']
    
    # 拼接出 nvidia 库的绝对路径
    cublas_dir = os.path.join(site_packages, 'nvidia', 'cublas', 'lib')
    cudnn_dir = os.path.join(site_packages, 'nvidia', 'cudnn', 'lib')
    
    # 1. 环境变量兜底（以防万一有些子进程需要）
    nvidia_libs = f"{cublas_dir}:{cudnn_dir}"
    os.environ["LD_LIBRARY_PATH"] = f"{nvidia_libs}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    
    # 2. 强制把 .so 文件加载进内存 (核心大招)
    # 使用 glob 匹配，防止因为具体小版本号不同（如 .12.9.2.10）导致找不到文件
    libs_to_load = []
    
    # 查找 libcublas.so.12*
    cublas_files = glob.glob(os.path.join(cublas_dir, "libcublas.so.12*"))
    libs_to_load.extend(cublas_files)
    
    # 查找 cudnn 相关的 .so (可能有多个，如 libcudnn.so.9, libcudnn_ops.so.9 等)
    cudnn_files = glob.glob(os.path.join(cudnn_dir, "libcudnn*.so.9*"))
    libs_to_load.extend(cudnn_files)
    
    if not libs_to_load:
        print("[警告] 未在虚拟环境中找到 NVIDIA .so 文件，请确认是否安装成功。")
        return

    for lib_path in libs_to_load:
        try:
            # 强制加载到内存
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            print(f"[预加载成功] {os.path.basename(lib_path)}")
        except OSError as e:
            print(f"[预加载失败] {lib_path}: {e}")

# 执行预加载（放在所有其他 import 之前）
preload_nvidia_libs()

import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from faster_whisper import BatchedInferencePipeline, WhisperModel
from pydantic import BaseModel, Field

# ==================== 关键参数 ====================
SERVER_HOST = "127.0.0.1"          # 服务监听地址
SERVER_PORT = 8000                 # 服务端口
MODEL_NAME = "/mnt/f/model"            # Whisper 模型(large-v3)
# MODEL_NAME = "large-v3"            # Whisper 模型(默认 large-v3)
DEVICE = "cuda"                    # 运行设备:cuda / cpu
COMPUTE_TYPE = "float16"           # 计算精度:GPU 用 float16,CPU 可改 int8

DEFAULT_LANGUAGE = "zh"            # 默认识别语言(ISO 码,中文为 zh);None 自动检测
BATCH_SIZE = 16                    # BatchedInferencePipeline 批大小,显存不足时调小
BEAM_SIZE = 5                      # beam search 大小
TEMPERATURE = 0.0                  # 采样温度,0 表示贪婪解码
VAD_FILTER = True                  # 是否开启 VAD 静音过滤(长音频强烈建议开启)
VAD_MIN_SILENCE_MS = 500           # VAD 判定静音的最短时长(毫秒)
CONDITION_ON_PREVIOUS_TEXT = False # 长音频建议 False,避免"复读机"式幻觉
MAX_CONCURRENT_REQUESTS = 1        # 同时占用 GPU 的请求数,默认串行防 OOM
# ==================================================

# 模型与流水线在进程内只加载一次
_model: Optional[WhisperModel] = None
_pipeline: Optional[BatchedInferencePipeline] = None
# 限制同时占用 GPU 的请求数量
_gpu_semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)


def load_model() -> None:
    """加载 Whisper 模型并构建批处理流水线(进程启动时执行一次)。"""
    global _model, _pipeline
    print(
        f"[启动] 加载 Faster-Whisper 模型 {MODEL_NAME} "
        f"(device={DEVICE}, compute_type={COMPUTE_TYPE}) "
        f"\nbatch_size={BATCH_SIZE}"
    )
    _model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    _pipeline = BatchedInferencePipeline(model=_model)
    print("[启动] 模型加载完成,服务已就绪。")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """FastAPI 生命周期:启动时加载模型。"""
    load_model()
    yield


app = FastAPI(title="Faster-Whisper 转录服务", lifespan=lifespan)


class TranscribeRequest(BaseModel):
    """转录请求参数。"""

    file_path: str = Field(..., description="待转录的音频/视频文件路径(服务器本机可访问)")
    language: Optional[str] = Field(DEFAULT_LANGUAGE, description="识别语言 ISO 码,如 zh;null 自动检测")
    batch_size: int = Field(BATCH_SIZE, ge=1, description="批大小,显存不足时调小")
    beam_size: int = Field(BEAM_SIZE, ge=1, description="beam search 大小")
    temperature: float = Field(TEMPERATURE, ge=0.0, description="采样温度,0 表示贪婪解码")
    vad_filter: bool = Field(VAD_FILTER, description="是否开启 VAD 静音过滤")
    condition_on_previous_text: bool = Field(
        CONDITION_ON_PREVIOUS_TEXT,
        description="是否参考前文,长音频建议 False",
    )
    initial_prompt: Optional[str] = Field(None, description="初始提示词,如 '以下是普通话的句子:'")


class Segment(BaseModel):
    """一条带时间戳的字幕分段。"""

    id: int
    start: float
    end: float
    text: str


class TranscribeResponse(BaseModel):
    """转录响应。"""

    status: str
    language: str
    language_probability: float
    duration: float
    text: str
    segments: list[Segment]


@app.get("/")
def index() -> dict:
    """返回服务信息与调用示例。"""
    return {
        "service": "faster-whisper",
        "model": MODEL_NAME,
        "endpoints": {
            "GET /health": "健康检查",
            "POST /transcribe": "转录单个音视频文件",
        },
        "example": {
            "file_path": "D:/audio/input.wav",
            "language": DEFAULT_LANGUAGE,
        },
    }


@app.get("/health")
def health() -> dict:
    """健康检查:模型加载完成后返回 ok。"""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe(req: TranscribeRequest) -> TranscribeResponse:
    """转录单个文件,返回带时间戳的分段结果。"""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")
    if not os.path.isfile(req.file_path):
        raise HTTPException(status_code=400, detail=f"文件不存在: {req.file_path}")

    # 串行占用 GPU:segments 是生成器,必须在持有信号量期间消费完
    with _gpu_semaphore:
        try:
            segment_iter, info = _pipeline.transcribe(
                req.file_path,
                language=req.language or None,
                batch_size=req.batch_size,
                beam_size=req.beam_size,
                temperature=req.temperature,
                vad_filter=req.vad_filter,
                vad_parameters={"min_silence_duration_ms": VAD_MIN_SILENCE_MS},
                condition_on_previous_text=req.condition_on_previous_text,
                initial_prompt=req.initial_prompt,
                without_timestamps=False,  # 关键：让模型输出时间戳 token
            )
            segments = [
                Segment(
                    id=index,
                    start=round(float(segment.start), 3),
                    end=round(float(segment.end), 3),
                    text=segment.text.strip(),
                )
                for index, segment in enumerate(segment_iter, start=1)
            ]
        except Exception as exc:  # 统一转成 HTTP 500,方便客户端定位
            print(exc)
            raise HTTPException(status_code=500, detail=f"转录失败: {exc}") from exc

    text = "\n".join(segment.text for segment in segments if segment.text)
    return TranscribeResponse(
        status="success",
        language=getattr(info, "language", "") or "",
        language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
        duration=float(getattr(info, "duration", 0.0) or 0.0),
        text=text,
        segments=segments,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
