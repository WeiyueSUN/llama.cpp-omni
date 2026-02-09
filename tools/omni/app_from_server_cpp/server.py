"""MiniCPM-o Demo — 基于 server.cpp HTTP API 的轻量前端

架构:
  Browser ──HTTP/SSE──→ FastAPI adapter (:9061) ──HTTP──→ llama-server (:8080)

与 pybind11 直连版 (app/server.py) 的区别:
  - 不加载 C++ 模块，全部通过 HTTP 调用已启动的 llama-server (server.cpp)
  - 文本通过 SSE 代理实时推流 (/v1/stream/decode → text_queue)
  - 音频通过轮询磁盘 WAV 文件渐进式加载 (T2W 写入 output_dir)
  - Prefill: 前端 base64 → 本地保存临时文件 → server.cpp 用文件路径

不可用功能 (server.cpp 未暴露):
  - 音频 SSE 实时推流 (server.cpp 仅推文本，音频写磁盘)

前置条件:
  先启动 llama-server:
    cd llama.cpp-omni
    ./build/bin/llama-server \\
        --model tools/omni/models/MiniCPM-o-4_5-Q4_K_M.gguf \\
        --port 8080 -ngl 99 -c 4096

启动本服务:
  cd llama.cpp-omni
  PYTHONPATH=. .venv/base/bin/python tools/omni/app_from_server_cpp/server.py
"""

import sys
import os
import re
import argparse
import json
import base64
import asyncio
import shutil
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, AsyncGenerator

import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ==================== 路径设置 ====================
# 目录层级: llama.cpp-omni / tools / omni / app_from_server_cpp / server.py

SCRIPT_DIR = Path(__file__).parent.resolve()
OMNI_DIR = SCRIPT_DIR.parent                          # tools/omni/
LLAMACPP_ROOT = OMNI_DIR.parent.parent                # llama.cpp-omni/

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("adapter_server")


# ==================== 全局状态 ====================

class AdapterState:
    """Adapter 全局状态"""

    def __init__(self) -> None:
        self.backend_url: str = "http://127.0.0.1:8080"
        self.output_dir: str = str(LLAMACPP_ROOT / "tools" / "omni" / "output")
        self.model_dir: str = str(OMNI_DIR / "models")
        self.temp_dir: str = str(LLAMACPP_ROOT / "tmp" / "audio_uploads")
        self.initialized: bool = False
        self.round_idx: int = 0


STATE = AdapterState()


# ==================== FastAPI ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动/关闭时的资源管理"""
    os.makedirs(STATE.temp_dir, exist_ok=True)
    logger.info(
        "adapter started | backend=%s | output_dir=%s",
        STATE.backend_url, STATE.output_dir,
    )
    yield


app = FastAPI(title="MiniCPM-o Adapter (server.cpp backend)", lifespan=lifespan)


# ==================== 静态文件 ====================

app.mount("/static", StaticFiles(directory=str(SCRIPT_DIR / "static")), name="static")


@app.get("/")
async def root() -> FileResponse:
    """前端 demo 页面"""
    return FileResponse(str(SCRIPT_DIR / "static" / "index.html"))


# ==================== API: 健康检查 ====================

@app.get("/health")
async def health() -> JSONResponse:
    """检查 llama-server 后端是否可达"""
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            resp = await client.get(f"{STATE.backend_url}/health")
            return JSONResponse({
                "status": "healthy",
                "initialized": STATE.initialized,
                "backend": "server.cpp",
                "backend_status": resp.json() if resp.status_code == 200 else "error",
            })
        except Exception as e:
            return JSONResponse(
                {"status": "unhealthy", "initialized": False, "error": str(e)},
                status_code=503,
            )


# ==================== API: 初始化引擎 ====================

@app.post("/init")
async def init_engine(request: Request) -> JSONResponse:
    """初始化 omni 引擎（代理 /v1/stream/omni_init）

    Body:
        media_type: int (1=audio, 2=omni)
        duplex_mode: bool
        language: str (暂不支持透传，server.cpp 无此参数)
    """
    body: Dict[str, Any] = await request.json()

    # 构造 server.cpp 所需参数
    # 🔧 tts_bin_dir 必须指向 models/tts/ (不是 token2wav-gguf)
    #    omni_init 从中加载 Projector，否则 Projector 加载失败 → 音质异常
    #    见 py_omni.cpp [CRITICAL FIX] 注释
    # 🔧 n_predict=256: 语音助手场景无需超长回复，256 tokens ≈ 25s 语音
    #    过大的 n_predict (如 2048) 会导致 LLM 进入重复循环时 TTS KV cache 溢出
    payload: Dict[str, Any] = {
        "media_type": body.get("media_type", 1),
        "use_tts": True,
        "duplex_mode": body.get("duplex_mode", False),
        "model_dir": STATE.model_dir + "/",
        "tts_bin_dir": STATE.model_dir + "/tts",
        "token2wav_device": "gpu:0",
        "output_dir": STATE.output_dir,
        "n_predict": 256,
    }

    # 参考音频 (voice cloning)
    ref_audio = str(OMNI_DIR / "assets" / "default_ref_audio" / "default_ref_audio.wav")
    if os.path.isfile(ref_audio):
        payload["voice_audio"] = ref_audio

    logger.info("init_engine: payload=%s", json.dumps(payload, ensure_ascii=False))

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            resp = await client.post(
                f"{STATE.backend_url}/v1/stream/omni_init",
                json=payload,
            )
            data: Dict[str, Any] = resp.json()
            if data.get("success"):
                STATE.initialized = True
                STATE.round_idx = 0
            logger.info("init_engine: response=%s", data)
            return JSONResponse(data)
        except Exception as e:
            logger.error("init_engine failed: %s", e)
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ==================== API: Prefill ====================

@app.post("/prefill")
async def prefill(request: Request) -> JSONResponse:
    """接收 base64 音频，保存临时文件，调用 server.cpp prefill

    Body:
        audio: str (base64 WAV)
        image: str (base64 image, 可选)
    """
    body: Dict[str, Any] = await request.json()
    audio_b64: str = body.get("audio", "")

    if not audio_b64:
        return JSONResponse(
            {"success": False, "error": "audio field is required"},
            status_code=400,
        )

    # 解码并保存到临时文件
    audio_bytes: bytes = base64.b64decode(audio_b64)
    temp_path: str = os.path.join(
        STATE.temp_dir, f"upload_{int(time.time() * 1000)}.wav"
    )
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)
    logger.info("prefill: saved %d bytes → %s", len(audio_bytes), temp_path)

    # 图像 (可选)
    img_path: str = ""
    image_b64: str = body.get("image", "")
    if image_b64:
        img_bytes: bytes = base64.b64decode(image_b64)
        img_path = os.path.join(
            STATE.temp_dir, f"upload_{int(time.time() * 1000)}.jpg"
        )
        with open(img_path, "wb") as f:
            f.write(img_bytes)

    # 调用 server.cpp prefill
    payload: Dict[str, Any] = {
        "audio_path_prefix": temp_path,
        "img_path_prefix": img_path,
        "cnt": 0,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.post(
                f"{STATE.backend_url}/v1/stream/prefill",
                json=payload,
            )
            data: Dict[str, Any] = resp.json()
            logger.info("prefill: response=%s", data)
            return JSONResponse(data)
        except Exception as e:
            logger.error("prefill failed: %s", e)
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ==================== API: Generate (SSE 代理) ====================

@app.post("/generate")
async def generate(request: Request) -> StreamingResponse:
    """代理 server.cpp 的 SSE 文本流 + 注入音频轮询提示

    server.cpp /v1/stream/decode SSE 格式:
        data: {"content": "text", "stop": false, "is_listen": false, "end_of_turn": false}
        data: [DONE]

    本端点原样转发 SSE 事件，并在 [DONE] 后附加 round_idx 信息供前端音频轮询使用。
    """
    current_round: int = STATE.round_idx

    # 清理该 round 的输出目录，避免旧文件污染
    round_wav_dir: str = os.path.join(
        STATE.output_dir, f"round_{current_round:03d}", "tts_wav"
    )
    if os.path.isdir(round_wav_dir):
        shutil.rmtree(round_wav_dir)
        logger.info("generate: cleaned stale output dir %s", round_wav_dir)

    payload: Dict[str, Any] = {
        "stream": True,
        "round_idx": current_round,
    }

    async def event_stream() -> AsyncGenerator[str, None]:
        # 🔧 [关键修复] 在 SSE 流最前面发送 meta（round_idx），
        # 前端收到后立即并行启动音频 polling，不用等文本流结束
        meta = json.dumps({
            "round_idx": current_round,
            "audio_poll_url": f"/audio/poll?round_idx={current_round}&from_idx=0",
        })
        yield f"data: {{\"meta\": {meta}}}\n\n"

        async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{STATE.backend_url}/v1/stream/decode",
                    json=payload,
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line.strip():
                            yield line + "\n\n"
                            # 检测 [DONE]
                            if "[DONE]" in line:
                                break

            except Exception as e:
                logger.error("generate SSE error: %s", e)
                yield f"data: {{\"error\": \"{e}\"}}\n\n"

        # decode 完成后递增 round_idx
        STATE.round_idx = current_round + 1
        logger.info("generate done, next round_idx=%d", STATE.round_idx)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ==================== API: 音频轮询 ====================

@app.get("/audio/poll")
async def audio_poll(round_idx: int = 0, from_idx: int = 0) -> JSONResponse:
    """轮询 output_dir 中新生成的 WAV 文件

    T2W 写入路径: {output_dir}/round_{round_idx:03d}/tts_wav/wav_{N}.wav
    注意: N 不一定从 0 开始！C++ 的 wav_turn_base 机制:
      round_0 → wav_0, wav_1, ...
      round_1 → wav_1000, wav_1001, ...
    因此必须扫描目录中实际存在的 wav 文件名。

    完成标记: generation_done.flag (内容为最后一个 wav 编号)

    Returns:
        files: list[{idx, url}]  新发现的 WAV 文件（按编号排序）
        done: bool               T2W 是否全部完成
        next_from: int           下次轮询的 from_idx (已加载文件数)
        last_wav_idx: int        done=true 时的最后 wav 编号
    """
    wav_dir: str = os.path.join(
        STATE.output_dir, f"round_{round_idx:03d}", "tts_wav"
    )

    if not os.path.isdir(wav_dir):
        return JSONResponse({
            "files": [],
            "done": False,
            "next_from": from_idx,
            "round_idx": round_idx,
        })

    # 扫描目录中所有 wav_*.wav 文件，提取编号并排序
    all_wav_indices: List[int] = []
    for fname in os.listdir(wav_dir):
        m = re.match(r"^wav_(\d+)\.wav$", fname)
        if m:
            wav_path: str = os.path.join(wav_dir, fname)
            if os.path.getsize(wav_path) > 44:
                all_wav_indices.append(int(m.group(1)))
    all_wav_indices.sort()

    # from_idx 表示已加载的文件数，返回之后的新文件
    new_files: List[Dict[str, Any]] = []
    for wav_idx in all_wav_indices[from_idx:]:
        new_files.append({
            "idx": wav_idx,
            "url": f"/audio/file?round_idx={round_idx}&wav_idx={wav_idx}",
        })

    # 检查 generation_done.flag
    done_flag_path: str = os.path.join(wav_dir, "generation_done.flag")
    done: bool = os.path.isfile(done_flag_path)
    last_wav_idx: int = -1
    if done:
        try:
            with open(done_flag_path, "r") as f:
                last_wav_idx = int(f.read().strip())
        except (ValueError, IOError):
            pass

    return JSONResponse({
        "files": new_files,
        "done": done,
        "next_from": from_idx + len(new_files),
        "round_idx": round_idx,
        "last_wav_idx": last_wav_idx,
    })


# ==================== API: 音频文件服务 ====================

@app.get("/audio/file")
async def audio_file(round_idx: int = 0, wav_idx: int = 0) -> FileResponse:
    """直接服务磁盘上的 WAV 文件（禁用缓存，避免旧音频残留）"""
    wav_path: str = os.path.join(
        STATE.output_dir,
        f"round_{round_idx:03d}",
        "tts_wav",
        f"wav_{wav_idx}.wav",
    )
    if os.path.isfile(wav_path):
        return FileResponse(
            wav_path,
            media_type="audio/wav",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
            },
        )
    return JSONResponse({"error": "file not found"}, status_code=404)


# ==================== API: 重置对话 ====================

@app.post("/reset")
async def reset(request: Request) -> JSONResponse:
    """重置对话 — 重新调用 omni_init 恢复完整上下文

    server.cpp 的 /v1/stream/reset 只清 KV cache，不恢复 system prompt。
    因此 reset 必须重新调用 omni_init（含 voice_audio），否则下次生成无上下文 → 垃圾输出。
    代价: ~4s（重新加载 TTS 模型），但保证语义正确。
    """
    # 构造 omni_init 参数（与 init_engine 一致）
    payload: Dict[str, Any] = {
        "media_type": 1,
        "use_tts": True,
        "duplex_mode": False,
        "model_dir": STATE.model_dir + "/",
        "tts_bin_dir": STATE.model_dir + "/tts",
        "token2wav_device": "gpu:0",
        "output_dir": STATE.output_dir,
        "n_predict": 256,
    }
    ref_audio: str = str(OMNI_DIR / "assets" / "default_ref_audio" / "default_ref_audio.wav")
    if os.path.isfile(ref_audio):
        payload["voice_audio"] = ref_audio

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            resp = await client.post(
                f"{STATE.backend_url}/v1/stream/omni_init",
                json=payload,
            )
            data: Dict[str, Any] = resp.json()
            STATE.round_idx = 0
            STATE.initialized = data.get("success", False)
            logger.info("reset (via omni_init): response=%s", data)
            return JSONResponse({"success": data.get("success", False)})
        except Exception as e:
            logger.error("reset failed: %s", e)
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ==================== API: 打断 ====================

@app.post("/stop")
async def stop(request: Request) -> JSONResponse:
    """打断当前生成（代理 /v1/stream/break）"""
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(
                f"{STATE.backend_url}/v1/stream/break",
                json={"reason": "user_interrupt"},
            )
            data: Dict[str, Any] = resp.json()
            logger.info("stop: response=%s", data)
            return JSONResponse(data)
        except httpx.TimeoutException:
            logger.warning("stop: timeout (server busy)")
            return JSONResponse({"success": False, "error": "timeout"}, status_code=504)
        except Exception as e:
            logger.error("stop failed: %s (type=%s)", e, type(e).__name__)
            return JSONResponse({"success": False, "error": f"{type(e).__name__}: {e}"}, status_code=500)


# ==================== main ====================

def main() -> None:
    parser = argparse.ArgumentParser(description="MiniCPM-o Adapter (server.cpp backend)")
    parser.add_argument("--port", type=int, default=9061, help="本服务端口 (default: 9061)")
    parser.add_argument("--backend", type=str, default="http://127.0.0.1:8080",
                        help="llama-server 地址 (default: http://127.0.0.1:8080)")
    parser.add_argument("--output-dir", type=str, default=str(LLAMACPP_ROOT / "tools" / "omni" / "output"),
                        help="T2W 音频输出目录")
    parser.add_argument("--model-dir", type=str, default=str(OMNI_DIR / "models"),
                        help="omni 模型目录 (传给 server.cpp omni_init)")
    args = parser.parse_args()

    STATE.backend_url = args.backend
    STATE.output_dir = args.output_dir
    STATE.model_dir = args.model_dir

    import uvicorn
    logger.info(
        "启动 adapter | port=%d | backend=%s | output_dir=%s | model_dir=%s",
        args.port, args.backend, args.output_dir, args.model_dir,
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
