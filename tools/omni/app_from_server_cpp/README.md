# MiniCPM-o Demo (server.cpp backend)

基于 `llama-server` (server.cpp) HTTP API 的轻量前端 demo，**不依赖 pybind11 模块**。

## 前置条件：模型文件

启动前需将模型文件放置到以下位置（`run.sh` 会自动检查并提示缺失项）：

```
llama.cpp-omni/
└── tools/omni/
    ├── models/
    │   ├── MiniCPM-o-4_5-Q4_K_M.gguf        # LLM 主模型 (~5GB, Q4量化)
    │   ├── tts/
    │   │   └── MiniCPM-o-4_5-tts-F16.gguf    # TTS 模型
    │   └── token2wav/
    │       ├── encoder.gguf                   # Token2Mel encoder
    │       ├── flow_matching.gguf             # Flow matching
    │       ├── flow_extra.gguf                # Flow extra
    │       ├── hifigan2.gguf                  # Vocoder (HiFi-GAN)
    │       └── prompt_cache.gguf              # Prompt cache (n_timesteps=5)
    └── assets/default_ref_audio/
        └── default_ref_audio.wav              # 参考音频 (音色克隆, 6s 16kHz mono)
```

## 架构

```
Browser ──HTTP/SSE──→ FastAPI adapter (:9061) ──HTTP──→ llama-server (:8080)
```

- 文本：SSE 实时流（代理 `/v1/stream/decode`）
- 音频：轮询磁盘 WAV 文件（T2W 写入 `output/round_XXX/tts_wav/`）

## 启动

### 一键启动（推荐）

```bash
cd llama.cpp-omni
bash tools/omni/app_from_server_cpp/run.sh
```

自动启动 llama-server (:8080) + adapter (:9061)，等模型加载完后打开浏览器。
Ctrl+C 退出时两个进程一起停止。

### 手动启动

```bash
cd llama.cpp-omni

# 终端 1: llama-server
./build/bin/llama-server \
    --model tools/omni/models/MiniCPM-o-4_5-Q4_K_M.gguf \
    --port 8080 -ngl 99 -c 4096

# 终端 2: adapter
PYTHONPATH=. .venv/base/bin/python tools/omni/app_from_server_cpp/server.py

# 浏览器打开 http://127.0.0.1:9061
```

## 编译 llama-server

```bash
cd llama.cpp-omni
cmake -B build -DGGML_METAL=ON -DBUILD_PYBIND=ON \
    -DPython3_EXECUTABLE=$(pwd)/.venv/base/bin/python -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target llama-server
```

## Limitations

### 1. 打断（Break）不能立即生效

`/v1/stream/break` 的 `break_event` 设置被 `octx_mutex` 阻塞。该 mutex 在 streaming decode 期间由 worker 线程持有（可达 17+ 秒）。因此：

- 点击「打断」后，break 请求会阻塞到当前 decode 自然结束
- 打断后下一轮 prefill 需等待 TTS 线程设置 `speek_done`（最多 30 秒超时）

**根因**：`handle_stream_break_impl` 在 `octx_mutex` 内部设置 `break_event`（atomic），但 `octx_mutex` 被 decode worker 长期持有。此外 TTS 线程的 break 路径未设置 `speek_done = true`。

**修复方案**（需改 C++，未实施）：
- `server.cpp`：break handler 不锁 `octx_mutex`，直接设置 atomic `break_event`
- `omni.cpp`：TTS `tts_thread_func` break 路径加 `speek_done = true; speek_cv.notify_all()`

### 2. 音频通过文件轮询，非实时流

server.cpp 的 SSE 只推文本，音频由 T2W 写入磁盘。前端通过 500ms 间隔轮询 `/audio/poll` 获取新 WAV 文件。

### 3. n_predict 限制为 256

为避免 LLM 进入重复循环时 TTS KV cache 溢出，`n_predict` 设为 256（约 25 秒语音）。如需更长回复可在 `server.py` 中调整。

## 与 pybind11 版本 (app/server.py) 的对比

| 特性 | pybind11 版 | server.cpp 版 |
|---|---|---|
| 依赖 | 需编译 omni_engine.so | 仅需 llama-server + Python |
| 音频传输 | SSE 实时 PCM 流 | 磁盘文件轮询 |
| 打断 | 即时（同进程 atomic） | 受 mutex 阻塞 |
| 首响延迟 | ~3.2s | ~4s（多一层 HTTP） |
