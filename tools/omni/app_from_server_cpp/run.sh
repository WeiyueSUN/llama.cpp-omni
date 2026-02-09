#!/bin/bash
# MiniCPM-o Demo — 一键启动 llama-server + adapter
#
# 使用方法 (从任意目录执行均可):
#   bash /path/to/llama.cpp-omni/tools/omni/app_from_server_cpp/run.sh
#
# Ctrl+C 退出时两个进程一起停止。

set -euo pipefail

# 从脚本位置计算项目根目录，并立即 cd 过去
# 这样无论用户从哪里执行脚本，后续操作都基于项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
echo "[INFO] 项目根目录: ${REPO_ROOT}"

VENV="${REPO_ROOT}/.venv/base"
SERVER_PORT=8080
ADAPTER_PORT=9061
MODEL_DIR="tools/omni/models"
LLM_MODEL="${MODEL_DIR}/MiniCPM-o-4_5-Q4_K_M.gguf"
TTS_DIR="${MODEL_DIR}/tts"
REF_AUDIO="tools/omni/assets/default_ref_audio/default_ref_audio.wav"
SERVER_BIN="build/bin/llama-server"

# ==================== 检查模型 ====================

MISSING=""
[ ! -f "${LLM_MODEL}" ] && MISSING="${MISSING}\n  - LLM 主模型: ${REPO_ROOT}/${LLM_MODEL} (~5GB Q4量化)"
[ ! -d "${TTS_DIR}" ]    && MISSING="${MISSING}\n  - TTS 模型目录: ${REPO_ROOT}/${TTS_DIR}/ (含 MiniCPM-o-4_5-tts-F16.gguf)"
[ ! -f "${REF_AUDIO}" ]  && MISSING="${MISSING}\n  - 参考音频: ${REPO_ROOT}/${REF_AUDIO}"

if [ -n "${MISSING}" ]; then
    echo "[ERROR] 缺少以下模型/资源文件:"
    echo -e "${MISSING}"
    echo ""
    echo "模型目录结构:"
    echo "  tools/omni/"
    echo "  ├── models/"
    echo "  │   ├── MiniCPM-o-4_5-Q4_K_M.gguf    # LLM 主模型"
    echo "  │   ├── tts/                           # TTS 模型"
    echo "  │   │   └── MiniCPM-o-4_5-tts-F16.gguf"
    echo "  │   └── token2wav/                     # Token2Wav 模型"
    echo "  │       ├── encoder.gguf"
    echo "  │       ├── flow_matching.gguf"
    echo "  │       ├── flow_extra.gguf"
    echo "  │       ├── hifigan2.gguf"
    echo "  │       └── prompt_cache.gguf"
    echo "  └── assets/default_ref_audio/"
    echo "      └── default_ref_audio.wav          # 参考音频 (音色克隆)"
    exit 1
fi

# ==================== 自动准备环境 ====================

# venv: 不存在则自动创建 + 安装依赖
if [ ! -d "${VENV}" ]; then
    echo "[INFO] 创建 venv: ${VENV}"
    python3 -m venv "${VENV}"
    echo "[INFO] 安装 Python 依赖..."
    "${VENV}/bin/pip" install -q fastapi uvicorn httpx
else
    # 已有 venv，静默确保依赖存在
    "${VENV}/bin/pip" install -q fastapi uvicorn httpx 2>/dev/null || true
fi

PYTHON="${VENV}/bin/python"

# llama-server: 不存在则自动编译
if [ ! -f "${SERVER_BIN}" ]; then
    echo "[INFO] llama-server 未编译，自动编译中（首次约 2-3 分钟）..."
    cmake -B build -DGGML_METAL=ON -DBUILD_PYBIND=ON \
        -DPython3_EXECUTABLE="${PYTHON}" -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j 8 --target llama-server
    echo "[INFO] 编译完成"
fi

# ==================== 清理旧进程 ====================

lsof -ti:${SERVER_PORT} | xargs kill -9 2>/dev/null || true
lsof -ti:${ADAPTER_PORT} | xargs kill -9 2>/dev/null || true
sleep 1

# ==================== trap: Ctrl+C 时一起退出 ====================

SERVER_PID=""
ADAPTER_PID=""

cleanup() {
    echo ""
    echo "[INFO] 正在退出..."
    [ -n "${ADAPTER_PID}" ] && kill "${ADAPTER_PID}" 2>/dev/null
    [ -n "${SERVER_PID}" ] && kill "${SERVER_PID}" 2>/dev/null
    wait 2>/dev/null
    echo "[INFO] 已退出"
    exit 0
}
trap cleanup INT TERM

# ==================== 启动 llama-server ====================

echo "=========================================="
echo " MiniCPM-o Demo (server.cpp backend)"
echo "=========================================="
echo ""
echo "[1/2] 启动 llama-server (:${SERVER_PORT})..."

./"${SERVER_BIN}" \
    --model "${LLM_MODEL}" \
    --port ${SERVER_PORT} \
    -ngl 99 -c 4096 &
SERVER_PID=$!

# 等待 llama-server 就绪
echo "[1/2] 等待模型加载..."
for i in $(seq 1 60); do
    if curl -s "http://127.0.0.1:${SERVER_PORT}/health" 2>/dev/null | grep -q '"ok"'; then
        echo "[1/2] llama-server 就绪"
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[ERROR] llama-server 启动失败"
        exit 1
    fi
    sleep 2
done

# ==================== 启动 adapter ====================

echo "[2/2] 启动 adapter (:${ADAPTER_PORT})..."

PYTHONPATH=. "${PYTHON}" tools/omni/app_from_server_cpp/server.py \
    --port ${ADAPTER_PORT} --backend "http://127.0.0.1:${SERVER_PORT}" &
ADAPTER_PID=$!

sleep 2
echo ""
echo "=========================================="
echo " 浏览器访问: http://127.0.0.1:${ADAPTER_PORT}"
echo " Ctrl+C 退出（两个进程一起停止）"
echo "=========================================="
echo ""

# macOS 自动打开浏览器
open "http://127.0.0.1:${ADAPTER_PORT}" 2>/dev/null || true

# 等待进程（macOS 默认 bash 不支持 wait -n，用 wait 等所有子进程）
wait
