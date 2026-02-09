#!/usr/bin/env bash
# =============================================================================
# MiniCPM-o macOS App — 一键启动 (pybind11 直连版)
#
# 从任意目录执行均可:
#   bash /path/to/llama.cpp-omni/tools/omni/app/run.sh --simplex
#   bash /path/to/llama.cpp-omni/tools/omni/app/run.sh --duplex
#
# 自动处理: 模型检查 → venv 创建 → 依赖安装 → pybind11 模块编译 → 启动服务
# =============================================================================
set -euo pipefail

# 从脚本位置计算项目根目录，立即 cd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

VENV="${REPO_ROOT}/.venv/base"
MODEL_DIR="tools/omni/models"
LLM_MODEL="${MODEL_DIR}/MiniCPM-o-4_5-Q4_K_M.gguf"
TTS_DIR="${MODEL_DIR}/tts"
REF_AUDIO="tools/omni/assets/default_ref_audio/default_ref_audio.wav"

# 解析 --port
PORT=9060
for arg in "$@"; do
    if [[ "${prev_arg:-}" == "--port" ]]; then PORT="$arg"; fi
    prev_arg="$arg"
done

echo ""
echo "============================================"
echo "  MiniCPM-o App (pybind11 直连)"
echo "============================================"
echo "  REPO_ROOT: ${REPO_ROOT}"
echo ""

# ==================== 1. 检查模型 ====================

MISSING=""
[ ! -f "${LLM_MODEL}" ] && MISSING="${MISSING}\n  - LLM 主模型: ${REPO_ROOT}/${LLM_MODEL}"
[ ! -d "${TTS_DIR}" ]    && MISSING="${MISSING}\n  - TTS 模型目录: ${REPO_ROOT}/${TTS_DIR}/"
[ ! -f "${REF_AUDIO}" ]  && MISSING="${MISSING}\n  - 参考音频: ${REPO_ROOT}/${REF_AUDIO}"

if [ -n "${MISSING}" ]; then
    echo "[ERROR] 缺少以下模型/资源文件:"
    echo -e "${MISSING}"
    echo ""
    echo "模型目录结构:"
    echo "  tools/omni/"
    echo "  ├── models/"
    echo "  │   ├── MiniCPM-o-4_5-Q4_K_M.gguf    # LLM 主模型 (~5GB)"
    echo "  │   ├── tts/                           # TTS 模型"
    echo "  │   └── token2wav/                     # Token2Wav 模型"
    echo "  └── assets/default_ref_audio/"
    echo "      └── default_ref_audio.wav          # 参考音频"
    exit 1
fi
echo "[1/3] 模型检查通过"

# ==================== 2. venv + 依赖 ====================

if [ ! -d "${VENV}" ]; then
    echo "[2/3] 创建 venv: ${VENV}"
    python3 -m venv "${VENV}"
    echo "  安装依赖..."
    "${VENV}/bin/pip" install --quiet --upgrade pip
    "${VENV}/bin/pip" install --quiet pybind11 numpy fastapi uvicorn
else
    echo "[2/3] venv 已存在"
    "${VENV}/bin/pip" install --quiet pybind11 numpy fastapi uvicorn 2>/dev/null || true
fi

PYTHON="${VENV}/bin/python"

# ==================== 3. 编译 omni_engine ====================

# 查找 omni_engine.so (文件名含 cpython 版本后缀)
OMNI_SO=$(find build/bin -name "omni_engine*.so" 2>/dev/null | head -1)

if [ -z "${OMNI_SO}" ]; then
    echo "[3/3] omni_engine.so 未编译，自动编译中（首次约 3-5 分钟）..."
    cmake -B build -DGGML_METAL=ON -DBUILD_PYBIND=ON \
        -DPython3_EXECUTABLE="${PYTHON}" -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j 8 --target omni_engine
    OMNI_SO=$(find build/bin -name "omni_engine*.so" 2>/dev/null | head -1)
    if [ -z "${OMNI_SO}" ]; then
        echo "[ERROR] 编译失败，未找到 omni_engine.so"
        exit 1
    fi
    echo "  编译完成: ${OMNI_SO}"
else
    echo "[3/3] omni_engine 已编译: ${OMNI_SO}"
fi

# ==================== 启动服务 ====================

echo ""
echo "============================================"
echo "  启动服务 (port=${PORT})"
echo "  Ctrl+C 退出"
echo "============================================"
echo ""

# 清理旧进程
lsof -ti:${PORT} | xargs kill -9 2>/dev/null || true
sleep 1

# 自动打开浏览器
(sleep 3 && open "http://localhost:${PORT}" 2>/dev/null || true) &

PYTHONPATH=. exec "${PYTHON}" tools/omni/app/server.py --port "${PORT}" "$@"
