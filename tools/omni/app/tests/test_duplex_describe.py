"""双工描述画面测试：用户说"实时描述画面中的内容"，模型持续描述两个场景

流程 (与 HF duplex demo 对齐):
  每 1s 一个 tick，共 20 tick:
    使用 /omni/duplex_tick 合并 endpoint (prefill + generate 单次 HTTP 调用)
    → 收集一个 chunk (≤10 LLM tokens)
  总计 20 次 duplex_tick 调用

输入:
  - 音频: user_query_description_camera.wav (2.46s, "实时描述画面中的内容")，0s 开始，之后静音
  - 图片A (0-10s): 夜晚房间    图片B (10-20s): 白天公园

输出:
  - results/duplex_describe/chunks/tick_NNN_{text,audio}.{txt,wav}
  - results/duplex_describe/output.mp4 (左声道=user, 右声道=AI, 按时间轴对齐)
  - results/duplex_describe/events.json

使用方式:
  cd llama.cpp-omni && PYTHONPATH=. .venv/base/bin/python tools/omni/app/tests/test_duplex_describe.py
"""
import os
import sys
import io
import json
import time
import base64
import struct
import subprocess
import wave
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

import numpy as np
import requests

# ==================== 配置 ====================

SERVER_URL = "http://127.0.0.1:9060"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OMNI_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # tools/omni/
USER_AUDIO = os.path.join(SCRIPT_DIR, "user_query_description_camera.wav")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results/duplex_describe")

SEND_INTERVAL_S = 1.0         # 每 1s 一个 tick
INPUT_SAMPLE_RATE = 16000      # 输入音频采样率
TTS_SAMPLE_RATE = 24000        # TTS 输出采样率
TOTAL_TICKS = 20               # 总 tick 数 = 20s
IMAGE_SWITCH_TICK = 10         # tick 10 (10s) 切换到第二张图
TIMEOUT = 60

# 需要从文本中过滤的特殊 token
SPECIAL_TOKEN_PATTERNS = [
    "<|tts_bos|>", "<|tts_eos|>", "<|speak|>", "<|listen|>",
    "<|turn_eos|>", "<|chunk_eos|>", "<|chunk_tts_eos|>",
    "<|im_start|>", "<|im_end|>", "</s>",
]


# ==================== 图片生成 ====================

def generate_test_image_night_room() -> Tuple[str, bytes]:
    """场景 A: 夜晚房间"""
    from PIL import Image, ImageDraw

    w, h = 640, 480
    img = Image.new("RGB", (w, h), (40, 45, 55))
    draw = ImageDraw.Draw(img)
    draw.rectangle([420, 30, 580, 200], fill=(20, 30, 60), outline=(120, 120, 120), width=3)
    draw.rectangle([430, 40, 500, 110], fill=(30, 40, 80))
    draw.rectangle([510, 40, 570, 110], fill=(30, 40, 80))
    draw.rectangle([430, 120, 500, 190], fill=(25, 35, 70))
    draw.rectangle([510, 120, 570, 190], fill=(25, 35, 70))
    for x, y in [(450, 60), (540, 80), (470, 150)]:
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 255, 200))
    draw.rectangle([50, 320, 400, 340], fill=(120, 80, 40))
    draw.rectangle([70, 340, 90, 450], fill=(100, 65, 30))
    draw.rectangle([360, 340, 380, 450], fill=(100, 65, 30))
    draw.rectangle([180, 240, 200, 320], fill=(80, 80, 80))
    draw.polygon([(140, 240), (240, 240), (220, 200), (160, 200)], fill=(255, 220, 100))
    draw.rectangle([260, 300, 340, 320], fill=(200, 50, 50))
    draw.rectangle([100, 290, 130, 320], fill=(200, 200, 220))
    draw.rectangle([0, 450, w, h], fill=(80, 60, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    raw_bytes = buf.getvalue()
    return base64.b64encode(raw_bytes).decode("utf-8"), raw_bytes


def generate_test_image_park() -> Tuple[str, bytes]:
    """场景 B: 白天公园"""
    from PIL import Image, ImageDraw

    w, h = 640, 480
    img = Image.new("RGB", (w, h), (135, 200, 250))
    draw = ImageDraw.Draw(img)
    for cx, cy, rx, ry in [(150, 60, 60, 25), (170, 50, 40, 18),
                            (450, 80, 55, 22), (470, 70, 35, 16)]:
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(255, 255, 255))
    draw.polygon([(0, 250), (100, 180), (200, 220), (320, 160), (450, 210),
                  (550, 170), (640, 230), (640, 280), (0, 280)], fill=(100, 160, 80))
    draw.rectangle([0, 270, w, h], fill=(80, 170, 60))
    for y in range(280, h, 30):
        draw.rectangle([0, y, w, y + 8], fill=(90, 180, 70))
    draw.rectangle([80, 200, 105, 370], fill=(100, 70, 30))
    draw.ellipse([30, 100, 155, 220], fill=(50, 130, 40))
    draw.ellipse([45, 130, 140, 240], fill=(60, 140, 50))
    draw.rectangle([500, 240, 515, 340], fill=(110, 75, 35))
    draw.ellipse([470, 170, 545, 260], fill=(55, 135, 45))
    draw.polygon([(280, 480), (360, 480), (340, 380), (310, 310),
                  (320, 280), (300, 280), (290, 310), (300, 380)], fill=(190, 175, 140))
    draw.rectangle([380, 330, 460, 340], fill=(140, 90, 40))
    draw.rectangle([380, 340, 390, 370], fill=(120, 75, 30))
    draw.rectangle([450, 340, 460, 370], fill=(120, 75, 30))
    draw.rectangle([375, 310, 465, 318], fill=(140, 90, 40))
    draw.ellipse([530, 20, 590, 80], fill=(255, 230, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    raw_bytes = buf.getvalue()
    return base64.b64encode(raw_bytes).decode("utf-8"), raw_bytes


# ==================== 数据结构 ====================

@dataclass
class TickResult:
    """一个 tick (1s) 的结果"""
    tick_idx: int
    tick_start_s: float          # 相对于 stream 开始的时间
    prefill_ms: float = 0.0
    generate_ms: float = 0.0
    text: str = ""               # LLM 生成文本 (已过滤特殊 token)
    is_listen: bool = False
    audio_chunks: List[bytes] = field(default_factory=list)  # PCM int16 bytes
    audio_dur_s: float = 0.0     # 总音频时长
    image_tag: str = ""


@dataclass
class TestState:
    """测试全局状态"""
    stream_start_time: float = 0.0
    ticks: List[TickResult] = field(default_factory=list)
    events: List[Dict] = field(default_factory=list)

    def log(self, event_type: str, detail: str) -> None:
        elapsed = time.time() - self.stream_start_time
        entry = {
            "elapsed_s": round(elapsed, 2),
            "type": event_type,
            "detail": detail,
            "ts": datetime.now().strftime("%H:%M:%S.%f")[:12],
        }
        self.events.append(entry)
        print(f"  [{elapsed:6.1f}s] [{event_type:<12}] {detail}", flush=True)


# ==================== 工具函数 ====================

def strip_special_tokens(text: str) -> str:
    """过滤文本中的特殊 token 标记"""
    for pattern in SPECIAL_TOKEN_PATTERNS:
        text = text.replace(pattern, "")
    return text.strip()


def load_wav_pcm16(wav_path: str) -> np.ndarray:
    """读取 WAV 返回 float32 [-1, 1]"""
    import soundfile as sf
    data, sr = sf.read(wav_path, dtype='float32')
    if sr != INPUT_SAMPLE_RATE:
        ratio = INPUT_SAMPLE_RATE / sr
        n_out = int(len(data) * ratio)
        indices = np.arange(n_out) / ratio
        data = np.interp(indices, np.arange(len(data)), data).astype(np.float32)
    return data


def float32_to_wav_base64(audio: np.ndarray, sr: int = INPUT_SAMPLE_RATE) -> str:
    """float32 → WAV base64"""
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    pcm_bytes = pcm16.tobytes()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(pcm_bytes)))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(pcm_bytes)))
    buf.write(pcm_bytes)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def write_wav(path: str, pcm_bytes: bytes, sr: int = TTS_SAMPLE_RATE) -> None:
    """PCM int16 bytes → WAV 文件"""
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_bytes)


def prepare_audio_chunks(audio_path: str) -> List[Tuple[str, np.ndarray]]:
    """用户音频 + zero pad 静音，按 1s 切块

    Returns:
        list of (label, chunk_float32)，共 TOTAL_TICKS 个，每个 1s = 16000 samples
    """
    chunk_samples = int(SEND_INTERVAL_S * INPUT_SAMPLE_RATE)
    total_samples = int(TOTAL_TICKS * SEND_INTERVAL_S * INPUT_SAMPLE_RATE)

    user_audio = load_wav_pcm16(audio_path)
    print(f"用户音频: {len(user_audio)} samples ({len(user_audio) / INPUT_SAMPLE_RATE:.2f}s)")

    full_stream = np.zeros(total_samples, dtype=np.float32)
    copy_len = min(len(user_audio), total_samples)
    full_stream[:copy_len] = user_audio[:copy_len]

    chunks: List[Tuple[str, np.ndarray]] = []
    for i in range(0, total_samples, chunk_samples):
        end = min(i + chunk_samples, total_samples)
        chunk = full_stream[i:end]
        start_s = i / INPUT_SAMPLE_RATE
        end_s = end / INPUT_SAMPLE_RATE
        has_audio = np.abs(chunk).max() > 0.001
        label = f"[{start_s:.0f}s-{end_s:.0f}s] {'SPEECH' if has_audio else 'SILENCE'}"
        chunks.append((label, chunk))

    return chunks


# ==================== 核心: 单次 duplex tick (prefill + generate 合并) ====================

def do_tick(state: TestState, tick_idx: int,
            audio_b64: str, image_b64: str, img_tag: str,
            ) -> Tuple[str, bool, List[bytes], float, float]:
    """执行一次 duplex_tick (prefill + generate 合并为单次 HTTP 调用)

    使用 /omni/duplex_tick endpoint，省去一次 HTTP round-trip。

    Returns:
        (text, is_listen, audio_chunks, audio_dur_s, total_ms)
    """
    t0 = time.time()
    resp = requests.post(
        f"{SERVER_URL}/omni/duplex_tick",
        json={"audio": audio_b64, "image": image_b64},
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()

    texts: List[str] = []
    is_listen = False
    audio_chunks: List[bytes] = []
    total_audio_s = 0.0

    buffer = ""
    for raw_chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
        if raw_chunk is None:
            continue
        buffer += raw_chunk

        while "\n\n" in buffer:
            idx = buffer.index("\n\n")
            event_block = buffer[:idx]
            buffer = buffer[idx + 2:]

            for line in event_block.split("\n"):
                if not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])

                    if "chunk_data" in data:
                        cd = data["chunk_data"]
                        wav_b64 = cd.get("wav", "")
                        raw_text = cd.get("text", "")
                        text = strip_special_tokens(raw_text)

                        if text:
                            texts.append(text)

                        if wav_b64:
                            pcm = base64.b64decode(wav_b64)
                            sr = cd.get("sample_rate", TTS_SAMPLE_RATE)
                            dur = len(pcm) / (sr * 2)
                            audio_chunks.append(pcm)
                            total_audio_s += dur

                    if "is_listen" in data and data["is_listen"]:
                        is_listen = True

                except json.JSONDecodeError:
                    pass

    total_ms = (time.time() - t0) * 1000
    full_text = "".join(texts)
    return full_text, is_listen, audio_chunks, total_audio_s, total_ms


# ==================== 主循环: 20 ticks ====================

def run_ticks(state: TestState, audio_chunks: List[Tuple[str, np.ndarray]],
              image_a_b64: str, image_b_b64: str) -> None:
    """核心循环: 每 tick 调用 /omni/duplex_tick (合并 prefill+generate)，共 TOTAL_TICKS 次"""

    for tick_idx in range(TOTAL_TICKS):
        tick_wall_start = time.time()
        tick_start_s = tick_wall_start - state.stream_start_time

        label, audio_data = audio_chunks[tick_idx]
        audio_b64 = float32_to_wav_base64(audio_data)
        image_b64 = image_a_b64 if tick_idx < IMAGE_SWITCH_TICK else image_b_b64
        img_tag = "IMG_A(夜晚房间)" if tick_idx < IMAGE_SWITCH_TICK else "IMG_B(白天公园)"

        state.log("TICK_START", f"tick#{tick_idx} {label} {img_tag}")

        # 单次 HTTP 调用: prefill + generate
        text, is_listen, audio_pcms, audio_dur, tick_ms = do_tick(
            state, tick_idx, audio_b64, image_b64, img_tag,
        )

        # 记录结果 (prefill_ms 不再单独可测，合入 tick_ms)
        tick = TickResult(
            tick_idx=tick_idx,
            tick_start_s=round(tick_start_s, 2),
            prefill_ms=0.0,  # 合并 endpoint 无法拆分
            generate_ms=round(tick_ms, 1),  # 整个 tick 耗时
            text=text,
            is_listen=is_listen,
            audio_chunks=audio_pcms,
            audio_dur_s=round(audio_dur, 3),
            image_tag=img_tag,
        )
        state.ticks.append(tick)

        detail = f"tick#{tick_idx}: {tick_ms:.0f}ms"
        if text:
            detail += f' | "{text[:40]}"'
        if is_listen:
            detail += " [LISTEN]"
        if audio_dur > 0:
            detail += f" [{audio_dur:.2f}s audio]"
        state.log("TICK_DONE", detail)


# ==================== 结果保存 ====================

def save_results(state: TestState, user_audio_path: str,
                 image_a_raw: bytes, image_b_raw: bytes) -> None:
    """保存 per-tick WAV/TXT + 合并音频 + MP4"""
    chunks_dir = os.path.join(OUTPUT_DIR, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    all_text = ""
    tick_manifest: List[Dict] = []

    for tick in state.ticks:
        tag = f"tick_{tick.tick_idx:03d}"

        # 保存文本
        if tick.text:
            with open(os.path.join(chunks_dir, f"{tag}_text.txt"), "w", encoding="utf-8") as f:
                f.write(tick.text)
            all_text += tick.text

        # 保存音频 (合并该 tick 的所有 audio chunks)
        if tick.audio_chunks:
            merged_pcm = b"".join(tick.audio_chunks)
            write_wav(os.path.join(chunks_dir, f"{tag}_audio.wav"), merged_pcm)

        tick_manifest.append({
            "tick_idx": tick.tick_idx,
            "tick_start_s": tick.tick_start_s,
            "prefill_ms": tick.prefill_ms,
            "generate_ms": tick.generate_ms,
            "text": tick.text,
            "is_listen": tick.is_listen,
            "audio_dur_s": tick.audio_dur_s,
            "image_tag": tick.image_tag,
        })

    # --- AI 音频按时间轴放置 (20s 总长) ---
    total_samples_24k = int(TOTAL_TICKS * SEND_INTERVAL_S * TTS_SAMPLE_RATE)
    ai_timeline = np.zeros(total_samples_24k, dtype=np.int16)

    for tick in state.ticks:
        if not tick.audio_chunks:
            continue
        merged_pcm = b"".join(tick.audio_chunks)
        samples = np.frombuffer(merged_pcm, dtype=np.int16)
        # 按 tick 的实际 elapsed 时间放置
        start_sample = int(tick.tick_start_s * TTS_SAMPLE_RATE)
        end_sample = min(start_sample + len(samples), total_samples_24k)
        copy_len = end_sample - start_sample
        if copy_len > 0 and start_sample < total_samples_24k:
            ai_timeline[start_sample:end_sample] = samples[:copy_len]

    ai_pcm_bytes = ai_timeline.tobytes()
    merged_path = os.path.join(OUTPUT_DIR, "ai_merged.wav")
    write_wav(merged_path, ai_pcm_bytes)
    active_audio = sum(t.audio_dur_s for t in state.ticks)
    print(f"  AI 音频轨: {merged_path} (时间轴 {TOTAL_TICKS}s, 有效音频 {active_audio:.2f}s)")

    # 保存文本
    if all_text:
        text_path = os.path.join(OUTPUT_DIR, "response_text.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(all_text)
        print(f"  文本: {all_text[:200]}")

    # 保存 manifest + events
    with open(os.path.join(OUTPUT_DIR, "tick_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(tick_manifest, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUTPUT_DIR, "events.json"), "w", encoding="utf-8") as f:
        json.dump(state.events, f, ensure_ascii=False, indent=2)

    # 保存图片
    for tag, raw in [("image_A_night_room.jpg", image_a_raw), ("image_B_day_park.jpg", image_b_raw)]:
        with open(os.path.join(OUTPUT_DIR, tag), "wb") as f:
            f.write(raw)

    # MP4
    synthesize_mp4(user_audio_path, ai_pcm_bytes, image_a_raw, image_b_raw)


def synthesize_mp4(user_audio_path: str, ai_pcm_timeline: bytes,
                   image_a_raw: bytes, image_b_raw: bytes) -> None:
    """ffmpeg 合成 MP4: 左声道=user(20s), 右声道=AI(20s 时间轴)"""
    if not ai_pcm_timeline:
        print("  [SKIP] 无 AI 音频，跳过 MP4 合成")
        return

    mp4_path = os.path.join(OUTPUT_DIR, "output.mp4")
    tmp_dir = os.path.join(OUTPUT_DIR, "_tmp_mp4")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        total_dur_s = TOTAL_TICKS * SEND_INTERVAL_S
        total_samples_24k = int(total_dur_s * TTS_SAMPLE_RATE)

        # 1. user 音频 → 24kHz, pad 到 20s
        user_wav_24k = os.path.join(tmp_dir, "user_24k.wav")
        user_data = load_wav_pcm16(user_audio_path)
        ratio = TTS_SAMPLE_RATE / INPUT_SAMPLE_RATE
        n_out = int(len(user_data) * ratio)
        user_24k = np.interp(np.arange(n_out) / ratio, np.arange(len(user_data)), user_data).astype(np.float32)
        user_padded = np.zeros(total_samples_24k, dtype=np.float32)
        copy_len = min(len(user_24k), total_samples_24k)
        user_padded[:copy_len] = user_24k[:copy_len]
        user_pcm16 = (np.clip(user_padded, -1.0, 1.0) * 32767).astype(np.int16)
        write_wav(user_wav_24k, user_pcm16.tobytes(), TTS_SAMPLE_RATE)

        # 2. AI 音频 (已按时间轴)
        ai_wav_path = os.path.join(tmp_dir, "ai_24k.wav")
        write_wav(ai_wav_path, ai_pcm_timeline, TTS_SAMPLE_RATE)

        # 3. 立体声 (左=user, 右=AI)
        stereo_wav = os.path.join(tmp_dir, "stereo.wav")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", user_wav_24k, "-i", ai_wav_path,
            "-filter_complex", "[0:a][1:a]amerge=inputs=2[aout]",
            "-map", "[aout]", "-ac", "2", stereo_wav,
        ], check=True, capture_output=True)

        # 4. 视频帧 (1fps)
        for sec in range(int(total_dur_s)):
            img_raw = image_a_raw if sec < IMAGE_SWITCH_TICK else image_b_raw
            with open(os.path.join(tmp_dir, f"frame_{sec:03d}.jpg"), "wb") as f:
                f.write(img_raw)

        # 5. MP4
        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", "1",
            "-i", os.path.join(tmp_dir, "frame_%03d.jpg"),
            "-i", stereo_wav,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "1",
            "-c:a", "aac", "-b:a", "128k", "-shortest",
            mp4_path,
        ], check=True, capture_output=True)

        print(f"  MP4: {mp4_path}")
        print(f"  (macOS: open {mp4_path})")

    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] ffmpeg: {e.stderr.decode()[:500]}")
    except Exception as e:
        print(f"  [ERROR] MP4: {e}")
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ==================== 主流程 ====================

def main() -> None:
    print("=" * 70)
    print("双工描述画面测试 (tick-based: 每 tick = 1 duplex_tick 调用)")
    print(f"服务器: {SERVER_URL}")
    print(f"用户音频: {USER_AUDIO}")
    print(f"总 tick 数: {TOTAL_TICKS} (每 tick {SEND_INTERVAL_S}s)")
    print(f"图片切换: tick #{IMAGE_SWITCH_TICK}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Health check
    try:
        health = requests.get(f"{SERVER_URL}/health", timeout=5).json()
        print(f"\n[HEALTH] {json.dumps(health, ensure_ascii=False)}")
    except Exception as e:
        print(f"\n[ERROR] 无法连接: {e}")
        sys.exit(1)

    if not os.path.exists(USER_AUDIO):
        print(f"\n[ERROR] 音频不存在: {USER_AUDIO}")
        sys.exit(1)

    # 准备音频
    audio_chunks = prepare_audio_chunks(USER_AUDIO)
    print(f"准备了 {len(audio_chunks)} 个音频 chunk:")
    for label, chunk in audio_chunks[:4]:
        rms = np.sqrt(np.mean(chunk ** 2))
        print(f"  {label}  RMS={rms:.6f}")
    print(f"  ... (后续 {len(audio_chunks) - 4} 个均为 SILENCE)")

    # 初始化或重置
    if not health.get("initialized"):
        print(f"\n初始化双工模式...")
        body = {"media_type": "omni", "duplex_mode": True, "language": "zh"}
        resp = requests.post(f"{SERVER_URL}/omni/init_sys_prompt", json=body, timeout=120)
        resp.raise_for_status()
        print(f"  Init: {json.dumps(resp.json(), ensure_ascii=False)}")
    else:
        print(f"\n已初始化，重置对话...")
        requests.post(f"{SERVER_URL}/omni/reset", timeout=10)

    # 生成图片
    print("\n生成模拟摄像头画面...")
    image_a_b64, image_a_raw = generate_test_image_night_room()
    image_b_b64, image_b_raw = generate_test_image_park()
    print(f"  图A (夜晚房间): {len(image_a_raw)} bytes")
    print(f"  图B (白天公园): {len(image_b_raw)} bytes")

    # 运行
    state = TestState()
    state.stream_start_time = time.time()

    print(f"\n{'=' * 70}")
    print(f"开始 {TOTAL_TICKS} 个 tick (每 tick: /omni/duplex_tick 合并调用)")
    print(f"{'=' * 70}")

    run_ticks(state, audio_chunks, image_a_b64, image_b_b64)

    # 停止
    try:
        requests.post(f"{SERVER_URL}/omni/stop", timeout=5)
    except Exception:
        pass

    total_time = time.time() - state.stream_start_time
    total_audio = sum(t.audio_dur_s for t in state.ticks)
    total_text = "".join(t.text for t in state.ticks)
    n_listen = sum(1 for t in state.ticks if t.is_listen)
    n_speak = sum(1 for t in state.ticks if t.text or t.audio_chunks)

    # 总结
    print(f"\n\n{'=' * 70}")
    print("测试总结")
    print(f"{'=' * 70}")
    print(f"  总运行时间: {total_time:.1f}s (目标: {TOTAL_TICKS}s)")
    print(f"  总 tick 数: {TOTAL_TICKS}")
    print(f"  speak ticks: {n_speak}, listen ticks: {n_listen}")
    print(f"  总有效音频: {total_audio:.2f}s")
    print(f"  模型输出: {total_text[:200]}")

    # per-tick 时间分布
    tick_times = [t.generate_ms for t in state.ticks]  # generate_ms = 整个 tick 耗时
    listen_times = [t.generate_ms for t in state.ticks if t.is_listen]
    speak_times = [t.generate_ms for t in state.ticks if not t.is_listen]
    print(f"\n  Tick 总体: avg={np.mean(tick_times):.0f}ms, "
          f"min={np.min(tick_times):.0f}ms, max={np.max(tick_times):.0f}ms")
    if listen_times:
        print(f"  LISTEN ticks: avg={np.mean(listen_times):.0f}ms, "
              f"min={np.min(listen_times):.0f}ms, max={np.max(listen_times):.0f}ms")
    if speak_times:
        print(f"  SPEAK ticks: avg={np.mean(speak_times):.0f}ms, "
              f"min={np.min(speak_times):.0f}ms, max={np.max(speak_times):.0f}ms")
    print(f"  目标: ≤1000ms/tick 实现实时")

    # 保存
    print(f"\n{'=' * 70}")
    print("保存结果...")
    save_results(state, USER_AUDIO, image_a_raw, image_b_raw)
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
