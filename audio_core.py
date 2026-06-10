"""
audio_core — audio-cleaner 的核心处理逻辑。

把 main.py 里所有 ffmpeg / 模型 / 文件路径操作抽出来，方便：
- main.py 只负责 UI（Streamlit）
- tests/test_audio_core.py 直接 import + 单测
- 其他前端（FastAPI / CLI）也能复用

边界：
- 不读/写仓库外的文件
- 不调用 ffmpeg/ffprobe 之外的 system 命令
- 不联网（除了 torch hub 拉模型权重，那是上游 init_df 的事）
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple

logger = logging.getLogger(__name__)

# ---- 安全 / 资源上限 --------------------------------------------------------

# 单文件 200 MB。超过 → 直接拒，避免 OOM（48kHz mono float32 ≈ 4.6 MB/s，
# 200MB = ~43s 音频 + ~800MB RAM（torchaudio load 进内存））。
MAX_INPUT_BYTES = int(os.getenv("AUDIO_CLEANER_MAX_INPUT_BYTES", 200 * 1024 * 1024))

# 单文件 30 分钟音频。再长 → 拆分上传。
MAX_AUDIO_SECONDS = float(os.getenv("AUDIO_CLEANER_MAX_AUDIO_SECONDS", 30 * 60))

# ffmpeg / ffprobe 子进程超时（秒）。恶意/损坏文件不能挂死主进程。
SUBPROCESS_TIMEOUT = int(os.getenv("AUDIO_CLEANER_SUBPROC_TIMEOUT", 120))

# 推理 chunk 大小（秒）。main.py 原来写死 30s。
DEFAULT_CHUNK_SECONDS = 30

# 工作采样率（DeepFilterNet 要求 48kHz）
TARGET_SR = 48000


# ---- 错误类型 ---------------------------------------------------------------

class AudioCleanerError(Exception):
    """audio-cleaner 业务异常的基类。"""


class FFmpegMissingError(AudioCleanerError):
    """ffmpeg 或 ffprobe 不在 PATH。"""


class FFmpegFailedError(AudioCleanerError):
    """ffmpeg/ffprobe 子进程返回非零或超时。"""


class FileTooLargeError(AudioCleanerError):
    """输入文件超过 MAX_INPUT_BYTES。"""


class AudioTooLongError(AudioCleanerError):
    """输入音频时长超过 MAX_AUDIO_SECONDS。"""


# ---- 工具函数 ---------------------------------------------------------------

def check_ffmpeg() -> None:
    """确认 ffmpeg + ffprobe 在 PATH 里。任何一个缺失 → 抛 FFmpegMissingError。"""
    missing = [cmd for cmd in ("ffmpeg", "ffprobe") if shutil.which(cmd) is None]
    if missing:
        raise FFmpegMissingError(
            f"ffmpeg dependency missing: {missing}. "
            "Install ffmpeg (e.g. `brew install ffmpeg` or `apt install ffmpeg`) "
            "and ensure it is on your PATH."
        )


def _run(cmd: list[str], *, timeout: int = SUBPROCESS_TIMEOUT, check: bool = True) -> subprocess.CompletedProcess:
    """subprocess.run 的薄包装：捕获 stderr、超时转 FFmpegFailedError。

    与原 main.py 关键区别：原代码 `stderr=subprocess.DEVNULL` 把所有错误吞了，
    用户只看到白屏。这里捕到 stderr 后包成异常抛出。
    """
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise FFmpegFailedError(
            f"Command {cmd[0]!r} timed out after {timeout}s. "
            "The input file may be corrupt or the audio too long."
        ) from e

    if check and result.returncode != 0:
        raise FFmpegFailedError(
            f"Command {cmd!r} failed with exit {result.returncode}: "
            f"{result.stderr.strip()[:500]}"
        )
    return result


def probe_duration(path: str | Path) -> float:
    """用 ffprobe 读音频时长（秒），失败 fallback 到 librosa。"""
    check_ffmpeg()
    try:
        result = _run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
        )
        return float(result.stdout.strip())
    except (FFmpegFailedError, ValueError) as e:
        logger.warning("ffprobe failed (%s), falling back to librosa", e)
        try:
            import librosa  # 延迟 import 避免冷启动慢
            return float(librosa.get_duration(filename=str(path)))
        except Exception as e2:
            raise FFmpegFailedError(
                f"Both ffprobe and librosa failed to read duration: {e2}"
            ) from e2


def convert_to_work_wav(
    input_path: str | Path,
    work_wav_path: str | Path,
) -> None:
    """用 ffmpeg 把任意输入转成 48kHz mono float32 WAV。失败抛 FFmpegFailedError。"""
    check_ffmpeg()
    _run(
        [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-ar", str(TARGET_SR),
            "-ac", "1",
            str(work_wav_path),
        ],
        check=True,
    )


def make_temp_paths(orig_filename: str) -> "TempPaths":
    """为一次处理生成唯一的临时文件路径。

    用 uuid4 替代原 `int(time.time())`，避免同秒并发上传冲突。
    """
    file_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    temp_dir = Path(tempfile.gettempdir())
    safe_name = Path(orig_filename).name  # 去掉任何路径成分
    return TempPaths(
        file_id=file_id,
        input_path=temp_dir / f"in_{file_id}_{safe_name}",
        work_wav_path=temp_dir / f"work_{file_id}.wav",
        out_wav_path=temp_dir / f"out_{file_id}.wav",
    )


@dataclass(frozen=True)
class TempPaths:
    """一次处理的临时文件路径集合。"""
    file_id: str
    input_path: Path
    work_wav_path: Path
    out_wav_path: Path

    def cleanup(self) -> None:
        """清理本次处理产生的所有临时文件。"""
        for p in (self.input_path, self.work_wav_path, self.out_wav_path):
            try:
                if p.exists():
                    p.unlink()
            except OSError as e:
                logger.warning("Failed to remove temp file %s: %s", p, e)


@contextmanager
def managed_temp_paths(orig_filename: str) -> Generator[TempPaths, None, None]:
    """上下文管理器：无论中途是否异常，都会清理临时文件。

    用法：
        with managed_temp_paths("foo.wav") as paths:
            paths.input_path.write_bytes(data)
            ...
    """
    paths = make_temp_paths(orig_filename)
    try:
        yield paths
    finally:
        paths.cleanup()


def validate_size(num_bytes: int) -> None:
    """检查上传文件大小。"""
    if num_bytes > MAX_INPUT_BYTES:
        mb = num_bytes / 1024 / 1024
        limit_mb = MAX_INPUT_BYTES / 1024 / 1024
        raise FileTooLargeError(
            f"Input file is {mb:.1f} MB, exceeds limit {limit_mb:.0f} MB. "
            "Please split the file or increase AUDIO_CLEANER_MAX_INPUT_BYTES."
        )


def validate_duration(seconds: float) -> None:
    """检查音频时长。"""
    if seconds > MAX_AUDIO_SECONDS:
        raise AudioTooLongError(
            f"Input audio is {seconds:.0f}s, exceeds limit {MAX_AUDIO_SECONDS:.0f}s. "
            "Please split the file or increase AUDIO_CLEANER_MAX_AUDIO_SECONDS."
        )


def reattach_audio_to_video(
    input_video: str | Path,
    audio_wav: str | Path,
    output_video: str | Path,
) -> None:
    """把处理后的 wav 重新封装回原视频容器（MOV → MP4）。"""
    check_ffmpeg()
    _run(
        [
            "ffmpeg",
            "-y",
            "-i", str(input_video),
            "-i", str(audio_wav),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            str(output_video),
        ],
        check=True,
    )


# ---- 兼容性 shim：保留旧 main.py 用的 _check_ffmpeg 名字 -------------------

def _check_ffmpeg() -> None:
    """兼容旧 main.py 的命名。"""
    check_ffmpeg()
