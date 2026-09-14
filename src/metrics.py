"""
Performance measurement.

This module is the actual deliverable of an embedded-systems project. The
separation itself is published work; what is being evaluated is whether it fits
inside a Raspberry Pi's compute and memory budget.

Definitions used throughout:

  RTF (real-time factor) = processing_time / audio_duration
      RTF < 1.0  -> faster than real time; real-time streaming is possible
      RTF = 2.0  -> a 3-minute song takes 6 minutes to process

  Peak RSS = the highest resident set size the process reached, sampled from
      a background thread. RSS is what actually has to fit in the Pi's physical
      RAM. Sampling (rather than reading once at the end) matters because
      Python frees the peak allocation before the run ends.
"""

from __future__ import annotations

import json
import os
import platform
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def _bytes_to_mb(n: float) -> float:
    return round(n / (1024 * 1024), 1)


def system_info() -> Dict[str, object]:
    """Describe the machine, so laptop and Pi runs are comparable in the report."""
    info: Dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }

    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
    except ImportError:
        pass

    if psutil is not None:
        vm = psutil.virtual_memory()
        info["total_ram_mb"] = _bytes_to_mb(vm.total)
        info["available_ram_mb"] = _bytes_to_mb(vm.available)

    # Raspberry Pi exposes its board name here; absent on a laptop.
    try:
        with open("/proc/device-tree/model", "rb") as fh:
            info["board"] = fh.read().decode("utf-8", "ignore").strip("\x00").strip()
    except OSError:
        pass

    return info


class MemorySampler:
    """
    Samples process RSS on a background thread to capture the true peak.

    Used as a context manager:
        with MemorySampler() as ms:
            ...work...
        print(ms.peak_mb)
    """

    def __init__(self, interval: float = 0.05) -> None:
        self.interval = interval
        self.peak_bytes = 0
        self.baseline_bytes = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc = psutil.Process(os.getpid()) if psutil is not None else None

    def _rss(self) -> int:
        if self._proc is None:
            return 0
        try:
            return self._proc.memory_info().rss
        except Exception:  # noqa: BLE001
            return 0

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self._rss())
            self._stop.wait(self.interval)

    def __enter__(self) -> "MemorySampler":
        self.baseline_bytes = self._rss()
        self.peak_bytes = self.baseline_bytes
        if self._proc is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.peak_bytes = max(self.peak_bytes, self._rss())

    @property
    def peak_mb(self) -> float:
        return _bytes_to_mb(self.peak_bytes)

    @property
    def baseline_mb(self) -> float:
        return _bytes_to_mb(self.baseline_bytes)

    @property
    def delta_mb(self) -> float:
        return _bytes_to_mb(max(0, self.peak_bytes - self.baseline_bytes))


@dataclass
class RunMetrics:
    """Everything measured during one separation or benchmark run."""

    backend: str = ""
    preset: str = ""
    is_pretrained: bool = False

    n_params: int = 0
    model_size_mb: float = 0.0

    audio_seconds: float = 0.0
    n_chunks: int = 0
    chunk_seconds: float = 0.0
    overlap_seconds: float = 0.0
    threads: int = 0

    model_load_seconds: float = 0.0
    inference_seconds: float = 0.0      # model forward passes only
    network_seconds: float = 0.0        # time spent waiting on a remote backend
    total_seconds: float = 0.0          # includes I/O and overlap-add

    chunk_times: List[float] = field(default_factory=list)

    peak_ram_mb: float = 0.0
    baseline_ram_mb: float = 0.0
    model_ram_mb: float = 0.0

    cpu_percent: float = 0.0
    system: Dict[str, object] = field(default_factory=dict)

    # --- derived ---

    @property
    def rtf(self) -> float:
        """Real-time factor over the whole pipeline."""
        if self.audio_seconds <= 0:
            return 0.0
        return round(self.total_seconds / self.audio_seconds, 3)

    @property
    def inference_rtf(self) -> float:
        """Real-time factor of the model alone, excluding file I/O."""
        if self.audio_seconds <= 0:
            return 0.0
        return round(self.inference_seconds / self.audio_seconds, 3)

    @property
    def mean_chunk_ms(self) -> float:
        if not self.chunk_times:
            return 0.0
        return round(1000 * sum(self.chunk_times) / len(self.chunk_times), 1)

    @property
    def max_chunk_ms(self) -> float:
        if not self.chunk_times:
            return 0.0
        return round(1000 * max(self.chunk_times), 1)

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["rtf"] = self.rtf
        d["inference_rtf"] = self.inference_rtf
        d["mean_chunk_ms"] = self.mean_chunk_ms
        d["max_chunk_ms"] = self.max_chunk_ms
        # Per-chunk timings are noisy detail; keep them out of the summary file.
        d.pop("chunk_times", None)
        return d

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)

    def report(self) -> str:
        """Human-readable summary; this is what you screenshot for the review."""
        sysinfo = self.system
        board = sysinfo.get("board") or sysinfo.get("processor", "unknown")
        ram = sysinfo.get("total_ram_mb", "?")

        lines = [
            "",
            "=" * 62,
            " PERFORMANCE REPORT",
            "=" * 62,
            f" Device            : {board}",
            f" CPU cores / RAM   : {sysinfo.get('cpu_count','?')} cores / {ram} MB",
            f" Torch threads     : {self.threads}",
            "-" * 62,
            f" Backend           : {self.backend}"
            + (f" ({self.preset})" if self.preset else ""),
            f" Pretrained weights: {'yes' if self.is_pretrained else 'NO - audio is not meaningful'}",
            f" Parameters        : {self.n_params/1e6:.2f} M",
            f" Model size (fp32) : {self.model_size_mb:.1f} MB",
            "-" * 62,
            f" Audio duration    : {self.audio_seconds:.1f} s",
            f" Chunk / overlap   : {self.chunk_seconds:.1f} s / {self.overlap_seconds:.1f} s",
            f" Chunks processed  : {self.n_chunks}",
            "-" * 62,
            f" Model load time   : {self.model_load_seconds:.2f} s",
            f" Inference time    : {self.inference_seconds:.2f} s",
        ]
        if self.network_seconds > 0:
            lines.append(
                f" Network time      : {self.network_seconds:.2f} s  "
                f"(round-trip to the remote host, included in inference time)"
            )
        lines += [
            f" Total wall time   : {self.total_seconds:.2f} s",
            f" Mean chunk time   : {self.mean_chunk_ms:.0f} ms",
            f" Max  chunk time   : {self.max_chunk_ms:.0f} ms",
            "-" * 62,
            f" Real-time factor  : {self.rtf:.2f}x  "
            f"({'faster' if self.rtf < 1 else 'slower'} than real time)",
            f" Inference-only RTF: {self.inference_rtf:.2f}x",
            "-" * 62,
            f" Baseline RAM      : {self.baseline_ram_mb:.1f} MB",
            f" Model RAM         : {self.model_ram_mb:.1f} MB",
            f" Peak RAM (RSS)    : {self.peak_ram_mb:.1f} MB",
            f" Mean CPU usage    : {self.cpu_percent:.0f} %",
            "=" * 62,
        ]
        return "\n".join(lines)


def model_size_mb(n_params: int, bytes_per_param: int = 4) -> float:
    """fp32 by default. Pass 1 to estimate an int8-quantised size."""
    return round(n_params * bytes_per_param / (1024 * 1024), 2)
