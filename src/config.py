"""
Central configuration.

Every tunable that affects embedded (Raspberry Pi) viability lives here so that
the laptop run and the Pi run differ only by a config profile, never by code.
"""

from dataclasses import dataclass, field, asdict
from typing import List
import os


# The four stems the model separates. Order is fixed by the model architecture.
STEM_NAMES: List[str] = ["vocals", "drums", "bass", "other"]

# Moises-Light operates on 44.1 kHz stereo. Changing this breaks the STFT config.
SAMPLE_RATE: int = 44100
CHANNELS: int = 2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "input")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output")
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


@dataclass
class RunConfig:
    """
    Runtime configuration for one separation or benchmark run.

    chunk_seconds / overlap_seconds are the two knobs that control the
    memory-vs-quality tradeoff on constrained hardware. Peak RAM scales with
    chunk_seconds, NOT with the length of the input song, because the separator
    streams chunks to disk instead of holding the whole song in memory.
    """

    # --- model selection ---
    backend: str = "moises_light"     # "moises_light" | "openunmix"
    preset: str = "paper_small"       # moises_light preset name
    checkpoint: str = ""              # path to .pth weights; empty = untrained

    # --- chunked inference ---
    chunk_seconds: float = 6.0        # audio seconds fed to the model at once
    overlap_seconds: float = 0.5      # crossfade region between chunks

    # --- compute ---
    threads: int = 0                  # 0 = auto (use all cores)
    device: str = "cpu"               # Pi has no CUDA; cpu is the only option

    # --- io ---
    output_dir: str = DEFAULT_OUTPUT_DIR
    output_format: str = "WAV"
    output_subtype: str = "PCM_16"    # 16-bit keeps stem files small on SD card

    # --- reporting ---
    stems: List[str] = field(default_factory=lambda: list(STEM_NAMES))
    save_metrics_json: bool = True

    def validate(self) -> None:
        if self.backend not in ("moises_light", "openunmix"):
            raise ValueError(f"unknown backend: {self.backend}")
        if self.overlap_seconds >= self.chunk_seconds:
            raise ValueError("overlap_seconds must be smaller than chunk_seconds")
        if self.chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive")
        for s in self.stems:
            if s not in STEM_NAMES:
                raise ValueError(f"unknown stem '{s}', expected one of {STEM_NAMES}")

    def as_dict(self) -> dict:
        return asdict(self)


# Preset profiles. "pi" trades chunk size for lower peak RAM.
PROFILES = {
    "laptop": RunConfig(preset="paper_small", chunk_seconds=6.0, threads=0),
    "pi5":    RunConfig(preset="paper_small", chunk_seconds=6.0, threads=4),
    "pi4":    RunConfig(preset="paper_small", chunk_seconds=4.0, threads=4),
    "pi_2gb": RunConfig(preset="paper_small", chunk_seconds=3.0, threads=4),
}
