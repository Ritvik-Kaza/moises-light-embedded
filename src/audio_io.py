"""
Audio input and output.

Design note for embedded use: the writer is *streaming*. Stems are appended to
disk chunk by chunk, so the process never holds four full-length stereo stems in
RAM. For a 4-minute song that is the difference between ~340 MB of buffers and
almost nothing.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

from .config import SAMPLE_RATE, CHANNELS


class AudioLoadError(RuntimeError):
    pass


def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """
    Load an audio file as float32 stereo at target_sr.

    Returns (audio, sample_rate) where audio has shape [channels, samples].

    soundfile handles WAV/FLAC/OGG natively. MP3 support depends on the
    installed libsndfile version (>= 1.1.0). If MP3 fails, convert with ffmpeg:
        ffmpeg -i song.mp3 -ar 44100 -ac 2 song.wav
    """
    if not os.path.isfile(path):
        raise AudioLoadError(f"input file not found: {path}")

    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001 - surface a clear message to the CLI
        raise AudioLoadError(
            f"could not decode '{path}': {exc}\n"
            "If this is an MP3, convert it first:\n"
            f"  ffmpeg -i '{path}' -ar 44100 -ac 2 converted.wav"
        ) from exc

    # soundfile returns [samples, channels]; the model wants [channels, samples].
    audio = audio.T

    audio = _to_stereo(audio)

    if sr != target_sr:
        audio = _resample_linear(audio, sr, target_sr)
        sr = target_sr

    return np.ascontiguousarray(audio, dtype=np.float32), sr


def _to_stereo(audio: np.ndarray) -> np.ndarray:
    """Force [channels, samples] to exactly 2 channels."""
    n_ch = audio.shape[0]
    if n_ch == CHANNELS:
        return audio
    if n_ch == 1:
        return np.repeat(audio, CHANNELS, axis=0)
    # More than 2 channels: downmix extras into stereo by averaging halves.
    left = audio[0::2].mean(axis=0)
    right = audio[1::2].mean(axis=0)
    return np.stack([left, right], axis=0)


def _resample_linear(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """
    Linear-interpolation resampler.

    Deliberately dependency-free: scipy and librosa are heavy installs on a
    Raspberry Pi and we only need resampling as a fallback for oddly-sampled
    input. Anything you actually benchmark should already be 44.1 kHz, so this
    path should not run during a normal measurement.
    """
    n_in = audio.shape[1]
    n_out = int(round(n_in * sr_out / sr_in))
    if n_out <= 0:
        raise AudioLoadError("resampling produced an empty signal")
    src_idx = np.linspace(0.0, n_in - 1.0, num=n_out, dtype=np.float64)
    out = np.empty((audio.shape[0], n_out), dtype=np.float32)
    grid = np.arange(n_in, dtype=np.float64)
    for c in range(audio.shape[0]):
        out[c] = np.interp(src_idx, grid, audio[c].astype(np.float64))
    return out


class StemWriter:
    """
    Opens one output file per stem and appends audio incrementally.

    Use as a context manager so files are always closed, even on error:
        with StemWriter(outdir, "song", ["vocals","drums"]) as w:
            w.append({"vocals": arr, "drums": arr})
    """

    def __init__(
        self,
        output_dir: str,
        basename: str,
        stems: List[str],
        sample_rate: int = SAMPLE_RATE,
        fmt: str = "WAV",
        subtype: str = "PCM_16",
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.paths: Dict[str, str] = {}
        self._files: Dict[str, sf.SoundFile] = {}
        self._stems = list(stems)

        for stem in self._stems:
            path = os.path.join(output_dir, f"{basename}_{stem}.wav")
            self.paths[stem] = path
            self._files[stem] = sf.SoundFile(
                path,
                mode="w",
                samplerate=sample_rate,
                channels=CHANNELS,
                format=fmt,
                subtype=subtype,
            )

    def append(self, block: Dict[str, np.ndarray]) -> None:
        """block maps stem name -> [channels, samples] float32."""
        for stem, data in block.items():
            if stem not in self._files:
                continue
            # soundfile writes [samples, channels]; clip to avoid wrap-around
            # distortion when the model outputs values slightly outside [-1, 1].
            self._files[stem].write(np.clip(data.T, -1.0, 1.0))

    def close(self) -> None:
        for f in self._files.values():
            try:
                f.close()
            except Exception:  # noqa: BLE001 - best effort on cleanup
                pass
        self._files.clear()

    def __enter__(self) -> "StemWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
