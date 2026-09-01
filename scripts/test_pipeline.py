#!/usr/bin/env python3
"""
Correctness tests for the pipeline.

The important one is test_overlap_add_is_exact. It swaps in a fake backend that
returns the input unchanged as all four stems. If the chunking and crossfade
logic is right, every output stem must equal the original mixture sample for
sample. Any windowing bug — wrong ramp, wrong hop, off-by-one on the flush —
shows up immediately as a nonzero error at the chunk boundaries.

Run:
    python scripts/test_pipeline.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import List

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio_io import load_audio  # noqa: E402
from src.config import SAMPLE_RATE, STEM_NAMES, RunConfig  # noqa: E402
from src.models.base import SeparatorBackend  # noqa: E402
from src.separator import Separator  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results: List[tuple] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, PASS if ok else FAIL, detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f"  — {detail}" if detail else ""))


class IdentityBackend(SeparatorBackend):
    """Returns the mixture unchanged as every stem. Used to test reconstruction."""

    name = "identity"
    is_pretrained = True

    @property
    def stems(self):
        return list(STEM_NAMES)

    def n_params(self) -> int:
        return 0

    def separate_chunk(self, mixture: torch.Tensor) -> torch.Tensor:
        # [1, 2, N] -> [1, 4, 2, N]
        return mixture.unsqueeze(1).repeat(1, len(STEM_NAMES), 1, 1)


def make_wav(path: str, seconds: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Deterministic broadband signal — worst case for boundary artefacts."""
    n = int(seconds * sr)
    rng = np.random.default_rng(1234)
    audio = (rng.standard_normal((2, n)) * 0.2).astype(np.float32)
    sf.write(path, audio.T, sr, subtype="FLOAT")
    return audio


def test_overlap_add_is_exact() -> None:
    """With an identity model, output must equal input exactly."""
    for seconds, chunk, overlap in [
        (15.0, 6.0, 0.5),   # last chunk needs padding
        (12.0, 6.0, 0.0),   # no crossfade
        (7.3, 2.0, 0.5),    # many chunks, ragged tail
        (1.0, 6.0, 0.5),    # input shorter than one chunk
        (18.0, 6.0, 3.0),   # 50% overlap
    ]:
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.wav")
            original = make_wav(src, seconds)

            cfg = RunConfig(
                chunk_seconds=chunk,
                overlap_seconds=overlap,
                output_dir=tmp,
                output_subtype="FLOAT",  # 16-bit would mask small errors
                save_metrics_json=False,
                threads=1,
            )
            cfg.validate()

            sep = Separator.__new__(Separator)  # bypass __init__ model loading
            sep.cfg = cfg
            from src.metrics import RunMetrics

            sep.metrics = RunMetrics()
            sep.backend = IdentityBackend()

            sep.separate_file(src, output_dir=tmp, verbose=False)

            out, _ = load_audio(os.path.join(tmp, "in_vocals.wav"))

            len_ok = out.shape[1] == original.shape[1]
            if not len_ok:
                check(
                    f"overlap-add {seconds}s/{chunk}s/{overlap}s length",
                    False,
                    f"got {out.shape[1]} samples, expected {original.shape[1]}",
                )
                continue

            err = float(np.max(np.abs(out - original)))
            check(
                f"overlap-add exact  dur={seconds}s chunk={chunk}s ov={overlap}s",
                err < 1e-5,
                f"max abs error {err:.2e}",
            )


def test_stereo_and_mono_input() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mono = os.path.join(tmp, "mono.wav")
        sf.write(mono, np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
        audio, sr = load_audio(mono)
        check("mono input upmixed to stereo", audio.shape[0] == 2, str(audio.shape))
        check("sample rate preserved", sr == SAMPLE_RATE, str(sr))


def test_resampling() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "22k.wav")
        sf.write(p, np.zeros((22050, 2), dtype=np.float32), 22050)
        audio, sr = load_audio(p, target_sr=SAMPLE_RATE)
        check("resampled 22.05k -> 44.1k", sr == SAMPLE_RATE, str(sr))
        check(
            "resampled length ~1 second",
            abs(audio.shape[1] - SAMPLE_RATE) <= 2,
            f"{audio.shape[1]} samples",
        )


def test_stem_subset() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.wav")
        make_wav(src, 4.0)
        cfg = RunConfig(
            chunk_seconds=2.0,
            overlap_seconds=0.25,
            output_dir=tmp,
            save_metrics_json=False,
            stems=["vocals", "drums"],
            threads=1,
        )
        sep = Separator.__new__(Separator)
        sep.cfg = cfg
        from src.metrics import RunMetrics

        sep.metrics = RunMetrics()
        sep.backend = IdentityBackend()
        sep.separate_file(src, output_dir=tmp, verbose=False)

        written = sorted(f for f in os.listdir(tmp) if f.startswith("in_"))
        check(
            "stem subset writes only requested stems",
            written == ["in_drums.wav", "in_vocals.wav"],
            str(written),
        )


def test_config_validation() -> None:
    bad = RunConfig(chunk_seconds=1.0, overlap_seconds=2.0)
    try:
        bad.validate()
        check("rejects overlap >= chunk", False, "no exception raised")
    except ValueError:
        check("rejects overlap >= chunk", True)

    bad2 = RunConfig(backend="nonexistent")
    try:
        bad2.validate()
        check("rejects unknown backend", False, "no exception raised")
    except ValueError:
        check("rejects unknown backend", True)


def test_moises_light_shapes() -> None:
    """The real model must accept a chunk and return [1, 4, 2, N]."""
    try:
        from src.models.moises_light_backend import MoisesLightBackend
    except ImportError as exc:
        check("moises-light importable", False, str(exc))
        return

    backend = MoisesLightBackend(preset="paper_small")
    n = int(2.0 * SAMPLE_RATE)
    x = torch.zeros(1, 2, n)
    y = backend.separate_chunk(x)
    check(
        "moises_light output shape",
        tuple(y.shape) == (1, 4, 2, n),
        str(tuple(y.shape)),
    )
    check("moises_light param count", backend.n_params() == 2_558_768,
          f"{backend.n_params():,}")


def main() -> int:
    print("Running pipeline tests\n" + "-" * 55)
    test_config_validation()
    test_stereo_and_mono_input()
    test_resampling()
    test_overlap_add_is_exact()
    test_stem_subset()
    test_moises_light_shapes()

    failed = [r for r in results if r[1] == FAIL]
    print("-" * 55)
    print(f"{len(results) - len(failed)}/{len(results)} passed")
    if failed:
        print("\nFailures:")
        for name, _, detail in failed:
            print(f"  - {name}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
