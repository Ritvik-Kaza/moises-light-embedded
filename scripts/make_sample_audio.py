#!/usr/bin/env python3
"""
Generate a synthetic multi-instrument mixture for pipeline testing.

This is a test fixture, not a music sample. It produces a bass line, a drum
pattern, a chord pad and a vocal-like tone, mixed into one stereo file, plus the
four ground-truth stems. Use it to verify that the pipeline runs end to end and
that overlap-add reconstruction is clean, without needing to source a real song.

Usage:
    python scripts/make_sample_audio.py
    python scripts/make_sample_audio.py --duration 20 --out data/input/test.wav
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SR = 44100


def _env(n: int, attack: float, decay: float) -> np.ndarray:
    """Simple attack/decay amplitude envelope."""
    a = max(1, int(attack * SR))
    d = max(1, int(decay * SR))
    env = np.zeros(n, dtype=np.float32)
    a = min(a, n)
    env[:a] = np.linspace(0, 1, a, dtype=np.float32)
    if n > a:
        tail = np.exp(-np.arange(n - a, dtype=np.float32) / d)
        env[a:] = tail
    return env


def make_bass(duration: float) -> np.ndarray:
    n = int(duration * SR)
    out = np.zeros(n, dtype=np.float32)
    notes = [55.0, 55.0, 73.42, 65.41]  # A1 A1 D2 C2
    step = int(0.5 * SR)
    for i, start in enumerate(range(0, n - step, step)):
        f = notes[i % len(notes)]
        t = np.arange(step, dtype=np.float32) / SR
        tone = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)
        out[start : start + step] += tone * _env(step, 0.005, 0.12) * 0.5
    return out


def make_drums(duration: float) -> np.ndarray:
    n = int(duration * SR)
    out = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(42)
    beat = int(0.25 * SR)

    for i, start in enumerate(range(0, n - beat, beat)):
        seg = np.arange(beat, dtype=np.float32) / SR
        if i % 4 == 0:  # kick
            f = 90 * np.exp(-seg * 30)
            hit = np.sin(2 * np.pi * f * seg) * _env(beat, 0.001, 0.03)
            out[start : start + beat] += hit * 0.7
        elif i % 4 == 2:  # snare
            noise = rng.standard_normal(beat).astype(np.float32)
            hit = noise * _env(beat, 0.001, 0.02)
            out[start : start + beat] += hit * 0.35
        # hi-hat on every beat
        hh = rng.standard_normal(beat).astype(np.float32) * _env(beat, 0.001, 0.005)
        out[start : start + beat] += hh * 0.12
    return out


def make_other(duration: float) -> np.ndarray:
    """A sustained chord pad standing in for 'other instruments'."""
    n = int(duration * SR)
    t = np.arange(n, dtype=np.float32) / SR
    out = np.zeros(n, dtype=np.float32)
    for f in (220.0, 277.18, 329.63):  # A major triad
        out += np.sin(2 * np.pi * f * t) * 0.12
    # slow tremolo so it is not perfectly stationary
    out *= 0.8 + 0.2 * np.sin(2 * np.pi * 0.7 * t)
    return out.astype(np.float32)


def make_vocals(duration: float) -> np.ndarray:
    """A vibrato tone with harmonics, standing in for a voice."""
    n = int(duration * SR)
    t = np.arange(n, dtype=np.float32) / SR
    f0 = 440.0 * (1.0 + 0.02 * np.sin(2 * np.pi * 5.0 * t))  # vibrato
    phase = 2 * np.pi * np.cumsum(f0) / SR
    out = np.sin(phase) * 0.4
    out += np.sin(2 * phase) * 0.15
    out += np.sin(3 * phase) * 0.07
    # phrase the tone on and off so it is not a continuous drone
    gate = (np.sin(2 * np.pi * 0.35 * t) > -0.2).astype(np.float32)
    kernel = np.ones(2048, dtype=np.float32) / 2048
    gate = np.convolve(gate, kernel, mode="same")
    return (out * gate).astype(np.float32)


def to_stereo(mono: np.ndarray, pan: float = 0.0) -> np.ndarray:
    """pan: -1 hard left, 0 centre, +1 hard right. Returns [2, n]."""
    left = mono * np.sqrt((1.0 - pan) / 2.0)
    right = mono * np.sqrt((1.0 + pan) / 2.0)
    return np.stack([left, right], axis=0).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=float, default=15.0, help="seconds")
    ap.add_argument("--out", default="data/input/test_mixture.wav")
    ap.add_argument(
        "--stems", action="store_true", help="also write the ground-truth stems"
    )
    args = ap.parse_args()

    d = args.duration
    parts = {
        "vocals": to_stereo(make_vocals(d), pan=0.0),
        "drums": to_stereo(make_drums(d), pan=0.0),
        "bass": to_stereo(make_bass(d), pan=0.0),
        "other": to_stereo(make_other(d), pan=0.2),
    }

    mix = sum(parts.values())
    peak = float(np.max(np.abs(mix)))
    if peak > 0:
        mix = mix / peak * 0.89  # leave headroom, avoid clipping

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    sf.write(args.out, mix.T, SR, subtype="PCM_16")
    print(f"wrote {args.out}  ({d:.1f}s, {SR} Hz, stereo)")

    if args.stems:
        base = os.path.splitext(args.out)[0]
        for name, data in parts.items():
            path = f"{base}_ref_{name}.wav"
            sf.write(path, (data / peak * 0.89).T, SR, subtype="PCM_16")
            print(f"wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
