"""
The separation pipeline.

Why chunking exists
-------------------
Feeding a whole 4-minute song to the model in one forward pass would allocate
intermediate STFT tensors of several gigabytes. That is fine on a workstation
and fatal on a Raspberry Pi. So the pipeline slices the mixture into fixed-length
chunks, runs the model on one chunk at a time, and crossfades the results back
together with overlap-add.

The consequence, and the reason this design is the right one for the project:
peak memory is a function of `chunk_seconds`, not of song length. A 30-second
clip and a 10-minute track have essentially the same memory profile. Song length
only costs time.

Overlap-add correctness
-----------------------
Chunk boundaries produce audible clicks if you just butt the outputs together,
because the model's output is not continuous across independent forward passes.
Each chunk therefore gets a linear fade-in over its first `overlap` samples
(except the first chunk) and a linear fade-out over its last `overlap` samples
(except the last chunk). The two ramps are complements, so in any overlap region
the two contributing chunks sum to exactly unity gain — no dip, no boost.

Finalised samples are written to disk as soon as they can no longer be modified
by a future chunk, which keeps output buffers bounded too.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional

import numpy as np
import torch

from .audio_io import StemWriter, load_audio
from .config import SAMPLE_RATE, RunConfig
from .metrics import (
    MemorySampler,
    RunMetrics,
    model_size_mb,
    system_info,
)
from .models import build_backend

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def configure_threads(n_threads: int) -> int:
    """
    Set the PyTorch CPU thread count and return what was actually applied.

    On a Raspberry Pi, pinning this to the core count (4) is worth doing
    explicitly: the default heuristic sometimes oversubscribes, and thread
    thrashing on a small ARM core costs more than the parallelism gains.
    """
    if n_threads and n_threads > 0:
        torch.set_num_threads(n_threads)
    return torch.get_num_threads()


def _make_ramps(overlap: int) -> tuple[np.ndarray, np.ndarray]:
    """Complementary linear fade-in / fade-out ramps that sum to 1.0."""
    if overlap <= 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty
    fade_in = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
    fade_out = 1.0 - fade_in
    return fade_in, fade_out


class Separator:
    """Runs a backend over an audio file and writes stems + metrics."""

    def __init__(self, cfg: RunConfig) -> None:
        cfg.validate()
        self.cfg = cfg
        self.metrics = RunMetrics()

        applied_threads = configure_threads(cfg.threads)
        self.metrics.threads = applied_threads

        t0 = time.perf_counter()
        mem_before = self._rss_mb()
        self.backend = build_backend(cfg)
        self.metrics.model_load_seconds = round(time.perf_counter() - t0, 3)
        self.metrics.model_ram_mb = round(max(0.0, self._rss_mb() - mem_before), 1)

        self.metrics.backend = self.backend.name
        self.metrics.preset = cfg.preset if self.backend.name == "moises_light" else ""
        self.metrics.is_pretrained = self.backend.is_pretrained
        self.metrics.n_params = self.backend.n_params()
        self.metrics.model_size_mb = model_size_mb(self.metrics.n_params)
        self.metrics.system = system_info()

    @staticmethod
    def _rss_mb() -> float:
        if psutil is None:
            return 0.0
        try:
            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except Exception:  # noqa: BLE001
            return 0.0

    def separate_file(
        self,
        input_path: str,
        output_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> RunMetrics:
        cfg = self.cfg
        out_dir = output_dir or cfg.output_dir
        basename = os.path.splitext(os.path.basename(input_path))[0]

        if verbose:
            print(f"[load] {input_path}")
        mixture, sr = load_audio(input_path, target_sr=SAMPLE_RATE)
        n_samples = mixture.shape[1]
        duration = n_samples / sr

        chunk_n = int(cfg.chunk_seconds * sr)
        overlap_n = int(cfg.overlap_seconds * sr)
        hop_n = chunk_n - overlap_n
        if hop_n <= 0:
            raise ValueError("chunk_seconds must exceed overlap_seconds")

        n_chunks = max(1, int(np.ceil(n_samples / hop_n)))

        self.metrics.audio_seconds = round(duration, 2)
        self.metrics.chunk_seconds = cfg.chunk_seconds
        self.metrics.overlap_seconds = cfg.overlap_seconds
        self.metrics.n_chunks = n_chunks

        if self.backend.is_query_driven:
            # The backend's single stem is named after the query, not one of
            # STEM_NAMES, so cfg.stems (a filter over the fixed vocabulary)
            # does not apply — there is nothing to filter.
            stems = list(self.backend.stems)
        else:
            stems = [s for s in self.backend.stems if s in cfg.stems]
        stem_index = {s: self.backend.stems.index(s) for s in stems}

        fade_in, fade_out = _make_ramps(overlap_n)

        if verbose:
            print(
                f"[plan] {duration:.1f}s audio -> {n_chunks} chunks "
                f"of {cfg.chunk_seconds}s (overlap {cfg.overlap_seconds}s)"
            )
            if not self.backend.is_pretrained:
                print(
                    "[warn] backend has NO trained weights: timing and memory "
                    "numbers are valid, output audio is noise."
                )

        # Rolling overlap-add accumulator: [n_stems, channels, chunk_n]
        acc = np.zeros((len(stems), mixture.shape[0], chunk_n), dtype=np.float32)

        proc = psutil.Process(os.getpid()) if psutil is not None else None
        if proc is not None:
            proc.cpu_percent(None)  # prime the CPU-percent counter

        remaining = n_samples
        inference_time = 0.0
        t_start = time.perf_counter()

        with MemorySampler() as sampler, StemWriter(
            out_dir,
            basename,
            stems,
            sample_rate=sr,
            fmt=cfg.output_format,
            subtype=cfg.output_subtype,
        ) as writer:

            for i in range(n_chunks):
                start = i * hop_n
                stop = start + chunk_n
                is_last = i == n_chunks - 1

                block = mixture[:, start:stop]
                if block.shape[1] < chunk_n:  # zero-pad the final short chunk
                    pad = chunk_n - block.shape[1]
                    block = np.pad(block, ((0, 0), (0, pad)), mode="constant")

                x = torch.from_numpy(np.ascontiguousarray(block)).unsqueeze(0)

                t_chunk = time.perf_counter()
                y = self.backend.separate_chunk(x)
                dt = time.perf_counter() - t_chunk

                inference_time += dt
                self.metrics.chunk_times.append(dt)
                net = getattr(self.backend, "last_network_seconds", None)
                if net is not None:
                    self.metrics.network_seconds += net

                out = y[0].detach().cpu().numpy().astype(np.float32, copy=False)
                out = out[[stem_index[s] for s in stems]]

                if overlap_n > 0:
                    if i > 0:
                        out[:, :, :overlap_n] *= fade_in
                    if not is_last:
                        out[:, :, -overlap_n:] *= fade_out

                acc += out

                # Samples in [start, start+hop) can no longer change: flush them.
                take = min(hop_n, remaining)
                if take > 0:
                    writer.append({s: acc[k, :, :take] for k, s in enumerate(stems)})
                    remaining -= take

                # Slide the window in place: keep the overlap tail, zero the rest.
                # Done in place rather than with np.concatenate so the loop makes
                # no per-chunk allocations, which matters on a memory-tight board.
                if overlap_n > 0:
                    acc[:, :, :overlap_n] = acc[:, :, hop_n:]
                acc[:, :, overlap_n:] = 0.0

                if verbose:
                    pct = 100.0 * (i + 1) / n_chunks
                    print(
                        f"\r[run ] chunk {i+1}/{n_chunks} "
                        f"({pct:5.1f}%)  {dt*1000:6.0f} ms",
                        end="",
                        flush=True,
                    )

            # Flush the trailing overlap region left in the accumulator.
            if remaining > 0:
                take = min(remaining, acc.shape[2])
                writer.append({s: acc[k, :, :take] for k, s in enumerate(stems)})
                remaining -= take

            self.metrics.peak_ram_mb = sampler.peak_mb
            self.metrics.baseline_ram_mb = sampler.baseline_mb

        if verbose:
            print()

        self.metrics.total_seconds = round(time.perf_counter() - t_start, 3)
        self.metrics.inference_seconds = round(inference_time, 3)
        if proc is not None:
            self.metrics.cpu_percent = round(proc.cpu_percent(None), 1)

        self.output_paths: Dict[str, str] = writer.paths

        if cfg.save_metrics_json:
            json_path = os.path.join(out_dir, f"{basename}_metrics.json")
            self.metrics.save_json(json_path)
            if verbose:
                print(f"[save] metrics -> {json_path}")

        if verbose:
            for stem, path in writer.paths.items():
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"[save] {stem:7s} -> {path}  ({size_mb:.1f} MB)")

        return self.metrics

    def benchmark(
        self,
        duration_seconds: float = 30.0,
        warmup: int = 1,
        verbose: bool = True,
    ) -> RunMetrics:
        """
        Measure throughput on synthetic audio, with no file I/O in the loop.

        Use this to characterise the hardware itself. It needs no input file and
        no trained weights, which makes it the one measurement you can run
        identically on the laptop and on the Pi from day one.
        """
        cfg = self.cfg
        sr = SAMPLE_RATE
        chunk_n = int(cfg.chunk_seconds * sr)
        n_chunks = max(1, int(np.ceil(duration_seconds / cfg.chunk_seconds)))

        self.metrics.audio_seconds = round(n_chunks * cfg.chunk_seconds, 2)
        self.metrics.chunk_seconds = cfg.chunk_seconds
        self.metrics.overlap_seconds = 0.0
        self.metrics.n_chunks = n_chunks

        rng = np.random.default_rng(0)
        x = torch.from_numpy(
            rng.standard_normal((1, 2, chunk_n)).astype(np.float32) * 0.1
        )

        if verbose:
            print(f"[bench] warmup x{warmup}")
        for _ in range(warmup):
            self.backend.separate_chunk(x)

        proc = psutil.Process(os.getpid()) if psutil is not None else None
        if proc is not None:
            proc.cpu_percent(None)

        inference_time = 0.0
        t_start = time.perf_counter()

        with MemorySampler() as sampler:
            for i in range(n_chunks):
                t_chunk = time.perf_counter()
                self.backend.separate_chunk(x)
                dt = time.perf_counter() - t_chunk
                inference_time += dt
                self.metrics.chunk_times.append(dt)
                net = getattr(self.backend, "last_network_seconds", None)
                if net is not None:
                    self.metrics.network_seconds += net
                if verbose:
                    print(
                        f"\r[bench] chunk {i+1}/{n_chunks}  {dt*1000:6.0f} ms",
                        end="",
                        flush=True,
                    )
            self.metrics.peak_ram_mb = sampler.peak_mb
            self.metrics.baseline_ram_mb = sampler.baseline_mb

        if verbose:
            print()

        self.metrics.total_seconds = round(time.perf_counter() - t_start, 3)
        self.metrics.inference_seconds = round(inference_time, 3)
        if proc is not None:
            self.metrics.cpu_percent = round(proc.cpu_percent(None), 1)

        return self.metrics
