#!/usr/bin/env python3
"""
Offload server: wraps AudioSep behind the wire protocol RemoteBackend speaks.

This script does NOT run inside this project's own virtualenv. AudioSep
(https://github.com/Audio-AGI/AudioSep) is not pip-installable — it is a git
clone with its own conda environment, its own checkpoint download, and its
own top-level `models`/`utils`/`pipeline` modules. Trying to import it into
this project's venv would fight over dependency versions and collide with
this project's own `src/models` package name. So instead this script:

  1. Runs standalone, inside AudioSep's own conda environment (see
     README.md's "Offload setup" section for how to set that up).
  2. Is pointed at your AudioSep checkout via --audiosep-repo (or the
     AUDIOSEP_REPO environment variable) and adds it to sys.path at
     startup, so it can `from pipeline import build_audiosep, separate_audio`
     without you having to copy this file into that checkout.
  3. Speaks the exact wire protocol RemoteBackend
     (src/models/remote_backend.py) expects: POST /separate with raw
     float32 PCM in, raw float32 PCM (always 2 channels) out.

Run it with:
    conda activate AudioSep
    pip install flask                 # only needed inside this env
    python serve_audiosep.py \
        --audiosep-repo ~/AudioSep \
        --checkpoint ~/AudioSep/checkpoint/audiosep_base_4M_steps.ckpt \
        --device cpu                  # or cuda, if your laptop has a GPU

SECURITY NOTE: this server has no authentication and accepts arbitrary text
queries and audio from anyone who can reach it. Only bind it to a trusted
LAN (the default host below), never expose it on the public internet.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

AUDIOSEP_SAMPLE_RATE = 32000  # fixed by the AudioSep model itself


def _load_audiosep(repo_path: str, config_yaml: str, checkpoint: str, device: str):
    """Import and build the AudioSep model from an external checkout."""
    repo_path = os.path.abspath(os.path.expanduser(repo_path))
    if not os.path.isdir(repo_path):
        raise SystemExit(f"--audiosep-repo not found: {repo_path}")
    sys.path.insert(0, repo_path)
    try:
        from pipeline import build_audiosep  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            f"could not import AudioSep's pipeline.py from {repo_path}.\n"
            "Make sure --audiosep-repo points at a full AudioSep checkout "
            "and that this script is running inside AudioSep's own conda "
            f"environment (its deps, not this project's venv).\nUnderlying error: {exc}"
        ) from exc

    if not os.path.isfile(checkpoint):
        raise SystemExit(f"--checkpoint not found: {checkpoint}")

    print(f"[serve_audiosep] loading model on {device} ...")
    model = build_audiosep(
        config_yaml=os.path.join(repo_path, config_yaml)
        if not os.path.isabs(config_yaml)
        else config_yaml,
        checkpoint_path=checkpoint,
        device=device,
    )
    print("[serve_audiosep] model ready")
    return model


def _separate(model, device: str, pcm: np.ndarray, channels: int, sr: int, query: str) -> np.ndarray:
    """
    pcm: float32, shape [channels, samples], as received over the wire.
    Returns float32, shape [2, samples] (always stereo, mono duplicated),
    resampled back to `sr` to match the caller's chunk length exactly.
    """
    from pipeline import separate_audio  # type: ignore

    # Downmix to mono for the temp file AudioSep reads — it resamples and
    # mixes down internally via librosa.load(sr=32000, mono=True) anyway,
    # but writing mono here keeps the round-trip file small.
    mono = pcm.mean(axis=0) if channels > 1 else pcm[0]

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "chunk.wav")
        out_path = os.path.join(tmp, "sep.wav")
        sf.write(in_path, mono, sr, subtype="FLOAT")

        separate_audio(
            model, in_path, query, out_path, device=device, use_chunk=True
        )

        sep, sep_sr = sf.read(out_path, dtype="float32", always_2d=False)

    if sep_sr != sr:
        # Linear resample back to the caller's sample rate so the response
        # is exactly `samples` long, matching the request. Good enough for
        # a chunk-boundary crossfade; not meant to be audiophile-grade.
        n_out = pcm.shape[1]
        src_idx = np.linspace(0.0, len(sep) - 1.0, num=n_out, dtype=np.float64)
        grid = np.arange(len(sep), dtype=np.float64)
        sep = np.interp(src_idx, grid, sep.astype(np.float64)).astype(np.float32)
    else:
        # Defend against off-by-a-few-samples rounding in the resample lib.
        n_out = pcm.shape[1]
        if len(sep) < n_out:
            sep = np.pad(sep, (0, n_out - len(sep)))
        elif len(sep) > n_out:
            sep = sep[:n_out]

    return np.stack([sep, sep], axis=0)  # mono -> stereo


def make_app(model, device: str):
    from flask import Flask, request, Response

    app = Flask(__name__)

    @app.post("/separate")
    def handle_separate():
        query = request.args.get("query", "")
        channels = int(request.args.get("channels", 2))
        sr = int(request.args.get("sr", 44100))
        if not query:
            return Response("missing 'query' parameter", status=400)

        body = request.get_data()
        expected_dtype_size = 4  # float32
        if len(body) % (channels * expected_dtype_size) != 0:
            return Response("malformed body: not a whole number of samples", status=400)
        samples = len(body) // (channels * expected_dtype_size)

        pcm = np.frombuffer(body, dtype=np.float32).reshape(channels, samples)

        try:
            out = _separate(model, device, pcm, channels, sr, query)
        except Exception as exc:  # noqa: BLE001 - surface it to the client, don't crash the server
            return Response(f"separation failed: {exc}", status=500)

        return Response(out.astype(np.float32).tobytes(), mimetype="application/octet-stream")

    @app.get("/health")
    def health():
        return {"status": "ok", "device": device}

    return app


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--audiosep-repo",
        default=os.environ.get("AUDIOSEP_REPO", ""),
        help="path to your AudioSep git checkout (or set AUDIOSEP_REPO)",
    )
    ap.add_argument("--config", default="config/audiosep_base.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--host", default="0.0.0.0", help="LAN-only; do not expose publicly")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    if not args.audiosep_repo:
        raise SystemExit("--audiosep-repo is required (or set AUDIOSEP_REPO)")

    model = _load_audiosep(args.audiosep_repo, args.config, args.checkpoint, args.device)
    app = make_app(model, args.device)
    print(f"[serve_audiosep] listening on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
