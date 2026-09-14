"""
Remote (offloaded) backend.

This backend runs no model locally. It is a thin HTTP client: each chunk is
sent to a server (typically your laptop, running AudioSep or any other
query-conditioned separator) and the separated audio comes back over the
network. The Pi supplies the text query, chunking, overlap-add, and file I/O;
the laptop supplies the compute.

Why this exists: text/audio-query-conditioned separators (e.g. AudioSep) are
far heavier than Moises-Light or even Open-Unmix, and are not realistically
Pi-viable on-device. Rather than skip the feature, the Pi offloads just the
forward pass. This is a deliberate design decision, not a workaround — see
README.md's "Offload setup" section for the full reasoning and the
server-side setup.

IMPORTANT: this backend does not make the offloaded model "run on the Pi".
The Pi is a network client. Report it as such — do not present it alongside
Moises-Light's on-device numbers as if they were the same kind of result.

Wire protocol (see scripts/serve_audiosep.py for the server-side contract):
    POST {remote_url}/separate?query=<text>&sr=<int>&channels=<int>
    body: raw float32 little-endian PCM, shape [channels, samples],
          channel-major (i.e. mixture.tobytes() for a [channels, samples]
          C-contiguous float32 array)
    response: raw float32 little-endian PCM, always 2 channels,
          channel-major, same sample count as the request
"""

from __future__ import annotations

import re
import time
from typing import List
from urllib.parse import urlencode

import numpy as np
import torch

from .base import SeparatorBackend

_RESPONSE_CHANNELS = 2


def _sanitize_stem_name(query: str) -> str:
    """Turn a free-text query into a filesystem-safe stem/file name."""
    name = query.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    return name or "query"


class RemoteBackend(SeparatorBackend):
    name = "remote"
    is_pretrained = True       # the remote model, if configured correctly, is trained
    is_query_driven = True

    def __init__(
        self,
        query: str,
        url: str,
        device: str = "cpu",
        connect_timeout: float = 5.0,
        read_timeout: float = 60.0,
    ) -> None:
        try:
            import requests
        except ImportError as exc:
            raise ImportError(
                "The 'requests' package is not installed.\n"
                "Install it with:  pip install requests"
            ) from exc
        self._requests = requests

        if not query.strip():
            raise ValueError("RemoteBackend requires a non-empty query")
        if not url.strip():
            raise ValueError("RemoteBackend requires a remote_url, e.g. http://<laptop-ip>:8765")

        self.query = query.strip()
        self.url = url.rstrip("/")
        self.timeout = (connect_timeout, read_timeout)
        self._stem_name = _sanitize_stem_name(self.query)

        # Populated after each separate_chunk() call; separator.py reads this
        # (via getattr, so it is a no-op for every other backend) to report
        # network time separately from on-device compute.
        self.last_network_seconds: float = 0.0

        # A remote backend holds no local weights, hence 0 — that absence is
        # itself the point of offloading, not a bug in the size report.
        self._n_params = 0

    @property
    def stems(self) -> List[str]:
        return [self._stem_name]

    def n_params(self) -> int:
        return self._n_params

    def separate_chunk(self, mixture: torch.Tensor) -> torch.Tensor:
        # mixture: [1, 2, samples] -> drop the batch dim for the wire format.
        x = mixture[0].detach().cpu().numpy().astype(np.float32, copy=False)
        x = np.ascontiguousarray(x)
        channels, samples = x.shape

        params = urlencode({"query": self.query, "sr": 44100, "channels": channels})
        endpoint = f"{self.url}/separate?{params}"

        t0 = time.perf_counter()
        try:
            resp = self._requests.post(
                endpoint,
                data=x.tobytes(),
                headers={"Content-Type": "application/octet-stream"},
                timeout=self.timeout,
            )
        except self._requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"could not reach remote separation server at '{self.url}'.\n"
                "Check that the server is running and that --remote-url points "
                "at the right host:port. See README.md's Offload setup section."
            ) from exc
        except self._requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"remote separation server at '{self.url}' did not respond within "
                f"{self.timeout[1]}s. A slow/CPU-only laptop may need a longer "
                "--remote-timeout, or the chunk may be too long."
            ) from exc
        self.last_network_seconds = time.perf_counter() - t0

        if resp.status_code != 200:
            raise RuntimeError(
                f"remote separation server returned HTTP {resp.status_code}: "
                f"{resp.text[:500]}"
            )

        out = np.frombuffer(resp.content, dtype=np.float32)
        expected = _RESPONSE_CHANNELS * samples
        if out.size != expected:
            raise RuntimeError(
                f"remote separation server returned {out.size} samples, "
                f"expected {expected} ({_RESPONSE_CHANNELS} channels x {samples} "
                "samples). Server and client disagree on the wire protocol — "
                "see scripts/serve_audiosep.py."
            )
        out = out.reshape(_RESPONSE_CHANNELS, samples)

        # [channels, samples] -> [1, 1, channels, samples]: batch=1, stems=1.
        return torch.from_numpy(out.copy()).unsqueeze(0).unsqueeze(0)
