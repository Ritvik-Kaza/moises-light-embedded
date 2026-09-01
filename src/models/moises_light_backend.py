"""
Moises-Light backend.

Wraps the `moises-light` package, an unofficial PyTorch implementation of
"Moises-Light: Resource-efficient Band-split U-Net for Music Source Separation"
(Hung, Pereira & Korzeniowski, WASPAA 2025).

IMPORTANT — read this before interpreting any output audio:

The Moises-Light paper does not release code or weights, and the unofficial
package ships the architecture only. If you construct this backend without a
checkpoint, the weights are randomly initialised. The model will run, and every
*performance* number it produces (parameter count, latency, real-time factor,
peak RSS) is valid, because compute cost is a function of the architecture and
input shape, not of the weight values. The *audio* it produces is noise.

That is the correct split for this project: the embedded contribution is the
resource characterisation, which needs no training. Use the `openunmix` backend
when you need stems a listener can actually evaluate.
"""

from __future__ import annotations

import os
import warnings
from typing import List

import torch

from ..config import STEM_NAMES
from .base import SeparatorBackend

# The package emits a CPU-autocast warning on every forward pass. It is benign
# (it just falls back to fp32) but it would flood a per-chunk progress log.
warnings.filterwarnings(
    "ignore",
    message=".*CPU autocast.*",
    category=UserWarning,
)


class MoisesLightBackend(SeparatorBackend):
    name = "moises_light"

    def __init__(
        self,
        preset: str = "paper_small",
        checkpoint: str = "",
        device: str = "cpu",
    ) -> None:
        try:
            from moises_light import MoisesLight, configs
        except ImportError as exc:
            raise ImportError(
                "The 'moises-light' package is not installed.\n"
                "Install it with:  pip install moises-light"
            ) from exc

        if preset not in configs:
            raise ValueError(
                f"unknown preset '{preset}'. Available: {sorted(configs.keys())}"
            )

        self.preset = preset
        self.device = torch.device(device)

        self.model = MoisesLight(**configs[preset])
        self.model.eval()

        self.checkpoint_path = checkpoint
        if checkpoint:
            self._load_checkpoint(checkpoint)
            self.is_pretrained = True
        else:
            self.is_pretrained = False

        self.model.to(self.device)

    def _load_checkpoint(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"checkpoint not found: {path}")

        # weights_only=True refuses to unpickle arbitrary objects. Keep it on;
        # a checkpoint that needs it off is a checkpoint you should not trust.
        state = torch.load(path, map_location="cpu", weights_only=True)

        # Accept both a bare state_dict and a training-framework wrapper.
        if isinstance(state, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break

        # Lightning-style checkpoints prefix every key with "model.".
        state = { k[len("model."):] if k.startswith("model.") else k: v
                  for k, v in state.items() }

        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(
                f"checkpoint is missing {len(missing)} keys "
                f"(first few: {missing[:5]}). Preset '{self.preset}' probably "
                "does not match the preset the checkpoint was trained with."
            )
        if unexpected:
            print(f"[moises_light] ignoring {len(unexpected)} unexpected keys")

    @property
    def stems(self) -> List[str]:
        return list(STEM_NAMES)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def separate_chunk(self, mixture: torch.Tensor) -> torch.Tensor:
        # MoisesLight returns [batch, stems, channels, samples] directly.
        return self.model(mixture.to(self.device))
