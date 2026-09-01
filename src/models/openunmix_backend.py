"""
Open-Unmix backend.

This exists because Moises-Light has no public weights (see
moises_light_backend.py). Open-Unmix is a published 4-stem separator that ships
downloadable pretrained weights and runs on CPU, so it gives you stems a human
can actually listen to during a review demo.

Roles in this project:
    moises_light -> the architecture under study; provides latency/memory numbers
    openunmix    -> a pretrained reference; provides audible separation quality

Do not present Open-Unmix numbers as Moises-Light numbers. It is a substantially
larger model (roughly 10M parameters per target, four targets) and will be
slower and heavier on a Pi than Moises-Light is.

Weights download automatically on first use and are cached in ~/.cache/torch.
Run it once on a machine with internet before taking the Pi offline.
"""

from __future__ import annotations

from typing import List

import torch

from .base import SeparatorBackend

# Open-Unmix's own target order, which we remap to the project's stem order.
_UMX_TARGETS = ["vocals", "drums", "bass", "other"]


class OpenUnmixBackend(SeparatorBackend):
    name = "openunmix"
    is_pretrained = True

    def __init__(
        self,
        variant: str = "umxl",
        device: str = "cpu",
        niter: int = 0,
    ) -> None:
        """
        Args:
            variant: "umxl" (best quality, largest), "umxhq", or "umx".
            niter:   Wiener-filter iterations. 0 = soft mask only, which is
                     markedly faster and is the right choice on a Pi. 1 is the
                     upstream default and sounds slightly better.
        """
        try:
            import openunmix
        except ImportError as exc:
            raise ImportError(
                "The 'openunmix' package is not installed.\n"
                "Install it with:  pip install openunmix"
            ) from exc

        if not hasattr(openunmix, variant):
            raise ValueError(
                f"unknown Open-Unmix variant '{variant}'. "
                "Expected one of: umx, umxhq, umxl"
            )

        self.variant = variant
        self.device = torch.device(device)

        try:
            self.model = getattr(openunmix, variant)(
                device=device, pretrained=True, niter=niter
            )
        except Exception as exc:  # noqa: BLE001 - network failures land here
            raise RuntimeError(
                f"could not load pretrained Open-Unmix weights ('{variant}').\n"
                "First use requires an internet connection to download them.\n"
                f"Underlying error: {exc}"
            ) from exc

        self.model.eval()

        # Read the real target order off the model rather than assuming it.
        target_models = getattr(self.model, "target_models", None)
        if target_models is not None:
            self._model_targets = list(target_models.keys())
        else:
            self._model_targets = list(_UMX_TARGETS)

    @property
    def stems(self) -> List[str]:
        return list(self._model_targets)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def separate_chunk(self, mixture: torch.Tensor) -> torch.Tensor:
        # Separator expects [batch, channels, samples] and returns
        # [batch, targets, channels, samples] — same layout as MoisesLight.
        out = self.model(mixture.to(self.device))
        return out.float()
