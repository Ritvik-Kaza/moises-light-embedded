"""
Backend registry.

Imports are lazy so that a machine with only one backend installed (a Pi that
never needs Open-Unmix, say) does not pay the import cost or fail at startup.
"""

from __future__ import annotations

from .base import SeparatorBackend

AVAILABLE_BACKENDS = ("moises_light", "openunmix", "remote")


def build_backend(cfg) -> SeparatorBackend:
    """Construct the backend named in cfg.backend."""
    if cfg.backend == "moises_light":
        from .moises_light_backend import MoisesLightBackend

        return MoisesLightBackend(
            preset=cfg.preset,
            checkpoint=cfg.checkpoint,
            device=cfg.device,
        )

    if cfg.backend == "openunmix":
        from .openunmix_backend import OpenUnmixBackend

        return OpenUnmixBackend(device=cfg.device)

    if cfg.backend == "remote":
        from .remote_backend import RemoteBackend

        return RemoteBackend(
            query=cfg.query,
            url=cfg.remote_url,
            device=cfg.device,
            read_timeout=cfg.remote_timeout,
        )

    raise ValueError(
        f"unknown backend '{cfg.backend}'. Available: {AVAILABLE_BACKENDS}"
    )


__all__ = ["SeparatorBackend", "build_backend", "AVAILABLE_BACKENDS"]
