"""
Backend interface.

The separator only ever talks to this interface, which is what lets the same
pipeline run the Moises-Light architecture (for embedded benchmarking) and a
pretrained model (for audible output) without changing any pipeline code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import torch


class SeparatorBackend(ABC):
    """A model that maps a mixture chunk to four stem chunks."""

    #: Human-readable name used in logs and the metrics report.
    name: str = "base"

    #: True when the loaded weights are trained. False means output audio is
    #: meaningless and only timing/memory numbers should be trusted.
    is_pretrained: bool = False

    #: True for backends whose `stems` is a single, query-derived name rather
    #: than the fixed vocals/drums/bass/other set (e.g. a text-conditioned
    #: separator). separator.py uses this to skip filtering `stems` against
    #: the fixed STEM_NAMES list, since a query-driven output is never in it.
    is_query_driven: bool = False

    @property
    @abstractmethod
    def stems(self) -> List[str]:
        """Stem names in the order the model's output dimension uses."""

    @abstractmethod
    def n_params(self) -> int:
        """Total parameter count, used for the model-size report."""

    @abstractmethod
    def separate_chunk(self, mixture: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mixture: [1, 2, samples] float32
        Returns:
            [1, n_stems, 2, samples] float32
        """
