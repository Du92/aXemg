"""
Parameter containers for physical and numerical settings.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalParameters:
    """
    Physical parameters.

    In Phase 1, only m_axion is dynamically relevant.
    The coupling g_agamma is included already for compatibility with later phases.
    """

    m_axion: float = 0.0
    g_agamma: float = 0.0


@dataclass(frozen=True)
class NumericalParameters:
    """
    Numerical parameters.
    """

    dt: float
    t_final: float
    output_every: int = 10
    derivative_order: int = 2

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be positive.")

        if self.t_final <= 0:
            raise ValueError("t_final must be positive.")

        if self.output_every <= 0:
            raise ValueError("output_every must be positive.")

        if self.derivative_order not in (2,):
            raise NotImplementedError(
                "Phase 1 currently supports only second-order finite differences."
            )
