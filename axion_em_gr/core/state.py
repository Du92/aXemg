"""
Dynamical state variables.

For Phase 1 we only evolve the axion field a and its conjugate momentum Pi.
The State object is nevertheless written in a way that can later include
E^i and B^i without changing the integrator interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class State:
    """
    Dynamical fields.

    Parameters
    ----------
    a:
        Axion field.
    Pi:
        Conjugate momentum of the axion field.
    E:
        Electric field E^i. Optional in Phase 1.
    B:
        Magnetic field B^i. Optional in Phase 1.
    """

    a: np.ndarray
    Pi: np.ndarray
    E: np.ndarray | None = None
    B: np.ndarray | None = None

    def copy(self) -> "State":
        return State(
            a=self.a.copy(),
            Pi=self.Pi.copy(),
            E=None if self.E is None else self.E.copy(),
            B=None if self.B is None else self.B.copy(),
        )

    def zeros_like(self) -> "State":
        return State(
            a=np.zeros_like(self.a),
            Pi=np.zeros_like(self.Pi),
            E=None if self.E is None else np.zeros_like(self.E),
            B=None if self.B is None else np.zeros_like(self.B),
        )

    def add_scaled(self, other: "State", scale: float) -> "State":
        """
        Return self + scale * other.
        """
        E_new = None
        B_new = None

        if self.E is not None and other.E is not None:
            E_new = self.E + scale * other.E

        if self.B is not None and other.B is not None:
            B_new = self.B + scale * other.B

        return State(
            a=self.a + scale * other.a,
            Pi=self.Pi + scale * other.Pi,
            E=E_new,
            B=B_new,
        )

    @staticmethod
    def linear_combination(
        base: "State",
        terms: list[tuple[float, "State"]],
    ) -> "State":
        """
        Return base + sum_i coeff_i * state_i.
        """
        result = base.copy()

        for coeff, state in terms:
            result.a += coeff * state.a
            result.Pi += coeff * state.Pi

            if result.E is not None and state.E is not None:
                result.E += coeff * state.E

            if result.B is not None and state.B is not None:
                result.B += coeff * state.B

        return result

    def assert_finite(self) -> None:
        """
        Raise an error if any evolved field contains NaNs or infinities.
        """
        if not np.all(np.isfinite(self.a)):
            raise FloatingPointError("Non-finite values detected in axion field a.")

        if not np.all(np.isfinite(self.Pi)):
            raise FloatingPointError("Non-finite values detected in axion momentum Pi.")

        if self.E is not None and not np.all(np.isfinite(self.E)):
            raise FloatingPointError("Non-finite values detected in electric field E.")

        if self.B is not None and not np.all(np.isfinite(self.B)):
            raise FloatingPointError("Non-finite values detected in magnetic field B.")
