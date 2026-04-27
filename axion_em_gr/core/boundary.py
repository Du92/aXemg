"""
Boundary conditions.

Phase 14 introduces a GhostZoneManager-based boundary system.

Supported:
- periodic
- dirichlet
- neumann
- outflow
- linear_extrapolation
- sommerfeld
- mixed field-wise boundary
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from axion_em_gr.core.ghost_zones import GhostZoneManager
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.state import State


class BoundaryCondition:
    """
    Base class for boundary conditions.
    """

    def apply_array(self, array: np.ndarray, grid: Grid) -> None:
        raise NotImplementedError

    def apply_state(self, state: State, grid: Grid) -> None:
        self.apply_array(state.a, grid)
        self.apply_array(state.Pi, grid)

        if state.E is not None:
            for i in range(3):
                self.apply_array(state.E[i], grid)

        if state.B is not None:
            for i in range(3):
                self.apply_array(state.B[i], grid)


class PeriodicBoundary(BoundaryCondition):
    """
    Periodic boundary condition.
    """

    def apply_array(self, array: np.ndarray, grid: Grid) -> None:
        GhostZoneManager(grid).apply_periodic(array)


class DirichletBoundary(BoundaryCondition):
    """
    Constant-value Dirichlet boundary condition.
    """

    def __init__(self, value: float = 0.0) -> None:
        self.value = float(value)

    def apply_array(self, array: np.ndarray, grid: Grid) -> None:
        GhostZoneManager(grid).apply_dirichlet(array, value=self.value)


class NeumannBoundary(BoundaryCondition):
    """
    Zero-gradient Neumann boundary condition.
    """

    def apply_array(self, array: np.ndarray, grid: Grid) -> None:
        GhostZoneManager(grid).apply_neumann(array)


class OutflowBoundary(BoundaryCondition):
    """
    Simple outflow boundary condition.

    In this first implementation, outflow is implemented as zero-gradient
    ghost-zone filling. This is robust and often preferable to periodic
    boundaries for radiative test problems.
    """

    def apply_array(self, array: np.ndarray, grid: Grid) -> None:
        GhostZoneManager(grid).apply_outflow(array)


class LinearExtrapolationBoundary(BoundaryCondition):
    """
    Linear extrapolation boundary condition.

    This is less diffusive than zero-gradient outflow but can amplify noise.
    """

    def apply_array(self, array: np.ndarray, grid: Grid) -> None:
        GhostZoneManager(grid).apply_linear_extrapolation(array)


class SommerfeldBoundary(BoundaryCondition):
    """
    Approximate Sommerfeld-like boundary condition.

    This implementation is a ghost-zone approximation designed for outgoing
    waves approaching an asymptotic value u_inf.

    For a scalar u, a simple outgoing asymptotic behaviour is approximated by

        u_ghost = u_inf + (u_boundary - u_inf) * r_boundary / r_ghost

    in 2D radial form, and by a linear outgoing extrapolation in 1D.

    This is not a full characteristic boundary condition for the coupled
    axion-Maxwell system, but it is a useful improvement over periodic
    boundaries for localized packets.
    """

    def __init__(
        self,
        asymptotic_value: float = 0.0,
        wave_speed: float = 1.0,
        center: tuple[float, float] = (0.0, 0.0),
        fallback: str = "outflow",
    ) -> None:
        self.asymptotic_value = float(asymptotic_value)
        self.wave_speed = float(wave_speed)
        self.center = center
        self.fallback = fallback

    def apply_array(self, array: np.ndarray, grid: Grid) -> None:
        if grid.ndim == 1:
            self._apply_1d(array, grid)
        elif grid.ndim == 2:
            self._apply_2d_radial(array, grid)
        else:
            raise NotImplementedError("SommerfeldBoundary supports 1D and 2D.")

    def _apply_1d(self, array: np.ndarray, grid: Grid) -> None:
        """
        1D Sommerfeld-like ghost fill.

        We use a damped linear extrapolation toward asymptotic_value.
        """
        g = grid.nghost
        n = grid.shape[0]
        u_inf = self.asymptotic_value

        left_boundary = array[g]
        left_next = array[g + 1]

        right_boundary = array[g + n - 1]
        right_next = array[g + n - 2]

        left_slope = left_boundary - left_next
        right_slope = right_boundary - right_next

        for k in range(1, g + 1):
            damping = 1.0 / (1.0 + k)

            array[g - k] = u_inf + damping * (
                left_boundary - u_inf + k * left_slope
            )

            array[g + n - 1 + k] = u_inf + damping * (
                right_boundary - u_inf + k * right_slope
            )

    def _apply_2d_radial(self, array: np.ndarray, grid: Grid) -> None:
        """
        2D radial Sommerfeld-like fill.

        For ghost cells, we approximate

            u - u_inf ~ 1/r

        using nearest interior boundary values.
        """
        g = grid.nghost
        nx, ny = grid.shape
        u_inf = self.asymptotic_value

        x = grid.axis_coordinates(axis=0)
        y = grid.axis_coordinates(axis=1)

        x0, y0 = self.center

        # First fill with outflow to get a robust baseline, including corners.
        GhostZoneManager(grid).apply_outflow(array)

        # Left/right ghost zones.
        for i in range(g):
            i_left = i
            i_ref = g

            i_right = g + nx + i
            i_ref_right = g + nx - 1

            for j in range(g, g + ny):
                # Left.
                r_ref = np.sqrt((x[i_ref] - x0) ** 2 + (y[j] - y0) ** 2)
                r_ghost = np.sqrt((x[i_left] - x0) ** 2 + (y[j] - y0) ** 2)

                if r_ghost > 0.0:
                    array[i_left, j] = u_inf + (array[i_ref, j] - u_inf) * (
                        r_ref / r_ghost
                    )

                # Right.
                r_ref = np.sqrt((x[i_ref_right] - x0) ** 2 + (y[j] - y0) ** 2)
                r_ghost = np.sqrt((x[i_right] - x0) ** 2 + (y[j] - y0) ** 2)

                if r_ghost > 0.0:
                    array[i_right, j] = u_inf + (
                        array[i_ref_right, j] - u_inf
                    ) * (r_ref / r_ghost)

        # Bottom/top ghost zones.
        for j in range(g):
            j_bottom = j
            j_ref = g

            j_top = g + ny + j
            j_ref_top = g + ny - 1

            for i in range(0, nx + 2 * g):
                # Bottom.
                r_ref = np.sqrt((x[i] - x0) ** 2 + (y[j_ref] - y0) ** 2)
                r_ghost = np.sqrt((x[i] - x0) ** 2 + (y[j_bottom] - y0) ** 2)

                if r_ghost > 0.0:
                    array[i, j_bottom] = u_inf + (array[i, j_ref] - u_inf) * (
                        r_ref / r_ghost
                    )

                # Top.
                r_ref = np.sqrt((x[i] - x0) ** 2 + (y[j_ref_top] - y0) ** 2)
                r_ghost = np.sqrt((x[i] - x0) ** 2 + (y[j_top] - y0) ** 2)

                if r_ghost > 0.0:
                    array[i, j_top] = u_inf + (
                        array[i, j_ref_top] - u_inf
                    ) * (r_ref / r_ghost)


@dataclass
class FieldBoundarySpec:
    """
    Boundary condition specification for one field group.
    """

    boundary: BoundaryCondition


class MixedBoundary(BoundaryCondition):
    """
    Field-wise boundary condition.

    This allows, for example:

        a, Pi -> Sommerfeld
        E, B  -> Outflow

    Field keys:
        a
        Pi
        E
        B

    E and B are applied component-wise.
    """

    def __init__(
        self,
        default: BoundaryCondition,
        field_boundaries: dict[str, BoundaryCondition] | None = None,
    ) -> None:
        self.default = default
        self.field_boundaries = field_boundaries or {}

    def _get(self, field_name: str) -> BoundaryCondition:
        return self.field_boundaries.get(field_name, self.default)

    def apply_array(self, array: np.ndarray, grid: Grid) -> None:
        self.default.apply_array(array, grid)

    def apply_state(self, state: State, grid: Grid) -> None:
        self._get("a").apply_array(state.a, grid)
        self._get("Pi").apply_array(state.Pi, grid)

        if state.E is not None:
            bc_E = self._get("E")
            for i in range(3):
                bc_E.apply_array(state.E[i], grid)

        if state.B is not None:
            bc_B = self._get("B")
            for i in range(3):
                bc_B.apply_array(state.B[i], grid)


def make_boundary_from_config_dict(config: dict[str, Any]) -> BoundaryCondition:
    """
    Build a boundary object from a dictionary.

    Examples
    --------
    boundary:
      type: periodic

    boundary:
      type: sommerfeld
      asymptotic_value: 0.0
      wave_speed: 1.0
      center: [0.0, 0.0]

    boundary:
      type: mixed
      default:
        type: outflow
      fields:
        a:
          type: sommerfeld
          asymptotic_value: 0.0
        Pi:
          type: sommerfeld
          asymptotic_value: 0.0
        E:
          type: outflow
        B:
          type: outflow
    """
    bc_type = config.get("type", "periodic")

    if bc_type == "periodic":
        return PeriodicBoundary()

    if bc_type == "dirichlet":
        return DirichletBoundary(value=float(config.get("value", 0.0)))

    if bc_type == "neumann":
        return NeumannBoundary()

    if bc_type == "outflow":
        return OutflowBoundary()

    if bc_type == "linear_extrapolation":
        return LinearExtrapolationBoundary()

    if bc_type == "sommerfeld":
        center_raw = config.get("center", [0.0, 0.0])
        center = (float(center_raw[0]), float(center_raw[1]))

        return SommerfeldBoundary(
            asymptotic_value=float(config.get("asymptotic_value", 0.0)),
            wave_speed=float(config.get("wave_speed", 1.0)),
            center=center,
            fallback=str(config.get("fallback", "outflow")),
        )

    if bc_type == "mixed":
        default_cfg = config.get("default", {"type": "outflow"})
        default_bc = make_boundary_from_config_dict(default_cfg)

        fields_cfg = config.get("fields", {})
        field_boundaries = {
            name: make_boundary_from_config_dict(field_cfg)
            for name, field_cfg in fields_cfg.items()
        }

        return MixedBoundary(
            default=default_bc,
            field_boundaries=field_boundaries,
        )

    raise ValueError(f"Unknown boundary type: {bc_type!r}")