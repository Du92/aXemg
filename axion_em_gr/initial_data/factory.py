"""
Initial-data factory.

Build State objects from YAML configuration dictionaries.
"""

from __future__ import annotations

from typing import Any

from axion_em_gr.core.grid import Grid
from axion_em_gr.initial_data.axion_profiles import (
    gaussian_axion_packet,
    sinusoidal_axion_mode,
)
from axion_em_gr.initial_data.combined_setups import (
    gaussian_axion_em_wave_background_Bx_1d,
    gaussian_axion_uniform_Bx_constraint_solved_1d,
    gaussian_axion_uniform_magnetic_field_1d,
)

from axion_em_gr.initial_data.combined_setups_2d import (
    gaussian_axion_em_ring_2d,
    gaussian_axion_uniform_Bxy_constraint_solved_2d,
    gaussian_axion_uniform_Bz_2d,
)

from axion_em_gr.initial_data.physical_scenarios_1d import (
    axion_halo_gradient_magnetized_1d,
)

from axion_em_gr.initial_data.ns_scenarios_2d import (
    axion_cloud_around_compact_object_2d,
    gaussian_axion_core_2d,
)


def _as_tuple(values, length: int, dtype=float):
    """
    Convert YAML list to tuple with validation.
    """
    if len(values) != length:
        raise ValueError(f"Expected {length} values, got {len(values)}.")

    return tuple(dtype(v) for v in values)


def build_initial_state(config: dict[str, Any], grid: Grid, metric=None):
    """
    Build initial State from YAML.

    Expected block:

        initial_data:
          type: gaussian_axion_packet
          ...
    """
    ic_cfg = config["initial_data"]
    ic_type = ic_cfg["type"]

    if ic_type == "gaussian_axion_packet_1d":
        return gaussian_axion_packet(
            grid=grid,
            amplitude=float(ic_cfg.get("amplitude", 1.0)),
            center=float(ic_cfg.get("center", 50.0)),
            width=float(ic_cfg.get("width", 5.0)),
            momentum_amplitude=float(ic_cfg.get("momentum_amplitude", 0.0)),
        )

    if ic_type == "sinusoidal_axion_mode_1d":
        return sinusoidal_axion_mode(
            grid=grid,
            amplitude=float(ic_cfg.get("amplitude", 1.0)),
            mode_number=int(ic_cfg.get("mode_number", 1)),
            mass=float(ic_cfg.get("mass", 0.0)),
        )

    if ic_type == "gaussian_axion_uniform_magnetic_field_1d":
        return gaussian_axion_uniform_magnetic_field_1d(
            grid=grid,
            axion_amplitude=float(ic_cfg.get("axion_amplitude", 1.0)),
            axion_center=float(ic_cfg.get("axion_center", 50.0)),
            axion_width=float(ic_cfg.get("axion_width", 5.0)),
            axion_momentum_amplitude=float(
                ic_cfg.get("axion_momentum_amplitude", 0.0)
            ),
            B0=_as_tuple(ic_cfg.get("B0", [1.0, 0.0, 0.0]), 3, float),
            E0=_as_tuple(ic_cfg.get("E0", [0.0, 0.0, 0.0]), 3, float),
        )

    if ic_type == "gaussian_axion_em_wave_background_Bx_1d":
        return gaussian_axion_em_wave_background_Bx_1d(
            grid=grid,
            axion_amplitude=float(ic_cfg.get("axion_amplitude", 1.0)),
            axion_center=float(ic_cfg.get("axion_center", 40.0)),
            axion_width=float(ic_cfg.get("axion_width", 7.0)),
            axion_momentum_amplitude=float(
                ic_cfg.get("axion_momentum_amplitude", 0.3)
            ),
            em_amplitude=float(ic_cfg.get("em_amplitude", 0.2)),
            em_center=float(ic_cfg.get("em_center", 45.0)),
            em_width=float(ic_cfg.get("em_width", 8.0)),
            background_Bx=float(ic_cfg.get("background_Bx", 1.0)),
            propagation=str(ic_cfg.get("propagation", "right")),
        )

    if ic_type == "gaussian_axion_uniform_Bz_2d":
        return gaussian_axion_uniform_Bz_2d(
            grid=grid,
            axion_amplitude=float(ic_cfg.get("axion_amplitude", 1.0)),
            axion_center=_as_tuple(ic_cfg.get("axion_center", [0.0, 0.0]), 2, float),
            axion_width=_as_tuple(ic_cfg.get("axion_width", [5.0, 5.0]), 2, float),
            axion_momentum_amplitude=float(
                ic_cfg.get("axion_momentum_amplitude", 0.3)
            ),
            B0=_as_tuple(ic_cfg.get("B0", [0.0, 0.0, 1.0]), 3, float),
            E0=_as_tuple(ic_cfg.get("E0", [0.0, 0.0, 0.0]), 3, float),
        )

    if ic_type == "gaussian_axion_em_ring_2d":
        return gaussian_axion_em_ring_2d(
            grid=grid,
            axion_amplitude=float(ic_cfg.get("axion_amplitude", 1.0)),
            axion_center=_as_tuple(ic_cfg.get("axion_center", [0.0, 0.0]), 2, float),
            axion_width=_as_tuple(ic_cfg.get("axion_width", [6.0, 6.0]), 2, float),
            axion_momentum_amplitude=float(
                ic_cfg.get("axion_momentum_amplitude", 0.3)
            ),
            em_amplitude=float(ic_cfg.get("em_amplitude", 0.2)),
            em_center=_as_tuple(ic_cfg.get("em_center", [0.0, 0.0]), 2, float),
            em_width=_as_tuple(ic_cfg.get("em_width", [8.0, 8.0]), 2, float),
            background_Bz=float(ic_cfg.get("background_Bz", 1.0)),
        )
    
    if ic_type == "gaussian_axion_uniform_Bx_constraint_solved_1d":
        physics_cfg = config.get("physics", {})

        return gaussian_axion_uniform_Bx_constraint_solved_1d(
            grid=grid,
            axion_amplitude=float(ic_cfg.get("axion_amplitude", 1.0)),
            axion_center=float(ic_cfg.get("axion_center", 50.0)),
            axion_width=float(ic_cfg.get("axion_width", 5.0)),
            axion_momentum_amplitude=float(
                ic_cfg.get("axion_momentum_amplitude", 0.0)
            ),
            g_agamma=float(ic_cfg.get("g_agamma", physics_cfg.get("g_agamma", 0.03))),
            Bx=float(ic_cfg.get("Bx", 1.0)),
            Ex_constant=float(ic_cfg.get("Ex_constant", 0.0)),
        )
    
    if ic_type == "gaussian_axion_uniform_Bxy_constraint_solved_2d":
        physics_cfg = config.get("physics", {})

        return gaussian_axion_uniform_Bxy_constraint_solved_2d(
            grid=grid,
            axion_amplitude=float(ic_cfg.get("axion_amplitude", 1.0)),
            axion_center=_as_tuple(ic_cfg.get("axion_center", [0.0, 0.0]), 2, float),
            axion_width=_as_tuple(ic_cfg.get("axion_width", [8.0, 8.0]), 2, float),
            axion_momentum_amplitude=float(
                ic_cfg.get("axion_momentum_amplitude", 0.3)
            ),
            g_agamma=float(ic_cfg.get("g_agamma", physics_cfg.get("g_agamma", 0.03))),
            B0=_as_tuple(ic_cfg.get("B0", [1.0, 0.5, 0.0]), 3, float),
            E0=_as_tuple(ic_cfg.get("E0", [0.0, 0.0, 0.0]), 3, float),
            dt_for_cleaning=float(ic_cfg.get("dt_for_cleaning", 0.01)),
            poisson_solver=str(ic_cfg.get("poisson_solver", "periodic_fft")),
            poisson_boundary=str(ic_cfg.get("poisson_boundary", "periodic")),
            dirichlet_value=float(ic_cfg.get("dirichlet_value", 0.0)),
            max_iterations=int(ic_cfg.get("max_iterations", 50_000)),
            tolerance=float(ic_cfg.get("tolerance", 1.0e-8)),
            omega=(
                None
                if ic_cfg.get("omega", None) is None
                else float(ic_cfg.get("omega"))
            ),
            cleaning_geometry=str(ic_cfg.get("cleaning_geometry", "flat")),
            metric=metric,
        )
    
    if ic_type == "axion_halo_gradient_magnetized_1d":
        physics_cfg = config.get("physics", {})

        return axion_halo_gradient_magnetized_1d(
            grid=grid,
            axion_amplitude=float(ic_cfg.get("axion_amplitude", 1.0)),
            axion_center=float(ic_cfg.get("axion_center", 45.0)),
            axion_width=float(ic_cfg.get("axion_width", 10.0)),
            axion_background=float(ic_cfg.get("axion_background", 0.0)),
            gradient_amplitude=float(ic_cfg.get("gradient_amplitude", 0.3)),
            gradient_width=float(ic_cfg.get("gradient_width", 12.0)),
            Pi_amplitude=float(ic_cfg.get("Pi_amplitude", 0.0)),
            Pi_width=(
                None
                if ic_cfg.get("Pi_width", None) is None
                else float(ic_cfg.get("Pi_width"))
            ),
            g_agamma=float(ic_cfg.get("g_agamma", physics_cfg.get("g_agamma", 0.03))),
            background_Bx=float(ic_cfg.get("background_Bx", 1.0)),
            background_By=float(ic_cfg.get("background_By", 0.0)),
            background_Bz=float(ic_cfg.get("background_Bz", 0.0)),
            em_pulse_amplitude=float(ic_cfg.get("em_pulse_amplitude", 0.15)),
            em_pulse_center=float(ic_cfg.get("em_pulse_center", 40.0)),
            em_pulse_width=float(ic_cfg.get("em_pulse_width", 8.0)),
            em_pulse_polarization=str(ic_cfg.get("em_pulse_polarization", "y")),
            em_pulse_propagation=str(ic_cfg.get("em_pulse_propagation", "right")),
            constraint_solved_Ex=bool(ic_cfg.get("constraint_solved_Ex", True)),
        )
    
    if ic_type == "axion_cloud_around_compact_object_2d":
        return axion_cloud_around_compact_object_2d(
            grid=grid,
            axion_amplitude=float(ic_cfg.get("axion_amplitude", 1.0)),
            cloud_radius=float(ic_cfg.get("cloud_radius", 15.0)),
            cloud_width=float(ic_cfg.get("cloud_width", 8.0)),
            center=_as_tuple(ic_cfg.get("center", [0.0, 0.0]), 2, float),
            axion_background=float(ic_cfg.get("axion_background", 0.0)),
            angular_modulation=float(ic_cfg.get("angular_modulation", 0.0)),
            azimuthal_mode=int(ic_cfg.get("azimuthal_mode", 1)),
            Pi_amplitude=float(ic_cfg.get("Pi_amplitude", 0.0)),
            Pi_width=(
                None
                if ic_cfg.get("Pi_width", None) is None
                else float(ic_cfg.get("Pi_width"))
            ),
        )

    if ic_type == "gaussian_axion_core_2d":
        return gaussian_axion_core_2d(
            grid=grid,
            axion_amplitude=float(ic_cfg.get("axion_amplitude", 1.0)),
            center=_as_tuple(ic_cfg.get("center", [0.0, 0.0]), 2, float),
            width=float(ic_cfg.get("width", 10.0)),
            Pi_amplitude=float(ic_cfg.get("Pi_amplitude", 0.0)),
        )

    raise ValueError(f"Unknown initial_data type: {ic_type!r}")
