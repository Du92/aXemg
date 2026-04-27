"""
YAML configuration loader.

This module reads a YAML configuration file and builds the main simulation
objects used by the solver.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from axion_em_gr.core.boundary import make_boundary_from_config_dict
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.rhs import RHSComputer
from axion_em_gr.geometry.flat import FlatMetric
from axion_em_gr.geometry.gw_tt import GWTTMetric1D
from axion_em_gr.physics.potentials import MassivePotential, ZeroPotential
from axion_em_gr.physics.sources import VacuumSources
from axion_em_gr.solvers.evolution import EvolutionSolver
from axion_em_gr.solvers.rk4 import RK4
from axion_em_gr.physics.background_em import (
    NoBackgroundEM,
    RotatingDipoleBackground2D,
    UniformMagneticBackground,
)
from axion_em_gr.geometry.schwarzschild_like import (
    SchwarzschildIsotropicMetric1D,
    SchwarzschildIsotropicMetric2D,
    SmoothCompactObjectMetric2D,
)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML configuration file.
    """
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return config


def build_grid(config: dict[str, Any]) -> Grid:
    """
    Build Grid from config.

    Expected YAML block:

        grid:
          ndim: 2
          shape: [256, 256]
          bounds:
            - [-50.0, 50.0]
            - [-50.0, 50.0]
          nghost: 3
    """
    grid_cfg = config["grid"]

    ndim = int(grid_cfg["ndim"])
    shape = tuple(int(v) for v in grid_cfg["shape"])
    bounds = tuple(tuple(float(x) for x in pair) for pair in grid_cfg["bounds"])
    nghost = int(grid_cfg.get("nghost", 3))

    return Grid(
        ndim=ndim,
        shape=shape,
        bounds=bounds,
        nghost=nghost,
    )


def build_physical_parameters(config: dict[str, Any]) -> PhysicalParameters:
    """
    Build physical parameter container.

    Expected YAML block:

        physics:
          m_axion: 0.2
          g_agamma: 0.03
    """
    physics_cfg = config.get("physics", {})

    return PhysicalParameters(
        m_axion=float(physics_cfg.get("m_axion", 0.0)),
        g_agamma=float(physics_cfg.get("g_agamma", 0.0)),
    )


def build_numerical_parameters(config: dict[str, Any]) -> NumericalParameters:
    """
    Build numerical parameter container.

    Expected YAML block:

        numerics:
          dt: 0.01
          t_final: 10.0
          output_every: 100
          derivative_order: 2
    """
    numerics_cfg = config["numerics"]

    return NumericalParameters(
        dt=float(numerics_cfg["dt"]),
        t_final=float(numerics_cfg["t_final"]),
        output_every=int(numerics_cfg.get("output_every", 100)),
        derivative_order=int(numerics_cfg.get("derivative_order", 2)),
    )


def build_metric(config: dict[str, Any]):
    """
    Build metric object from config.

    Supported types:
        flat
        gw_tt_1d
        diagonal_1d, if you placed DiagonalMetric1D in geometry/custom_metric.py
    """
    geometry_cfg = config.get("geometry", {})
    metric_type = geometry_cfg.get("type", "flat")

    if metric_type == "flat":
        return FlatMetric()

    if metric_type == "gw_tt_1d":
        return GWTTMetric1D(
            h_plus_amplitude=float(geometry_cfg.get("h_plus_amplitude", 1.0e-3)),
            h_cross_amplitude=float(geometry_cfg.get("h_cross_amplitude", 0.0)),
            wavelength=float(geometry_cfg.get("wavelength", 50.0)),
            omega=(
                None
                if geometry_cfg.get("omega", None) is None
                else float(geometry_cfg["omega"])
            ),
            phase_plus=float(geometry_cfg.get("phase_plus", 0.0)),
            phase_cross=float(geometry_cfg.get("phase_cross", 0.0)),
            packet=bool(geometry_cfg.get("packet", False)),
            packet_center=float(geometry_cfg.get("packet_center", 50.0)),
            packet_width=float(geometry_cfg.get("packet_width", 15.0)),
            direction=int(geometry_cfg.get("direction", +1)),
            compute_K_exact=bool(geometry_cfg.get("compute_K_exact", True)),
        )

    if metric_type == "diagonal_1d":
        try:
            from axion_em_gr.geometry.custom_metric import DiagonalMetric1D
        except ImportError as exc:
            raise ImportError(
                "geometry.type='diagonal_1d' requires DiagonalMetric1D "
                "to be defined in axion_em_gr.geometry.custom_metric."
            ) from exc

        return DiagonalMetric1D(
            lapse_amplitude=float(geometry_cfg.get("lapse_amplitude", 0.1)),
            metric_amplitude=float(geometry_cfg.get("metric_amplitude", 0.1)),
            shift_amplitude=float(geometry_cfg.get("shift_amplitude", 0.0)),
            center=float(geometry_cfg.get("center", 50.0)),
            width=float(geometry_cfg.get("width", 10.0)),
            K_value=float(geometry_cfg.get("K_value", 0.0)),
        )
    
    if metric_type == "schwarzschild_isotropic_1d":
        return SchwarzschildIsotropicMetric1D(
            mass=float(geometry_cfg.get("mass", 1.0)),
            center=float(geometry_cfg.get("center", 0.0)),
            use_absolute_radius=bool(geometry_cfg.get("use_absolute_radius", False)),
            radial_floor=float(geometry_cfg.get("radial_floor", 1.0e-6)),
            lapse_floor=float(geometry_cfg.get("lapse_floor", 1.0e-4)),
            horizon_buffer=float(geometry_cfg.get("horizon_buffer", 1.0e-3)),
        )

    if metric_type == "schwarzschild_isotropic_2d":
        center_raw = geometry_cfg.get("center", [0.0, 0.0])

        return SchwarzschildIsotropicMetric2D(
            mass=float(geometry_cfg.get("mass", 1.0)),
            center=(float(center_raw[0]), float(center_raw[1])),
            plane=str(geometry_cfg.get("plane", "xy")),
            plane_offset=float(geometry_cfg.get("plane_offset", 0.0)),
            radial_floor=float(geometry_cfg.get("radial_floor", 1.0e-6)),
            lapse_floor=float(geometry_cfg.get("lapse_floor", 1.0e-4)),
            horizon_buffer=float(geometry_cfg.get("horizon_buffer", 1.0e-3)),
        )

    if metric_type == "smooth_compact_object_2d":
        center_raw = geometry_cfg.get("center", [0.0, 0.0])

        return SmoothCompactObjectMetric2D(
            conformal_amplitude=float(geometry_cfg.get("conformal_amplitude", 1.0)),
            compactness=float(geometry_cfg.get("compactness", 0.2)),
            radius=float(geometry_cfg.get("radius", 10.0)),
            center=(float(center_raw[0]), float(center_raw[1])),
            plane=str(geometry_cfg.get("plane", "xy")),
            plane_offset=float(geometry_cfg.get("plane_offset", 0.0)),
            lapse_floor=float(geometry_cfg.get("lapse_floor", 0.2)),
        )

    raise ValueError(f"Unknown geometry type: {metric_type!r}")


def build_potential(config: dict[str, Any], physical: PhysicalParameters):
    """
    Build axion potential from config.

    Expected YAML block:

        potential:
          type: massive

    Supported:
        massive
        zero
    """
    potential_cfg = config.get("potential", {})
    potential_type = potential_cfg.get("type", "massive")

    if potential_type == "massive":
        m = float(potential_cfg.get("m", physical.m_axion))
        return MassivePotential(m=m)

    if potential_type == "zero":
        return ZeroPotential()

    raise ValueError(f"Unknown potential type: {potential_type!r}")


def build_sources(config: dict[str, Any]):
    """
    Build source model.

    Phase 10 supports vacuum sources.
    """
    sources_cfg = config.get("sources", {})
    source_type = sources_cfg.get("type", "vacuum")

    if source_type == "vacuum":
        return VacuumSources()

    raise ValueError(f"Unknown source type: {source_type!r}")


def build_boundary(config: dict[str, Any]):
    """
    Build boundary condition object.

    Supported examples:

        boundary:
          type: periodic

        boundary:
          type: outflow

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
    boundary_cfg = config.get("boundary", {"type": "periodic"})
    return make_boundary_from_config_dict(boundary_cfg)


def build_integrator(config: dict[str, Any]):
    """
    Build time integrator.

    Phase 10 supports RK4.
    """
    integrator_cfg = config.get("integrator", {})
    integrator_type = integrator_cfg.get("type", "rk4")

    if integrator_type == "rk4":
        return RK4()

    raise ValueError(f"Unknown integrator type: {integrator_type!r}")


def build_rhs_computer(
    config: dict[str, Any],
    grid: Grid,
    metric,
    potential,
    numerics: NumericalParameters,
    physical: PhysicalParameters,
    sources,
    background_em=None,
) -> RHSComputer:
    """
    Build RHSComputer.

    Expected YAML block:

        evolution:
          evolve_axion: true
          evolve_maxwell: true
          include_axion_em_coupling: true
    """
    evolution_cfg = config.get("evolution", {})

    return RHSComputer(
        grid=grid,
        metric=metric,
        potential=potential,
        numerics=numerics,
        physical=physical,
        sources=sources,
        evolve_axion=bool(evolution_cfg.get("evolve_axion", True)),
        evolve_maxwell=bool(evolution_cfg.get("evolve_maxwell", False)),
        include_axion_em_coupling=bool(
            evolution_cfg.get("include_axion_em_coupling", False)
        ),
        background_em=background_em,
        background_em_mode=str(evolution_cfg.get("background_em_mode", "replace")),
    )


def build_solver(
    config: dict[str, Any],
    grid: Grid,
    rhs_computer: RHSComputer,
    integrator,
    boundary,
    numerics: NumericalParameters,
) -> EvolutionSolver:
    """
    Build EvolutionSolver.

    Expected YAML block:

        snapshots:
          save: true
          every: 100
    """
    snapshots_cfg = config.get("snapshots", {})

    save_snapshots = bool(snapshots_cfg.get("save", False))
    snapshot_every = snapshots_cfg.get("every", None)

    if snapshot_every is not None:
        snapshot_every = int(snapshot_every)

    return EvolutionSolver(
        grid=grid,
        rhs_computer=rhs_computer,
        integrator=integrator,
        boundary=boundary,
        numerics=numerics,
        save_snapshots=save_snapshots,
        snapshot_every=snapshot_every,
    )


def build_simulation_objects(config: dict[str, Any]) -> dict[str, Any]:
    """
    Build all main simulation objects except the initial state.

    Returns a dictionary with:
        grid
        physical
        numerics
        metric
        potential
        sources
        boundary
        integrator
        rhs
        solver
    """
    grid = build_grid(config)
    physical = build_physical_parameters(config)
    numerics = build_numerical_parameters(config)
    metric = build_metric(config)
    potential = build_potential(config, physical)
    sources = build_sources(config)
    background_em = build_background_em(config)
    boundary = build_boundary(config)
    integrator = build_integrator(config)

    rhs = build_rhs_computer(
        config=config,
        grid=grid,
        metric=metric,
        potential=potential,
        numerics=numerics,
        physical=physical,
        sources=sources,
        background_em=background_em,
    )

    solver = build_solver(
        config=config,
        grid=grid,
        rhs_computer=rhs,
        integrator=integrator,
        boundary=boundary,
        numerics=numerics,
    )

    return {
        "grid": grid,
        "physical": physical,
        "numerics": numerics,
        "metric": metric,
        "potential": potential,
        "sources": sources,
        "boundary": boundary,
        "integrator": integrator,
        "rhs": rhs,
        "solver": solver,
        "background_em": background_em,
    }

def build_background_em(config: dict[str, Any]):
    """
    Build prescribed EM background from config.

    Supported:

        background_em:
          type: none

        background_em:
          type: uniform
          B0: [0.0, 0.0, 1.0]
          E0: [0.0, 0.0, 0.0]

        background_em:
          type: rotating_dipole_2d
          mu0: 1.0
          omega: 0.2
          inclination: 0.5
          phase0: 0.0
          center: [0.0, 0.0]
          plane: xy
          plane_offset: 0.0
          B_scale: 1.0
          softening_radius: 2.0
          star_radius: 5.0
          include_induced_E: true
          electric_scale: 1.0
    """
    bg_cfg = config.get("background_em", {"type": "none"})
    bg_type = bg_cfg.get("type", "none")

    if bg_type in ("none", None):
        return None

    if bg_type == "uniform":
        return UniformMagneticBackground(
            B0=tuple(float(v) for v in bg_cfg.get("B0", [0.0, 0.0, 1.0])),
            E0=tuple(float(v) for v in bg_cfg.get("E0", [0.0, 0.0, 0.0])),
        )

    if bg_type == "rotating_dipole_2d":
        center_raw = bg_cfg.get("center", [0.0, 0.0])

        return RotatingDipoleBackground2D(
            mu0=float(bg_cfg.get("mu0", 1.0)),
            omega=float(bg_cfg.get("omega", 0.2)),
            inclination=float(bg_cfg.get("inclination", 0.5)),
            phase0=float(bg_cfg.get("phase0", 0.0)),
            center=(float(center_raw[0]), float(center_raw[1])),
            plane=str(bg_cfg.get("plane", "xy")),
            plane_offset=float(bg_cfg.get("plane_offset", 0.0)),
            B_scale=float(bg_cfg.get("B_scale", 1.0)),
            softening_radius=float(bg_cfg.get("softening_radius", 2.0)),
            star_radius=float(bg_cfg.get("star_radius", 5.0)),
            include_induced_E=bool(bg_cfg.get("include_induced_E", True)),
            electric_scale=float(bg_cfg.get("electric_scale", 1.0)),
            light_cylinder_limit=bool(bg_cfg.get("light_cylinder_limit", True)),
            max_velocity=float(bg_cfg.get("max_velocity", 0.95)),
        )

    raise ValueError(f"Unknown background_em type: {bg_type!r}")