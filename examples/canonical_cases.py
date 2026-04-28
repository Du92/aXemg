"""
Registry of canonical examples.

This file maps short case names to YAML configuration files and metadata.
It is used by examples/run_canonical.py and documentation scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CanonicalCase:
    name: str
    config_path: Path
    description: str
    dimension: str
    category: str
    recommended_animation: str | None = None


CANONICAL_CASES: dict[str, CanonicalCase] = {
    "flat_axion_1d": CanonicalCase(
        name="flat_axion_1d",
        config_path=Path("config/canonical/flat_axion_1d.yaml"),
        description="Free massive axion evolution in flat 1D spacetime.",
        dimension="1D",
        category="baseline",
        recommended_animation="spacetime",
    ),
    "flat_axion_maxwell_1d": CanonicalCase(
        name="flat_axion_maxwell_1d",
        config_path=Path("config/canonical/flat_axion_maxwell_1d.yaml"),
        description="Coupled axion-Maxwell evolution in flat 1D spacetime.",
        dimension="1D",
        category="baseline",
        recommended_animation="spacetime",
    ),
    "gw_axion_halo_1d": CanonicalCase(
        name="gw_axion_halo_1d",
        config_path=Path("config/canonical/gw_axion_halo_1d.yaml"),
        description="Gravitational wave crossing a magnetized axion halo in 1D.",
        dimension="1D",
        category="gw",
        recommended_animation="spacetime",
    ),
    "schwarzschild_axion_1d": CanonicalCase(
        name="schwarzschild_axion_1d",
        config_path=Path("config/canonical/schwarzschild_axion_1d.yaml"),
        description="Axion evolution on a fixed isotropic Schwarzschild background.",
        dimension="1D",
        category="compact_object",
        recommended_animation="spacetime",
    ),
    "rotating_dipole_axion_2d": CanonicalCase(
        name="rotating_dipole_axion_2d",
        config_path=Path("config/canonical/rotating_dipole_axion_2d.yaml"),
        description="Axion cloud driven by a prescribed rotating dipole background.",
        dimension="2D",
        category="compact_object",
        recommended_animation="multipanel",
    ),
    "curved_axion_2d": CanonicalCase(
        name="curved_axion_2d",
        config_path=Path("config/canonical/curved_axion_2d.yaml"),
        description="Axion evolution on a smooth compact-object metric using curved 2D operators.",
        dimension="2D",
        category="curved",
        recommended_animation="multipanel",
    ),
    "curved_axion_maxwell_2d": CanonicalCase(
        name="curved_axion_maxwell_2d",
        config_path=Path("config/canonical/curved_axion_maxwell_2d.yaml"),
        description="Coupled axion-Maxwell evolution on a fixed compact-object metric.",
        dimension="2D",
        category="curved",
        recommended_animation="multipanel",
    ),
    "curved_constraint_cleaned_2d": CanonicalCase(
        name="curved_constraint_cleaned_2d",
        config_path=Path("config/canonical/curved_constraint_cleaned_2d.yaml"),
        description="Curved 2D axion-Maxwell case with metric-compatible electric constraint cleaning.",
        dimension="2D",
        category="constraints",
        recommended_animation="multipanel",
    ),
}


def list_cases() -> list[CanonicalCase]:
    return list(CANONICAL_CASES.values())


def get_case(name: str) -> CanonicalCase:
    try:
        return CANONICAL_CASES[name]
    except KeyError as exc:
        available = ", ".join(sorted(CANONICAL_CASES))
        raise KeyError(
            f"Unknown canonical case {name!r}. Available cases: {available}"
        ) from exc
