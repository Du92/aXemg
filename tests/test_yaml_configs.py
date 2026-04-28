"""
YAML configuration smoke tests.

These tests only verify that important YAML files can be loaded and that
simulation objects can be built.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from axion_em_gr.io.config_loader import load_yaml_config, build_simulation_objects


CONFIGS_TO_TEST = [
    "config/canonical/flat_axion_1d.yaml",
    "config/canonical/flat_axion_maxwell_1d.yaml",
    "config/canonical/gw_axion_halo_1d.yaml",
    "config/canonical/schwarzschild_axion_1d.yaml",
    "config/canonical/rotating_dipole_axion_2d.yaml",
    "config/canonical/curved_axion_2d.yaml",
    "config/canonical/curved_axion_maxwell_2d.yaml",
    "config/canonical/curved_constraint_cleaned_2d.yaml",
]


@pytest.mark.parametrize("config_path", CONFIGS_TO_TEST)
def test_yaml_config_builds_objects(config_path):
    path = Path(config_path)

    if not path.exists():
        pytest.skip(f"Config file not present yet: {config_path}")

    config = load_yaml_config(path)
    objects = build_simulation_objects(config)

    assert "grid" in objects
    assert "metric" in objects
    assert "solver" in objects
    assert "numerics" in objects
    assert "physical" in objects