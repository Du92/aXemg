"""
Utilities for modifying nested configuration dictionaries.

Useful for parameter sweeps where one wants to override entries such as:

    physics.g_agamma
    physics.m_axion
    initial_data.background_Bz
    geometry.h_plus_amplitude
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def get_nested(config: dict[str, Any], dotted_key: str) -> Any:
    """
    Get a nested value using a dotted key.

    Example
    -------
    get_nested(config, "physics.g_agamma")
    """
    keys = dotted_key.split(".")
    current = config

    for key in keys:
        current = current[key]

    return current


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> dict[str, Any]:
    """
    Set a nested value using a dotted key.

    This modifies the dictionary in place and also returns it.

    Example
    -------
    set_nested(config, "physics.g_agamma", 0.03)
    """
    keys = dotted_key.split(".")
    current = config

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value

    return config


def with_nested_override(
    config: dict[str, Any],
    dotted_key: str,
    value: Any,
) -> dict[str, Any]:
    """
    Return a deep copy of config with one nested value changed.
    """
    new_config = deepcopy(config)
    set_nested(new_config, dotted_key, value)
    return new_config


def with_nested_overrides(
    config: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """
    Return a deep copy of config with several nested values changed.

    Parameters
    ----------
    overrides:
        Dictionary like:
            {
                "physics.g_agamma": 0.03,
                "physics.m_axion": 0.2,
            }
    """
    new_config = deepcopy(config)

    for key, value in overrides.items():
        set_nested(new_config, key, value)

    return new_config