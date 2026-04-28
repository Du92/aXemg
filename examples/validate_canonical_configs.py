"""
Validate canonical YAML configurations.

This script checks that all canonical YAML files:
    - exist
    - load correctly
    - build simulation objects
    - build initial state

It does not run full evolutions.

Run:
    python examples/validate_canonical_configs.py
"""

from __future__ import annotations

from examples.canonical_cases import list_cases
from axion_em_gr.initial_data.factory import build_initial_state
from axion_em_gr.io.config_loader import build_simulation_objects, load_yaml_config


def main() -> None:
    failed = []

    for case in list_cases():
        print(f"\nChecking {case.name}")
        print(f"  config: {case.config_path}")

        try:
            if not case.config_path.exists():
                raise FileNotFoundError(case.config_path)

            config = load_yaml_config(case.config_path)
            objects = build_simulation_objects(config)

            grid = objects["grid"]
            metric = objects["metric"]

            state0 = build_initial_state(config, grid, metric=metric)

            print("  OK")
            print(f"  ndim: {grid.ndim}")
            print(f"  shape: {grid.shape}")

        except Exception as exc:
            print(f"  FAILED: {exc}")
            failed.append((case.name, exc))

    print("\nSummary")
    print("-------")

    if not failed:
        print("All canonical configs validated successfully.")
        return

    print(f"{len(failed)} canonical configs failed:")

    for name, exc in failed:
        print(f"  {name}: {exc}")

    raise SystemExit(1)


if __name__ == "__main__":
    main()
