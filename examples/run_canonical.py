"""
Run canonical examples by name.

Examples
--------
List cases:

    python examples/run_canonical.py --list

Run one case:

    python examples/run_canonical.py --case curved_axion_maxwell_2d

Run and animate:

    python examples/run_canonical.py --case curved_axion_maxwell_2d --animate

Run faster by overriding final time:

    python examples/run_canonical.py --case curved_axion_maxwell_2d --t-final 1.0
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from examples.canonical_cases import get_case, list_cases
from axion_em_gr.initial_data.factory import build_initial_state
from axion_em_gr.io.config_loader import build_simulation_objects, load_yaml_config
from axion_em_gr.visualization.scientific_animations import (
    animate_axion_em_summary_2d,
    ensure_output_dir,
    make_default_spacetime_maps_1d,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run canonical axion_em_gr examples.")

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available canonical cases.",
    )

    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Canonical case name.",
    )

    parser.add_argument(
        "--animate",
        action="store_true",
        help="Generate recommended animations after running.",
    )

    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=50,
        help="Snapshot interval used when --animate is active.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="Animation FPS.",
    )

    parser.add_argument(
        "--t-final",
        type=float,
        default=None,
        help="Override numerics.t_final.",
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Override numerics.dt.",
    )

    return parser.parse_args()


def print_cases() -> None:
    print("\nCanonical cases:\n")

    for case in list_cases():
        print(f"  {case.name}")
        print(f"    config:      {case.config_path}")
        print(f"    dimension:   {case.dimension}")
        print(f"    category:    {case.category}")
        print(f"    description: {case.description}")
        print()


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(config)

    cfg.setdefault("numerics", {})

    if args.t_final is not None:
        cfg["numerics"]["t_final"] = float(args.t_final)

    if args.dt is not None:
        cfg["numerics"]["dt"] = float(args.dt)

    return cfg


def run_case(case_name: str, args: argparse.Namespace) -> None:
    case = get_case(case_name)

    if not case.config_path.exists():
        raise FileNotFoundError(
            f"Canonical config not found: {case.config_path}\n"
            f"Create/copy it into config/canonical/ first."
        )

    config = load_yaml_config(case.config_path)
    config = apply_overrides(config, args)

    objects = build_simulation_objects(config)

    grid = objects["grid"]
    metric = objects["metric"]
    solver = objects["solver"]

    if args.animate:
        solver.save_snapshots = True
        solver.snapshot_every = args.snapshot_every

    state0 = build_initial_state(config, grid, metric=metric)

    print(f"\nRunning canonical case: {case.name}")
    print(f"Description: {case.description}")
    print(f"Config: {case.config_path}\n")

    final_state, history = solver.evolve(state0)

    print("\nRun completed.")

    if args.animate:
        output_dir_cfg = config.get("output", {}).get("directory", "outputs/from_config")
        anim_dir = ensure_output_dir(Path(output_dir_cfg) / "canonical_animations")

        if grid.ndim == 1:
            paths = make_default_spacetime_maps_1d(
                history=history,
                grid=grid,
                output_dir=anim_dir / "spacetime_maps",
                metric=metric,
            )
        elif grid.ndim == 2:
            paths = [
                animate_axion_em_summary_2d(
                    history=history,
                    grid=grid,
                    output_path=anim_dir / "axion_em_summary.mp4",
                    metric=metric,
                    overlay="lapse",
                    radius=_guess_radius(config),
                    fps=args.fps,
                    title=case.name,
                )
            ]
        else:
            paths = []

        print("\nCreated animation products:")
        for path in paths:
            print(path)


def _guess_radius(config: dict) -> float | None:
    """
    Try to infer a reasonable radius marker from the config.
    """
    geometry = config.get("geometry", {})
    background = config.get("background_em", {})

    if "star_radius" in background:
        return float(background["star_radius"])

    if "radius" in geometry:
        return float(geometry["radius"])

    return None


def main() -> None:
    args = parse_args()

    if args.list:
        print_cases()
        return

    if args.case is None:
        raise ValueError("Use --case CASE_NAME or --list.")

    run_case(args.case, args)


if __name__ == "__main__":
    main()
