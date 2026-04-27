"""
Specialized animation script for compact-object / rotating-dipole 2D cases.

It creates:
    - axion_em_summary.mp4
    - axion_summary.mp4
    - custom_EdotB_B_a.mp4

Examples
--------
    python examples/animate_compact_object_case.py \
        --config config/curved_2d_rotating_dipole.yaml \
        --snapshot-every 50 \
        --star-radius 6.0

    python examples/animate_compact_object_case.py \
        --config config/curved_axion_maxwell_2d.yaml \
        --snapshot-every 50 \
        --star-radius 12.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from axion_em_gr.initial_data.factory import build_initial_state
from axion_em_gr.io.config_loader import build_simulation_objects, load_yaml_config
from axion_em_gr.visualization.scientific_animations import (
    animate_axion_em_summary_2d,
    animate_axion_summary_2d,
    animate_multipanel_2d,
    ensure_output_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate compact-object 2D simulation."
    )

    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to YAML configuration.",
    )

    parser.add_argument(
        "--snapshot-every",
        default=50,
        type=int,
        help="Snapshot interval.",
    )

    parser.add_argument(
        "--fps",
        default=12,
        type=int,
        help="Frames per second.",
    )

    parser.add_argument(
        "--overlay",
        default="lapse",
        choices=["none", "lapse", "sqrt_gamma"],
        help="Geometry overlay.",
    )

    parser.add_argument(
        "--star-radius",
        default=None,
        type=float,
        help="Radius of compact object / star marker.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_yaml_config(args.config)
    objects = build_simulation_objects(config)

    grid = objects["grid"]
    metric = objects["metric"]
    solver = objects["solver"]

    state0 = build_initial_state(config, grid, metric=metric)

    solver.save_snapshots = True
    solver.snapshot_every = args.snapshot_every

    final_state, history = solver.evolve(state0)

    output_dir_cfg = config.get("output", {}).get("directory", "outputs/from_config")
    run_name = config.get("run", {}).get("name", "compact_object_case")

    output_dir = ensure_output_dir(Path(output_dir_cfg) / "compact_object_animations")

    overlay = None if args.overlay == "none" else args.overlay

    paths = []

    paths.append(
        animate_axion_em_summary_2d(
            history=history,
            grid=grid,
            output_path=output_dir / "axion_em_summary.mp4",
            metric=metric,
            overlay=overlay,
            radius=args.star_radius,
            radius_center=(0.0, 0.0),
            fps=args.fps,
            title=f"{run_name}: axion--EM summary",
        )
    )

    paths.append(
        animate_axion_summary_2d(
            history=history,
            grid=grid,
            output_path=output_dir / "axion_summary.mp4",
            metric=metric,
            overlay=overlay,
            radius=args.star_radius,
            radius_center=(0.0, 0.0),
            fps=args.fps,
            title=f"{run_name}: axion response",
        )
    )

    paths.append(
        animate_multipanel_2d(
            history=history,
            grid=grid,
            quantities=["EdotB", "a", "B", "Pi"],
            output_path=output_dir / "custom_EdotB_a_B_Pi.mp4",
            metric=metric,
            overlay=overlay,
            radius=args.star_radius,
            radius_center=(0.0, 0.0),
            fps=args.fps,
            title=f"{run_name}: source and response",
        )
    )

    print("\nCreated compact-object animations:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()