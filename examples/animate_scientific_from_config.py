"""
Run a YAML-configured simulation and generate scientific animations.

Examples
--------
Curved 2D axion-Maxwell:

    python examples/animate_scientific_from_config.py \
        --config config/curved_axion_maxwell_2d.yaml \
        --snapshot-every 50 \
        --overlay lapse \
        --radius 12.0

Curved 2D rotating dipole:

    python examples/animate_scientific_from_config.py \
        --config config/curved_2d_rotating_dipole.yaml \
        --snapshot-every 50 \
        --overlay sqrt_gamma \
        --radius 6.0

1D GW halo:

    python examples/animate_scientific_from_config.py \
        --config config/gw_axion_halo_1d.yaml \
        --snapshot-every 50 \
        --spacetime
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
    make_default_spacetime_maps_1d,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate scientific animations from a YAML simulation."
    )

    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to YAML configuration file.",
    )

    parser.add_argument(
        "--fps",
        default=12,
        type=int,
        help="Frames per second for mp4/gif animations.",
    )

    parser.add_argument(
        "--snapshot-every",
        default=None,
        type=int,
        help="Override snapshot interval.",
    )

    parser.add_argument(
        "--overlay",
        default="none",
        choices=["none", "lapse", "sqrt_gamma"],
        help="Static geometry overlay for 2D animations.",
    )

    parser.add_argument(
        "--radius",
        default=None,
        type=float,
        help="Optional circle radius to draw, e.g. star radius.",
    )

    parser.add_argument(
        "--radius-center",
        default="0.0,0.0",
        type=str,
        help="Circle center as x,y.",
    )

    parser.add_argument(
        "--mode",
        default="em",
        choices=["em", "axion", "custom"],
        help=(
            "Animation mode. "
            "'em' -> a, EdotB, |E|, |B|. "
            "'axion' -> a, Pi, EdotB, |B|. "
            "'custom' uses --quantities."
        ),
    )

    parser.add_argument(
        "--quantities",
        default="a,EdotB,E,B",
        type=str,
        help="Comma-separated quantities for custom mode.",
    )

    parser.add_argument(
        "--spacetime",
        action="store_true",
        help="For 1D cases, also generate static spacetime maps.",
    )

    return parser.parse_args()


def _parse_center(text: str) -> tuple[float, float]:
    parts = text.split(",")

    if len(parts) != 2:
        raise ValueError("--radius-center must have form x,y")

    return float(parts[0]), float(parts[1])


def main() -> None:
    args = parse_args()

    config = load_yaml_config(args.config)
    objects = build_simulation_objects(config)

    grid = objects["grid"]
    metric = objects["metric"]
    solver = objects["solver"]

    state0 = build_initial_state(config, grid, metric=metric)

    # Force snapshots for animation.
    solver.save_snapshots = True

    if args.snapshot_every is not None:
        solver.snapshot_every = args.snapshot_every
    elif getattr(solver, "snapshot_every", None) is None:
        solver.snapshot_every = max(1, objects["numerics"].output_every)

    final_state, history = solver.evolve(state0)

    output_dir_cfg = config.get("output", {}).get("directory", "outputs/from_config")
    run_name = config.get("run", {}).get("name", "run")

    output_dir = ensure_output_dir(Path(output_dir_cfg) / "scientific_animations")

    overlay = None if args.overlay == "none" else args.overlay
    radius_center = _parse_center(args.radius_center)

    print(f"\nGenerating scientific animations for run: {run_name}")

    paths = []

    if grid.ndim == 1:
        if args.spacetime:
            paths.extend(
                make_default_spacetime_maps_1d(
                    history=history,
                    grid=grid,
                    output_dir=output_dir / "spacetime_maps",
                    metric=metric,
                )
            )
        else:
            print(
                "This script mainly produces improved 2D multipanel animations. "
                "For 1D, use --spacetime to generate spacetime maps."
            )

    elif grid.ndim == 2:
        if args.mode == "em":
            paths.append(
                animate_axion_em_summary_2d(
                    history=history,
                    grid=grid,
                    output_path=output_dir / "axion_em_summary.mp4",
                    metric=metric,
                    overlay=overlay,
                    radius=args.radius,
                    radius_center=radius_center,
                    fps=args.fps,
                    title=run_name,
                )
            )

        elif args.mode == "axion":
            paths.append(
                animate_axion_summary_2d(
                    history=history,
                    grid=grid,
                    output_path=output_dir / "axion_summary.mp4",
                    metric=metric,
                    overlay=overlay,
                    radius=args.radius,
                    radius_center=radius_center,
                    fps=args.fps,
                    title=run_name,
                )
            )

        elif args.mode == "custom":
            quantities = [q.strip() for q in args.quantities.split(",") if q.strip()]

            paths.append(
                animate_multipanel_2d(
                    history=history,
                    grid=grid,
                    quantities=quantities,
                    output_path=output_dir / "custom_summary.mp4",
                    metric=metric,
                    overlay=overlay,
                    radius=args.radius,
                    radius_center=radius_center,
                    fps=args.fps,
                    title=run_name,
                )
            )

    else:
        raise NotImplementedError("Only 1D and 2D animations are supported.")

    print("\nCreated:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()