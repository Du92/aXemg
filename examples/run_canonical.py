"""
Run canonical examples by name.

Examples
--------
List cases:

    python examples/run_canonical.py --list

Run one case:

    python examples/run_canonical.py --case curved_axion_maxwell_2d

Run a short version:

    python examples/run_canonical.py --case curved_axion_maxwell_2d --t-final 0.5

Run and animate:

    python examples/run_canonical.py --case curved_axion_maxwell_2d --animate

Show output directory only:

    python examples/run_canonical.py --case curved_axion_maxwell_2d --show-output

Dry run:

    python examples/run_canonical.py --case curved_axion_maxwell_2d --dry-run
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from datetime import datetime

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

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config and build objects, but do not evolve.",
    )

    parser.add_argument(
        "--show-output",
        action="store_true",
        help="Only print the output directory for the selected case.",
    )

    return parser.parse_args()


def print_cases() -> None:
    print("\nCanonical cases:\n")

    for case in list_cases():
        exists = "yes" if case.config_path.exists() else "NO"
        print(f"  {case.name}")
        print(f"    config:      {case.config_path}")
        print(f"    exists:      {exists}")
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


def get_output_dir(config: dict) -> Path:
    output_cfg = config.get("output", {})
    output_dir = output_cfg.get("directory", "outputs/from_canonical")
    return Path(output_dir)


def write_run_summary(
    output_dir: Path,
    case_name: str,
    config_path: Path,
    config: dict,
    success: bool,
    error_message: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    numerics = config.get("numerics", {})
    physics = config.get("physics", {})
    geometry = config.get("geometry", {})
    evolution = config.get("evolution", {})

    path = output_dir / "run_summary.txt"

    lines = []
    lines.append("axion_em_gr canonical run summary")
    lines.append("=================================")
    lines.append("")
    lines.append(f"case:        {case_name}")
    lines.append(f"config:      {config_path}")
    lines.append(f"timestamp:   {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"success:     {success}")
    lines.append("")
    lines.append("[numerics]")
    lines.append(f"dt:          {numerics.get('dt')}")
    lines.append(f"t_final:     {numerics.get('t_final')}")
    lines.append(f"output_every:{numerics.get('output_every')}")
    lines.append("")
    lines.append("[physics]")
    lines.append(f"m_axion:     {physics.get('m_axion')}")
    lines.append(f"g_agamma:    {physics.get('g_agamma')}")
    lines.append("")
    lines.append("[geometry]")
    lines.append(f"type:        {geometry.get('type')}")
    lines.append("")
    lines.append("[evolution]")
    lines.append(f"evolve_axion:              {evolution.get('evolve_axion')}")
    lines.append(f"evolve_maxwell:            {evolution.get('evolve_maxwell')}")
    lines.append(f"include_axion_em_coupling: {evolution.get('include_axion_em_coupling')}")
    lines.append("")

    if error_message is not None:
        lines.append("[error]")
        lines.append(error_message)
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")

    return path


def try_write_history_plot(history, output_dir: Path) -> Path | None:
    """
    Try to save a basic history plot if plot_history is available.
    """
    try:
        from axion_em_gr.visualization.plots_1d import plot_history

        path = plot_history(
            history,
            output_dir=output_dir,
            filename="history.png",
        )
        return Path(path)

    except Exception as exc:
        print(f"History plot skipped: {exc}")
        return None


def _guess_radius(config: dict) -> float | None:
    geometry = config.get("geometry", {})
    background = config.get("background_em", {})

    if "star_radius" in background:
        return float(background["star_radius"])

    if "radius" in geometry:
        return float(geometry["radius"])

    return None


def run_case(case_name: str, args: argparse.Namespace) -> None:
    case = get_case(case_name)

    if not case.config_path.exists():
        raise FileNotFoundError(
            f"Canonical config not found: {case.config_path}\n"
            f"Create/copy it into config/canonical/ first."
        )

    config = load_yaml_config(case.config_path)
    config = apply_overrides(config, args)

    output_dir = get_output_dir(config)

    if args.show_output:
        print(output_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning canonical case: {case.name}")
    print(f"Description: {case.description}")
    print(f"Config:      {case.config_path}")
    print(f"Output dir:  {output_dir}\n")

    objects = build_simulation_objects(config)

    grid = objects["grid"]
    metric = objects["metric"]
    solver = objects["solver"]

    if args.animate:
        solver.save_snapshots = True
        solver.snapshot_every = args.snapshot_every

    state0 = build_initial_state(config, grid, metric=metric)

    if args.dry_run:
        summary = write_run_summary(
            output_dir=output_dir,
            case_name=case.name,
            config_path=case.config_path,
            config=config,
            success=True,
            error_message="Dry run only. No evolution performed.",
        )
        print("Dry run completed.")
        print(f"Summary written to: {summary}")
        return

    try:
        final_state, history = solver.evolve(state0)

    except Exception as exc:
        summary = write_run_summary(
            output_dir=output_dir,
            case_name=case.name,
            config_path=case.config_path,
            config=config,
            success=False,
            error_message=repr(exc),
        )

        print("\nRun failed.")
        print(f"Output dir: {output_dir}")
        print(f"Summary written to: {summary}")
        raise

    summary = write_run_summary(
        output_dir=output_dir,
        case_name=case.name,
        config_path=case.config_path,
        config=config,
        success=True,
    )

    print("\nRun completed.")
    print(f"Output dir: {output_dir}")
    print(f"Summary written to: {summary}")

    history_plot = try_write_history_plot(history, output_dir)

    if history_plot is not None:
        print(f"History plot written to: {history_plot}")

    if args.animate:
        anim_dir = ensure_output_dir(output_dir / "canonical_animations")

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