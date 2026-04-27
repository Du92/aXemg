"""
Parameter sweep utilities.

A sweep config is a normal simulation YAML plus a 'sweep' block.

Example
-------
sweep:
  mode: grid
  parameters:
    physics.g_agamma: [0.0, 0.01, 0.03]
    physics.m_axion: [0.1, 0.2]
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import csv
import traceback

from axion_em_gr.initial_data.factory import build_initial_state
from axion_em_gr.io.config_loader import build_simulation_objects
from axion_em_gr.io.config_utils import with_nested_overrides
from axion_em_gr.io.metrics import compute_final_metrics, metrics_to_dict
from axion_em_gr.io.output_manager import save_basic_outputs


@dataclass
class SweepCase:
    """
    One parameter-sweep case.
    """

    run_id: str
    overrides: dict[str, Any]
    config: dict[str, Any]
    output_dir: Path


def _format_value_for_id(value: Any) -> str:
    """
    Convert a value into a filesystem-safe string.
    """
    if isinstance(value, float):
        text = f"{value:.6g}"
    else:
        text = str(value)

    text = text.replace("-", "m")
    text = text.replace("+", "p")
    text = text.replace(".", "p")
    text = text.replace("/", "_")
    text = text.replace(" ", "")

    return text


def make_run_id(index: int, overrides: dict[str, Any]) -> str:
    """
    Create a short run id from parameter overrides.
    """
    pieces = [f"case{index:04d}"]

    for key, value in overrides.items():
        short_key = key.replace(".", "_")
        pieces.append(f"{short_key}_{_format_value_for_id(value)}")

    return "__".join(pieces)


def generate_grid_sweep_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Generate parameter combinations using Cartesian product.

    Expected YAML:

        sweep:
          mode: grid
          parameters:
            physics.g_agamma: [0.0, 0.01, 0.03]
            physics.m_axion: [0.1, 0.2]
    """
    sweep_cfg = config.get("sweep", {})
    params = sweep_cfg.get("parameters", {})

    if not params:
        raise ValueError("Sweep has no parameters.")

    keys = list(params.keys())
    value_lists = [params[key] for key in keys]

    cases = []

    for values in product(*value_lists):
        overrides = dict(zip(keys, values))
        cases.append(overrides)

    return cases


def generate_list_sweep_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Generate cases from an explicit list.

    Expected YAML:

        sweep:
          mode: list
          cases:
            - physics.g_agamma: 0.01
              physics.m_axion: 0.1
            - physics.g_agamma: 0.03
              physics.m_axion: 0.2
    """
    sweep_cfg = config.get("sweep", {})
    cases = sweep_cfg.get("cases", [])

    if not cases:
        raise ValueError("Sweep mode 'list' requires non-empty sweep.cases.")

    return [dict(case) for case in cases]


def generate_sweep_overrides(config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Generate override dictionaries for all sweep cases.
    """
    sweep_cfg = config.get("sweep", {})
    mode = sweep_cfg.get("mode", "grid")

    if mode == "grid":
        return generate_grid_sweep_cases(config)

    if mode == "list":
        return generate_list_sweep_cases(config)

    raise ValueError(f"Unknown sweep mode: {mode!r}")


def build_sweep_cases(config: dict[str, Any]) -> list[SweepCase]:
    """
    Build SweepCase objects from one sweep config.
    """
    base_output_dir = Path(
        config.get("sweep", {}).get(
            "output_directory",
            config.get("output", {}).get("directory", "outputs/sweep"),
        )
    )

    base_output_dir.mkdir(parents=True, exist_ok=True)

    overrides_list = generate_sweep_overrides(config)

    cases: list[SweepCase] = []

    for index, overrides in enumerate(overrides_list):
        run_id = make_run_id(index, overrides)

        case_config = with_nested_overrides(config, overrides)

        case_output_dir = base_output_dir / run_id
        case_output_dir.mkdir(parents=True, exist_ok=True)

        if "output" not in case_config:
            case_config["output"] = {}

        case_config["output"]["directory"] = str(case_output_dir)

        cases.append(
            SweepCase(
                run_id=run_id,
                overrides=overrides,
                config=case_config,
                output_dir=case_output_dir,
            )
        )

    return cases


def write_sweep_summary_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
) -> Path:
    """
    Write sweep metrics to CSV.
    """
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("Cannot write empty sweep summary.")

    fieldnames = list(rows[0].keys())

    # Include any fields that appear in later rows.
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    return csv_path


def run_single_sweep_case(
    case: SweepCase,
    save_outputs: bool = True,
) -> dict[str, Any]:
    """
    Run one sweep case and return a metrics row.
    """
    try:
        objects = build_simulation_objects(case.config)

        grid = objects["grid"]
        physical = objects["physical"]
        numerics = objects["numerics"]
        metric = objects["metric"]
        potential = objects["potential"]
        sources = objects["sources"]
        solver = objects["solver"]

        state0 = build_initial_state(case.config, grid, metric=metric)

        final_state, history = solver.evolve(state0)

        geom_final = metric.evaluate(numerics.t_final, grid)

        include_axion_coupling = bool(
            case.config.get("evolution", {}).get("include_axion_em_coupling", False)
        )

        metrics = compute_final_metrics(
            run_id=case.run_id,
            status="ok",
            state=final_state,
            grid=grid,
            geom=geom_final,
            potential=potential,
            sources=sources,
            numerics=numerics,
            physical=physical,
            include_axion_coupling=include_axion_coupling,
        )

        row = metrics_to_dict(metrics)

        for key, value in case.overrides.items():
            row[f"param.{key}"] = value

        row["output_dir"] = str(case.output_dir)

        if save_outputs:
            save_basic_outputs(
                config=case.config,
                grid=grid,
                state0=state0,
                final_state=final_state,
                history=history,
                metric=metric,
                potential=potential,
                sources=sources,
                numerics=numerics,
                physical=physical,
            )

        return row

    except Exception as exc:
        error_path = case.output_dir / "error.txt"

        with error_path.open("w", encoding="utf-8") as f:
            f.write(f"Run failed: {case.run_id}\n")
            f.write(str(exc))
            f.write("\n\n")
            f.write(traceback.format_exc())

        row = {
            "run_id": case.run_id,
            "status": "failed",
            "error": str(exc),
            "output_dir": str(case.output_dir),
        }

        for key, value in case.overrides.items():
            row[f"param.{key}"] = value

        return row


def run_sweep(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], Path]:
    """
    Run all sweep cases.

    Returns
    -------
    rows:
        List of metrics rows.
    summary_path:
        Path to CSV summary.
    """
    sweep_cfg = config.get("sweep", {})

    save_outputs = bool(sweep_cfg.get("save_outputs", True))

    cases = build_sweep_cases(config)

    rows: list[dict[str, Any]] = []

    for i, case in enumerate(cases):
        print("\n" + "=" * 80)
        print(f"Running sweep case {i + 1}/{len(cases)}")
        print(f"run_id: {case.run_id}")
        print("overrides:")
        for key, value in case.overrides.items():
            print(f"  {key}: {value}")
        print("=" * 80 + "\n")

        row = run_single_sweep_case(
            case=case,
            save_outputs=save_outputs,
        )

        rows.append(row)

    base_output_dir = Path(
        sweep_cfg.get(
            "output_directory",
            config.get("output", {}).get("directory", "outputs/sweep"),
        )
    )

    summary_path = write_sweep_summary_csv(
        rows=rows,
        path=base_output_dir / "sweep_summary.csv",
    )

    return rows, summary_path