"""
Output manager for simulations launched from YAML config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from axion_em_gr.io.snapshot import save_history_snapshots
from axion_em_gr.physics.constraints import electric_constraint, magnetic_constraint
from axion_em_gr.visualization.diagnostics_2d import make_full_2d_diagnostic_report
from axion_em_gr.visualization.plots_1d import (
    plot_EdotB_profile,
    plot_axion_state,
    plot_constraints_1d,
    plot_coupling_history,
    plot_em_state,
    plot_gw_metric_1d,
    plot_history,
)
from axion_em_gr.visualization.plots_2d import (
    plot_EdotB_2d,
    plot_axion_state_2d,
    plot_constraint_heatmaps_2d,
    plot_em_summary_2d,
)


def get_output_dir(config: dict[str, Any]) -> Path:
    """
    Get and create output directory from config.

    YAML block:

        output:
          directory: outputs/my_run
    """
    output_cfg = config.get("output", {})
    output_dir = Path(output_cfg.get("directory", "outputs/run_from_config"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_basic_outputs(
    config: dict[str, Any],
    grid,
    state0,
    final_state,
    history,
    metric,
    potential,
    sources,
    numerics,
    physical,
) -> list[Path]:
    """
    Save standard plots and snapshots depending on grid dimensionality.
    """
    output_dir = get_output_dir(config)
    paths: list[Path] = []

    geom0 = metric.evaluate(0.0, grid)
    geom_final = metric.evaluate(numerics.t_final, grid)

    if grid.ndim == 1:
        paths.append(
            plot_axion_state(
                grid,
                state0,
                title="Initial axion field",
                output_dir=output_dir,
                filename="initial_axion_state.png",
            )
        )

        paths.append(
            plot_axion_state(
                grid,
                final_state,
                title="Final axion field",
                output_dir=output_dir,
                filename="final_axion_state.png",
            )
        )

        if state0.E is not None and state0.B is not None:
            paths.append(
                plot_em_state(
                    grid,
                    state0,
                    title="Initial electromagnetic field",
                    output_dir=output_dir,
                    filename="initial_em_state.png",
                )
            )

            paths.append(
                plot_em_state(
                    grid,
                    final_state,
                    title="Final electromagnetic field",
                    output_dir=output_dir,
                    filename="final_em_state.png",
                )
            )

            paths.append(
                plot_EdotB_profile(
                    grid,
                    state0,
                    geom0,
                    output_dir=output_dir,
                    filename="initial_EdotB.png",
                    title=r"Initial $E_iB^i$",
                )
            )

            paths.append(
                plot_EdotB_profile(
                    grid,
                    final_state,
                    geom_final,
                    output_dir=output_dir,
                    filename="final_EdotB.png",
                    title=r"Final $E_iB^i$",
                )
            )

            div_B = magnetic_constraint(
                state=final_state,
                grid=grid,
                geom=geom_final,
                numerics=numerics,
            )

            div_E = electric_constraint(
                state=final_state,
                t=numerics.t_final,
                grid=grid,
                geom=geom_final,
                sources=sources,
                numerics=numerics,
                physical=physical,
                include_axion_coupling=config.get("evolution", {}).get(
                    "include_axion_em_coupling", False
                ),
            )

            paths.append(
                plot_constraints_1d(
                    grid=grid,
                    div_B=div_B,
                    div_E=div_E,
                    output_dir=output_dir,
                    filename="constraints_final.png",
                )
            )

        geometry_cfg = config.get("geometry", {})
        if geometry_cfg.get("type", "flat") == "gw_tt_1d":
            paths.append(
                plot_gw_metric_1d(
                    grid,
                    geom0,
                    output_dir=output_dir,
                    filename="gw_metric_initial.png",
                    title="Initial GW metric",
                )
            )

            paths.append(
                plot_gw_metric_1d(
                    grid,
                    geom_final,
                    output_dir=output_dir,
                    filename="gw_metric_final.png",
                    title="Final GW metric",
                )
            )

    elif grid.ndim == 2:
        paths.extend(
            plot_axion_state_2d(
                grid=grid,
                state=state0,
                output_dir=output_dir,
                prefix="initial_axion",
            )
        )

        paths.extend(
            plot_axion_state_2d(
                grid=grid,
                state=final_state,
                output_dir=output_dir,
                prefix="final_axion",
            )
        )

        if state0.E is not None and state0.B is not None:
            paths.extend(
                plot_em_summary_2d(
                    grid=grid,
                    state=state0,
                    output_dir=output_dir,
                    prefix="initial_em",
                )
            )

            paths.extend(
                plot_em_summary_2d(
                    grid=grid,
                    state=final_state,
                    output_dir=output_dir,
                    prefix="final_em",
                )
            )

            paths.append(
                plot_EdotB_2d(
                    grid=grid,
                    state=state0,
                    geom=geom0,
                    output_dir=output_dir,
                    filename="initial_EdotB.png",
                )
            )

            paths.append(
                plot_EdotB_2d(
                    grid=grid,
                    state=final_state,
                    geom=geom_final,
                    output_dir=output_dir,
                    filename="final_EdotB.png",
                )
            )

            div_B = magnetic_constraint(
                state=final_state,
                grid=grid,
                geom=geom_final,
                numerics=numerics,
            )

            div_E = electric_constraint(
                state=final_state,
                t=numerics.t_final,
                grid=grid,
                geom=geom_final,
                sources=sources,
                numerics=numerics,
                physical=physical,
                include_axion_coupling=config.get("evolution", {}).get(
                    "include_axion_em_coupling", False
                ),
            )

            paths.extend(
                plot_constraint_heatmaps_2d(
                    grid=grid,
                    div_B=div_B,
                    div_E=div_E,
                    output_dir=output_dir,
                    prefix="final_constraints",
                )
            )

        diagnostics_cfg = config.get("diagnostics", {})

        if diagnostics_cfg.get("full_2d_report", False):
            report_dir = output_dir / "final_diagnostics"
            report_dir.mkdir(parents=True, exist_ok=True)

            paths.extend(
                make_full_2d_diagnostic_report(
                    grid=grid,
                    state=final_state,
                    geom=geom_final,
                    potential=potential,
                    sources=sources,
                    numerics=numerics,
                    physical=physical,
                    time=numerics.t_final,
                    output_dir=report_dir,
                    prefix="final",
                    center=tuple(diagnostics_cfg.get("center", [0.0, 0.0])),
                )
            )

    else:
        raise NotImplementedError("Output manager currently supports 1D and 2D.")

    paths.append(
        plot_history(
            history,
            output_dir=output_dir,
            filename="history.png",
        )
    )

    if state0.E is not None and state0.B is not None:
        paths.append(
            plot_coupling_history(
                history,
                output_dir=output_dir,
                filename="coupling_history.png",
            )
        )

    snapshots_cfg = config.get("snapshots", {})

    if snapshots_cfg.get("save", False):
        snapshot_dir = output_dir / snapshots_cfg.get("directory", "snapshots")
        snapshot_paths = save_history_snapshots(
            history=history,
            output_dir=snapshot_dir,
            prefix=snapshots_cfg.get("prefix", "snapshot"),
        )
        paths.extend(snapshot_paths)

    return paths
