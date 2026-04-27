"""
Snapshot input/output utilities.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from axion_em_gr.core.state import State


def save_state_npz(
    state: State,
    path: str | Path,
    time: float,
) -> Path:
    """
    Save a State object to a compressed NumPy file.
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "time": np.array(time),
        "a": state.a,
        "Pi": state.Pi,
    }

    if state.E is not None:
        data["E"] = state.E

    if state.B is not None:
        data["B"] = state.B

    np.savez_compressed(save_path, **data)

    return save_path


def load_state_npz(path: str | Path) -> tuple[State, float]:
    """
    Load a State object from a compressed NumPy file.
    """
    data = np.load(path)

    a = data["a"]
    Pi = data["Pi"]

    E = data["E"] if "E" in data.files else None
    B = data["B"] if "B" in data.files else None

    time = float(data["time"])

    return State(a=a, Pi=Pi, E=E, B=B), time


def save_history_snapshots(
    history,
    output_dir: str | Path,
    prefix: str = "snapshot",
) -> list[Path]:
    """
    Save all snapshots stored in an EvolutionHistory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paths = []

    for i, (time, state) in enumerate(zip(history.snapshot_times, history.snapshots)):
        path = output_path / f"{prefix}_{i:04d}_t{time:.6f}.npz"
        paths.append(save_state_npz(state, path, time))

    return paths

def list_snapshot_files(
    snapshot_dir: str | Path,
    pattern: str = "*.npz",
) -> list[Path]:
    """
    Return sorted snapshot files from a directory.

    The default sorting is lexicographic, which works with files named like:

        snapshot_2d_0000_t0.000000.npz
        snapshot_2d_0001_t1.000000.npz
        ...
    """
    snapshot_path = Path(snapshot_dir)

    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot directory does not exist: {snapshot_path}")

    files = sorted(snapshot_path.glob(pattern))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No snapshot files matching pattern {pattern!r} in {snapshot_path}"
        )

    return files