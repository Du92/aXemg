"""
Command-line entry point for plotting sweep summaries.

Installed command:

    axemg-sweep-plot \
        --csv outputs/sweep_compactness_2d/sweep_summary.csv \
        --x param.geometry.compactness \
        --y axion_energy
"""

from __future__ import annotations

from examples.plot_sweep_summary import main


if __name__ == "__main__":
    main()