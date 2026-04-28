# Architecture

This document describes the modular structure of `axion_em_gr`.

---

## 1. Package layout

```text
axion_em_gr/
├── core/
├── geometry/
├── physics/
├── initial_data/
├── solvers/
├── io/
├── visualization/
├── examples/
├── config/
├── tests/
└── docs/

core/
├── grid.py
├── state.py
├── parameters.py
├── rhs.py
├── derivatives.py
├── covariant_derivatives.py
├── boundary.py
├── tensors.py
├── levi_civita.py
└── integrators.py

geometry/
├── base_metric.py
├── flat.py
├── gw_tt.py
└── schwarzschild_like.py

metric.evaluate(t, grid)
lapse
shift
gamma_down
gamma_up
sqrt_gamma
K

physics/
├── axion.py
├── maxwell.py
├── potentials.py
├── sources.py
├── constraints.py
├── constraint_cleaning.py
├── curved_poisson.py
├── background_em.py
├── diagnostics.py
└── comparison.py

initial_data/
├── axion_profiles.py
├── electromagnetic_profiles.py
├── combined_setups.py
├── combined_setups_2d.py
├── ns_scenarios_2d.py
├── physical_scenarios_1d.py
└── factory.py

solvers/
├── evolution.py
└── rk4.py

io/
├── config_loader.py
├── metrics.py
├── sweep.py
├── output.py
└── checkpoint.py

visualization/
├── plots_1d.py
├── plots_2d.py
├── diagnostics_2d.py
├── curved_diagnostics_2d.py
├── geometry_plots.py
├── background_em_plots.py
├── sweep_plots.py
├── animations.py
└── scientific_animations.py


examples/run_from_config.py
examples/run_canonical.py
examples/run_sweep_from_config.py
examples/plot_sweep_summary.py
examples/animate_from_config.py
examples/animate_scientific_from_config.py
examples/validate_canonical_configs.py

python examples/run_canonical.py --list
python examples/run_canonical.py --case curved_axion_maxwell_2d
python examples/animate_scientific_from_config.py --config config/canonical/curved_axion_maxwell_2d.yaml