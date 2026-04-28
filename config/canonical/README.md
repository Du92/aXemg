# Canonical configuration files

This directory contains stable, reproducible YAML configuration files for the
main physical scenarios implemented in `axion_em_gr`.

## Available cases

| File | Description |
|---|---|
| `flat_axion_1d.yaml` | Free massive axion/Klein-Gordon evolution in flat 1D spacetime. |
| `flat_axion_maxwell_1d.yaml` | Coupled axion-Maxwell evolution in flat 1D spacetime. |
| `gw_axion_halo_1d.yaml` | Gravitational wave crossing a magnetized axion halo in 1D. |
| `schwarzschild_axion_1d.yaml` | Axion evolution on a fixed isotropic Schwarzschild background. |
| `rotating_dipole_axion_2d.yaml` | Axion cloud driven by a prescribed rotating magnetic dipole in 2D. |
| `curved_axion_2d.yaml` | Axion evolution on a smooth compact-object metric using curved 2D operators. |
| `curved_axion_maxwell_2d.yaml` | Coupled axion-Maxwell evolution on a fixed compact-object metric. |
| `curved_constraint_cleaned_2d.yaml` | Curved 2D axion-Maxwell initial data with metric-compatible electric constraint cleaning. |

## Running a case

From the project root:

```bash
python examples/run_from_config.py --config config/canonical/curved_axion_maxwell_2d.yaml
