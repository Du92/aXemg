# Canonical examples

This file is generated from `examples/canonical_cases.py`.

## Quick commands

List all cases:

```bash
python examples/run_canonical.py --list
```

Run a case:

```bash
python examples/run_canonical.py --case curved_axion_maxwell_2d
```

Run and animate:

```bash
python examples/run_canonical.py --case curved_axion_maxwell_2d --animate
```

## Cases

| Name | Dimension | Category | Config | Description |
|---|---:|---|---|---|
| `flat_axion_1d` | 1D | baseline | `config/canonical/flat_axion_1d.yaml` | Free massive axion evolution in flat 1D spacetime. |
| `flat_axion_maxwell_1d` | 1D | baseline | `config/canonical/flat_axion_maxwell_1d.yaml` | Coupled axion-Maxwell evolution in flat 1D spacetime. |
| `gw_axion_halo_1d` | 1D | gw | `config/canonical/gw_axion_halo_1d.yaml` | Gravitational wave crossing a magnetized axion halo in 1D. |
| `schwarzschild_axion_1d` | 1D | compact_object | `config/canonical/schwarzschild_axion_1d.yaml` | Axion evolution on a fixed isotropic Schwarzschild background. |
| `rotating_dipole_axion_2d` | 2D | compact_object | `config/canonical/rotating_dipole_axion_2d.yaml` | Axion cloud driven by a prescribed rotating dipole background. |
| `curved_axion_2d` | 2D | curved | `config/canonical/curved_axion_2d.yaml` | Axion evolution on a smooth compact-object metric using curved 2D operators. |
| `curved_axion_maxwell_2d` | 2D | curved | `config/canonical/curved_axion_maxwell_2d.yaml` | Coupled axion-Maxwell evolution on a fixed compact-object metric. |
| `curved_constraint_cleaned_2d` | 2D | constraints | `config/canonical/curved_constraint_cleaned_2d.yaml` | Curved 2D axion-Maxwell case with metric-compatible electric constraint cleaning. |

## Notes

- 1D cases are usually best visualized using spacetime maps.
- 2D cases are best visualized using multipanel animations.
- Curved 2D cases use fixed background metrics, not dynamical Einstein evolution.
- Rotating dipole examples use prescribed electromagnetic backgrounds unless otherwise specified.
