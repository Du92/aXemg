# Examples

This document explains the main example families included in `axion_em_gr`.

---

## 1. Listing canonical cases

```bash
python examples/run_canonical.py --list

python examples/run_canonical.py --case flat_axion_1d
python examples/run_canonical.py \
  --case flat_axion_1d \
  --animate \
  --snapshot-every 50


python examples/run_canonical.py --case flat_axion_maxwell_1d



python examples/run_canonical.py --case gw_axion_halo_1d
python examples/run_from_config.py \
  --config config/canonical/gw_axion_halo_1d.yaml

python examples/animate_scientific_from_config.py \
  --config config/canonical/gw_axion_halo_1d.yaml \
  --snapshot-every 50 \
  --spacetime



python examples/run_canonical.py --case schwarzschild_axion_1d



python examples/run_canonical.py --case rotating_dipole_axion_2d
python examples/animate_compact_object_case.py \
  --config config/canonical/rotating_dipole_axion_2d.yaml \
  --snapshot-every 50 \
  --overlay lapse \
  --star-radius 6.0



python examples/run_canonical.py --case curved_axion_2d


python examples/run_canonical.py --case curved_axion_maxwell_2d
python examples/animate_scientific_from_config.py \
  --config config/canonical/curved_axion_maxwell_2d.yaml \
  --snapshot-every 50 \
  --overlay lapse \
  --radius 12.0


python examples/run_canonical.py --case curved_constraint_cleaned_2d




python examples/run_sweep_from_config.py \
  --config config/sweeps/sweep_compactness_2d.yaml \
  --no-plots



python examples/plot_sweep_summary.py \
  --csv outputs/sweep_compactness_2d/sweep_summary.csv \
  --x param.geometry.compactness \
  --y axion_energy
