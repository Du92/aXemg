# User Manual

This manual explains how to use `axion_em_gr` to run modular simulations of axion electrodynamics in fixed 3+1 backgrounds.

---

## 1. Installation

Create a virtual environment:

```bash
python3 -m venv aXemg-env
source aXemg-env/bin/activate

pip install numpy matplotlib pyyaml pytest

Optional for animations

sudo apt install ffmpeg

Run cannonical examples 
python examples/run_canonical.py --list

