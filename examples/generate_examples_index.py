"""
Generate a Markdown index of canonical examples.

Run:
    python examples/generate_examples_index.py
"""

from __future__ import annotations

from pathlib import Path

from examples.canonical_cases import list_cases


def main() -> None:
    output_path = Path("EXAMPLES.md")

    lines = []
    lines.append("# Canonical examples")
    lines.append("")
    lines.append("This file is generated from `examples/canonical_cases.py`.")
    lines.append("")
    lines.append("## Quick commands")
    lines.append("")
    lines.append("List all cases:")
    lines.append("")
    lines.append("```bash")
    lines.append("python examples/run_canonical.py --list")
    lines.append("```")
    lines.append("")
    lines.append("Run a case:")
    lines.append("")
    lines.append("```bash")
    lines.append("python examples/run_canonical.py --case curved_axion_maxwell_2d")
    lines.append("```")
    lines.append("")
    lines.append("Run and animate:")
    lines.append("")
    lines.append("```bash")
    lines.append("python examples/run_canonical.py --case curved_axion_maxwell_2d --animate")
    lines.append("```")
    lines.append("")
    lines.append("## Cases")
    lines.append("")
    lines.append("| Name | Dimension | Category | Config | Description |")
    lines.append("|---|---:|---|---|---|")

    for case in list_cases():
        lines.append(
            f"| `{case.name}` | {case.dimension} | {case.category} | "
            f"`{case.config_path}` | {case.description} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- 1D cases are usually best visualized using spacetime maps.")
    lines.append("- 2D cases are best visualized using multipanel animations.")
    lines.append("- Curved 2D cases use fixed background metrics, not dynamical Einstein evolution.")
    lines.append("- Rotating dipole examples use prescribed electromagnetic backgrounds unless otherwise specified.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(output_path)


if __name__ == "__main__":
    main()
