"""Create a clean source archive while excluding caches and generated outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "demo_output",
    "local_data",
    "research_output",
    "synthetic_output",
    "build",
    "dist",
    ".venv",
    "venv",
    ".idea",
    ".vscode",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if any(part.startswith(".venv") or part.endswith(".egg-info") for part in relative.parts):
                continue
            if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
                continue
            if path.suffix.lower() in {".pem", ".key", ".pyc", ".pyo"}:
                continue
            if not path.is_file() or EXCLUDED_PARTS.intersection(relative.parts):
                continue
            if path.resolve() == output or path.suffix.lower() == ".zip":
                continue
            archive.write(path, Path(root.name) / relative)
    print(output)


if __name__ == "__main__":
    main()
