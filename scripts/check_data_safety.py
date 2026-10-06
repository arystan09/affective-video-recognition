"""Reject obvious private dataset artifacts in Git-tracked content."""

import argparse
import subprocess
from pathlib import Path


def violations(root: Path, paths: list[str]) -> list[str]:
    errors = []
    for name in paths:
        path = Path(name)
        private = (
            name.startswith(("data/", "reports/figures/dataset/", "reports/dataset/", "private/"))
            and name != "data/README.md"
            and path.name != ".gitkeep"
        )
        if private or path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".mpg", ".arff"):
            errors.append(f"restricted data path: {name}")
        if (root / path).is_file() and (root / path).stat().st_size > 10 * 1024 * 1024:
            errors.append(f"file exceeds 10 MiB: {name}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-untracked", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    command = ["git", "-C", str(root), "ls-files", "--cached"]
    if arguments.include_untracked:
        command += ["--others", "--exclude-standard"]
    paths = subprocess.run(command, capture_output=True, text=True, check=True).stdout.splitlines()
    errors = violations(root, paths)
    print("\n".join(errors) if errors else f"Data safety check passed ({len(paths)} files)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
