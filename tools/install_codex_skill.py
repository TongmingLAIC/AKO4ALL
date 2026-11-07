#!/usr/bin/env python3
"""Install a self-contained AKO4ALL skill without changing Codex configuration."""

import argparse
import shutil
import tempfile
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1]
CONTENTS = (
    "SKILL.md", "HINTS.md", "ITERATIONS.md", "bench-wrapper.sh",
    "workspace.gitignore", "bench", "references", "agents", "LICENSE",
)


def install(source, destination):
    source = source.resolve()
    destination = destination.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Destination already exists; leaving it unchanged: {destination}")
    if destination.resolve() == source or source in destination.resolve().parents:
        raise ValueError("Install outside the source checkout to avoid recursive discovery")
    for name in CONTENTS:
        if not (source / name).exists():
            raise ValueError(f"Missing skill resource: {source / name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".ako4all-install-", dir=destination.parent))
    try:
        for name in CONTENTS:
            item = source / name
            if item.is_dir():
                shutil.copytree(item, staging / name,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                shutil.copy2(item, staging / name)
        # Reserve the name before publishing; never replace another installation.
        destination.mkdir()
        try:
            for item in staging.iterdir():
                shutil.move(str(item), destination / item.name)
        except BaseException:
            shutil.rmtree(destination)
            raise
    finally:
        shutil.rmtree(staging)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path,
                        default=Path.home() / ".agents/skills/ako4all",
                        help="Exact skill folder (default: ~/.agents/skills/ako4all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the destination without creating files")
    args = parser.parse_args()
    if args.dry_run:
        print(f"Would install {SOURCE} -> {args.dest.expanduser().absolute()}")
        return
    try:
        destination = install(SOURCE, args.dest)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Install failed: {error}\n")
    print(f"Installed AKO4ALL at {destination}\nInvoke $ako4all in your kernel workspace.")


if __name__ == "__main__":
    main()
