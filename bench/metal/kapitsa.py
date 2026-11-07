#!/usr/bin/env python3
"""Adapt Kapitsa's existing output check and quick benchmark to AKO records."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


def require_lock():
    value = os.environ.get("AKO_GPU_LOCK_FD", "")
    if not value.isdigit():
        raise ValueError("Run through evaluate.py; an inherited GPU lock is required")
    fd = int(value)
    lock_path = os.environ.get("BENCH_GPU_LOCK", "/tmp/kapitsa-gpu.lock")
    held, expected = os.fstat(fd), os.stat(lock_path)
    if (held.st_dev, held.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError("Inherited descriptor does not identify BENCH_GPU_LOCK")
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.update(NI_LOCK_HELD="1", NACE_GPU_LOCK_HELD="1",
                      KAPITSA_NO_AUTO_UPDATE="1", KAPITSA_NO_PROMPT="1")
    return fd


def benchmark_metrics(result):
    if not isinstance(result, dict) or result.get("kind") != "kapitsa-bench" or any(
        not isinstance(result.get(key), list) for key in ("invalid", "flags")
    ):
        raise ValueError("Expected a complete kapitsa bench report")
    # Flags include contention/power/thermal warnings. Retain the report and
    # invalidate the measurement instead of silently selecting clean samples.
    if result["invalid"] or result["flags"]:
        raise ValueError(f"Invalid/flagged benchmark: {result['invalid']} {result['flags']}")
    data = result["results"]
    if data.get("tool_calls") != "pass":
        raise ValueError("Kapitsa's benchmark checks did not pass")
    metrics = {"decode_tps": data["decode"]["median_tps"]}
    for row in data["prefill"]:
        size = row["size"]
        metrics[f"prefill_{size}_tps"] = row["tps"]
        metrics[f"ttft_{size}_ms"] = row["ttft_s"] * 1000
        metrics[f"followup_{size}_ms"] = row["followup_median_s"] * 1000
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    check = sub.add_parser("check", help="Old/new logit dumps and repository numerical gate")
    check.add_argument("--baseline-build", type=Path, required=True)
    check.add_argument("--candidate-build", type=Path, required=True)
    check.add_argument("--repo", type=Path, required=True)
    check.add_argument("--evidence", type=Path, required=True)
    bench = sub.add_parser("bench", help="One quick inference benchmark")
    bench.add_argument("--binary", type=Path, required=True)
    bench.add_argument("--out", type=Path, required=True)
    for command in (check, bench):
        command.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    try:
        lock_fd = require_lock()
        model = args.model.resolve()
        if not model.is_dir() or not (model / "config.json").is_file():
            raise ValueError("--model must be an existing local model folder; no downloads")
        if args.mode == "check":
            args.evidence.mkdir(parents=True, exist_ok=False)
            reference, candidate = args.evidence.resolve() / "reference", args.evidence.resolve() / "candidate"
            subprocess.run([str(args.baseline_build.resolve() / "logit_dump"), str(model), str(reference)],
                           check=True, pass_fds=(lock_fd,))
            subprocess.run([str(args.candidate_build.resolve() / "logit_dump"), str(model), str(candidate),
                            "--from", str(reference)], check=True, pass_fds=(lock_fd,))
            subprocess.run([sys.executable, str(args.repo.resolve() / "tools/logit_compare.py"),
                            str(reference), str(candidate)], check=True, cwd=args.repo, pass_fds=(lock_fd,))
            # A case here is one complete model's repository-defined output suite.
            print('AKO_CHECK={"passed":true,"cases":1,"kind":"kapitsa-logit-suite"}')
        else:
            if args.out.exists():
                raise ValueError("Benchmark output already exists; choose a new path")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([str(args.binary.resolve()), "bench", "--model", str(model),
                            "--quick", "--json", "--out", str(args.out.resolve())], check=True, pass_fds=(lock_fd,))
            result = json.loads(args.out.read_text())
            print("AKO_METRICS=" + json.dumps({"valid": True, "metrics": benchmark_metrics(result),
                                             "context": {k: result[k] for k in ("model", "protocol")}}))
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(f"Kapitsa adapter failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
