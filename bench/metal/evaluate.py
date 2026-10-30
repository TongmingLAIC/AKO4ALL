#!/usr/bin/env python3
"""Run correctness commands and serialized A/B/A/B measurements on a Mac GPU."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys


class EvaluationError(ValueError):
    pass


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def strict_json(text):
    def invalid(value):
        raise EvaluationError(f"Non-finite JSON value: {value}")
    return json.loads(text, parse_constant=invalid)


def read_config(path):
    config = strict_json(path.read_text())
    if not isinstance(config, dict):
        raise EvaluationError("Configuration must be a JSON object")
    if config.get("version") != 1 or config.get("scope") not in ("kernel", "inference"):
        raise EvaluationError("Expected version 1 and scope kernel or inference")
    contract = config.get("contract")
    if not isinstance(contract, dict) or any(not contract.get(k) for k in
                                           ("workload", "inputs", "precision", "timing")):
        raise EvaluationError("contract needs workload, inputs, precision, and timing")
    metrics = config.get("metrics", {})
    if not isinstance(metrics, dict) or not metrics or config.get("primary_metric") not in metrics:
        raise EvaluationError("primary_metric must name a configured metric")
    for name, rule in metrics.items():
        if not isinstance(rule, dict) or rule.get("direction") not in ("higher", "lower"):
            raise EvaluationError(f"Invalid direction for {name}")
        for key in ("min_speedup_pct", "max_regression_pct"):
            value = rule.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise EvaluationError(f"{name}.{key} must be finite and non-negative")
    pairs = config.get("pairs", 2)
    if type(pairs) is not int or pairs < 2:
        raise EvaluationError("At least two A/B pairs are required")
    if not positive(config.get("timeout_seconds", 900)):
        raise EvaluationError("timeout_seconds must be positive")
    checks = config.get("checks")
    if not isinstance(checks, list) or not checks:
        raise EvaluationError("At least one independent correctness check is required")
    for spec in [*checks, config.get("baseline", {}), config.get("candidate", {})]:
        if not isinstance(spec, dict) or not isinstance(spec.get("cwd", "."), str):
            raise EvaluationError("Command specifications must be objects with a string cwd")
        if type(spec.get("shader_validation", False)) is not bool:
            raise EvaluationError("shader_validation must be a boolean")
        command = spec.get("command")
        if not isinstance(command, list) or not command or any(
            not isinstance(arg, str) or not arg for arg in command
        ):
            raise EvaluationError("Commands must be non-empty argv arrays; no implicit shell")
        if not isinstance(spec.get("env", {}), dict) or any(
            not isinstance(k, str) or not isinstance(v, str)
            for k, v in spec.get("env", {}).items()
        ):
            raise EvaluationError("Command env must map strings to strings")
    if not isinstance(config.get("artifacts"), dict):
        raise EvaluationError("artifacts must identify baseline and candidate files")
    for variant in ("baseline", "candidate"):
        files = config.get("artifacts", {}).get(variant)
        if not isinstance(files, list) or not files or any(not isinstance(p, str) for p in files):
            raise EvaluationError(f"artifacts.{variant} must list measured source/binary files")
    return config


def payload(text, marker):
    lines = [line[len(marker):] for line in text.splitlines() if line.startswith(marker)]
    if len(lines) != 1:
        raise EvaluationError(f"Expected exactly one {marker} record")
    value = strict_json(lines[0])
    if not isinstance(value, dict):
        raise EvaluationError(f"{marker} must contain an object")
    return value


def fingerprints(config, root):
    result = {}
    for variant in ("baseline", "candidate"):
        names = config["artifacts"][variant]
        result[variant] = {}
        for name in names:
            path = (root / name).resolve()
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            result[variant][str(path)] = digest.hexdigest()
    return result


@contextmanager
def gpu_lock(path):
    # Never trust a caller's NI_LOCK_HELD flag: this process must own the flock.
    with open(path, "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield lock.fileno()
        # Closing the descriptor releases the lock after child cleanup.


def terminate_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def execute(spec, root, run_dir, label, timeout, lock_fd, validation=False):
    command = [arg.replace("{run_dir}", str(run_dir)).replace("{label}", label)
               for arg in spec["command"]]
    cwd = (root / spec.get("cwd", ".")).resolve()
    env = {**os.environ, **spec.get("env", {}), "NI_LOCK_HELD": "1",
           "NACE_GPU_LOCK_HELD": "1", "KAPITSA_NO_AUTO_UPDATE": "1",
           "AKO_GPU_LOCK_FD": str(lock_fd),
           "BENCH_GPU_LOCK": str(Path(os.environ.get("BENCH_GPU_LOCK", "/tmp/kapitsa-gpu.lock")).resolve())}
    if validation:
        env.update(MTL_SHADER_VALIDATION="1", MTL_DEBUG_LAYER="1")
    elif any(env.get(key, "0") not in ("0", "") for key in
             ("MTL_SHADER_VALIDATION", "MTL_DEBUG_LAYER")):
        raise EvaluationError("Disable Metal validation/debug instrumentation for timing")
    stdout_path, stderr_path = run_dir / f"{label}.out", run_dir / f"{label}.err"
    (run_dir / f"{label}.command.json").write_text(json.dumps({"argv": command, "cwd": str(cwd)}))
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                                   start_new_session=True, pass_fds=(lock_fd,))
        try:
            code = process.wait(timeout=timeout)
        finally:
            # Commands must be synchronous; reap any leftover descendants as well.
            terminate_group(process)
    (run_dir / f"{label}.exit").write_text(str(code))
    if code != 0:
        raise EvaluationError(f"{label} exited {code}; see {stderr_path.name}")
    return stdout_path.read_text()


def summarize(config, runs):
    summaries = {}
    regressions = []
    for name, rule in config["metrics"].items():
        values = {v: [r["metrics"][name] for r in runs if r["variant"] == v]
                  for v in ("baseline", "candidate")}
        med = {v: statistics.median(nums) for v, nums in values.items()}
        ratio = med["candidate"] / med["baseline"]
        if rule["direction"] == "lower":
            ratio = 1 / ratio
        gain = (ratio - 1) * 100
        noise = max((max(nums) - min(nums)) / med[v] * 100 for v, nums in values.items())
        summaries[name] = {"samples": values, "median": med, "speedup": ratio,
                           "speedup_pct": gain, "observed_spread_pct": noise}
        if gain < -rule["max_regression_pct"]:
            regressions.append(name)
    primary = summaries[config["primary_metric"]]
    threshold = config["metrics"][config["primary_metric"]]["min_speedup_pct"]
    if regressions:
        return "rejected", summaries, f"Regression in: {', '.join(regressions)}"
    if primary["speedup_pct"] < threshold or primary["speedup_pct"] <= primary["observed_spread_pct"]:
        return "inconclusive", summaries, "Gain does not clear the declared threshold and observed spread"
    return "accepted", summaries, "Configured correctness and performance gates passed"


def evaluate(config, root, output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    report = {"status": "incomplete", "scope": config["scope"], "contract": config["contract"],
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "host": {"system": platform.platform(), "machine": platform.machine()},
              "checks": [], "runs": []}
    try:
        before = fingerprints(config, root)
        report["artifacts"] = before
        with gpu_lock(os.environ.get("BENCH_GPU_LOCK", "/tmp/kapitsa-gpu.lock")) as fd:
            timeout = config.get("timeout_seconds", 900)
            for i, check in enumerate(config["checks"]):
                raw = execute(check, root, output, f"check-{i + 1}", timeout, fd,
                              validation=check.get("shader_validation", False))
                result = payload(raw, "AKO_CHECK=")
                report["checks"].append(result)
                if result.get("passed") is not True or type(result.get("cases")) is not int or result["cases"] < 1:
                    raise EvaluationError("Correctness check failed or checked no cases")
            for pair in range(1, config.get("pairs", 2) + 1):
                for variant, prefix in (("baseline", "A"), ("candidate", "B")):
                    label = f"{prefix}{pair}"
                    raw = execute(config[variant], root, output, label, timeout, fd)
                    result = payload(raw, "AKO_METRICS=")
                    if result.get("valid") is not True:
                        raise EvaluationError(f"{label} reported an invalid/contended run")
                    metrics = result.get("metrics", {})
                    if not isinstance(metrics, dict) or any(not positive(metrics.get(name))
                                                            for name in config["metrics"]):
                        raise EvaluationError(f"{label} has missing/non-finite/non-positive metrics")
                    context = result.get("context")
                    if "measurement_context" not in report:
                        report["measurement_context"] = context
                    elif context != report["measurement_context"]:
                        raise EvaluationError(f"{label} changed workload/model/protocol context")
                    report["runs"].append({"label": label, "variant": variant, "metrics": metrics})
            if fingerprints(config, root) != before:
                raise EvaluationError("Measured artifacts changed during evaluation; results are invalid")
            report["status"], report["metrics"], report["reason"] = summarize(config, report["runs"])
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        report.update(status="error", reason=str(error))
    except KeyboardInterrupt:
        report.update(status="incomplete", reason="Interrupted; no candidate accepted")
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", required=True, type=Path, help="New evidence directory; never overwritten")
    parser.add_argument("--allow-gpu", action="store_true", help="Acknowledge task-authorized GPU use")
    parser.add_argument("--dry-run", action="store_true", help="Validate config only; no command or GPU use")
    args = parser.parse_args()
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        config = read_config(args.config)
        if args.dry_run:
            print(json.dumps(config, indent=2))
            return
        if not args.allow_gpu:
            raise EvaluationError("GPU use needs task authorization and --allow-gpu")
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise EvaluationError("Real measurements require an Apple Silicon Mac")
        report = evaluate(config, args.config.resolve().parent, args.out)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if report["status"] == "accepted" else 1
    except (OSError, ValueError) as error:
        parser.exit(2, f"Evaluation refused: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Evaluation interrupted; evidence retained without acceptance\n")


if __name__ == "__main__":
    sys.exit(main())
