"""Exercise evaluation control flow with CPU fixtures; never touch a real GPU."""

import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "bench/metal" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("evaluate")
kapitsa = load("kapitsa")


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lock = self.root / "test-gpu.lock"
        self.env = patch.dict(os.environ, {"BENCH_GPU_LOCK": str(self.lock),
                                          "MTL_SHADER_VALIDATION": "0", "MTL_DEBUG_LAYER": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)
        (self.root / "kernel.txt").write_text("original kernel")
        self.config = {
            "version": 1, "scope": "kernel", "pairs": 2, "timeout_seconds": 5,
            "contract": {"workload": "CPU fixture", "inputs": "synthetic",
                         "precision": "exact", "timing": "simulated; not GPU evidence"},
            "primary_metric": "latency",
            "metrics": {"latency": {"direction": "lower", "min_speedup_pct": 3,
                                    "max_regression_pct": 0}},
            "artifacts": {"baseline": ["kernel.txt"], "candidate": ["kernel.txt"]},
            "checks": [self.command('print(\'AKO_CHECK={"passed":true,"cases":3}\')')],
            "baseline": self.measure(10), "candidate": self.measure(8),
        }

    def command(self, code):
        return {"command": [sys.executable, "-c", code]}

    def measure(self, value):
        return self.command(f"print('AKO_METRICS=' + {json.dumps(json.dumps({'valid': True, 'metrics': {'latency': value}}))})")

    def evaluate(self):
        self.config_path = self.root / "experiment.json"
        self.config_path.write_text(json.dumps(self.config))
        config = runner.read_config(self.config_path)
        return runner.evaluate(config, self.root, self.root / "evidence")

    def test_accepts_and_preserves_abab_order(self):
        report = self.evaluate()
        self.assertEqual(report["status"], "accepted", report)
        self.assertEqual([r["label"] for r in report["runs"]], ["A1", "B1", "A2", "B2"])
        self.assertEqual(report["metrics"]["latency"]["speedup"], 1.25)
        self.assertEqual((self.root / "evidence/report.json").exists(), True)
        with self.assertRaises(FileExistsError):
            runner.evaluate(self.config, self.root, self.root / "evidence")

    def test_failed_check_prevents_any_timing(self):
        self.config["checks"] = [self.command('print(\'AKO_CHECK={"passed":false,"cases":3}\')')]
        report = self.evaluate()
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["runs"], [])
        self.assertFalse((self.root / "evidence/A1.command.json").exists())

    def test_zero_cases_and_nonzero_exit_cannot_pass(self):
        for code in ['print(\'AKO_CHECK={"passed":true,"cases":0}\')',
                     'print(\'AKO_CHECK={"passed":true,"cases":1}\'); exit(7)']:
            with self.subTest(code=code):
                self.config["checks"] = [self.command(code)]
                report = runner.evaluate(self.config, self.root, self.root / str(len(list(self.root.iterdir()))))
                self.assertEqual(report["status"], "error")
                self.assertEqual(report["runs"], [])

    def test_malformed_metric_records_cannot_pass(self):
        for record in [{"valid": True, "metrics": {"latency": float("nan")}},
                       {"valid": True, "metrics": {"latency": 0}},
                       {"valid": True, "metrics": {"latency": True}},
                       {"valid": True, "metrics": {}},
                       {"valid": False, "metrics": {"latency": 1}}]:
            with self.subTest(record=record):
                command = self.command("print(" + repr("AKO_METRICS=" + json.dumps(record)) + ")")
                self.config["candidate"] = command
                report = runner.evaluate(self.config, self.root, self.root / str(len(list(self.root.iterdir()))))
                self.assertEqual(report["status"], "error")
        with self.assertRaises(ValueError):
            runner.payload('AKO_CHECK={}\nAKO_CHECK={}', 'AKO_CHECK=')

    def test_lock_contention_refuses_even_with_inherited_flag(self):
        with self.lock.open("w") as lock, patch.dict(os.environ, {"NI_LOCK_HELD": "1"}):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            report = self.evaluate()
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["checks"], [])

    def test_shader_validation_only_on_check_and_lock_inherited(self):
        self.config["checks"] = [self.command('''import os,fcntl
assert os.environ['MTL_SHADER_VALIDATION'] == '1'
assert os.environ['MTL_DEBUG_LAYER'] == '1'
assert os.environ['NI_LOCK_HELD'] == '1'
fd = int(os.environ['AKO_GPU_LOCK_FD'])
fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
print('AKO_CHECK={"passed":true,"cases":1}')''')]
        self.config["checks"][0]["shader_validation"] = True
        self.assertEqual(self.evaluate()["status"], "accepted")

    def test_timeout_keeps_evidence_and_releases_lock(self):
        self.config["timeout_seconds"] = 0.1
        self.config["checks"] = [self.command('import time; print("started", flush=True); time.sleep(30)')]
        report = self.evaluate()
        self.assertEqual(report["status"], "error")
        self.assertIn("started", (self.root / "evidence/check-1.out").read_text())
        with self.lock.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_artifact_mutation_cannot_be_accepted(self):
        self.config["candidate"] = self.command('''from pathlib import Path
Path('kernel.txt').write_text('changed during measurement')
print('AKO_METRICS={"valid":true,"metrics":{"latency":8}}')''')
        report = self.evaluate()
        self.assertEqual(report["status"], "error")
        self.assertIn("artifacts changed", report["reason"])

    def test_context_change_and_instrumented_timing_are_refused(self):
        self.config["candidate"] = self.command('print(\'AKO_METRICS={"valid":true,"metrics":{"latency":8},"context":{"dtype":"half"}}\')')
        self.assertEqual(self.evaluate()["status"], "error")
        self.config["candidate"] = self.measure(8)
        with patch.dict(os.environ, {"MTL_DEBUG_LAYER": "1"}):
            report = runner.evaluate(self.config, self.root, self.root / "instrumented")
        self.assertEqual(report["status"], "error")

    def test_performance_noise_and_secondary_regressions(self):
        runs = [{"variant": v, "metrics": {"latency": value}}
                for v, value in [("baseline", 9), ("candidate", 8.5), ("baseline", 11), ("candidate", 10.5)]]
        self.assertEqual(runner.summarize(self.config, runs)[0], "inconclusive")
        for row in runs:
            row["metrics"]["latency"] = 10 if row["variant"] == "baseline" else 8
            row["metrics"]["throughput"] = 100 if row["variant"] == "baseline" else 90
        self.config["metrics"]["throughput"] = {"direction": "higher", "min_speedup_pct": 0,
                                                 "max_regression_pct": 0}
        self.assertEqual(runner.summarize(self.config, runs)[0], "rejected")

    def test_cli_dry_run_and_missing_authorization_execute_nothing(self):
        config = self.root / "config.json"
        config.write_text(json.dumps(self.config))
        args = [sys.executable, str(ROOT / "bench/metal/evaluate.py"), str(config),
                "--out", str(self.root / "cli-output")]
        self.assertEqual(subprocess.run([*args, "--dry-run"], capture_output=True).returncode, 0)
        self.assertEqual(subprocess.run(args, capture_output=True).returncode, 2)
        self.assertFalse((self.root / "cli-output").exists())

    def test_config_rejects_weak_measurement_and_missing_oracle(self):
        for field, value in [("pairs", 1), ("checks", []), ("timeout_seconds", -1)]:
            with self.subTest(field=field):
                config = copy.deepcopy(self.config)
                config[field] = value
                path = self.root / "bad.json"
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    runner.read_config(path)

    def test_kapitsa_adapter_runs_native_commands_through_gate(self):
        model = self.root / "local-model"
        model.mkdir()
        (model / "config.json").write_text("{}")
        repo = self.root / "native-repo"
        (repo / "tools").mkdir(parents=True)
        gate = repo / "tools/logit_compare.py"
        gate.write_text("import sys\nfrom pathlib import Path\nassert all(Path(p).is_dir() for p in sys.argv[1:])\n")
        adapter = ROOT / "bench/metal/kapitsa.py"
        for variant, rate in (("baseline", 100), ("candidate", 125)):
            build = self.root / variant
            build.mkdir()
            dump = build / "logit_dump"
            dump.write_text(f"#!{sys.executable}\nimport os,sys\nfrom pathlib import Path\n"
                            "assert os.environ['NI_LOCK_HELD'] == '1'\n"
                            "Path(sys.argv[2]).mkdir()\n")
            dump.chmod(0o755)
            binary = build / "kapitsa"
            native_result = {"kind": "kapitsa-bench", "flags": [], "invalid": [],
                             "model": {"name": "fixture"}, "protocol": {"seed": 1},
                             "results": {"tool_calls": "pass", "decode": {"median_tps": rate}, "prefill": []}}
            binary.write_text(f"#!{sys.executable}\nimport json,os,sys\nfrom pathlib import Path\n"
                              "assert os.environ['NACE_GPU_LOCK_HELD'] == '1'\n"
                              "output = Path(sys.argv[sys.argv.index('--out') + 1])\n"
                              f"output.write_text(json.dumps({native_result!r}))\n")
            binary.chmod(0o755)
            self.config[variant] = {"command": [sys.executable, str(adapter), "bench",
                "--binary", str(binary), "--model", str(model), "--out", "{run_dir}/{label}.bench.json"]}
            self.config["artifacts"][variant] = [str(binary), str(dump)]
        self.config["scope"] = "inference"
        self.config["primary_metric"] = "decode_tps"
        self.config["metrics"] = {"decode_tps": {"direction": "higher", "min_speedup_pct": 3,
                                                   "max_regression_pct": 0}}
        self.config["checks"] = [{"command": [sys.executable, str(adapter), "check",
            "--baseline-build", str(self.root / "baseline"), "--candidate-build", str(self.root / "candidate"),
            "--repo", str(repo), "--model", str(model), "--evidence", "{run_dir}/logits"]}]
        report = self.evaluate()
        self.assertEqual(report["status"], "accepted", report)
        self.assertTrue((self.root / "evidence/logits/reference").is_dir())
        self.assertTrue((self.root / "evidence/B2.bench.json").is_file())
        gate.write_text("raise SystemExit(1)\n")
        failed = runner.evaluate(self.config, self.root, self.root / "failed-gate")
        self.assertEqual(failed["status"], "error")
        self.assertEqual(failed["runs"], [])


class KapitsaTests(unittest.TestCase):
    def test_report_mapping_and_flagged_result_rejection(self):
        # Fabricated values matching the published schema; no frozen data edited.
        result = {"kind": "kapitsa-bench", "flags": [], "invalid": [], "results": {
            "tool_calls": "pass", "decode": {"median_tps": 100},
            "prefill": [{"size": 4000, "tps": 2000, "ttft_s": 2,
                         "followup_median_s": 0.1}]}}
        metrics = kapitsa.benchmark_metrics(result)
        self.assertEqual(metrics["ttft_4000_ms"], 2000)
        self.assertEqual(metrics["followup_4000_ms"], 100)
        result["flags"] = ["CONTENTION"]
        with self.assertRaises(ValueError):
            kapitsa.benchmark_metrics(result)

    def test_adapter_refuses_unlocked_invocation(self):
        with patch.dict(os.environ, {"AKO_GPU_LOCK_FD": ""}):
            with self.assertRaises(ValueError):
                kapitsa.require_lock()


if __name__ == "__main__":
    unittest.main()
