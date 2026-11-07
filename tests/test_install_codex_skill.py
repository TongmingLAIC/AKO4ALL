"""CPU-only tests for installation isolation and runnable scaffold packaging."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "tools/install_codex_skill.py"


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dest = self.root / "skills with spaces/ako4all"

    def install(self, *extra):
        return subprocess.run(
            [sys.executable, str(INSTALLER), "--dest", str(self.dest), *extra],
            cwd=self.root, text=True, capture_output=True,
        )

    def test_dry_run_does_not_create_parent(self):
        self.assertEqual(self.install("--dry-run").returncode, 0)
        self.assertFalse(self.dest.parent.exists())

    def test_existing_install_and_broken_symlink_are_preserved(self):
        self.dest.mkdir(parents=True)
        (self.dest / "SKILL.md").write_text("user customization")
        self.assertNotEqual(self.install().returncode, 0)
        self.assertEqual((self.dest / "SKILL.md").read_text(), "user customization")
        link = self.root / "broken"
        link.symlink_to(self.root / "missing")
        self.dest = link
        self.assertNotEqual(self.install().returncode, 0)
        self.assertTrue(link.is_symlink())

    def test_nested_install_is_rejected(self):
        self.dest = ROOT / ".agents/skills/ako4all"
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside the source checkout", result.stderr)

    def test_installed_scaffold_preserves_failed_measurement(self):
        self.assertEqual(self.install().returncode, 0)
        workspace = self.root / "independent workspace"
        (workspace / "scripts").mkdir(parents=True)
        (workspace / "solution").mkdir()
        (workspace / "solution/kernel.txt").write_text("candidate")
        # Avoid importing any GPU library; this fake executable is test-only.
        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        (fake_bin / "python").write_text("#!/bin/sh\nexit 1\n")
        (fake_bin / "python").chmod(0o755)
        wrapper = (self.dest / "bench-wrapper.sh").read_text()
        command = "(printf 'CPU fixture failed\\n'; exit 7) 2>&1 | tee _bench_output.txt"
        (workspace / "scripts/bench.sh").write_text(wrapper.replace("{{BENCH_COMMAND}}", command))
        result = subprocess.run(
            ["bash", str(workspace / "scripts/bench.sh"), "iter-1"], cwd=self.root,
            env={**os.environ, "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"]},
            text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 7, result.stderr)
        runs = list((workspace / "trajectory").iterdir())
        self.assertEqual(len(runs), 1)
        self.assertEqual((runs[0] / "kernel.txt").read_text(), "candidate")
        self.assertIn("CPU fixture failed", (runs[0] / "output.txt").read_text())
        # The installation remains independent of both source and experiment state.
        self.assertFalse((self.dest / ".git").exists())
        self.assertFalse((self.dest / "trajectory").exists())


if __name__ == "__main__":
    unittest.main()
