from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = PACKAGE_ROOT / "launchers"
if str(LAUNCHERS) not in sys.path:
    sys.path.insert(0, str(LAUNCHERS))

import repro_runtime  # noqa: E402


class PackageControlTests(unittest.TestCase):
    def test_internal_checksums_match(self) -> None:
        results = repro_runtime.verify_package_checksums(PACKAGE_ROOT)
        self.assertTrue(results)
        self.assertTrue(all(row["status"] == "ok" for row in results))

    def test_descriptors_and_artifacts_are_unique_and_closed(self) -> None:
        experiments = repro_runtime.load_experiments(PACKAGE_ROOT)
        artifacts = repro_runtime.load_artifact_specs(PACKAGE_ROOT)
        self.assertGreaterEqual(len(experiments), 1)
        self.assertGreaterEqual(len(artifacts), 1)
        referenced = {
            artifact_id
            for experiment in experiments.values()
            for artifact_id in experiment.get("required_artifacts", [])
        }
        self.assertEqual(sorted(referenced - artifacts.keys()), [])

    def test_historical_descriptors_are_guarded(self) -> None:
        experiments = repro_runtime.load_experiments(PACKAGE_ROOT)
        historical = [
            row for row in experiments.values() if row.get("category") == "historical_failed"
        ]
        self.assertTrue(historical)
        for experiment in historical:
            self.assertTrue(experiment.get("blocked_reason"))

    def test_control_plane_has_no_host_runtime_defaults(self) -> None:
        roots = [
            PACKAGE_ROOT / "experiments",
            PACKAGE_ROOT / "artifact_specs",
            PACKAGE_ROOT / "configs",
            PACKAGE_ROOT / "environments",
            PACKAGE_ROOT / "launchers",
        ]
        legacy_linux_root = "/root/" + "autodl-tmp"
        patterns = [
            re.compile(r"(?i)(?<![A-Za-z0-9_])[CDE]:[\\/]"),
            re.compile(re.escape(legacy_linux_root) + r"(?:/|\b)"),
        ]
        violations: list[str] = []
        for root in roots:
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix not in {".json", ".md", ".py", ".sh", ".env", ".txt", ".yml"}:
                    continue
                text = path.read_text(encoding="utf-8")
                if any(pattern.search(text) for pattern in patterns):
                    violations.append(path.relative_to(PACKAGE_ROOT).as_posix())
        self.assertEqual(violations, [])

    def test_manifest_declares_lightweight_linux_target(self) -> None:
        manifest = json.loads((PACKAGE_ROOT / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["target_platform"], "linux-x86_64-nvidia-gpu")
        self.assertFalse(manifest["heavy_artifacts_included"])


if __name__ == "__main__":
    unittest.main()
