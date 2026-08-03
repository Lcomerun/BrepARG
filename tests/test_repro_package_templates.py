import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_reproducibility_delivery_templates_are_complete() -> None:
    required = [
        "reproducibility/docs/START_HERE.md",
        "reproducibility/environments/environment.linux-gpu.yml",
        "reproducibility/environments/requirements.linux-gpu.lock.txt",
        "reproducibility/environments/bootstrap.sh",
        "reproducibility/environments/probe_environment.py",
        "reproducibility/environments/install_occ_optional.sh",
        "reproducibility/environments/probe_occ.py",
        "reproducibility/reports/current_conclusions.md",
        "reproducibility/project_history/00_READ_ME_FIRST.md",
        "reproducibility/project_history/02_timeline/project_timeline.md",
        "reproducibility/project_history/03_experiment_ledger/experiment_ledger.md",
        "reproducibility/project_history/03_experiment_ledger/experiment_ledger.json",
        "reproducibility/project_history/04_plans_and_decisions/README.md",
        "reproducibility/project_history/06_failure_incidents/incident_register.md",
        "reproducibility/project_history/06_failure_incidents/conversation_derived_incidents.md",
        "reproducibility/project_history/07_data_and_protocol/data_and_evaluation_protocol.md",
        "reproducibility/project_history/08_evidence_index/README.md",
        "reproducibility/tests/test_package_control.py",
    ]
    missing = [relative for relative in required if not (ROOT / relative).is_file()]
    assert missing == []


def test_environment_lock_uses_cu128_and_excludes_stale_occ_pin() -> None:
    lock = (ROOT / "reproducibility/environments/requirements.linux-gpu.lock.txt").read_text(
        encoding="utf-8"
    )
    assert "torch==2.8.0+cu128" in lock
    assert "transformers==4.57.3" in lock
    assert "diffusers==0.35.1" in lock
    assert "numpy==2.2.6" in lock
    assert "OCC==0.0.2" not in lock
    assert not re.search(r"^[A-Za-z0-9_.-]+\s*(?:$|#)", lock, re.MULTILINE)


def test_entry_document_states_current_scientific_limits() -> None:
    text = (ROOT / "reproducibility/docs/START_HERE.md").read_text(encoding="utf-8")
    for phrase in (
        "复杂曲面",
        "parent-CAD",
        "BrepARG",
        "不是官方权重复现",
        "external artifact",
        "reproduce.sh",
    ):
        assert phrase in text


def test_package_control_test_does_not_embed_legacy_host_path() -> None:
    text = (ROOT / "reproducibility/tests/test_package_control.py").read_text(
        encoding="utf-8"
    )
    assert "/root/autodl-tmp" not in text


def test_experiment_ledger_is_machine_readable_and_evidence_linked() -> None:
    ledger = json.loads(
        (
            ROOT
            / "reproducibility/project_history/03_experiment_ledger/experiment_ledger.json"
        ).read_text(encoding="utf-8")
    )
    assert ledger["schema_version"] == 1
    assert len(ledger["experiments"]) >= 12
    assert all(row.get("evidence_ids") for row in ledger["experiments"])
    assert {row["status"] for row in ledger["experiments"]} >= {
        "completed",
        "failed",
        "blocked",
    }
