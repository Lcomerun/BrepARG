"""Run the signed seven-CAD, seven-stage assembly census.

The coordinator in this module is deliberately Open-CASCADE free.  It binds
the frozen calibration and selector, registers seven unchanged controls plus
three curve-interpolation *reachability* bridges, and launches each task in a
fresh child.  A bridge is never scored as a repair.  Native execution,
including the temporary STEP roundtrip, happens only in the child.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import platform
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


try:
    from .assembly_stage_lineage import (
        ASSESSMENT_SCHEMA,
        GLOBAL_VERTEX_IDENTITY_PROOF_METHOD,
        STAGE_NAMES,
        STAGE_ORDER,
        STAGE_RECORD_SCHEMA,
        TOPOLOGY_FIELDS,
        assert_path_free_finite,
        assess_stage_lineage,
        make_not_reached_stage,
        normalize_stage_record,
        redact_path_and_native_text,
    )
except ImportError:  # pragma: no cover - direct script execution
    from assembly_stage_lineage import (
        ASSESSMENT_SCHEMA,
        GLOBAL_VERTEX_IDENTITY_PROOF_METHOD,
        STAGE_NAMES,
        STAGE_ORDER,
        STAGE_RECORD_SCHEMA,
        TOPOLOGY_FIELDS,
        assert_path_free_finite,
        assess_stage_lineage,
        make_not_reached_stage,
        normalize_stage_record,
        redact_path_and_native_text,
    )


SCHEMA = "source-bound-stage-census-attempt-v1"
RUN_SCHEMA = "source-bound-stage-census-run-v1"
SUMMARY_SCHEMA = "source-bound-stage-census-summary-v1"
RUN_NAME = "source_bound_stage_census_run.json"
ROWS_NAME = "source_bound_stage_census_attempts.jsonl"
SUMMARY_NAME = "source_bound_stage_census_summary.json"
LOCK_NAME = ".source_bound_stage_census_writer.lock"
WORKER_MARKER = "__SOURCE_BOUND_STAGE_CENSUS_RESULT__="

PRIMARY_ARM = "primary_control"
BRIDGE_ARM = "curve_interpolate_bridge"
PRIMARY_PROFILE_NAME = "directed_trim_local_intersection_topology"
BRIDGE_PROFILE_NAME = "directed_trim_curve_interpolate"
PRIMARY_SWITCHES = ("directed_trim", "local_intersection_topology")
BRIDGE_SWITCHES = ("directed_trim", "curve_interpolate")

CAD_51602 = "00051602_7f1947595ae247e0a4a32f43_step_000"
CAD_61931 = "00061931_dcdd8a95feac4121adfd341f_step_000"
CAD_67160 = "00067160_2a27016aa44f42c69c1079f7_step_000"
CAD_87341 = "00087341_6a73c5e821934d3fe4d0d555_step_000"
CAD_76198 = "00076198_7fde7438ca5d3ccb8a1dd1f4_step_000"
CAD_95733 = "00095733_8b325d2fcb27ec9e79388602_step_000"
CAD_32101 = "00032101_674d8fea687f4d9bbca6599b_step_000"
CAD_47472 = "00047472_197769bbdd814278b715d88a_step_000"
CAD_63055 = "00063055_e309c689b9b44f0686f47966_step_000"

SELECTOR_RESIDUAL_CAD_IDS = frozenset(
    {
        CAD_51602, CAD_61931, CAD_67160, CAD_63055, CAD_47472,
        CAD_87341, CAD_76198, CAD_95733, CAD_32101,
    }
)
EXCLUDED_EXACT_NEGATIVE_CAD_IDS = frozenset({CAD_47472, CAD_63055})
TARGET_CAD_IDS = (
    CAD_51602, CAD_61931, CAD_67160, CAD_87341, CAD_76198, CAD_95733,
    CAD_32101,
)
BRIDGE_CAD_IDS = (CAD_51602, CAD_61931, CAD_87341)

EXACT_NEGATIVE_EVIDENCE = {
    "archive_commit": "afafeb81e1674078aa4e08c2987f4343d4734808",
    "run_signature": "1d4f68839aadc8b3f8fb38eea642a1f7ea4f6d8d51b61152f943c725832ffcad",
    "rows_sha256": "f158f2ca7f9bf2adceb7a56434ca4925bed99e34d5791e8867ca476f32d70a34",
    "summary_sha256": "545802b4e783a3f3f76039d70e983fba1ab5eb29af0748e5db032e731c925f60",
}
FROZEN_BREPARG_UTILS_SHA256 = (
    "e2509a844db0a9e0f8eaf670fffb9d4ad9e240af755155d25891d37b4468d521"
)
FROZEN_CALIBRATION_MANIFEST_SHA256 = (
    "426809e39cf2f4ee13c2e86b542c76a5d1c80ce6abfa4ac4a2e84135f580f4ef"
)
FROZEN_SELECTOR_MATRIX_SHA256 = (
    "d3cb1ba56fbc67cdb4db3828cc1ba3036e800ccd32b20b47071144c081b65fe8"
)
FROZEN_SELECTOR_RUN_SHA256 = (
    "9e77ba4271f284effc61e2e33f7deaeffc273faafeb82b1d0f5df78b1ac52da5"
)
FULL_GIT_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")
STEP_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
RUNTIME_IDENTITY_SCHEMA = "source-bound-runtime-abi-sentinel-v1"
RUNTIME_ABI_SENTINEL_SCOPE = (
    "representative_abi_sentinel_not_complete_module_inventory"
)
WORKER_BOOTSTRAP_SOURCE = (
    "import runpy,sys;"
    "repo=sys.argv.pop(1);"
    "sys.path.insert(0,repo);"
    "runpy.run_module('tools.probe_source_bound_stage_census',run_name='__main__')"
)

# A formal census may be resumed over several invocations.  Bind the actual
# Python executable, NumPy, pythonocc wrapper binary, and loaded OCCT kernel so
# an in-place environment upgrade cannot silently mix native runtimes in one
# ten-row ledger.  The probe is source text executed in an isolated child;
# importing this coordinator remains OCC- and NumPy-free.
RUNTIME_PROBE_SOURCE = r'''import ctypes, hashlib, json, platform, sys
from ctypes import wintypes
from pathlib import Path
import numpy
import OCC
from OCC.Core import _Standard

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            value.update(chunk)
    return value.hexdigest()

def loaded_module(name):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    kernel32.GetModuleFileNameW.argtypes = [
        wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD
    ]
    kernel32.GetModuleFileNameW.restype = wintypes.DWORD
    handle = kernel32.GetModuleHandleW(name)
    if not handle:
        raise OSError(ctypes.get_last_error(), "loaded module missing")
    buffer = ctypes.create_unicode_buffer(32768)
    count = kernel32.GetModuleFileNameW(handle, buffer, len(buffer))
    if not count or count >= len(buffer):
        raise OSError(ctypes.get_last_error(), "loaded module path unavailable")
    return Path(buffer.value)

class FixedFileInfo(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint32)
        for name in (
            "dwSignature", "dwStrucVersion", "dwFileVersionMS",
            "dwFileVersionLS", "dwProductVersionMS", "dwProductVersionLS",
            "dwFileFlagsMask", "dwFileFlags", "dwFileOS", "dwFileType",
            "dwFileSubtype", "dwFileDateMS", "dwFileDateLS",
        )
    ]

def pe_versions(path):
    version = ctypes.WinDLL("version", use_last_error=True)
    dummy = wintypes.DWORD()
    version.GetFileVersionInfoSizeW.argtypes = [
        wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)
    ]
    version.GetFileVersionInfoSizeW.restype = wintypes.DWORD
    size = version.GetFileVersionInfoSizeW(str(path), ctypes.byref(dummy))
    if not size:
        raise OSError(ctypes.get_last_error(), "version size unavailable")
    buffer = ctypes.create_string_buffer(size)
    version.GetFileVersionInfoW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID
    ]
    version.GetFileVersionInfoW.restype = wintypes.BOOL
    if not version.GetFileVersionInfoW(str(path), 0, size, buffer):
        raise OSError(ctypes.get_last_error(), "version bytes unavailable")
    pointer = ctypes.c_void_p()
    length = wintypes.UINT()
    version.VerQueryValueW.argtypes = [
        wintypes.LPVOID, wintypes.LPCWSTR,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.UINT)
    ]
    version.VerQueryValueW.restype = wintypes.BOOL
    if (
        not version.VerQueryValueW(
            buffer, "\\", ctypes.byref(pointer), ctypes.byref(length)
        )
        or length.value < ctypes.sizeof(FixedFileInfo)
    ):
        raise OSError(ctypes.get_last_error(), "fixed version unavailable")
    info = ctypes.cast(pointer, ctypes.POINTER(FixedFileInfo)).contents
    if info.dwSignature != 0xFEEF04BD:
        raise RuntimeError("invalid fixed version signature")
    def render(ms, ls):
        return ".".join(
            str(value) for value in
            (ms >> 16, ms & 65535, ls >> 16, ls & 65535)
        )
    return (
        render(info.dwFileVersionMS, info.dwFileVersionLS),
        render(info.dwProductVersionMS, info.dwProductVersionLS),
    )

if sys.platform != "win32":
    raise RuntimeError("formal OCC runtime probe requires Windows")
executable = Path(sys.executable)
wrapper = Path(_Standard.__file__)
kernel = loaded_module("TKernel.dll")
file_version, product_version = pe_versions(kernel)
result = {
    "schema": "source-bound-runtime-abi-sentinel-v1",
    "scope": "representative_abi_sentinel_not_complete_module_inventory",
    "process_isolation": {
        "isolated": bool(sys.flags.isolated),
        "ignore_environment": bool(sys.flags.ignore_environment),
        "no_user_site": bool(sys.flags.no_user_site),
        "sitecustomize_loaded": "sitecustomize" in sys.modules,
        "usercustomize_loaded": "usercustomize" in sys.modules,
    },
    "python": {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "executable_name": executable.name,
        "executable_bytes": executable.stat().st_size,
        "executable_sha256": digest(executable),
    },
    "numpy": {"version": str(numpy.__version__)},
    "pythonocc": {
        "version": str(OCC.VERSION),
        "wrapper_binary_name": wrapper.name,
        "wrapper_binary_bytes": wrapper.stat().st_size,
        "wrapper_binary_sha256": digest(wrapper),
    },
    "occt": {
        "version_source": "TKernel.dll PE VS_FIXEDFILEINFO",
        "file_version": file_version,
        "product_version": product_version,
        "kernel_binary_name": kernel.name,
        "kernel_binary_bytes": kernel.stat().st_size,
        "kernel_binary_sha256": digest(kernel),
    },
}
if globals().get("__emit_runtime_abi_sentinel__", True):
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))'''

FROZEN_RUNTIME_IDENTITY = {
    "schema": RUNTIME_IDENTITY_SCHEMA,
    "scope": RUNTIME_ABI_SENTINEL_SCOPE,
    "process_isolation": {
        "isolated": True,
        "ignore_environment": True,
        "no_user_site": True,
        "sitecustomize_loaded": False,
        "usercustomize_loaded": False,
    },
    "python": {
        "implementation": "CPython",
        "version": "3.9.23",
        "executable_name": "python.exe",
        "executable_bytes": 96256,
        "executable_sha256": (
            "102dc158e0428a9ee945f010b33acf67091575725042ba537dcc8371efef0e14"
        ),
    },
    "numpy": {"version": "1.26.4"},
    "pythonocc": {
        "version": "7.7.2",
        "wrapper_binary_name": "_Standard.pyd",
        "wrapper_binary_bytes": 262144,
        "wrapper_binary_sha256": (
            "ef992b7d8450aaeaf75ac57f39549634c867a5802adfa5f2b9feb5652ac4b8b5"
        ),
    },
    "occt": {
        "version_source": "TKernel.dll PE VS_FIXEDFILEINFO",
        "file_version": "7.7.2.0",
        "product_version": "7.7.2.0",
        "kernel_binary_name": "TKernel.dll",
        "kernel_binary_bytes": 1618432,
        "kernel_binary_sha256": (
            "17685d115de41b426f3e79865548960cb584124d5eac9f91eb0eec961c3096a3"
        ),
    },
}

WORKER_FAILURE_STATUSES = frozenset(
    {
        "worker_timeout", "worker_process_exit", "worker_spawn_error",
        "worker_protocol_error", "worker_error", "source_binding_mismatch",
    }
)
ATTEMPT_STATUSES = WORKER_FAILURE_STATUSES | frozenset(
    {"completed", "scientific_inconclusive"}
)


@dataclass(frozen=True)
class TaskSpec:
    ordinal: int
    cad_id: str
    arm: str
    profile_name: str
    switches: tuple[str, ...]
    is_reachability_bridge: bool

    @property
    def task_id(self) -> str:
        return f"{self.ordinal:02d}::{self.cad_id}::{self.arm}"


def _task(ordinal: int, cad_id: str, arm: str) -> TaskSpec:
    bridge = arm == BRIDGE_ARM
    if arm not in {PRIMARY_ARM, BRIDGE_ARM}:
        raise ValueError("task arm is not registered")
    return TaskSpec(
        int(ordinal), str(cad_id), arm,
        BRIDGE_PROFILE_NAME if bridge else PRIMARY_PROFILE_NAME,
        BRIDGE_SWITCHES if bridge else PRIMARY_SWITCHES,
        bridge,
    )


TASKS: tuple[TaskSpec, ...] = tuple(
    _task(index, cad_id, PRIMARY_ARM)
    for index, cad_id in enumerate(TARGET_CAD_IDS, 1)
) + tuple(
    _task(index, cad_id, BRIDGE_ARM)
    for index, cad_id in enumerate(BRIDGE_CAD_IDS, len(TARGET_CAD_IDS) + 1)
)
TASKS_BY_ID = {task.task_id: task for task in TASKS}


def strict_json_loads(value: str | bytes | bytearray, *, label: str = "JSON") -> Any:
    """Decode standards-compliant JSON without lossy or ambiguous constructs.

    Python's default decoder accepts duplicate object members (last one wins),
    JavaScript non-finite constants, and finite-looking exponents that overflow
    to infinity.  None of those are acceptable for signed census evidence.
    Keeping this decoder here also gives every trust boundary the same rules.
    """

    def object_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate object key {key!r}")
            result[key] = item
        return result

    def reject_constant(token: str) -> Any:
        raise ValueError(f"{label} contains non-finite constant {token!r}")

    def finite_float(token: str) -> float:
        result = float(token)
        if not math.isfinite(result):
            raise ValueError(f"{label} contains an overflowing number")
        return result

    return json.loads(
        value,
        object_pairs_hook=object_pairs,
        parse_constant=reject_constant,
        parse_float=finite_float,
    )


def _json_exact_equal(left: Any, right: Any) -> bool:
    """Compare decoded JSON values without Python's bool/int/float coercions."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return (
            set(left) == set(right)
            and all(_json_exact_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_exact_equal(first, second)
            for first, second in zip(left, right)
        )
    return bool(left == right)


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_nonempty_string(value: Any, *, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _require_binary_name(value: Any, *, label: str) -> str:
    name = _require_nonempty_string(value, label=label)
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError(f"{label} must be a path-free binary name")
    return name


def normalize_runtime_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the path-free representative ABI sentinel with exact types."""

    if type(value) is not dict or set(value) != {
        "schema", "scope", "process_isolation", "python", "numpy", "pythonocc", "occt"
    }:
        raise ValueError("runtime identity has an unexpected root schema")
    if type(value.get("schema")) is not str or value["schema"] != RUNTIME_IDENTITY_SCHEMA:
        raise ValueError("runtime identity schema is not registered")
    if value.get("scope") != RUNTIME_ABI_SENTINEL_SCOPE:
        raise ValueError("runtime ABI sentinel scope is not registered")
    isolation = value.get("process_isolation")
    expected_isolation = {
        "isolated": True,
        "ignore_environment": True,
        "no_user_site": True,
        "sitecustomize_loaded": False,
        "usercustomize_loaded": False,
    }
    if type(isolation) is not dict or set(isolation) != set(expected_isolation):
        raise ValueError("runtime process-isolation evidence has unexpected fields")
    if any(type(isolation.get(key)) is not bool for key in expected_isolation):
        raise ValueError("runtime process-isolation evidence must be boolean")
    if isolation != expected_isolation:
        raise ValueError("runtime process is not isolated from external Python state")

    python = value.get("python")
    if type(python) is not dict or set(python) != {
        "implementation", "version", "executable_name", "executable_bytes",
        "executable_sha256",
    }:
        raise ValueError("runtime Python identity has unexpected fields")
    if type(python.get("executable_bytes")) is not int or python["executable_bytes"] <= 0:
        raise ValueError("runtime Python executable bytes must be a positive integer")
    normalized_python = {
        "implementation": _require_nonempty_string(
            python.get("implementation"), label="runtime Python implementation"
        ),
        "version": _require_nonempty_string(
            python.get("version"), label="runtime Python version"
        ),
        "executable_name": _require_binary_name(
            python.get("executable_name"), label="runtime Python executable name"
        ),
        "executable_bytes": python["executable_bytes"],
        "executable_sha256": _require_sha256(
            python.get("executable_sha256"), label="runtime Python executable hash"
        ),
    }

    numpy = value.get("numpy")
    if type(numpy) is not dict or set(numpy) != {"version"}:
        raise ValueError("runtime NumPy identity has unexpected fields")
    normalized_numpy = {
        "version": _require_nonempty_string(
            numpy.get("version"), label="runtime NumPy version"
        )
    }

    pythonocc = value.get("pythonocc")
    if type(pythonocc) is not dict or set(pythonocc) != {
        "version", "wrapper_binary_name", "wrapper_binary_bytes",
        "wrapper_binary_sha256",
    }:
        raise ValueError("runtime pythonocc identity has unexpected fields")
    if (
        type(pythonocc.get("wrapper_binary_bytes")) is not int
        or pythonocc["wrapper_binary_bytes"] <= 0
    ):
        raise ValueError("runtime pythonocc wrapper bytes must be a positive integer")
    normalized_pythonocc = {
        "version": _require_nonempty_string(
            pythonocc.get("version"), label="runtime pythonocc version"
        ),
        "wrapper_binary_name": _require_binary_name(
            pythonocc.get("wrapper_binary_name"),
            label="runtime pythonocc wrapper name",
        ),
        "wrapper_binary_bytes": pythonocc["wrapper_binary_bytes"],
        "wrapper_binary_sha256": _require_sha256(
            pythonocc.get("wrapper_binary_sha256"),
            label="runtime pythonocc wrapper hash",
        ),
    }

    occt = value.get("occt")
    if type(occt) is not dict or set(occt) != {
        "version_source", "file_version", "product_version",
        "kernel_binary_name", "kernel_binary_bytes", "kernel_binary_sha256",
    }:
        raise ValueError("runtime OCCT identity has unexpected fields")
    if type(occt.get("kernel_binary_bytes")) is not int or occt["kernel_binary_bytes"] <= 0:
        raise ValueError("runtime OCCT kernel bytes must be a positive integer")
    normalized_occt = {
        "version_source": _require_nonempty_string(
            occt.get("version_source"), label="runtime OCCT version source"
        ),
        "file_version": _require_nonempty_string(
            occt.get("file_version"), label="runtime OCCT file version"
        ),
        "product_version": _require_nonempty_string(
            occt.get("product_version"), label="runtime OCCT product version"
        ),
        "kernel_binary_name": _require_binary_name(
            occt.get("kernel_binary_name"), label="runtime OCCT kernel name"
        ),
        "kernel_binary_bytes": occt["kernel_binary_bytes"],
        "kernel_binary_sha256": _require_sha256(
            occt.get("kernel_binary_sha256"), label="runtime OCCT kernel hash"
        ),
    }
    return {
        "schema": RUNTIME_IDENTITY_SCHEMA,
        "scope": RUNTIME_ABI_SENTINEL_SCOPE,
        "process_isolation": dict(isolation),
        "python": normalized_python,
        "numpy": normalized_numpy,
        "pythonocc": normalized_pythonocc,
        "occt": normalized_occt,
    }


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json_value(value: Any) -> Any:
    """Return the exact JSON data model used on disk (for example tuple→list)."""

    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    )
    return strict_json_loads(encoded, label="canonical JSON value")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_binding(path: Path) -> dict[str, Any]:
    target = Path(path)
    return {"bytes": target.stat().st_size, "sha256": sha256_file(target)}


def _runtime_identity() -> dict[str, Any]:
    """Measure and freeze a representative Python/pythonocc/OCCT ABI sentinel.

    The isolated child imports the native stack and identifies the TKernel
    module that Windows actually loaded.  This parent independently binds the
    interpreter executable, rejects noisy or malformed probe output, and then
    requires the frozen protocol sentinel.  No absolute path is serialized.
    This intentionally does not claim to inventory every lazily loaded OCCT
    module; the scope field makes that evidence boundary explicit.
    """

    executable = Path(sys.executable)
    parent_python = {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "executable_name": executable.name,
        "executable_bytes": executable.stat().st_size,
        "executable_sha256": sha256_file(executable),
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", RUNTIME_PROBE_SOURCE],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=60.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("could not probe the native runtime identity") from exc
    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or completed.stderr.strip() or len(stdout_lines) != 1:
        raise RuntimeError("native runtime identity probe failed or was noisy")
    try:
        identity = strict_json_loads(
            stdout_lines[0], label="native runtime identity probe"
        )
        identity = normalize_runtime_identity(identity)
        frozen = normalize_runtime_identity(FROZEN_RUNTIME_IDENTITY)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("native runtime identity probe returned invalid JSON") from exc
    if not _json_exact_equal(identity["python"], parent_python):
        raise RuntimeError("native runtime identity probe used another interpreter")
    if not _json_exact_equal(identity, frozen):
        raise RuntimeError("native runtime differs from the frozen census runtime")
    return identity


def _measure_runtime_abi_sentinel_current_process() -> dict[str, Any]:
    """Measure the representative ABI sentinel in the calling worker.

    Unlike :func:`_runtime_identity`, this function never launches a child.
    Formal workers are bootstrapped with ``python -I`` and execute the same
    frozen measurement source before they deserialize a CAD or import any
    scientific assembly code.  Keeping the result in the worker row proves
    that the process doing the native work—not merely a probe grandchild—used
    the signed representative ABI sentinel.
    """

    namespace: dict[str, Any] = {
        "__name__": "__source_bound_runtime_abi_sentinel__",
        "__emit_runtime_abi_sentinel__": False,
    }
    try:
        exec(
            compile(
                RUNTIME_PROBE_SOURCE,
                "<source-bound-runtime-abi-sentinel>",
                "exec",
            ),
            namespace,
            namespace,
        )
        identity = normalize_runtime_identity(namespace.get("result"))
    except Exception as exc:
        raise RuntimeError(
            "could not measure the worker-process runtime ABI sentinel"
        ) from exc
    return identity


def runtime_abi_sentinel_from_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct and validate the signed representative ABI sentinel."""

    native = payload.get("native_runtime")
    expected_native_keys = {
        "schema", "scope", "process_isolation", "numpy", "pythonocc", "occt"
    }
    if type(native) is not dict or set(native) != expected_native_keys:
        raise ValueError("signed native runtime ABI sentinel is malformed")
    return normalize_runtime_identity(
        {
            "schema": native["schema"],
            "scope": native["scope"],
            "process_isolation": native["process_isolation"],
            "python": payload.get("python"),
            "numpy": native["numpy"],
            "pythonocc": native["pythonocc"],
            "occt": native["occt"],
        }
    )


def current_source_binding_failures(
    sources: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Return CAD IDs whose current pickle bytes differ from the signed run.

    This check deliberately reopens each source path.  Row-level five-point
    binding proves what a worker consumed, while this terminal check proves
    that a later resume or audit is still anchored to the same local bytes.
    Missing and unreadable files fail closed without rewriting old evidence.
    """

    failures: list[str] = []
    for source in sources:
        cad_id = str(source["cad_id"])
        try:
            expected = normalize_binding(bindings[cad_id])
            current = source_binding(Path(str(source["source_path"])))
        except (KeyError, OSError, TypeError, ValueError):
            failures.append(cad_id)
            continue
        if current != expected:
            failures.append(cad_id)
    return failures


def payload_binding(payload: bytes) -> dict[str, Any]:
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def normalize_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"bytes", "sha256"}:
        raise ValueError("source binding must contain exactly bytes and sha256")
    count, digest = value.get("bytes"), value.get("sha256")
    if type(count) is not int or count <= 0:
        raise ValueError("source binding bytes must be positive")
    if (
        not isinstance(digest, str) or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("source binding sha256 must be lowercase hex")
    return {"bytes": count, "sha256": digest}


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(dict(value), indent=2, sort_keys=True,
                                    ensure_ascii=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def read_rows(path: Path, *, recover_truncated_tail: bool = False) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        return []
    payload = target.read_bytes()
    rows: list[dict[str, Any]] = []
    offset = 0
    physical_rows = payload.splitlines(keepends=True)
    for index, raw in enumerate(physical_rows):
        is_last = index == len(physical_rows) - 1
        committed_by_newline = raw.endswith((b"\n", b"\r"))
        if recover_truncated_tail and is_last and not committed_by_newline:
            # append_row commits a JSONL record with its newline.  Even a
            # syntactically complete JSON object without that delimiter may
            # be the surviving prefix of a crashed write; accepting it would
            # make the next append concatenate two objects as `}{`.
            with target.open("r+b") as handle:
                handle.truncate(offset)
                handle.flush()
                os.fsync(handle.fileno())
            return rows
        if not raw.strip():
            offset += len(raw)
            continue
        try:
            value = strict_json_loads(
                raw.decode("utf-8"), label=f"census JSONL row {index + 1}"
            )
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
            raise
        if not isinstance(value, dict):
            raise ValueError("census JSONL must contain objects")
        rows.append(value)
        offset += len(raw)
    return rows


def append_row(path: Path, row: Mapping[str, Any]) -> None:
    assert_path_free_finite(row, label="census row")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=True,
                                allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def output_writer_lock(output_dir: Path) -> Iterator[None]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / LOCK_NAME
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    handle = os.fdopen(descriptor, "r+b", buffering=0)
    acquired = False
    try:
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b"\0")
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as exc:
            raise RuntimeError("census output already has an active writer") from exc
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def parse_worker_result(stdout: str) -> dict[str, Any] | None:
    lines = [line for line in str(stdout).splitlines() if line.strip()]
    positions = [index for index, line in enumerate(lines)
                 if line.startswith(WORKER_MARKER)]
    if not lines or positions != [len(lines) - 1]:
        return None
    try:
        value = strict_json_loads(
            lines[-1][len(WORKER_MARKER):], label="worker sentinel"
        )
    except (ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _unique_by_cad(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        cad_id = row.get("cad_id") if isinstance(row, Mapping) else None
        if not isinstance(cad_id, str) or not cad_id or cad_id in result:
            raise ValueError(f"{label} CAD identities must be unique strings")
        result[cad_id] = row
    return result


def _require_zero_optional_counter(row: Mapping[str, Any], key: str, label: str) -> None:
    if key in row and (type(row.get(key)) is not int or row.get(key) != 0):
        raise ValueError(f"{label} {key} must be integer zero")


def _validate_selector_health(row: Mapping[str, Any]) -> None:
    """Reject operational failures while retaining scientific assembly failures."""

    status = row.get("status")
    if status not in {"both_valid", "step_invalid", "assembly_error"}:
        raise ValueError("selector row contains a worker/protocol or unknown status")
    if status == "assembly_error" and row.get("error_type") != "RuntimeError":
        raise ValueError("selector assembly_error evidence type drifted")
    if isinstance(status, str) and status.startswith("worker_"):
        raise ValueError("selector row contains a worker failure")
    if "worker_returncode" in row and row.get("worker_returncode") != 0:
        raise ValueError("selector row worker return code is nonzero")
    for key in ("nonfinite_count", "worker_or_protocol_failures", "protocol_failure_count"):
        _require_zero_optional_counter(row, key, "selector row")
    if "protocol_failures" in row and row.get("protocol_failures") != []:
        raise ValueError("selector row protocol failures must be empty")

    selection = row.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("selector row lacks selection evidence")
    candidates = selection.get("candidates")
    attempted = selection.get("attempted_profiles")
    if (
        not isinstance(candidates, Sequence)
        or isinstance(candidates, (str, bytes))
        or not isinstance(attempted, Sequence)
        or isinstance(attempted, (str, bytes))
        or len(candidates) != len(attempted)
        or not candidates
    ):
        raise ValueError("selector candidate evidence is malformed")
    for index, candidate in enumerate(candidates):
        label = f"selector candidate {index}"
        if not isinstance(candidate, Mapping):
            raise ValueError(f"{label} is malformed")
        candidate_status = candidate.get("status")
        if candidate_status not in {"both_valid", "step_invalid", "assembly_error"}:
            raise ValueError(f"{label} contains a worker/protocol or unknown status")
        if candidate_status == "assembly_error" and candidate.get("error_type") != "RuntimeError":
            raise ValueError(f"{label} assembly_error evidence type drifted")
        if candidate.get("worker_returncode") != 0:
            raise ValueError(f"{label} worker return code is not zero")
        if candidate.get("profile") != attempted[index]:
            raise ValueError(f"{label} profile order drifted")
        for key in ("nonfinite_count", "worker_or_protocol_failures", "protocol_failure_count"):
            _require_zero_optional_counter(candidate, key, label)
        if "protocol_failures" in candidate and candidate.get("protocol_failures") != []:
            raise ValueError(f"{label} protocol failures must be empty")
        for text in (
            candidate_status,
            candidate.get("error_type"),
            *(candidate.get("rejection_reasons") or []),
        ):
            normalized = str(text or "").lower()
            if "worker_" in normalized or "worker protocol" in normalized:
                raise ValueError(f"{label} contains worker/protocol failure evidence")


def select_census_sources(
    calibration_rows: Sequence[Mapping[str, Any]],
    selector_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Prove frozen 100 -> selector residual 9 -> exclude exact negatives -> 7."""
    if len(calibration_rows) not in {100, 300}:
        raise ValueError("calibration must contain 100 originals or formal 3x100 arms")
    originals = [row for row in calibration_rows if row.get("arm") == "original"]
    if len(originals) != 100 or sum(row.get("brep_valid") is True for row in originals) != 84:
        raise ValueError("calibration must contain exactly 100 originals and 84 valid")
    original_by_id = _unique_by_cad(originals, "calibration original")
    original_order = [str(row["cad_id"]) for row in originals]
    for arm in sorted({str(row.get("arm")) for row in calibration_rows}):
        arm_rows = [row for row in calibration_rows if str(row.get("arm")) == arm]
        arm_by_id = _unique_by_cad(arm_rows, f"calibration {arm}")
        if len(arm_rows) != 100 or set(arm_by_id) != set(original_by_id):
            raise ValueError("calibration arms change the frozen cohort")
        if [str(row["cad_id"]) for row in arm_rows] != original_order:
            raise ValueError("calibration arms change the frozen order")
    if len(selector_rows) != 100:
        raise ValueError("selector must contain exactly 100 rows")
    for row in selector_rows:
        if not isinstance(row, Mapping):
            raise ValueError("selector rows must be mappings")
        _validate_selector_health(row)
    selector_by_id = _unique_by_cad(selector_rows, "selector")
    if set(selector_by_id) != set(original_by_id):
        raise ValueError("selector and calibration cohorts differ")
    if [str(row["cad_id"]) for row in selector_rows] != original_order:
        raise ValueError("selector and calibration order differ")
    if sum(row.get("strict_brep_valid") is True for row in selector_rows) != 91:
        raise ValueError("selector must contain exactly 91 strict-valid CADs")
    for cad_id, original in original_by_id.items():
        selected = selector_by_id[cad_id]
        if selected.get("parent_id") != original.get("parent_id"):
            raise ValueError("selector parent identity drifted")
        if selected.get("historical_strict_valid") is not original.get("brep_valid"):
            raise ValueError("selector historical validity drifted")
        if original.get("brep_valid") is True and selected.get("strict_brep_valid") is not True:
            raise ValueError("selector regressed a historical control")
    residual = frozenset(
        str(row["cad_id"]) for row in selector_rows
        if row.get("strict_brep_valid") is False
    )
    if residual != SELECTOR_RESIDUAL_CAD_IDS:
        raise ValueError("selector residual identity set drifted")
    if residual - EXCLUDED_EXACT_NEGATIVE_CAD_IDS != frozenset(TARGET_CAD_IDS):
        raise ValueError("seven-CAD exclusion result drifted")
    for cad_id in TARGET_CAD_IDS:
        selection = selector_by_id[cad_id].get("selection")
        if not isinstance(selection, Mapping) or selection.get("primary_profile") != PRIMARY_PROFILE_NAME:
            raise ValueError(f"selector primary profile drifted for {cad_id}")
    return [dict(original_by_id[cad_id]) for cad_id in TARGET_CAD_IDS]


def _git_identity(repo_root: Path, *, allow_dirty_development: bool) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=repo_root, capture_output=True,
        text=True, encoding="utf-8", errors="replace", check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo_root,
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    upstream = subprocess.run(
        ["git", "rev-parse", "--verify", "@{upstream}^{commit}"],
        cwd=repo_root, capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=False,
    )
    if revision.returncode or status.returncode or upstream.returncode:
        raise RuntimeError("could not bind repository identity")
    commit = revision.stdout.strip()
    upstream_commit = upstream.stdout.strip()
    if (
        FULL_GIT_COMMIT.fullmatch(commit) is None
        or FULL_GIT_COMMIT.fullmatch(upstream_commit) is None
    ):
        raise RuntimeError("repository HEAD is not a verified 40-hex commit object")
    dirty = bool(status.stdout.strip())
    if dirty and not allow_dirty_development:
        raise RuntimeError("formal census requires a clean Git worktree")
    if not dirty and commit.lower() != upstream_commit.lower():
        raise RuntimeError("formal census requires HEAD to match its upstream")
    return {
        "commit": commit.lower(), "dirty": dirty,
        "formal": not dirty and not allow_dirty_development,
        "upstream_commit": upstream_commit.lower(),
        "head_matches_upstream": commit.lower() == upstream_commit.lower(),
        "status_sha256": hashlib.sha256(status.stdout.encode("utf-8")).hexdigest(),
    }


def _selector_run_binding(
    selector_run: Path, *, calibration_manifest: Path, selector_matrix: Path,
    source_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if sha256_file(selector_run) != FROZEN_SELECTOR_RUN_SHA256:
        raise ValueError("selector run does not match the frozen SHA-256")
    if sha256_file(calibration_manifest) != FROZEN_CALIBRATION_MANIFEST_SHA256:
        raise ValueError("calibration manifest does not match the frozen SHA-256")
    if sha256_file(selector_matrix) != FROZEN_SELECTOR_MATRIX_SHA256:
        raise ValueError("selector matrix does not match the frozen SHA-256")
    value = strict_json_loads(
        Path(selector_run).read_text(encoding="utf-8"), label="selector run"
    )
    payload = value.get("payload") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping) or value.get("schema") != "assembly-repair-run-v2"
        or value.get("status") != "COMPLETED" or value.get("attempts") != 100
        or not isinstance(payload, Mapping)
        or canonical_sha256(payload) != value.get("signature")
        or payload.get("calibration_manifest_sha256") != sha256_file(calibration_manifest)
        or value.get("final_matrix_sha256") != sha256_file(selector_matrix)
        or payload.get("run_kind") != "assembly-repair-selector-v1"
        or payload.get("matrix_schema") != "assembly-repair-selector-v1"
        or payload.get("candidate_schema") != "assembly-selector-candidate-v1"
        or payload.get("full_cohort_count") != 100
        or payload.get("selected_cohort_count") != 100
        or payload.get("historical_invalid_only") is not False
    ):
        raise ValueError("selector run is not the signed completed 100-CAD run")
    registered = payload.get("selected_source_pickles")
    if not isinstance(registered, Mapping):
        raise ValueError("selector run lacks source bindings")
    for item in source_bindings:
        if normalize_binding(registered.get(str(item["cad_id"])) or {}) != normalize_binding(item["binding"]):
            raise ValueError("selector run source binding mismatch")
    return {
        "bytes": Path(selector_run).stat().st_size,
        "sha256": FROZEN_SELECTOR_RUN_SHA256, "signature": value["signature"],
        "status": "COMPLETED",
    }


def _source_hashes(repo_root: Path) -> dict[str, str]:
    relative = (
        "tools/probe_source_bound_stage_census.py",
        "tools/assembly_stage_lineage.py",
        "tools/directed_trim_assembly.py",
        "tools/assembly_repair.py",
        "tools/assembly_selector_geometry.py",
        "tools/diagnose_assembly_face_wires.py",
        "tools/diagnose_step_validity_components.py",
        "tools/probe_downstream_bad_wire_lineage.py",
        "tools/run_assembly_calibration_oracle.py",
        "tools/run_assembly_repair_matrix.py",
    )
    return {name: sha256_file(repo_root / name) for name in relative}


def build_run_payload(
    args: argparse.Namespace, *, sources: Sequence[Mapping[str, Any]],
    selector_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    runtime_utils = Path(args.breparg_root).resolve() / "utils.py"
    if not runtime_utils.is_file():
        raise FileNotFoundError(runtime_utils)
    runtime_utils_sha256 = sha256_file(runtime_utils)
    if runtime_utils_sha256 != FROZEN_BREPARG_UTILS_SHA256:
        raise ValueError("BrepARG runtime utils.py does not match the frozen selector")
    calibration_sha256 = sha256_file(args.calibration_manifest)
    selector_matrix_sha256 = sha256_file(args.selector_matrix)
    selector_run_sha256 = sha256_file(args.selector_run)
    if calibration_sha256 != FROZEN_CALIBRATION_MANIFEST_SHA256:
        raise ValueError("calibration manifest does not match the frozen SHA-256")
    if selector_matrix_sha256 != FROZEN_SELECTOR_MATRIX_SHA256:
        raise ValueError("selector matrix does not match the frozen SHA-256")
    if selector_run_sha256 != FROZEN_SELECTOR_RUN_SHA256:
        raise ValueError("selector run does not match the frozen SHA-256")
    runtime_identity = _runtime_identity()
    bindings = [
        {"cad_id": str(source["cad_id"]), "binding": source_binding(Path(str(source["source_path"])))}
        for source in sources
    ]
    repository = {
        **_git_identity(repo_root, allow_dirty_development=bool(args.development_allow_dirty)),
        "source_sha256": _source_hashes(repo_root),
    }
    selector_by_id = {str(row["cad_id"]): row for row in selector_rows}
    selector_run = _selector_run_binding(
        args.selector_run, calibration_manifest=args.calibration_manifest,
        selector_matrix=args.selector_matrix, source_bindings=bindings,
    )
    return {
        "schema": RUN_SCHEMA,
        "run_kind": (
            "formal"
            if repository["formal"]
            else "development_dirty_nonformal_nonresumable"
        ),
        "calibration_manifest_sha256": calibration_sha256,
        "selector_matrix_sha256": selector_matrix_sha256,
        "selector_run": selector_run,
        "selector": {
            "cohort_count": 100, "strict_valid": 91,
            "historical_valid_preserved": 84, "regressions": 0,
            "residual_cad_ids": sorted(SELECTOR_RESIDUAL_CAD_IDS),
        },
        "exact_negative_evidence": dict(EXACT_NEGATIVE_EVIDENCE),
        "excluded_exact_negative_cad_ids": sorted(EXCLUDED_EXACT_NEGATIVE_CAD_IDS),
        "ordered_target_cad_ids": list(TARGET_CAD_IDS),
        "ordered_tasks": [asdict(task) | {"task_id": task.task_id} for task in TASKS],
        "sources": [
            {
                "cad_id": str(source["cad_id"]), "parent_id": str(source["parent_id"]),
                "historical_strict_valid": bool(source["brep_valid"]),
                "selector_strict_valid": bool(selector_by_id[str(source["cad_id"])] ["strict_brep_valid"]),
                "binding": item["binding"],
            }
            for source, item in zip(sources, bindings)
        ],
        "stages": [{"stage": stage, "phase": STAGE_NAMES[stage]} for stage in STAGE_ORDER],
        "stage_record_schema": STAGE_RECORD_SCHEMA,
        "stage_assessment_schema": ASSESSMENT_SCHEMA,
        "schema_v2": {
            "identity": "assembly-selector-geometry-gate-v2",
            "max_bbox_relative_delta": 0.02,
            "max_edge_length_relative_delta": 0.05,
            "max_edge_sample_rms_normalized": 0.01,
            "max_edge_sample_max_normalized": 0.05,
            "unchanged": True,
        },
        "joint_iterations": int(args.joint_iterations),
        "worker_timeout_seconds": float(args.worker_timeout_seconds),
        "python": runtime_identity["python"],
        "native_runtime": {
            key: runtime_identity[key]
            for key in (
                "schema", "scope", "process_isolation", "numpy", "pythonocc",
                "occt",
            )
        },
        "repository": repository,
        "breparg_runtime": {"utils_sha256": runtime_utils_sha256},
        "authorization_ceiling": "exact_candidate_design_only",
    }


def bind_run_manifest(output_dir: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(output_dir)
    path = root / RUN_NAME
    expected_payload = _canonical_json_value(payload)
    if type(expected_payload) is not dict:
        raise ValueError("census run payload must be a JSON object")
    signature = canonical_sha256(expected_payload)
    if path.is_file():
        if payload.get("run_kind") != "formal":
            raise RuntimeError("dirty development census output is non-resumable")
        try:
            current = strict_json_loads(
                path.read_text(encoding="utf-8"), label="census run manifest"
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("census run manifest is malformed") from exc
        if type(current) is not dict:
            raise RuntimeError("census run manifest is not an object")
        status = current.get("status")
        if type(status) is not str or status not in {
            "RUNNING", "COMPLETED", "INCONCLUSIVE"
        }:
            raise RuntimeError("census run status is not registered")
        expected_keys = {"schema", "signature", "payload", "status"}
        if status in {"COMPLETED", "INCONCLUSIVE"}:
            expected_keys.update({"attempts", "rows_sha256", "summary_sha256"})
            if type(current.get("attempts")) is not int or current["attempts"] != len(TASKS):
                raise RuntimeError("terminal census manifest attempts are malformed")
            try:
                _require_sha256(
                    current.get("rows_sha256"), label="terminal rows hash"
                )
                _require_sha256(
                    current.get("summary_sha256"), label="terminal summary hash"
                )
            except ValueError as exc:
                raise RuntimeError("terminal census manifest hashes are malformed") from exc
        if set(current) != expected_keys:
            raise RuntimeError("census run manifest has unexpected fields")
        stored_payload = current.get("payload")
        stored_signature = current.get("signature")
        try:
            _require_sha256(stored_signature, label="census run signature")
            recomputed_signature = canonical_sha256(stored_payload)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("census run manifest signature is malformed") from exc
        if (
            type(current.get("schema")) is not str
            or current.get("schema") != RUN_SCHEMA
            or stored_signature != recomputed_signature
            or stored_signature != signature
            or not _json_exact_equal(stored_payload, expected_payload)
        ):
            raise RuntimeError("output root belongs to another census signature")
        return dict(current)
    unexpected = [item for item in root.iterdir() if item.name not in {LOCK_NAME, RUN_NAME}]
    if unexpected:
        raise RuntimeError("unsigned census output root is not empty")
    value = {"schema": RUN_SCHEMA, "signature": signature,
             "payload": expected_payload, "status": "RUNNING"}
    atomic_json(path, value)
    return value


def _source_topology(parsed: Mapping[str, Any]) -> dict[str, Any]:
    face_edges = [list(map(int, values)) for values in parsed["faceEdge_adj"]]
    edge_vertices = [sorted(map(int, values)) for values in parsed["edgeCorner_adj"]]
    if not face_edges or not edge_vertices or any(len(values) != 2 for values in edge_vertices):
        raise ValueError("source topology is empty or malformed")
    vertex_ids = sorted({value for edge in edge_vertices for value in edge})
    if vertex_ids != list(range(len(vertex_ids))):
        raise ValueError("source vertex ids are not contiguous")
    face_incidence = sorted(len(values) for values in face_edges)
    edge_faces: list[list[int]] = [[] for _ in edge_vertices]
    for values in face_edges:
        for edge_id in values:
            if not 0 <= edge_id < len(edge_vertices):
                raise ValueError("source face edge id is out of range")
    for face_id, values in enumerate(face_edges):
        # Preserve repeated uses.  A seam appears twice in both canonical
        # directions and is distinguished later by its duplicate ordinal.
        for edge_id in values:
            edge_faces[edge_id].append(face_id)
    vertex_edge = [0] * len(vertex_ids)
    for first, second in edge_vertices:
        vertex_edge[first] += 1
        vertex_edge[second] += 1
    return {
        "face_count": len(face_edges), "edge_count": len(edge_vertices),
        "vertex_count": len(vertex_ids),
        "face_edge_occurrence_count": sum(face_incidence),
        "face_edge_incidence_counts": face_incidence,
        "edge_face_incidence_counts": sorted(map(len, edge_faces)),
        "vertex_edge_incidence_counts": sorted(vertex_edge),
        "face_edge_source_ids": face_edges,
        "edge_face_source_ids": edge_faces,
        "edge_vertex_source_ids": edge_vertices,
    }


def _occurrence_keys(face_edge_source_ids: Sequence[Sequence[int]]) -> list[list[int]]:
    result: list[list[int]] = []
    for face_id, edge_ids in enumerate(face_edge_source_ids):
        duplicate_ordinals: Counter[int] = Counter()
        for edge_id in edge_ids:
            edge = int(edge_id)
            result.append([int(face_id), edge, int(duplicate_ordinals[edge])])
            duplicate_ordinals[edge] += 1
    return sorted(result)


def _lineage_entities(source_topology: Mapping[str, Any], *, faces: int | None = None,
                      edges: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if faces is not None:
        result["faces"] = {
            "source_count": int(source_topology["face_count"]), "observed_count": int(faces),
            "mapped_source_count": int(faces), "mapped_observed_count": int(faces),
            "max_observed_per_source": 1, "max_source_per_observed": 1,
            "solution_count": 1,
        }
    if edges is not None:
        result["edges"] = {
            "source_count": int(source_topology["edge_count"]), "observed_count": int(edges),
            "mapped_source_count": int(edges), "mapped_observed_count": int(edges),
            "max_observed_per_source": 1, "max_source_per_observed": 1,
            "solution_count": 1,
        }
    return result


def _lineage(status: str, source_topology: Mapping[str, Any], *, faces: int | None = None,
             edges: int | None = None, failures: Sequence[str] = (),
             source_face_ids: Sequence[int] | None = None,
             source_edge_ids: Sequence[int] | None = None,
             source_edge_occurrence_keys: Sequence[Sequence[int]] | None = None,
             distributed_scope: Mapping[str, Any] | None = None,
             whole_stage_terminal: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result = {
        "status": status, "proof_method": "source_identity_or_authoritative_history",
        "solution_count": 1,
        "failure_codes": list(dict.fromkeys(map(str, failures))),
        "entities": _lineage_entities(source_topology, faces=faces, edges=edges),
    }
    if source_face_ids is not None:
        result["source_face_ids"] = sorted(map(int, source_face_ids))
    if source_edge_ids is not None:
        result["source_edge_ids"] = sorted(map(int, source_edge_ids))
    if source_edge_occurrence_keys is not None:
        result["source_edge_occurrence_keys"] = sorted(
            [list(map(int, value)) for value in source_edge_occurrence_keys]
        )
    if distributed_scope is not None:
        result["distributed_scope"] = dict(distributed_scope)
    if whole_stage_terminal is not None:
        result["whole_stage_terminal"] = dict(whole_stage_terminal)
    return result


def _stage_local_occ_topology_proof(
    stage: str, *, scope_count: int, source_edge_count: int,
    constraint_occurrence_count: int, failures: Sequence[str] = (),
) -> dict[str, Any]:
    """Return the path-free residue of identity checks local to an OCC owner."""

    failure_codes = list(dict.fromkeys(map(str, failures)))
    return {
        "status": (
            "exact_stage_local_topology" if not failure_codes else "ambiguous"
        ),
        "proof_method": (
            "source_edge_endpoint_labels_to_stage_local_occ_identity_classes_v1"
        ),
        "scope_kind": "source_edge" if stage == "S2" else "source_face",
        "scope_count": int(scope_count),
        "source_edge_count": int(source_edge_count),
        "constraint_occurrence_count": int(constraint_occurrence_count),
        "max_observed_per_source_within_scope": 1 if not failure_codes else 0,
        "max_source_per_observed_within_scope": 1 if not failure_codes else 0,
        "failure_codes": failure_codes,
    }


def _prove_stage_local_occ_topology(
    scopes: Sequence[Sequence[tuple[int, Sequence[int], Any]]],
    *, scope_kind: str,
) -> dict[str, Any]:
    """Prove endpoint-label identity only inside each owning OCC scope.

    MakeEdge and face construction may lawfully copy endpoint handles across
    independent edges/faces.  Consequently no handle is compared with a
    different scope or with a prior stage.  Within one scope, equal source
    labels must share one OCC identity class and distinct labels must not
    merge.
    """

    failures: list[str] = []
    source_edge_ids: set[int] = set()
    constraint_count = 0
    for scope_index, occurrences in enumerate(scopes):
        representatives: list[Any] = []
        normalized: list[tuple[tuple[int, int], tuple[int, int]]] = []
        source_labels: set[int] = set()
        for occurrence_index, occurrence in enumerate(occurrences):
            try:
                edge_id, labels, edge = occurrence
                edge_id = int(edge_id)
                labels = tuple(map(int, labels))
                if len(labels) != 2:
                    raise ValueError("source endpoint pair must have length two")
                endpoints = _edge_endpoint_handles(edge)
                if len(endpoints) != 2:
                    raise ValueError("observed endpoint pair must have length two")
            except Exception as exc:
                failures.append(
                    f"scope_{scope_index}_occurrence_{occurrence_index}_malformed:"
                    f"{type(exc).__name__}"
                )
                continue
            source_edge_ids.add(edge_id)
            constraint_count += 1
            observed_classes: list[int] = []
            identity_measurement_failed = False
            for endpoint_index, endpoint in enumerate(endpoints):
                try:
                    matches = [
                        index
                        for index, representative in enumerate(representatives)
                        if _same_occ_identity(endpoint, representative)
                    ]
                except Exception as exc:
                    failures.append(
                        f"scope_{scope_index}_occurrence_{occurrence_index}_"
                        "identity_measurement_failed:"
                        f"{type(exc).__name__}"
                    )
                    identity_measurement_failed = True
                    break
                if len(matches) > 1:
                    failures.append(
                        f"scope_{scope_index}_identity_class_nonunique"
                    )
                    identity_measurement_failed = True
                    break
                if matches:
                    class_id = matches[0]
                else:
                    class_id = len(representatives)
                    representatives.append(endpoint)
                observed_classes.append(class_id)
            if not identity_measurement_failed and len(observed_classes) == 2:
                source_labels.update(labels)
                normalized.append(
                    ((int(labels[0]), int(labels[1])),
                     (observed_classes[0], observed_classes[1]))
                )
        labels = sorted(source_labels)
        label_index = {label: index for index, label in enumerate(labels)}
        if len(labels) != len(representatives):
            failures.append(
                f"scope_{scope_index}_source_vertex_split_or_merge"
            )
            continue
        constraints = [
            (
                label_index[source_pair[0]], label_index[source_pair[1]],
                observed_pair[0], observed_pair[1],
            )
            for source_pair, observed_pair in normalized
        ]
        compatibility: list[set[int]] = [
            set(range(len(representatives))) for _ in labels
        ]
        for source_first, source_second, observed_first, observed_second in constraints:
            endpoints = {observed_first, observed_second}
            compatibility[source_first].intersection_update(endpoints)
            compatibility[source_second].intersection_update(endpoints)
        order = sorted(range(len(labels)), key=lambda value: len(compatibility[value]))
        assignment = [-1] * len(labels)
        solution_found = False

        def search(depth: int, used: set[int]) -> None:
            nonlocal solution_found
            if solution_found:
                return
            if depth == len(order):
                solution_found = all(
                    sorted((assignment[source_first], assignment[source_second]))
                    == sorted((observed_first, observed_second))
                    for source_first, source_second, observed_first, observed_second
                    in constraints
                )
                return
            source_id = order[depth]
            for observed_id in sorted(compatibility[source_id] - used):
                assignment[source_id] = observed_id
                if all(
                    assignment[other] < 0
                    or sorted((observed_id, assignment[other]))
                    == sorted((observed_first, observed_second))
                    for first, second, observed_first, observed_second in constraints
                    for other in (
                        [second] if first == source_id
                        else [first] if second == source_id
                        else []
                    )
                ):
                    search(depth + 1, {*used, observed_id})
                assignment[source_id] = -1

        search(0, set())
        if not solution_found:
            failures.append(f"scope_{scope_index}_source_topology_not_bijective")
    return _stage_local_occ_topology_proof(
        "S2" if scope_kind == "source_edge" else "S3",
        scope_count=len(scopes),
        source_edge_count=len(source_edge_ids),
        constraint_occurrence_count=constraint_count,
        failures=failures,
    )


def _shape_topology(shape: Any) -> dict[str, Any]:
    from tools.assembly_selector_geometry import _shape_counts

    counts = _shape_counts(shape)
    return {
        "face_count": int(counts["face_count"]), "edge_count": int(counts["edge_count"]),
        "vertex_count": int(counts["vertex_count"]),
        "face_edge_occurrence_count": int(counts["face_edge_occurrences"]),
        "face_edge_incidence_counts": list(counts["face_edge_incidence_counts"]),
        "edge_face_incidence_counts": list(counts["edge_face_incidence_counts"]),
        "vertex_edge_incidence_counts": list(counts["vertex_edge_incidence_counts"]),
    }


def _merge_native_counts_with_source_relations(
    native_topology: Mapping[str, Any],
    mapped_topology: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach source labels only when native OCC census agrees exactly.

    Native counts remain the observation.  Mapping-derived relations are
    labels for those native entities; disagreement is retained as topology
    drift rather than replacing either side with expected source values.
    """

    result = dict(native_topology)
    for key in (
        "face_edge_source_ids",
        "edge_face_source_ids",
        "edge_vertex_source_ids",
    ):
        result[key] = mapped_topology[key]
    return result


def _observed_source_topology(
    source_topology: Mapping[str, Any],
    face_edge_source_ids: Sequence[Sequence[int]],
    edge_vertex_source_ids: Mapping[int, Sequence[int]],
) -> dict[str, Any]:
    """Reduce proof-bearing face assignments to canonical source-id relations.

    Counts here come from actual observed occurrence assignments.  Endpoint
    labels are carried by the uniquely assigned source edge, not copied as a
    claim that an arbitrary OCC explorer ordinal is correspondence.
    """

    rows = [list(map(int, values)) for values in face_edge_source_ids]
    edge_count = int(source_topology["edge_count"])
    edge_faces: list[list[int]] = [[] for _ in range(edge_count)]
    for face_id, edge_ids in enumerate(rows):
        for edge_id in edge_ids:
            if not 0 <= edge_id < edge_count:
                raise ValueError("observed mapping contains an out-of-range source edge")
            edge_faces[edge_id].append(int(face_id))
    observed_edges = sorted({edge_id for row in rows for edge_id in row})
    if observed_edges != list(range(edge_count)):
        raise ValueError("observed mapping does not cover every source edge")
    if sorted(map(int, edge_vertex_source_ids)) != observed_edges:
        raise ValueError("observed endpoint identity does not cover every mapped edge")
    edge_vertices = [
        sorted(map(int, edge_vertex_source_ids[edge_id]))
        for edge_id in observed_edges
    ]
    vertex_count = int(source_topology["vertex_count"])
    vertex_edge = [0] * vertex_count
    for endpoints in edge_vertices:
        if len(endpoints) != 2:
            raise ValueError("observed source edge lacks two canonical endpoints")
        for vertex_id in endpoints:
            if not 0 <= vertex_id < vertex_count:
                raise ValueError("observed mapping contains an out-of-range source vertex")
            vertex_edge[vertex_id] += 1
    return {
        "face_count": len(rows),
        "edge_count": len(observed_edges),
        "vertex_count": vertex_count,
        "face_edge_occurrence_count": sum(map(len, rows)),
        "face_edge_incidence_counts": sorted(map(len, rows)),
        "edge_face_incidence_counts": sorted(map(len, edge_faces)),
        "vertex_edge_incidence_counts": sorted(vertex_edge),
        "face_edge_source_ids": rows,
        "edge_face_source_ids": edge_faces,
        "edge_vertex_source_ids": edge_vertices,
    }


def _edge_endpoint_handles(edge: Any) -> tuple[Any, Any]:
    """Return the two endpoint occurrences of one private OCC edge handle."""

    from OCC.Extend.TopologyUtils import TopologyExplorer

    endpoints = list(TopologyExplorer(edge, ignore_orientation=False).vertices())
    if len(endpoints) != 2:
        raise ValueError("edge_endpoint_occurrence_count_not_two")
    return endpoints[0], endpoints[1]


def _same_occ_identity(first: Any, second: Any) -> bool:
    try:
        return bool(first.IsSame(second))
    except Exception as exc:
        raise RuntimeError("occ_endpoint_identity_measurement_failed") from exc


def _endpoint_pair_is_same(
    observed: Sequence[Any], reference: Sequence[Any]
) -> bool:
    """Compare unordered endpoint occurrence pairs, preserving self-loops."""

    if len(observed) != 2 or len(reference) != 2:
        return False
    return bool(
        (
            _same_occ_identity(observed[0], reference[0])
            and _same_occ_identity(observed[1], reference[1])
        )
        or (
            _same_occ_identity(observed[0], reference[1])
            and _same_occ_identity(observed[1], reference[0])
        )
    )


def _source_vertex_lineage_failure(
    *,
    source_vertex_count: int,
    observed_vertex_count: int,
    constraint_occurrence_count: int,
    failure_codes: Sequence[str],
    solution_count: int | None = None,
) -> dict[str, Any]:
    """Return one native-handle-free global endpoint-lineage failure proof."""

    return {
        "status": "ambiguous",
        "proof_method": GLOBAL_VERTEX_IDENTITY_PROOF_METHOD,
        "solution_count": solution_count,
        "solution_count_capped_at_two": True,
        "source_vertex_count": int(source_vertex_count),
        "observed_vertex_count": int(observed_vertex_count),
        "mapped_source_vertex_count": 0,
        "mapped_observed_vertex_count": 0,
        "max_observed_per_source": 0,
        "max_source_per_observed": 0,
        "constraint_occurrence_count": int(constraint_occurrence_count),
        "failure_codes": list(dict.fromkeys(map(str, failure_codes))),
    }


def _unique_occ_identity_representatives(values: Sequence[Any]) -> list[Any]:
    """Collapse borrowed OCC vertices into unique ``IsSame`` identity classes."""

    representatives: list[Any] = []
    for value in values:
        matches = [
            index
            for index, representative in enumerate(representatives)
            if _same_occ_identity(value, representative)
        ]
        if len(matches) > 1:
            raise RuntimeError("occ_vertex_identity_class_not_unique")
        if not matches:
            representatives.append(value)
    return representatives


def _occ_identity_class_index(value: Any, representatives: Sequence[Any]) -> int:
    matches = [
        index
        for index, representative in enumerate(representatives)
        if _same_occ_identity(value, representative)
    ]
    if len(matches) != 1:
        raise RuntimeError("endpoint_observed_vertex_identity_not_unique")
    return int(matches[0])


def _prove_endpoint_occurrence_vertex_lineage(
    occurrence_rows: Sequence[Mapping[str, Any]],
    source_topology: Mapping[str, Any],
    *,
    expected_occurrence_edge_ids: Sequence[int],
) -> dict[str, Any]:
    """Prove one unique source-to-OCC vertex map over completed occurrences.

    S2 contributes one occurrence per completed source edge.  S3/S4 contribute
    every source-bound face-edge occurrence in the completed face prefix.  The
    unordered endpoint constraints must admit exactly one global bijection;
    this catches shared-vertex splits and distinct-vertex merges that per-edge
    endpoint checks cannot detect.  Borrowed native handles remain local.
    """

    expected_ids = [int(value) for value in expected_occurrence_edge_ids]
    edge_vertex_rows = source_topology.get("edge_vertex_source_ids") or []
    source_vertex_ids = sorted(
        {
            int(vertex_id)
            for edge_id in expected_ids
            if 0 <= edge_id < len(edge_vertex_rows)
            for vertex_id in edge_vertex_rows[edge_id]
        }
    )
    source_vertex_count = len(source_vertex_ids)
    source_index = {
        source_vertex_id: index
        for index, source_vertex_id in enumerate(source_vertex_ids)
    }
    failures: list[str] = []
    normalized: list[tuple[int, tuple[Any, Any]]] = []
    observed_handles: list[Any] = []

    observed_ids: list[int] = []
    for row in occurrence_rows:
        if not isinstance(row, Mapping) or type(row.get("source_edge_id")) is not int:
            failures.append("source_vertex_occurrence_row_malformed")
            continue
        edge_id = int(row["source_edge_id"])
        observed_ids.append(edge_id)
        endpoints = row.get("endpoints")
        if (
            not isinstance(endpoints, Sequence)
            or isinstance(endpoints, (str, bytes))
            or len(endpoints) != 2
        ):
            failures.append(f"source_edge_{edge_id}_endpoint_occurrences_malformed")
            continue
        normalized.append((edge_id, (endpoints[0], endpoints[1])))
        observed_handles.extend((endpoints[0], endpoints[1]))

    if Counter(observed_ids) != Counter(expected_ids):
        failures.append("source_vertex_constraint_occurrence_coverage_incomplete")
    if any(
        edge_id < 0
        or edge_id >= len(edge_vertex_rows)
        or len(edge_vertex_rows[edge_id]) != 2
        for edge_id in expected_ids
    ):
        failures.append("source_edge_vertex_relation_malformed")

    try:
        observed_vertices = _unique_occ_identity_representatives(observed_handles)
    except Exception as exc:
        failures.append(
            "observed_vertex_identity_census_failed:" + type(exc).__name__
        )
        observed_vertices = []
    observed_vertex_count = len(observed_vertices)
    constraints: list[tuple[int, int, int, int]] = []
    for edge_id, endpoints in normalized:
        if not 0 <= edge_id < len(edge_vertex_rows):
            continue
        labels = tuple(map(int, edge_vertex_rows[edge_id]))
        if len(labels) != 2 or any(value not in source_index for value in labels):
            failures.append(f"source_edge_{edge_id}_source_endpoints_malformed")
            continue
        try:
            observed_first = _occ_identity_class_index(
                endpoints[0], observed_vertices
            )
            observed_second = _occ_identity_class_index(
                endpoints[1], observed_vertices
            )
        except Exception as exc:
            failures.append(
                f"source_edge_{edge_id}_endpoint_identity_failed:{type(exc).__name__}"
            )
            continue
        source_first, source_second = labels
        if source_first == source_second and observed_first != observed_second:
            failures.append(f"source_edge_{edge_id}_self_loop_split")
        if source_first != source_second and observed_first == observed_second:
            failures.append(f"source_edge_{edge_id}_distinct_endpoints_merged")
        constraints.append(
            (
                source_index[source_first],
                source_index[source_second],
                observed_first,
                observed_second,
            )
        )

    if len(constraints) != len(expected_ids):
        failures.append("source_vertex_constraint_occurrence_coverage_incomplete")
    if observed_vertex_count > source_vertex_count:
        failures.append("source_vertex_split_or_extra_observed_vertex")
    elif observed_vertex_count < source_vertex_count:
        failures.append("source_vertex_merge_or_missing_observed_vertex")
    if failures:
        return _source_vertex_lineage_failure(
            source_vertex_count=source_vertex_count,
            observed_vertex_count=observed_vertex_count,
            constraint_occurrence_count=len(constraints),
            failure_codes=failures,
        )

    constraints_by_source: list[list[tuple[int, int, int]]] = [
        [] for _ in source_vertex_ids
    ]
    for source_first, source_second, observed_first, observed_second in constraints:
        constraints_by_source[source_first].append(
            (source_second, observed_first, observed_second)
        )
        if source_second != source_first:
            constraints_by_source[source_second].append(
                (source_first, observed_first, observed_second)
            )
    compatibility = [
        sorted(
            set(range(observed_vertex_count)).intersection(
                *(
                    {observed_first, observed_second}
                    for _other, observed_first, observed_second in rows
                )
            )
        )
        if rows
        else list(range(observed_vertex_count))
        for rows in constraints_by_source
    ]
    order = sorted(
        range(source_vertex_count), key=lambda value: len(compatibility[value])
    )
    solutions: list[list[int]] = []
    assignment = [-1] * source_vertex_count

    def search(depth: int, used: set[int]) -> None:
        if len(solutions) >= 2:
            return
        if depth == len(order):
            if all(
                sorted((assignment[source_first], assignment[source_second]))
                == sorted((observed_first, observed_second))
                for source_first, source_second, observed_first, observed_second
                in constraints
            ):
                solutions.append(list(assignment))
            return
        source_id = order[depth]
        for observed_id in compatibility[source_id]:
            if observed_id in used:
                continue
            if any(
                assignment[other_source] >= 0
                and sorted((observed_id, assignment[other_source]))
                != sorted((observed_first, observed_second))
                for other_source, observed_first, observed_second
                in constraints_by_source[source_id]
            ):
                continue
            assignment[source_id] = observed_id
            search(depth + 1, {*used, observed_id})
            assignment[source_id] = -1

    search(0, set())
    if len(solutions) != 1:
        return _source_vertex_lineage_failure(
            source_vertex_count=source_vertex_count,
            observed_vertex_count=observed_vertex_count,
            constraint_occurrence_count=len(constraints),
            solution_count=len(solutions),
            failure_codes=[
                "source_vertex_assignment_missing"
                if not solutions
                else "source_vertex_assignment_nonunique"
            ],
        )
    assignment = solutions[0]
    if any(
        sorted((assignment[source_first], assignment[source_second]))
        != sorted((observed_first, observed_second))
        for source_first, source_second, observed_first, observed_second
        in constraints
    ):
        return _source_vertex_lineage_failure(
            source_vertex_count=source_vertex_count,
            observed_vertex_count=observed_vertex_count,
            constraint_occurrence_count=len(constraints),
            solution_count=1,
            failure_codes=["source_vertex_assignment_constraint_replay_failed"],
        )
    return {
        "status": "exact_identity",
        "proof_method": GLOBAL_VERTEX_IDENTITY_PROOF_METHOD,
        "solution_count": 1,
        "solution_count_capped_at_two": True,
        "source_vertex_count": source_vertex_count,
        "observed_vertex_count": observed_vertex_count,
        "mapped_source_vertex_count": source_vertex_count,
        "mapped_observed_vertex_count": observed_vertex_count,
        "max_observed_per_source": 1,
        "max_source_per_observed": 1,
        "constraint_occurrence_count": len(constraints),
        "failure_codes": [],
    }


def _build_s2_endpoint_references(
    entries: Sequence[Mapping[str, Any]],
    source_topology: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    """Keep private S2 edge/endpoints and validate their source endpoint labels.

    The returned native handles never enter a stage record.  Only stable counts
    and failure codes are serialized by the caller.
    """

    references: dict[int, dict[str, Any]] = {}
    failures: list[str] = []
    expected_rows = source_topology["edge_vertex_source_ids"]
    for entry in entries:
        metadata = entry.get("metadata")
        if not isinstance(metadata, Mapping):
            failures.append("s2_endpoint_metadata_missing")
            continue
        if metadata.get("boundary_event", "completed") != "completed":
            continue
        edge_id = metadata.get("source_edge_id")
        if type(edge_id) is not int or not 0 <= edge_id < len(expected_rows):
            failures.append("s2_endpoint_source_edge_id_invalid")
            continue
        if edge_id in references:
            failures.append(f"source_edge_{edge_id}_s2_endpoint_reference_duplicate")
            continue
        effective_ids = metadata.get("effective_vertex_ids")
        if (
            not isinstance(effective_ids, Sequence)
            or isinstance(effective_ids, (str, bytes))
            or len(effective_ids) != 2
            or any(type(value) is not int for value in effective_ids)
        ):
            failures.append(f"source_edge_{edge_id}_s2_endpoint_labels_malformed")
            continue
        source_vertex_ids = sorted(map(int, effective_ids))
        if source_vertex_ids != sorted(map(int, expected_rows[edge_id])):
            failures.append(f"source_edge_{edge_id}_s2_endpoint_labels_drifted")
            continue
        edge = entry.get("target")
        if edge is None:
            failures.append(f"source_edge_{edge_id}_s2_edge_handle_missing")
            continue
        try:
            endpoints = _edge_endpoint_handles(edge)
            endpoints_are_same = _same_occ_identity(endpoints[0], endpoints[1])
        except Exception as exc:
            failures.append(
                f"source_edge_{edge_id}_s2_endpoint_extraction_failed:{type(exc).__name__}"
            )
            continue
        if endpoints_are_same != (source_vertex_ids[0] == source_vertex_ids[1]):
            failures.append(f"source_edge_{edge_id}_s2_endpoint_identity_relation_drifted")
            continue
        references[edge_id] = {
            "edge": edge,
            "endpoints": endpoints,
            "source_vertex_ids": source_vertex_ids,
        }
    return references, list(dict.fromkeys(failures))


def _diagnose_face_defects(face: Any, face_index: int, mapping: Mapping[str, Any]) -> list[dict[str, Any]]:
    from tools.diagnose_assembly_face_wires import diagnose_face_wires_v2

    diagnosis = diagnose_face_wires_v2(
        face, face_index=int(face_index), source_face_index=int(face_index),
        source_mapping=dict(mapping),
    )
    defects = []
    for occurrence in diagnosis.get("occurrences") or []:
        if occurrence.get("status") != "detected":
            continue
        defects.append({
            "code": f"wire_{occurrence.get('kind', 'unknown')}",
            "source_face_index": int(face_index),
            "wire_index": occurrence.get("wire_index"),
            "source_edge_ids": list(occurrence.get("source_edge_ids") or []),
            "mapping_status": occurrence.get("source_mapping_status"),
        })
    return defects


def _aggregate_face_stage(
    stage: str, entries: Sequence[Mapping[str, Any]],
    source_topology: Mapping[str, Any],
    *,
    s2_endpoint_references: Mapping[int, Mapping[str, Any]] | None = None,
    expected_face_ids: Sequence[int] | None = None,
    source_vertex_lineage: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    expected_faces = int(source_topology["face_count"])
    expected_ids = (
        list(range(expected_faces))
        if expected_face_ids is None
        else list(map(int, expected_face_ids))
    )
    if expected_ids != sorted(set(expected_ids)) or any(
        value < 0 or value >= expected_faces for value in expected_ids
    ):
        raise ValueError("expected face prefix ids are noncanonical")
    population_complete = expected_ids == list(range(expected_faces))
    indices = [int(entry["source_face_index"]) for entry in entries]
    references: dict[int, dict[str, Any]] = {}
    failures: list[str] = []
    defects: list[dict[str, Any]] = []
    observed_face_edges: dict[int, list[int]] = {}
    observed_edge_vertices: dict[int, list[int]] = {}
    stage_local_scopes: list[list[tuple[int, Sequence[int], Any]]] = []
    sewing_stage = stage in {"S5", "S6"}
    accepted_mapping_statuses = (
        {"exact_sewing_history", "exact_sewing_face_local_geometry"}
        if sewing_stage
        else {"exact_identity", "exact_face_local_geometry"}
    )
    for entry in entries:
        face_index = int(entry["source_face_index"])
        mapping = entry.get("source_mapping")
        if not isinstance(mapping, Mapping):
            failures.append(f"source_face_{face_index}_mapping_missing")
            mapping = {}
        status = str(mapping.get("status") or "unavailable")
        if status not in accepted_mapping_statuses:
            failures.extend(str(value) for value in mapping.get("failures") or [])
            failures.append(f"source_face_{face_index}_mapping_{status}")
        else:
            references[face_index] = {"face": entry["face"], "source_mapping": mapping}
        mapped_edge_ids: list[int] = []
        stage_local_occurrences: list[tuple[int, Sequence[int], Any]] = []
        wire_rows = mapping.get("wire_rows") or []
        if not isinstance(wire_rows, Sequence) or isinstance(wire_rows, (str, bytes)):
            failures.append(f"source_face_{face_index}_wire_rows_malformed")
        else:
            for wire_row in wire_rows:
                if not isinstance(wire_row, Mapping):
                    failures.append(f"source_face_{face_index}_wire_row_malformed")
                    continue
                candidates = wire_row.get("source_edge_candidates") or []
                if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
                    failures.append(f"source_face_{face_index}_edge_candidates_malformed")
                    continue
                for candidate in candidates:
                    if not isinstance(candidate, Mapping) or type(candidate.get("source_edge_id")) is not int:
                        failures.append(f"source_face_{face_index}_edge_candidate_unbound")
                        continue
                    edge_id = int(candidate["source_edge_id"])
                    mapped_edge_ids.append(edge_id)
                    if sewing_stage:
                        # Sewing is allowed to produce authoritative copies, so
                        # post-sewing handles need not be IsSame as the S2 edge
                        # endpoints.  Their source endpoint relation is backed
                        # by the constructor's global unique-assignment proof.
                        if 0 <= edge_id < len(source_topology["edge_vertex_source_ids"]):
                            observed_edge_vertices[edge_id] = list(
                                map(int, source_topology["edge_vertex_source_ids"][edge_id])
                            )
                        else:
                            failures.append(f"source_edge_{edge_id}_out_of_range")
                        continue
                    observed_edge = candidate.get("observed_edge")
                    if observed_edge is None:
                        failures.append(f"source_edge_{edge_id}_observed_edge_missing")
                        continue
                    labels = (
                        source_topology["edge_vertex_source_ids"][edge_id]
                        if 0 <= edge_id < len(
                            source_topology["edge_vertex_source_ids"]
                        )
                        else None
                    )
                    if (
                        not isinstance(labels, Sequence)
                        or isinstance(labels, (str, bytes))
                        or len(labels) != 2
                        or any(type(value) is not int for value in labels)
                    ):
                        failures.append(f"source_edge_{edge_id}_endpoint_labels_missing")
                        continue
                    canonical_labels = sorted(map(int, labels))
                    previous_labels = observed_edge_vertices.get(edge_id)
                    if previous_labels is not None and previous_labels != canonical_labels:
                        failures.append(f"source_edge_{edge_id}_endpoint_labels_inconsistent")
                        continue
                    observed_edge_vertices[edge_id] = canonical_labels
                    stage_local_occurrences.append(
                        (edge_id, canonical_labels, observed_edge)
                    )
        observed_face_edges[face_index] = mapped_edge_ids
        if not sewing_stage:
            stage_local_scopes.append(stage_local_occurrences)
        try:
            defects.extend(_diagnose_face_defects(entry["face"], face_index, mapping))
        except Exception as exc:
            failures.append(f"source_face_{face_index}_diagnosis_failed:{type(exc).__name__}")
    if sorted(indices) != expected_ids or len(indices) != len(set(indices)):
        failures.append("source_face_coverage_incomplete_or_duplicate")
    edge_occurrences = sum(map(len, observed_face_edges.values()))
    expected_rows = source_topology["face_edge_source_ids"]
    expected_occurrences = sum(len(expected_rows[index]) for index in expected_ids)
    if edge_occurrences != expected_occurrences:
        failures.append("source_edge_occurrence_coverage_mismatch")
    for face_index in expected_ids:
        if sorted(observed_face_edges.get(face_index, [])) != sorted(expected_rows[face_index]):
            failures.append(f"source_face_{face_index}_edge_occurrence_relation_mismatch")
    vertex_lineage: Mapping[str, Any] | None = source_vertex_lineage
    if not sewing_stage:
        expected_occurrence_edge_ids = [
            int(edge_id)
            for face_index in expected_ids
            for edge_id in expected_rows[face_index]
        ]
        vertex_lineage = _prove_stage_local_occ_topology(
            stage_local_scopes, scope_kind="source_face"
        )
        local_proof_failures = list(vertex_lineage.get("failure_codes") or [])
        if (
            len(stage_local_scopes) != len(expected_ids)
            or int(vertex_lineage.get("constraint_occurrence_count") or 0)
            != len(expected_occurrence_edge_ids)
            or int(vertex_lineage.get("source_edge_count") or 0)
            != len(set(expected_occurrence_edge_ids))
        ):
            local_proof_failures.append(
                "stage_local_endpoint_occurrence_coverage_incomplete"
            )
        if local_proof_failures:
            failures.extend(local_proof_failures)
            vertex_lineage = _stage_local_occ_topology_proof(
                stage,
                scope_count=len(stage_local_scopes),
                source_edge_count=int(vertex_lineage.get("source_edge_count") or 0),
                constraint_occurrence_count=int(
                    vertex_lineage.get("constraint_occurrence_count") or 0
                ),
                failures=local_proof_failures,
            )
    exact = not failures
    if sewing_stage:
        if not isinstance(source_vertex_lineage, Mapping):
            failures.append("global_source_vertex_lineage_missing")
        elif (
            source_vertex_lineage.get("status") != "exact_identity"
            or source_vertex_lineage.get("solution_count") != 1
            or source_vertex_lineage.get("failure_codes") not in ([], ())
        ):
            failures.append("global_source_vertex_lineage_not_exact")
        exact = not failures
    topology = None
    if exact and population_complete:
        topology = _observed_source_topology(
            source_topology,
            [observed_face_edges[index] for index in range(expected_faces)],
            observed_edge_vertices,
        )
    face_ids = sorted(references)
    edge_ids = sorted({edge for values in observed_face_edges.values() for edge in values})
    occurrence_keys = _occurrence_keys(
        [observed_face_edges[index] for index in sorted(observed_face_edges)]
    )
    record = {
        "stage": stage, "phase": STAGE_NAMES[stage], "status": "observed",
        "lineage": _lineage(
            "exact_identity" if exact else "ambiguous", source_topology,
            faces=len(references), edges=(int(source_topology["edge_count"]) if exact else 0),
            failures=failures,
            source_face_ids=face_ids,
            source_edge_ids=edge_ids,
            source_edge_occurrence_keys=occurrence_keys,
        ),
        "topology": topology,
        "defects": defects,
        "evidence": {
            "observed_source_face_count": len(entries),
            "exact_source_face_count": len(references),
            "source_edge_occurrence_count": edge_occurrences,
            **(
                {"stage_local_occ_topology_proof": dict(vertex_lineage)}
                if isinstance(vertex_lineage, Mapping)
                else {}
            ),
        },
    }
    if sewing_stage:
        # The global proof supersedes S2-handle identity after sewing.  Remove
        # claims that are only meaningful at S3/S4 and retain the proof
        # verbatim after path/native-safety validation by the pure normalizer.
        record["evidence"] = {"source_vertex_lineage": dict(source_vertex_lineage or {})}
    return record, references


def _validate_face_event_stream(
    event_stream: Sequence[Mapping[str, Any]], source_topology: Mapping[str, Any]
) -> tuple[list[str], str | None, int | None]:
    """Validate S3/S4 distributed face events and locate a strict next stage.

    The constructor may throw between two callbacks.  Only a canonical prefix
    makes that absence localizable; an arbitrary/missing event is never
    interpreted from exception text.
    """

    expected_faces = int(source_topology["face_count"])
    failures: list[str] = []
    terminal_stage: str | None = None
    terminal_face: int | None = None
    for position, entry in enumerate(event_stream):
        metadata = entry.get("metadata")
        if not isinstance(metadata, Mapping):
            failures.append("global_face_event_metadata_missing")
            continue
        stage = str(metadata.get("stage"))
        face_id = metadata.get("source_face_index")
        sequence = metadata.get("event_sequence_position")
        expected_stage = "S3" if position % 2 == 0 else "S4"
        expected_face = position // 2
        if stage != expected_stage:
            failures.append("global_face_stage_interleave_mismatch")
        if type(face_id) is not int or int(face_id) != expected_face:
            failures.append("global_face_source_interleave_mismatch")
        if type(sequence) is not int or int(sequence) != position:
            failures.append("global_face_event_position_mismatch")
        boundary = metadata.get("boundary_event", "completed")
        if boundary not in {"completed", "terminal_failure"}:
            failures.append("global_face_boundary_event_invalid")
        if boundary == "terminal_failure":
            if position != len(event_stream) - 1 or terminal_stage is not None:
                failures.append("global_face_terminal_event_not_last_unique")
            terminal_stage, terminal_face = stage, int(face_id)
    if not failures and terminal_stage is None and len(event_stream) not in {
        2 * expected_faces,
        *range(0, 2 * expected_faces),
    }:
        failures.append("global_face_event_stream_invalid_length")
    return list(dict.fromkeys(failures)), terminal_stage, terminal_face


def _localize_face_failure_from_event_prefix(
    event_stream: Sequence[Mapping[str, Any]],
    source_topology: Mapping[str, Any],
    construction_error: BaseException | None,
) -> tuple[list[str], str | None, int | None]:
    """Return a source-bound S3/S4 terminal only from a canonical event prefix.

    An explicit terminal callback is authoritative after stream validation.
    For an ordinary Python construction exception, the next callback required
    by an otherwise exact prefix identifies the failed boundary.  Observer
    failures and malformed/duplicate/out-of-order streams are deliberately
    inconclusive; exception text is never parsed for a face id.
    """

    failures, explicit_stage, explicit_face = _validate_face_event_stream(
        event_stream, source_topology
    )
    if failures or explicit_stage is not None:
        return failures, explicit_stage, explicit_face
    if construction_error is None:
        return failures, None, None
    if str(construction_error).startswith("assembly_stage_observer_failed"):
        return failures, None, None

    completed_s3 = sum(
        str((entry.get("metadata") or {}).get("stage")) == "S3"
        for entry in event_stream
    )
    completed_s4 = sum(
        str((entry.get("metadata") or {}).get("stage")) == "S4"
        for entry in event_stream
    )
    face_count = int(source_topology["face_count"])
    if completed_s3 == completed_s4 and completed_s3 < face_count:
        return failures, "S3", completed_s3
    if completed_s3 == completed_s4 + 1:
        return failures, "S4", completed_s4
    return failures, None, None


def _face_terminal_record(
    stage: str,
    source_topology: Mapping[str, Any],
    completed_entries: Sequence[Mapping[str, Any]],
    terminal_face_id: int,
    *,
    failure_code: str,
    s2_endpoint_references: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_completed = list(range(int(terminal_face_id)))
    reduced = [
        {
            "source_face_index": int(entry["metadata"]["source_face_index"]),
            "source_mapping": entry["metadata"].get("source_mapping") or {},
            "face": entry["target"],
        }
        for entry in completed_entries
    ]
    if reduced:
        prefix_record, _references = _aggregate_face_stage(
            stage,
            reduced,
            source_topology,
            s2_endpoint_references=s2_endpoint_references,
            expected_face_ids=expected_completed,
        )
        completed = sorted(
            map(int, prefix_record["lineage"].get("source_face_ids") or [])
        )
        failures = list(prefix_record["lineage"].get("failure_codes") or [])
        evidence = dict(prefix_record.get("evidence") or {})
        defects = list(prefix_record.get("defects") or [])
    else:
        completed = []
        failures = [] if not expected_completed else [
            "source_face_terminal_prefix_missing"
        ]
        evidence = {
            "endpoint_identity_proof_method": (
                "s2_to_stage_occ_unordered_pair_IsSame"
            ),
            "endpoint_identity_source_edge_count": 0,
            "endpoint_identity_occurrence_count": 0,
        }
        defects = []
    if completed != expected_completed:
        failures.append("source_face_terminal_prefix_coverage_mismatch")
    exact_terminal = not failures
    evidence.update(
        {
            "failure_localized_by_strict_event_state_machine": True,
            "terminal_failure_source_face_id": int(terminal_face_id),
        }
    )
    return {
        "stage": stage,
        "phase": STAGE_NAMES[stage],
        "status": "observed",
        "lineage": _lineage(
            "local_exact_failure" if exact_terminal else "ambiguous",
            source_topology,
            failures=failures,
            source_face_ids=completed,
            distributed_scope={
                "entity_kind": "source_face",
                "expected_ids": list(range(int(source_topology["face_count"]))),
                "completed_ids": completed,
                "terminal_failure_entity_id": int(terminal_face_id),
                "preceding_stage_prefix_verified": True,
                "event_sequence_proof": {
                    "events": [
                        *(
                            {"entity_id": face_id, "event": "post_boundary_ok"}
                            for face_id in completed
                        ),
                        {
                            "entity_id": int(terminal_face_id),
                            "event": "terminal_failure",
                        },
                    ]
                },
            },
        ),
        "topology": None,
        "defects": defects,
        "failure": {"kind": str(failure_code), "reason": str(failure_code)},
        "evidence": evidence,
    }


def _face_prefix_record(
    stage: str,
    source_topology: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    *,
    next_stage_terminal_face_id: int,
    s2_endpoint_references: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    reduced = [
        {
            "source_face_index": int(entry["metadata"]["source_face_index"]),
            "source_mapping": entry["metadata"].get("source_mapping") or {},
            "face": entry["target"],
        }
        for entry in entries
    ]
    record, references = _aggregate_face_stage(
        stage,
        reduced,
        source_topology,
        s2_endpoint_references=s2_endpoint_references,
        expected_face_ids=list(range(int(next_stage_terminal_face_id) + 1)),
    )
    completed = sorted(references)
    expected_completed = list(range(int(next_stage_terminal_face_id) + 1))
    failures = list(record["lineage"].get("failure_codes") or [])
    if completed != expected_completed:
        failures.append("source_face_prefix_before_next_stage_failure_mismatch")
    record["lineage"] = _lineage(
        "exact_prefix" if not failures else "ambiguous",
        source_topology,
        failures=failures,
        source_face_ids=completed,
        distributed_scope={
            "entity_kind": "source_face",
            "expected_ids": list(range(int(source_topology["face_count"]))),
            "completed_ids": completed,
            "terminal_failure_entity_id": None,
            "preceding_stage_prefix_verified": True,
            "event_sequence_proof": {
                "events": [
                    {"entity_id": face_id, "event": "post_boundary_ok"}
                    for face_id in completed
                ]
            },
        },
    )
    record["topology"] = None
    record["evidence"]["prefix_pass_before_next_stage_failure"] = not failures
    return record, references


def _compact_failure(exc: BaseException) -> dict[str, str]:
    text = redact_path_and_native_text(str(exc))
    return {"kind": type(exc).__name__, "reason": text or type(exc).__name__}


def _whole_stage_terminal_record(
    stage: str, source_topology: Mapping[str, Any], exc: BaseException,
) -> dict[str, Any]:
    prerequisite = {"S5": "S4", "S6": "S5"}.get(stage)
    if prerequisite is None:
        raise ValueError("whole-stage terminal is supported only for S5/S6")
    return {
        "stage": stage,
        "phase": STAGE_NAMES[stage],
        "status": "observed",
        "lineage": _lineage(
            "local_exact_failure",
            source_topology,
            whole_stage_terminal={
                "scope_kind": "whole_shape_boundary_failure",
                "boundary_stage": stage,
                "prerequisite_stage": prerequisite,
                "prerequisite_exact": True,
                "construction_exception_observed": True,
            },
        ),
        "topology": None,
        "defects": [],
        "failure": _compact_failure(exc),
    }


def _whole_stage_failure_is_localizable(
    records: Sequence[Mapping[str, Any]], *, stage: str,
    construction_error: BaseException | None,
    raw: Mapping[str, Sequence[Mapping[str, Any]]],
    source_topology: Mapping[str, Any],
) -> bool:
    """Prove an exception occurred exactly at an unobserved S5/S6 boundary."""

    if stage not in {"S5", "S6"} or construction_error is None:
        return False
    if str(construction_error).startswith("assembly_stage_observer_failed"):
        return False
    target_index = STAGE_ORDER.index(stage)
    if raw.get(stage) or any(raw.get(name) for name in STAGE_ORDER[target_index + 1 : -1]):
        return False
    if len(records) != target_index:
        return False
    try:
        normalized = [
            normalize_stage_record(
                record, expected_stage=expected,
                source_topology=source_topology,
            )
            for record, expected in zip(records, STAGE_ORDER[:target_index])
        ]
    except (KeyError, TypeError, ValueError):
        return False
    if any(
        record.get("status") != "observed"
        or (record.get("lineage") or {}).get("exact") is not True
        or record.get("failure") is not None
        or record.get("scientifically_bad") is not False
        for record in normalized
    ):
        return False
    prerequisite = normalized[-1]
    topology = prerequisite.get("topology")
    return bool(
        isinstance(topology, Mapping)
        and topology.get("complete") is True
        and topology.get("matches_source") is True
    )


def _not_reached_after(records: list[dict[str, Any]], failure_stage: str, reason: str) -> None:
    start = STAGE_ORDER.index(failure_stage) + 1
    for stage in STAGE_ORDER[start:]:
        records.append(make_not_reached_stage(stage, reason, blocked_by_stage=failure_stage))


def _aggregate_edge_stage(
    stage: str,
    entries: Sequence[Mapping[str, Any]],
    source_topology: Mapping[str, Any],
    *,
    paired_entries: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Reduce per-edge S1/S2 callbacks without inventing a global snapshot."""
    expected_edges = int(source_topology["edge_count"])
    indices: list[int] = []
    failures: list[str] = []
    events: list[dict[str, Any]] = []
    terminal_failure: int | None = None
    observed_endpoints: dict[int, list[int]] = {}
    for entry in entries:
        metadata = entry.get("metadata")
        if not isinstance(metadata, Mapping):
            failures.append("edge_observation_metadata_missing")
            continue
        value = metadata.get("source_edge_id")
        if type(value) is not int:
            failures.append("source_edge_id_missing")
            continue
        indices.append(int(value))
        boundary_event = metadata.get("boundary_event", "completed")
        if boundary_event not in {"completed", "terminal_failure"}:
            failures.append(f"source_edge_{value}_boundary_event_invalid")
        elif boundary_event == "terminal_failure":
            if terminal_failure is not None:
                failures.append("multiple_terminal_edge_failures")
            terminal_failure = int(value)
        events.append(
            {
                "entity_id": int(value),
                "event": (
                    "terminal_failure"
                    if boundary_event == "terminal_failure"
                    else "pre_boundary_ok" if stage == "S1" else "post_boundary_ok"
                ),
            }
        )
        if metadata.get("observation_scope") != "distributed_source_edge_event":
            failures.append(f"source_edge_{value}_scope_mismatch")
        expected_sequence = 2 * int(value) + (1 if stage == "S2" else 0)
        if metadata.get("event_sequence_position") != expected_sequence:
            failures.append(f"source_edge_{value}_event_sequence_mismatch")
        expected_prefix = int(value) + 1
        expected_fit_count = (
            int(value)
            if stage == "S1" and boundary_event == "terminal_failure"
            else expected_prefix
        )
        if metadata.get("fitted_curve_prefix_count") != expected_fit_count:
            failures.append(f"source_edge_{value}_fit_prefix_mismatch")
        expected_built_prefix = int(value) + (
            1 if stage == "S2" and boundary_event != "terminal_failure" else 0
        )
        if metadata.get("built_edge_prefix_count") != expected_built_prefix:
            failures.append(f"source_edge_{value}_build_prefix_mismatch")
        # A curve-fit terminal event has no constructed edge and therefore no
        # endpoint handle/label obligation.  Every completed S1 event and all
        # S2 events do carry the immutable effective endpoint labels.
        if not (stage == "S1" and boundary_event == "terminal_failure"):
            endpoints = metadata.get("effective_vertex_ids")
            if (
                not isinstance(endpoints, Sequence)
                or isinstance(endpoints, (str, bytes))
                or len(endpoints) != 2
                or any(type(endpoint) is not int for endpoint in endpoints)
            ):
                failures.append(f"source_edge_{value}_effective_vertex_ids_malformed")
            else:
                canonical_endpoints = list(map(int, endpoints))
                observed_endpoints[int(value)] = canonical_endpoints
                expected_endpoint_rows = source_topology.get(
                    "edge_vertex_source_ids"
                ) or []
                if (
                    not 0 <= int(value) < len(expected_endpoint_rows)
                    or sorted(canonical_endpoints)
                    != sorted(map(int, expected_endpoint_rows[int(value)]))
                ):
                    failures.append(
                        f"source_edge_{value}_effective_vertex_ids_drifted"
                    )
    paired_terminal: int | None = None
    for paired in paired_entries:
        paired_metadata = paired.get("metadata")
        if (
            isinstance(paired_metadata, Mapping)
            and paired_metadata.get("boundary_event") == "terminal_failure"
            and type(paired_metadata.get("source_edge_id")) is int
        ):
            paired_terminal = int(paired_metadata["source_edge_id"])
            break
    unique_indices = sorted(set(indices))
    expected_prefix = (
        list(range(int(terminal_failure) + 1))
        if terminal_failure is not None
        else list(range(int(paired_terminal) + 1))
        if paired_terminal is not None and stage == "S1"
        else list(range(expected_edges))
    )
    if unique_indices != expected_prefix or len(indices) != len(set(indices)):
        failures.append("source_edge_coverage_incomplete_or_duplicate")
    completed_ids = (
        [edge_id for edge_id in unique_indices if edge_id != terminal_failure]
        if terminal_failure is not None else unique_indices
    )
    stage_local_proof: dict[str, Any] | None = None
    if stage == "S2" and completed_ids:
        local_scopes: list[list[tuple[int, Sequence[int], Any]]] = []
        for entry in entries:
            metadata = entry.get("metadata") or {}
            edge_id = metadata.get("source_edge_id")
            if (
                metadata.get("boundary_event", "completed") != "completed"
                or type(edge_id) is not int
            ):
                continue
            target = entry.get("target")
            if target is None:
                failures.append(f"source_edge_{edge_id}_s2_edge_handle_missing")
                continue
            try:
                endpoints = _edge_endpoint_handles(target)
            except Exception as exc:
                failures.append(
                    f"source_edge_{edge_id}_s2_endpoint_extraction_failed:"
                    f"{type(exc).__name__}"
                )
                continue
            labels = observed_endpoints.get(int(edge_id))
            if not isinstance(labels, Sequence) or len(labels) != 2:
                failures.append(
                    f"source_edge_{edge_id}_stage_local_labels_missing"
                )
                continue
            local_scopes.append([(int(edge_id), list(map(int, labels)), target)])
        stage_local_proof = _prove_stage_local_occ_topology(
            local_scopes, scope_kind="source_edge"
        )
        local_failures = list(stage_local_proof.get("failure_codes") or [])
        if (
            len(local_scopes) != len(completed_ids)
            or int(stage_local_proof.get("constraint_occurrence_count") or 0)
            != len(completed_ids)
            or int(stage_local_proof.get("source_edge_count") or 0)
            != len(set(completed_ids))
        ):
            local_failures.append("stage_local_endpoint_coverage_incomplete")
        if local_failures:
            failures.extend(local_failures)
            stage_local_proof = _stage_local_occ_topology_proof(
                stage,
                scope_count=len(local_scopes),
                source_edge_count=int(
                    stage_local_proof.get("source_edge_count") or 0
                ),
                constraint_occurrence_count=int(
                    stage_local_proof.get("constraint_occurrence_count") or 0
                ),
                failures=local_failures,
            )
    full_exact = terminal_failure is None and paired_terminal is None and not failures
    local_exact = terminal_failure is not None and not failures
    prefix_pass = (
        terminal_failure is None
        and paired_terminal is not None
        and stage == "S1"
        and not failures
    )
    if stage == "S1":
        topology = {"face_count": int(source_topology["face_count"])}
    else:
        # S2 observes every built edge by source identity.  Vertex incidence is
        # the immutable effective input topology for these primary/bridge arms.
        completed_vertex_incidence = Counter(
            endpoint
            for edge_id in completed_ids
            for endpoint in observed_endpoints.get(edge_id, [])
        )
        topology = {
            "edge_count": len(completed_ids),
            # For a terminal prefix this is the number of vertices actually
            # incident to completed edges, not the full source population.
            # Keeping the full source count beside a partial incidence list
            # would be a malformed topology census rather than useful failure
            # evidence.
            "vertex_count": len(completed_vertex_incidence),
            "vertex_edge_incidence_counts": sorted(
                completed_vertex_incidence.values()
            ),
        }
    scope_events = [
        {
            "entity_id": edge_id,
            "event": "pre_boundary_ok" if stage == "S1" else "post_boundary_ok",
        }
        for edge_id in completed_ids
    ]
    if terminal_failure is not None:
        scope_events.append(
            {"entity_id": terminal_failure, "event": "terminal_failure"}
        )
    lineage_status = (
        "local_exact_failure" if local_exact
        else "exact_prefix" if prefix_pass
        else "exact_identity" if full_exact
        else "ambiguous"
    )
    return {
        "stage": stage,
        "phase": STAGE_NAMES[stage],
        "status": "observed",
        "lineage": _lineage(
            lineage_status,
            source_topology,
            # Population cardinalities describe a complete bijection.  A
            # distributed prefix or a local terminal failure is instead
            # proved by ``distributed_scope``; presenting the prefix count as
            # a full entity census would correctly normalize to ``missing``.
            edges=(len(completed_ids) if full_exact else None),
            failures=failures,
            source_face_ids=(
                list(range(int(source_topology["face_count"]))) if stage == "S1" and full_exact else None
            ),
            source_edge_ids=completed_ids,
            distributed_scope={
                "entity_kind": "source_edge",
                "expected_ids": list(range(expected_edges)),
                "completed_ids": completed_ids,
                "terminal_failure_entity_id": terminal_failure,
                "preceding_stage_prefix_verified": bool(
                    full_exact or prefix_pass or local_exact
                ),
                "event_sequence_proof": {"events": scope_events},
            },
        ),
        "topology": topology,
        "defects": [],
        "evidence": {
            "observation_granularity": "per_source_edge",
            "observed_source_edge_count": len(indices),
            "unique_source_edge_count": len(set(indices)),
            "complete_order_independent_source_edge_coverage": full_exact,
            "terminal_failure_source_edge_id": terminal_failure,
            "paired_stage_terminal_failure_source_edge_id": paired_terminal,
            **(
                {"stage_local_occ_topology_proof": stage_local_proof}
                if stage_local_proof is not None
                else {}
            ),
        },
    }


def _validate_edge_event_stream(
    event_stream: Sequence[Mapping[str, Any]], source_topology: Mapping[str, Any]
) -> list[str]:
    """Prove constructor callbacks preserve fit(i), edge(i), fit(i+1)."""

    expected_edges = int(source_topology["edge_count"])
    failures: list[str] = []
    terminal_seen = False
    for position, entry in enumerate(event_stream):
        metadata = entry.get("metadata")
        if not isinstance(metadata, Mapping):
            failures.append("global_edge_event_metadata_missing")
            continue
        stage = str(metadata.get("stage"))
        source_edge_id = metadata.get("source_edge_id")
        sequence = metadata.get("event_sequence_position")
        if type(source_edge_id) is not int or type(sequence) is not int:
            failures.append("global_edge_event_identity_missing")
            continue
        if sequence != position:
            failures.append("global_edge_event_position_mismatch")
        if stage != ("S1" if position % 2 == 0 else "S2"):
            failures.append("global_edge_stage_interleave_mismatch")
        if source_edge_id != position // 2:
            failures.append("global_edge_source_interleave_mismatch")
        terminal = metadata.get("boundary_event", "completed") == "terminal_failure"
        if terminal_seen:
            failures.append("global_edge_event_after_terminal_failure")
        terminal_seen = terminal_seen or terminal
    if not terminal_seen and len(event_stream) != 2 * expected_edges:
        failures.append("global_edge_event_stream_incomplete")
    return list(dict.fromkeys(failures))


def _stage_records_from_native(
    parsed: Mapping[str, Any], surf_wcs: Any, edge_wcs: Any, *, task: TaskSpec,
    breparg_root: Path,
) -> tuple[list[dict[str, Any]], Any | None, dict[int, dict[str, Any]], dict[str, Any]]:
    """Construct and reduce S1-S6 while retaining private S5 references for S7."""
    import numpy as np
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from tools.assembly_repair import (
        DIRECTED_CURVE_INTERPOLATE_PROFILE, DIRECTED_LOCAL_TOPOLOGY_PROFILE,
    )
    from tools.directed_trim_assembly import construct_brep_directed
    from tools.run_assembly_repair_matrix import profile_kwargs

    source = _source_topology(parsed)
    raw: dict[str, list[dict[str, Any]]] = {stage: [] for stage in STAGE_ORDER[:-1]}
    raw_event_stream: list[dict[str, Any]] = []
    private_s5: dict[int, dict[str, Any]] = {}

    def observer(target: Any, metadata: Mapping[str, Any]) -> None:
        stage = str(metadata["stage"])
        entry = {"target": target, "metadata": dict(metadata)}
        raw[stage].append(entry)
        raw_event_stream.append(entry)

    profile = (
        DIRECTED_CURVE_INTERPOLATE_PROFILE
        if task.is_reachability_bridge else DIRECTED_LOCAL_TOPOLOGY_PROFILE
    )
    constructor_kwargs = profile_kwargs(profile)
    if constructor_kwargs.get("solid_topology_repair") is not False:
        raise RuntimeError(
            "source-bound census requires solid_topology_repair=False"
        )
    solid = None
    diagnostics: dict[str, Any] = {}
    construction_error: BaseException | None = None
    try:
        solid, diagnostics = construct_brep_directed(
            np.asarray(surf_wcs), np.asarray(edge_wcs),
            [list(map(int, values)) for values in parsed["faceEdge_adj"]],
            np.asarray(parsed["edgeCorner_adj"], dtype=np.int64),
            breparg_root=Path(breparg_root), **constructor_kwargs,
            assembly_stage_observer=observer,
            post_pcurve_face_mutator=None, post_sewing_shape_mutator=None,
        )
    except Exception as exc:
        construction_error = exc

    records: list[dict[str, Any]] = []
    edge_stream_failures = _validate_edge_event_stream(
        [
            entry for entry in raw_event_stream
            if str((entry.get("metadata") or {}).get("stage")) in {"S1", "S2"}
        ],
        source,
    )
    s2_endpoint_references, s2_endpoint_failures = _build_s2_endpoint_references(
        raw["S2"], source
    )
    if len(s2_endpoint_references) != sum(
        (entry.get("metadata") or {}).get("boundary_event", "completed") == "completed"
        for entry in raw["S2"]
    ):
        s2_endpoint_failures.append("s2_endpoint_reference_coverage_incomplete")
    s2_endpoint_failures = list(dict.fromkeys(s2_endpoint_failures))
    face_events = [
        entry for entry in raw_event_stream
        if str((entry.get("metadata") or {}).get("stage")) in {"S3", "S4"}
    ]
    face_stream_failures, synthesized_terminal_stage, synthesized_terminal_id = (
        _localize_face_failure_from_event_prefix(
            face_events, source, construction_error
        )
    )
    # A face-bound failure is not source-local unless the preceding S1/S2
    # stream is itself exact.  Preserve the face callbacks as evidence but do
    # not synthesize a scientific terminal over an ambiguous edge prefix.
    if edge_stream_failures and synthesized_terminal_stage is not None:
        synthesized_terminal_stage = None
        synthesized_terminal_id = None

    edge_terminal_stage = next(
        (
            stage
            for stage in ("S1", "S2")
            if any(
                (entry.get("metadata") or {}).get("boundary_event") == "terminal_failure"
                for entry in raw[stage]
            )
        ),
        None,
    )
    for stage in STAGE_ORDER[:-1]:
        entries = raw[stage]
        if edge_terminal_stage is not None and STAGE_ORDER.index(stage) > STAGE_ORDER.index(edge_terminal_stage):
            records.append(
                make_not_reached_stage(
                    stage,
                    f"blocked_by_{edge_terminal_stage}",
                    blocked_by_stage=edge_terminal_stage,
                )
            )
            continue
        if (
            synthesized_terminal_stage in {"S3", "S4"}
            and STAGE_ORDER.index(stage) > STAGE_ORDER.index(synthesized_terminal_stage)
        ):
            records.append(
                make_not_reached_stage(
                    stage,
                    f"blocked_by_{synthesized_terminal_stage}",
                    blocked_by_stage=synthesized_terminal_stage,
                )
            )
            continue
        if stage == synthesized_terminal_stage and synthesized_terminal_id is not None:
            completed_entries = [
                entry for entry in entries
                if (entry.get("metadata") or {}).get("boundary_event", "completed") == "completed"
            ]
            records.append(
                _face_terminal_record(
                    stage,
                    source,
                    completed_entries,
                    synthesized_terminal_id,
                    failure_code=str(
                        ((entries[-1].get("metadata") or {}).get("failure_code"))
                        if entries and entries[-1].get("target") is None
                        else f"{stage.lower()}_construction_failed"
                    ),
                    s2_endpoint_references=s2_endpoint_references,
                )
            )
            continue
        if not entries:
            if _whole_stage_failure_is_localizable(
                records,
                stage=stage,
                construction_error=construction_error,
                raw=raw,
                source_topology=source,
            ):
                records.append(
                    _whole_stage_terminal_record(stage, source, construction_error)
                )
                _not_reached_after(records, stage, f"blocked_by_{stage}")
                return records, None, {}, diagnostics
            failure = _compact_failure(construction_error or RuntimeError("stage_not_emitted"))
            records.append({
                "stage": stage, "phase": STAGE_NAMES[stage], "status": "observed",
                "lineage": _lineage(
                    "ambiguous", source,
                    failures=[
                        "unexpected_stage_missing_without_construction_error"
                        if construction_error is None else "stage_failure_not_localizable"
                    ],
                ), "topology": None,
                "defects": [], "failure": failure,
            })
            _not_reached_after(records, stage, failure["reason"])
            return records, None, {}, diagnostics
        if stage in {"S1", "S2"}:
            record = _aggregate_edge_stage(
                stage,
                entries,
                source,
                paired_entries=(raw["S2"] if stage == "S1" else raw["S1"]),
            )
            if edge_stream_failures:
                record["lineage"] = _lineage(
                    "ambiguous",
                    source,
                    failures=edge_stream_failures,
                    source_edge_ids=sorted(
                        {
                            int(entry["metadata"]["source_edge_id"])
                            for entry in entries
                            if isinstance(entry.get("metadata"), Mapping)
                            and type(entry["metadata"].get("source_edge_id")) is int
                        }
                    ),
                )
            if stage == "S2" and s2_endpoint_failures:
                record["lineage"] = _lineage(
                    "ambiguous",
                    source,
                    failures=s2_endpoint_failures,
                    source_edge_ids=sorted(s2_endpoint_references),
                )
            records.append(record)
        elif stage in {"S3", "S4"}:
            paired_terminal = (
                synthesized_terminal_id
                if stage == "S3" and synthesized_terminal_stage == "S4"
                else None
            )
            if paired_terminal is not None:
                record, _references = _face_prefix_record(
                    stage,
                    source,
                    entries,
                    next_stage_terminal_face_id=paired_terminal,
                    s2_endpoint_references=s2_endpoint_references,
                )
            else:
                reduced = [
                    {"source_face_index": int(entry["metadata"]["source_face_index"]),
                     "source_mapping": entry["metadata"].get("source_mapping") or {},
                     "face": entry["target"]}
                    for entry in entries
                ]
                record, _references = _aggregate_face_stage(
                    stage,
                    reduced,
                    source,
                    s2_endpoint_references=s2_endpoint_references,
                )
            if face_stream_failures:
                record["lineage"] = _lineage(
                    "ambiguous", source, failures=face_stream_failures
                )
                record["topology"] = None
            records.append(record)
        elif stage in {"S5", "S6"}:
            if len(entries) != 1:
                record = {
                    "stage": stage,
                    "phase": STAGE_NAMES[stage],
                    "status": "observed",
                    "lineage": _lineage(
                        "ambiguous", source,
                        failures=[f"{stage.lower()}_event_count_not_one"],
                    ),
                    "topology": None,
                    "defects": [],
                    "failure": {
                        "kind": "stage_event_count_invalid",
                        "reason": "stage_event_count_invalid",
                    },
                }
                records.append(record)
                _not_reached_after(records, stage, "stage_event_count_invalid")
                return records, None, {}, diagnostics
            metadata = entries[0]["metadata"]
            bindings = metadata.get("source_face_bindings") or []
            reduced = []
            sewing_failures: list[str] = []
            for binding in bindings:
                if not isinstance(binding, Mapping):
                    sewing_failures.append("sewing_binding_not_mapping")
                    continue
                face_index = int(binding.get("source_face_index", -1))
                sewing_lineage = binding.get("sewing_lineage") or {}
                mapping = binding.get("source_mapping") or {}
                if sewing_lineage.get("status") != "mapped":
                    sewing_failures.extend(str(value) for value in sewing_lineage.get("failure_codes") or [])
                reduced.append({"source_face_index": face_index,
                                "source_mapping": mapping, "face": binding.get("face")})
            record, references = _aggregate_face_stage(
                stage,
                reduced,
                source,
                source_vertex_lineage=metadata.get("source_vertex_lineage"),
            )
            if sewing_failures:
                record["lineage"] = _lineage("ambiguous", source, failures=sewing_failures)
                record["topology"] = None
            else:
                try:
                    native_topology = _shape_topology(entries[0]["target"])
                    if record.get("topology") is None:
                        raise ValueError("source-bound mapped topology unavailable")
                    record["topology"] = _merge_native_counts_with_source_relations(
                        native_topology, record["topology"]
                    )
                except Exception as exc:
                    record["failure"] = _compact_failure(exc)
            if stage == "S6":
                record["construction_native_valid"] = bool(BRepCheck_Analyzer(entries[0]["target"], True).IsValid())
            else:
                private_s5 = references
            records.append(record)
    return records, solid, private_s5, diagnostics


def _s7_record(
    solid: Any, *, output_dir: Path, breparg_root: Path,
    source_topology: Mapping[str, Any], face_edge_adj: Sequence[Sequence[int]],
    source_face_references: Mapping[int, Mapping[str, Any]],
    surf_wcs: Any, edge_wcs: Any, edge_vertex_adj: Any,
    effective_topology: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from OCC.Extend.DataExchange import write_step_file
    from tools.diagnose_step_validity_components import diagnose_step
    from tools.probe_downstream_bad_wire_lineage import _step_observation
    from tools.run_assembly_repair_matrix import strict_validate_step
    from tools.assembly_selector_geometry import (
        candidate_step_signature,
        geometry_topology_gate,
        input_geometry_signature,
        sample_input_edge_points,
    )

    step_path = Path(output_dir) / "roundtrip.step"
    try:
        write_step_file(solid, str(step_path))
        if not step_path.is_file() or step_path.stat().st_size <= 0:
            raise RuntimeError("step_writer_produced_no_nonempty_file")
        identity = {"bytes": step_path.stat().st_size, "sha256": sha256_file(step_path)}
        lineage = _step_observation(
            step_path, breparg_root=Path(breparg_root),
            source_face_references=source_face_references,
            face_edge_adj=face_edge_adj,
            edge_vertex_adj=edge_vertex_adj,
            source_edge_wcs=edge_wcs,
            require_vertex_proof=True,
            fail_on_matching_exception=True,
        )
        status = str(lineage.get("lineage_status") or "unavailable")
        failures = [str(value) for value in lineage.get("mapping_failures") or []]
        diagnosis = lineage.get("diagnosis") or {}
        step_geometry_incidence_proof = diagnosis.get(
            "geometry_incidence_proof"
        )
        defects = [
            {
                "code": f"wire_{item.get('kind', 'unknown')}",
                "source_face_index": item.get("source_face_index"),
                "wire_index": item.get("wire_index"),
                "source_edge_ids": list(item.get("source_edge_ids") or []),
            }
            for item in diagnosis.get("occurrences") or [] if item.get("status") == "detected"
        ]
        validity = strict_validate_step(step_path, breparg_root=Path(breparg_root))
        components = diagnose_step(step_path, breparg_root=Path(breparg_root))
        from tools.probe_downstream_bad_wire_lineage import _read_step_faces
        reimport_shape, _faces = _read_step_faces(step_path)
        topology = _shape_topology(reimport_shape)
        input_signature = input_geometry_signature(
            surf_wcs,
            edge_wcs,
            face_edge_adj,
            edge_vertex_adj,
            effective_vertex_count=(effective_topology or {}).get("vertex_count"),
            effective_vertex_edge_incidence_counts=(effective_topology or {}).get(
                "vertex_edge_incidence_counts"
            ),
        )
        candidate_signature = candidate_step_signature(
            step_path,
            input_edge_samples=sample_input_edge_points(edge_wcs),
            input_edge_polylines=edge_wcs,
            input_signature=input_signature,
            validity_components=validity.get("validity_components") or components,
        )
        schema_v2_gate = geometry_topology_gate(input_signature, candidate_signature)
        exact = status == "exact_geometry_incidence" and not failures
        record = {
            "stage": "S7", "phase": STAGE_NAMES["S7"], "status": "observed",
            "lineage": _lineage(
                status, source_topology,
                faces=(source_topology["face_count"] if exact else 0),
                edges=(source_topology["edge_count"] if exact else 0),
                failures=failures,
                source_face_ids=(list(range(int(source_topology["face_count"]))) if exact else []),
                source_edge_ids=(list(range(int(source_topology["edge_count"]))) if exact else []),
                source_edge_occurrence_keys=(
                    _occurrence_keys(source_topology["face_edge_source_ids"]) if exact else []
                ),
            ),
            "topology": topology, "defects": defects,
            "reimport_native_valid": bool(validity["native_brep_valid"]),
            "strict_valid": bool(validity["strict_brep_valid"]),
            "evidence": {
                "step_bytes": identity["bytes"], "step_sha256": identity["sha256"],
                "validity_components": {
                    key: components.get(key) for key in (
                        "status", "wire_count", "wire_order_failures",
                        "wire_self_intersections", "shell_count",
                        "shells_with_bad_edges", "free_edges", "solid_count",
                    )
                },
                # This is a measurement only.  Even if the unchanged gate says
                # accepted, the census is not a selectable repair candidate.
                "schema_v2": {
                    "applicable_to_census_authorization": False,
                    "measurement": schema_v2_gate,
                },
                "step_geometry_incidence_proof": dict(
                    step_geometry_incidence_proof or {}
                ),
            },
        }
        return record, {
            **identity,
            "artifact_id": Path(output_dir).name,
        }
    except Exception:
        # A failed roundtrip has no successful artifact identity.  The caller
        # records S7 failure while retaining any partial file only as local,
        # unreferenced forensic residue in this unique attempt directory.
        raise


def _base_row(source: Mapping[str, Any], task: TaskSpec, *, run_signature: str,
              expected_binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA, "task_id": task.task_id, "task_ordinal": task.ordinal,
        "cad_id": task.cad_id, "parent_id": source.get("parent_id"),
        "arm": task.arm, "profile_name": task.profile_name,
        "switches": list(task.switches),
        "is_reachability_bridge": task.is_reachability_bridge,
        "counts_as_repair": False, "denominator": True,
        "historical_strict_valid": bool(source.get("brep_valid")),
        "run_signature": run_signature,
        "source_binding_expected": normalize_binding(expected_binding),
        "source_binding_before_load": None,
        "source_binding_loaded_bytes": None,
        "source_binding_after_load": None,
        "source_binding_after_measurement": None,
        "source_binding_parent_after_child": None,
        "worker_runtime_abi_sentinel": None,
        "status": "worker_error", "stage_records": [], "assessment": None,
        "step_roundtrip": {
            "saved_to_persistent_output": False,
            "artifact_id": None,
            "bytes": None,
            "sha256": None,
        },
        "nonfinite_count": 0,
    }


def failure_row(source: Mapping[str, Any], task: TaskSpec, *, run_signature: str,
                expected_binding: Mapping[str, Any], status: str,
                error_type: str, returncode: int | None = None,
                parent_after_child: Mapping[str, Any] | None = None) -> dict[str, Any]:
    row = _base_row(source, task, run_signature=run_signature,
                    expected_binding=expected_binding)
    row.update(status=status, error_type=str(error_type), worker_returncode=returncode)
    if parent_after_child is not None:
        row["source_binding_parent_after_child"] = normalize_binding(parent_after_child)
    return row


def run_worker(
    source: Mapping[str, Any], task: TaskSpec, *, output_dir: Path,
    breparg_root: Path, joint_iterations: int,
    expected_binding: Mapping[str, Any], run_signature: str,
    expected_runtime_abi_sentinel: Mapping[str, Any],
    worker_runtime_abi_sentinel: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one native cell and keep all five source-byte observations."""
    started = time.perf_counter()
    expected = normalize_binding(expected_binding)
    row = _base_row(source, task, run_signature=run_signature,
                    expected_binding=expected)
    source_path = Path(str(source["source_path"]))
    try:
        signed_runtime = normalize_runtime_identity(expected_runtime_abi_sentinel)
        frozen_runtime = normalize_runtime_identity(FROZEN_RUNTIME_IDENTITY)
        if not _json_exact_equal(signed_runtime, frozen_runtime):
            raise RuntimeError("signed runtime ABI sentinel differs from frozen")
        worker_runtime = normalize_runtime_identity(worker_runtime_abi_sentinel)
        if (
            not _json_exact_equal(worker_runtime, signed_runtime)
            or not _json_exact_equal(worker_runtime, frozen_runtime)
        ):
            raise RuntimeError("worker runtime ABI sentinel differs from signed/frozen")

        # Native/scientific imports occur only after the same process has
        # measured and accepted its representative ABI sentinel.
        import numpy as np
        from tools.run_assembly_calibration_oracle import cpu_joint_optimize

        row["source_binding_before_load"] = source_binding(source_path)
        if row["source_binding_before_load"] != expected:
            raise RuntimeError("source_binding_mismatch_before_load")
        payload = source_path.read_bytes()
        row["source_binding_loaded_bytes"] = payload_binding(payload)
        if row["source_binding_loaded_bytes"] != expected:
            raise RuntimeError("source_binding_mismatch_loaded_bytes")
        parsed = pickle.loads(payload)
        row["source_binding_after_load"] = source_binding(source_path)
        if row["source_binding_after_load"] != expected:
            raise RuntimeError("source_binding_mismatch_after_load")
        face_edge_adj = [list(map(int, values)) for values in parsed["faceEdge_adj"]]
        edge_vertex_adj = np.asarray(parsed["edgeCorner_adj"], dtype=np.int64)
        surf_wcs, edge_wcs = cpu_joint_optimize(
            np.asarray(parsed["surf_ncs"], dtype=np.float32),
            np.asarray(parsed["edge_ncs"], dtype=np.float32),
            np.asarray(parsed["surf_bbox_wcs"], dtype=np.float32),
            np.asarray(parsed["corner_unique"], dtype=np.float32),
            edge_vertex_adj, face_edge_adj, iterations=int(joint_iterations),
        )
        stages, solid, references, _diagnostics = _stage_records_from_native(
            parsed, surf_wcs, edge_wcs, task=task, breparg_root=breparg_root,
        )
        step_identity = None
        if solid is not None and len(stages) == 6:
            from tools.probe_downstream_bad_wire_lineage import (
                StepGeometryIncidenceMatchingError,
            )
            try:
                s7, step_identity = _s7_record(
                    solid, output_dir=output_dir, breparg_root=breparg_root,
                    source_topology=_source_topology(parsed),
                    face_edge_adj=face_edge_adj, source_face_references=references,
                    surf_wcs=surf_wcs, edge_wcs=edge_wcs,
                    edge_vertex_adj=edge_vertex_adj,
                    effective_topology=_diagnostics.get("effective_input_topology") or {},
                )
                stages.append(s7)
            except StepGeometryIncidenceMatchingError:
                # A formal matching implementation/runtime fault is not a
                # scientific S7 ``unavailable`` observation.
                raise
            except Exception as exc:
                stages.append({
                    "stage": "S7", "phase": STAGE_NAMES["S7"], "status": "observed",
                    "lineage": _lineage("unavailable", _source_topology(parsed), failures=[f"step_roundtrip_failed:{type(exc).__name__}"]),
                    "topology": None, "defects": [], "failure": _compact_failure(exc),
                })
        assessment = assess_stage_lineage(stages, source_topology=_source_topology(parsed))
        row["stage_records"] = assessment["stages"]
        row["assessment"] = assessment
        row["step_roundtrip"] = {
            "saved_to_persistent_output": step_identity is not None,
            "artifact_id": (
                None if step_identity is None else step_identity["artifact_id"]
            ),
            "bytes": None if step_identity is None else step_identity["bytes"],
            "sha256": None if step_identity is None else step_identity["sha256"],
        }
        row["status"] = (
            "completed" if assessment["conclusive"]
            else "scientific_inconclusive"
        )
        row["source_binding_after_measurement"] = source_binding(source_path)
        if row["source_binding_after_measurement"] != expected:
            raise RuntimeError("source_binding_mismatch_after_measurement")
        # Commit the sentinel only for a scientifically completed row.  Every
        # worker/protocol failure remains explicitly null.
        row["worker_runtime_abi_sentinel"] = worker_runtime
    except RuntimeError as exc:
        row["status"] = "source_binding_mismatch" if str(exc).startswith("source_binding_mismatch") else "worker_error"
        row["error_type"] = type(exc).__name__
    except Exception as exc:
        row["status"] = "worker_error"
        row["error_type"] = type(exc).__name__
    row["elapsed_seconds"] = float(time.perf_counter() - started)
    return row


def validate_attempt_row(
    row: Mapping[str, Any], *, source: Mapping[str, Any], task: TaskSpec,
    run_signature: str, expected_binding: Mapping[str, Any],
    expected_runtime_abi_sentinel: Mapping[str, Any] | None = None,
) -> None:
    required = {
        "schema": SCHEMA, "task_id": task.task_id, "task_ordinal": task.ordinal,
        "cad_id": task.cad_id, "parent_id": source.get("parent_id"),
        "arm": task.arm, "profile_name": task.profile_name,
        "switches": list(task.switches),
        "is_reachability_bridge": task.is_reachability_bridge,
        "counts_as_repair": False, "denominator": True,
        "historical_strict_valid": bool(source.get("brep_valid")),
        "run_signature": run_signature,
        "source_binding_expected": normalize_binding(expected_binding),
    }
    for key, value in required.items():
        if row.get(key) != value:
            raise ValueError(f"attempt {key} mismatches signed task")
    status = row.get("status")
    if status not in ATTEMPT_STATUSES:
        raise ValueError("attempt status is not registered")
    if "worker_runtime_abi_sentinel" not in row:
        raise ValueError("attempt lacks worker-process runtime ABI sentinel field")
    signed_runtime = normalize_runtime_identity(
        FROZEN_RUNTIME_IDENTITY
        if expected_runtime_abi_sentinel is None
        else expected_runtime_abi_sentinel
    )
    frozen_runtime = normalize_runtime_identity(FROZEN_RUNTIME_IDENTITY)
    if not _json_exact_equal(signed_runtime, frozen_runtime):
        raise ValueError("signed runtime ABI sentinel differs from frozen")
    if status not in WORKER_FAILURE_STATUSES:
        try:
            worker_runtime = normalize_runtime_identity(
                row.get("worker_runtime_abi_sentinel")
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "scientific task lacks a valid worker-process runtime ABI sentinel"
            ) from exc
        if (
            not _json_exact_equal(worker_runtime, signed_runtime)
            or not _json_exact_equal(worker_runtime, frozen_runtime)
        ):
            raise ValueError(
                "worker-process runtime ABI sentinel differs from signed/frozen"
            )
    elif row.get("worker_runtime_abi_sentinel") is not None:
        raise ValueError("worker failure cannot claim a runtime ABI sentinel")
    if type(row.get("nonfinite_count")) is not int or row["nonfinite_count"] != 0:
        raise ValueError("attempt must report zero nonfinite values")
    expected = normalize_binding(expected_binding)
    binding_fields = (
        "source_binding_before_load", "source_binding_loaded_bytes",
        "source_binding_after_load", "source_binding_after_measurement",
        "source_binding_parent_after_child",
    )
    if status not in WORKER_FAILURE_STATUSES:
        if any(row.get(key) != expected for key in binding_fields):
            raise ValueError("completed task lacks the five exact source bindings")
    else:
        for key in binding_fields:
            if row.get(key) is not None:
                normalize_binding(row[key])
    assessment = row.get("assessment")
    if status in {"completed", "scientific_failure", "scientific_inconclusive"}:
        if not isinstance(assessment, Mapping) or assessment.get("schema") != ASSESSMENT_SCHEMA:
            raise ValueError("scientific task lacks a lineage assessment")
        recomputed = assess_stage_lineage(
            row.get("stage_records") or [],
            source_topology=assessment.get("source_topology") or {},
        )
        # Normalization is intentionally idempotent semantically, but the
        # lineage normalizer retains the originally reported status for audit
        # (for example ``exact_identity`` becomes reported ``exact`` on a
        # second pass).  Compare a canonical second pass on both sides.
        canonical_stored = assess_stage_lineage(
            assessment.get("stages") or [],
            source_topology=assessment.get("source_topology") or {},
        )
        if canonical_stored != recomputed:
            raise ValueError("stored lineage assessment differs from stage evidence")
        expected_status = "completed" if recomputed["conclusive"] else "scientific_inconclusive"
        if status != expected_status:
            raise ValueError("scientific task status contradicts its assessment")
    elif row.get("stage_records") not in ([], None) or assessment is not None:
        raise ValueError("worker failure cannot claim stage evidence")
    step = row.get("step_roundtrip")
    if not isinstance(step, Mapping) or set(step) != {
        "saved_to_persistent_output", "artifact_id", "bytes", "sha256"
    }:
        raise ValueError("census STEP contract is malformed")
    saved = step.get("saved_to_persistent_output")
    if type(saved) is not bool:
        raise ValueError("census STEP saved flag must be boolean")
    if saved:
        artifact_id = step.get("artifact_id")
        if (
            not isinstance(artifact_id, str)
            or artifact_id in {".", ".."}
            or STEP_ARTIFACT_ID.fullmatch(artifact_id) is None
            or Path(artifact_id).is_absolute()
            or len(Path(artifact_id).parts) != 1
        ):
            raise ValueError("census STEP artifact id is unsafe")
        normalize_binding({"bytes": step.get("bytes"), "sha256": step.get("sha256")})
    elif any(step.get(key) is not None for key in ("artifact_id", "bytes", "sha256")):
        raise ValueError("unsaved census STEP identity must be null")
    assert_path_free_finite(row, label="census row")


def validate_local_step_artifact(
    row: Mapping[str, Any], run_root: Path,
) -> Path | None:
    """Revalidate one attempt-bound STEP without trusting a serialized path."""

    step = row.get("step_roundtrip")
    if not isinstance(step, Mapping) or set(step) != {
        "saved_to_persistent_output", "artifact_id", "bytes", "sha256"
    }:
        raise ValueError("census STEP contract is malformed")
    if step.get("saved_to_persistent_output") is False:
        if any(step.get(key) is not None for key in ("artifact_id", "bytes", "sha256")):
            raise ValueError("unsaved census STEP identity must be null")
        return None
    if step.get("saved_to_persistent_output") is not True:
        raise ValueError("census STEP saved flag must be boolean")
    artifact_id = step.get("artifact_id")
    if (
        not isinstance(artifact_id, str)
        or artifact_id in {".", ".."}
        or STEP_ARTIFACT_ID.fullmatch(artifact_id) is None
        or Path(artifact_id).is_absolute()
        or len(Path(artifact_id).parts) != 1
    ):
        raise ValueError("census STEP artifact id is unsafe")
    expected = normalize_binding(
        {"bytes": step.get("bytes"), "sha256": step.get("sha256")}
    )
    attempts_root = (Path(run_root).resolve() / ".attempts").resolve()
    target = (attempts_root / artifact_id / "roundtrip.step").resolve()
    try:
        target.relative_to(attempts_root)
    except ValueError as exc:
        raise ValueError("census STEP artifact escapes attempts root") from exc
    if not target.is_file() or source_binding(target) != expected:
        raise ValueError("census STEP artifact binding mismatch")
    return target


def _subprocess_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def run_isolated(
    source: Mapping[str, Any], task: TaskSpec, *, args: argparse.Namespace,
    run_signature: str, expected_binding: Mapping[str, Any],
    expected_runtime_abi_sentinel: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(args.output_dir).resolve()
    work_root = root / ".attempts"
    work_root.mkdir(parents=True, exist_ok=True)
    slug = hashlib.sha256(task.task_id.encode("utf-8")).hexdigest()[:16]
    attempt_dir = Path(tempfile.mkdtemp(prefix=f"{slug}-", dir=work_root))
    log_root = root / "worker_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"{attempt_dir.name}.stdout.log"
    stderr_path = log_root / f"{attempt_dir.name}.stderr.log"
    repo_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable, "-I", "-c", WORKER_BOOTSTRAP_SOURCE, str(repo_root),
        "--calibration-manifest", str(Path(args.calibration_manifest).resolve()),
        "--selector-matrix", str(Path(args.selector_matrix).resolve()),
        "--selector-run", str(Path(args.selector_run).resolve()),
        "--breparg-root", str(Path(args.breparg_root).resolve()),
        "--output-dir", str(attempt_dir),
        "--joint-iterations", str(int(args.joint_iterations)),
        "--worker-timeout-seconds", str(float(args.worker_timeout_seconds)),
        "--worker-task-id", task.task_id,
        "--worker-run-signature", run_signature,
        "--worker-source-binding-json", json.dumps(dict(expected_binding), sort_keys=True, separators=(",", ":")),
        "--worker-source-path", str(Path(str(source["source_path"])).resolve()),
        "--worker-parent-id", str(source["parent_id"]),
    ]
    if args.development_allow_dirty:
        command.append("--development-allow-dirty")

    def parent_binding_or_failure(
        *, status: str, error_type: str, returncode: int | None = None,
    ) -> dict[str, Any]:
        try:
            parent = source_binding(Path(str(source["source_path"])))
        except OSError as exc:
            return failure_row(
                source, task, run_signature=run_signature,
                expected_binding=expected_binding, status="source_binding_mismatch",
                error_type=f"ParentSourceRehash{type(exc).__name__}",
                returncode=returncode,
            )
        if parent != normalize_binding(expected_binding):
            return failure_row(
                source, task, run_signature=run_signature,
                expected_binding=expected_binding, status="source_binding_mismatch",
                error_type="ParentSourceBindingMismatch", returncode=returncode,
                parent_after_child=parent,
            )
        return failure_row(
            source, task, run_signature=run_signature,
            expected_binding=expected_binding, status=status,
            error_type=error_type, returncode=returncode,
            parent_after_child=parent,
        )
    try:
        completed = subprocess.run(
            command, cwd=Path(__file__).resolve().parents[1], capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=float(args.worker_timeout_seconds), check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(_subprocess_text(exc.stdout), encoding="utf-8")
        stderr_path.write_text(_subprocess_text(exc.stderr), encoding="utf-8")
        return parent_binding_or_failure(
            status="worker_timeout", error_type="TimeoutExpired"
        )
    except OSError as exc:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(type(exc).__name__, encoding="utf-8")
        return parent_binding_or_failure(
            status="worker_spawn_error", error_type=type(exc).__name__
        )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    try:
        parent = source_binding(Path(str(source["source_path"])))
    except OSError as exc:
        return failure_row(
            source, task, run_signature=run_signature,
            expected_binding=expected_binding, status="source_binding_mismatch",
            error_type=f"ParentSourceRehash{type(exc).__name__}",
            returncode=int(completed.returncode),
        )
    if parent != normalize_binding(expected_binding):
        return failure_row(
            source, task, run_signature=run_signature,
            expected_binding=expected_binding, status="source_binding_mismatch",
            error_type="ParentSourceBindingMismatch",
            returncode=int(completed.returncode), parent_after_child=parent,
        )
    if completed.returncode != 0:
        return failure_row(source, task, run_signature=run_signature,
                           expected_binding=expected_binding, status="worker_process_exit",
                           error_type="NonzeroWorkerExit", returncode=int(completed.returncode),
                           parent_after_child=parent)
    row = parse_worker_result(completed.stdout)
    if row is None:
        return failure_row(source, task, run_signature=run_signature,
                           expected_binding=expected_binding, status="worker_protocol_error",
                           error_type="InvalidWorkerSentinel", returncode=int(completed.returncode),
                           parent_after_child=parent)
    row["source_binding_parent_after_child"] = parent
    row["worker_returncode"] = int(completed.returncode)
    row["worker_stdout_log"] = f"worker_logs/{stdout_path.name}"
    row["worker_stderr_log"] = f"worker_logs/{stderr_path.name}"
    try:
        validate_attempt_row(row, source=source, task=task,
                             run_signature=run_signature,
                             expected_binding=expected_binding,
                             expected_runtime_abi_sentinel=(
                                 expected_runtime_abi_sentinel
                             ))
        validate_local_step_artifact(row, root)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return failure_row(source, task, run_signature=run_signature,
                           expected_binding=expected_binding, status="worker_protocol_error",
                           error_type=type(exc).__name__, returncode=int(completed.returncode),
                           parent_after_child=parent)
    return row


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = [task.task_id for task in TASKS]
    if len(rows) != len(TASKS) or [row.get("task_id") for row in rows] != expected:
        raise ValueError("summary requires the exact ordered ten-task ledger")
    failures = sum(row.get("status") in WORKER_FAILURE_STATUSES for row in rows)
    binding_failures = sum(
        row.get("status") == "source_binding_mismatch"
        or any(
            row.get(key) is not None
            and row.get(key) != row.get("source_binding_expected")
            for key in (
                "source_binding_before_load", "source_binding_loaded_bytes",
                "source_binding_after_load", "source_binding_after_measurement",
                "source_binding_parent_after_child",
            )
        )
        for row in rows
    )
    nonfinite = sum(int(row.get("nonfinite_count") or 0) for row in rows)
    primary = [row for row in rows if row.get("arm") == PRIMARY_ARM]
    bridges = [row for row in rows if row.get("arm") == BRIDGE_ARM]
    first_bad = [
        {
            "task_id": row["task_id"], "cad_id": row["cad_id"], "arm": row["arm"],
            "conclusive": bool((row.get("assessment") or {}).get("conclusive")),
            "first_bad_stage": (row.get("assessment") or {}).get("first_bad_stage"),
            "first_bad_phase": (row.get("assessment") or {}).get("first_bad_phase"),
            "first_bad_reasons": (row.get("assessment") or {}).get("first_bad_reasons") or [],
        }
        for row in rows
    ]
    localizable_primary = [
        item for item in first_bad[:len(primary)]
        if item["conclusive"] and item["first_bad_stage"] is not None
    ]
    protocol_health = bool(
        len(rows) == len(TASKS)
        and all(row.get("denominator") is True for row in rows)
        and failures == binding_failures == nonfinite == 0
    )
    census_conclusive = bool(
        protocol_health
        and all((row.get("assessment") or {}).get("conclusive") is True for row in rows)
    )
    authorizes_design = bool(census_conclusive and localizable_primary)
    return {
        "schema": SUMMARY_SCHEMA, "attempts": len(rows),
        "denominator_rows": sum(row.get("denominator") is True for row in rows),
        "primary_controls": len(primary), "curve_interpolate_bridges": len(bridges),
        "primary_completed_or_scientific": sum(row.get("status") not in WORKER_FAILURE_STATUSES for row in primary),
        "bridge_completed_or_scientific": sum(row.get("status") not in WORKER_FAILURE_STATUSES for row in bridges),
        "status_counts": dict(sorted(Counter(str(row.get("status")) for row in rows).items())),
        "worker_or_protocol_failures": failures,
        "source_binding_failures": binding_failures,
        "nonfinite_count": nonfinite,
        "first_bad": first_bad,
        "protocol_health": protocol_health,
        "census_conclusive": census_conclusive,
        # Compatibility alias.  It now deliberately has the stronger census
        # meaning and cannot hide a scientifically inconclusive cell.
        "protocol_conclusive": census_conclusive,
        "decision": "AUTHORIZE_EXACT_CANDIDATE_DESIGN" if authorizes_design else "INCONCLUSIVE_NO_CANDIDATE_AUTHORIZATION",
        "bridge_results_are_reachability_only": True,
        "bridge_repairs_counted": 0,
        "selector_strict_valid_before": 91,
        "selector_strict_valid_after": 91,
        "authorizes_exact_candidate_design": authorizes_design,
        "authorizes_repair": False,
        "authorizes_residual_expansion": False,
        "authorizes_full_100cad": False,
        "authorizes_selector_score_change": False,
        "authorizes_schema_v2_relaxation": False,
        "authorizes_training": False,
        "authorizes_sequence_generation": False,
        "authorizes_ar": False,
    }


def validate_terminal_artifact_hashes(
    run: Mapping[str, Any], *, rows_path: Path, summary_path: Path,
) -> dict[str, Any]:
    if run.get("rows_sha256") != sha256_file(rows_path):
        raise RuntimeError("terminal rows hash mismatched")
    if run.get("summary_sha256") != sha256_file(summary_path):
        raise RuntimeError("terminal summary hash mismatched")
    value = strict_json_loads(
        Path(summary_path).read_text(encoding="utf-8"),
        label="terminal census summary",
    )
    if not isinstance(value, Mapping):
        raise RuntimeError("terminal summary is not an object")
    return dict(value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("calibration-manifest", "selector-matrix", "selector-run",
                 "breparg-root", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--joint-iterations", type=int, default=200)
    parser.add_argument("--worker-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--development-allow-dirty", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--worker-task-id", help=argparse.SUPPRESS)
    parser.add_argument("--worker-run-signature", help=argparse.SUPPRESS)
    parser.add_argument("--worker-source-binding-json", help=argparse.SUPPRESS)
    parser.add_argument("--worker-source-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-parent-id", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.joint_iterations < 0:
        parser.error("--joint-iterations must be nonnegative")
    if args.worker_timeout_seconds <= 0:
        parser.error("--worker-timeout-seconds must be positive")
    return args


def _load_inputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    calibration = read_rows(args.calibration_manifest)
    selector = read_rows(args.selector_matrix)
    return select_census_sources(calibration, selector), selector


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    # In worker mode, bind the actual process before reading even one source
    # pickle (including the input-selection hashes below).  A measurement
    # failure therefore exits without touching CAD data and is retained by the
    # coordinator as a worker-process failure with a null row sentinel.
    worker_runtime_abi_sentinel = None
    if args.worker_task_id is not None:
        worker_runtime_abi_sentinel = (
            _measure_runtime_abi_sentinel_current_process()
        )
        if not _json_exact_equal(
            worker_runtime_abi_sentinel,
            normalize_runtime_identity(FROZEN_RUNTIME_IDENTITY),
        ):
            raise SystemExit("worker runtime ABI sentinel differs from frozen")
    sources, selector_rows = _load_inputs(args)
    payload = build_run_payload(args, sources=sources, selector_rows=selector_rows)
    signature = canonical_sha256(payload)
    expected_runtime_abi_sentinel = runtime_abi_sentinel_from_payload(payload)
    sources_by_id = {str(source["cad_id"]): source for source in sources}
    bindings = {str(item["cad_id"]): dict(item["binding"]) for item in payload["sources"]}
    if args.worker_task_id is not None:
        task = TASKS_BY_ID.get(str(args.worker_task_id))
        if task is None or not args.worker_run_signature or not args.worker_source_binding_json or args.worker_source_path is None or not args.worker_parent_id:
            raise SystemExit("worker mode requires one exact signed task")
        source = sources_by_id[task.cad_id]
        try:
            supplied = normalize_binding(
                strict_json_loads(
                    args.worker_source_binding_json, label="worker source binding"
                )
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit("worker binding is malformed") from exc
        if (
            args.worker_run_signature != signature or supplied != bindings[task.cad_id]
            or args.worker_source_path.resolve() != Path(str(source["source_path"])).resolve()
            or args.worker_parent_id != str(source["parent_id"])
        ):
            raise SystemExit("worker arguments mismatch signed task")
        try:
            row = run_worker(
                source, task, output_dir=args.output_dir,
                breparg_root=args.breparg_root, joint_iterations=args.joint_iterations,
                expected_binding=supplied, run_signature=signature,
                expected_runtime_abi_sentinel=expected_runtime_abi_sentinel,
                worker_runtime_abi_sentinel=worker_runtime_abi_sentinel,
            )
        except Exception as exc:
            row = failure_row(source, task, run_signature=signature,
                              expected_binding=supplied, status="worker_error",
                              error_type=type(exc).__name__, returncode=0)
        print(WORKER_MARKER + json.dumps(row, sort_keys=True, ensure_ascii=True,
                                         allow_nan=False), flush=True)
        return 0

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with output_writer_lock(output):
        run = bind_run_manifest(output, payload)
        rows_path, summary_path = output / ROWS_NAME, output / SUMMARY_NAME
        terminal_reopen = run.get("status") in {"COMPLETED", "INCONCLUSIVE"}
        rows = read_rows(
            rows_path, recover_truncated_tail=not terminal_reopen
        )
        seen: set[str] = set()
        expected_prefix = [task.task_id for task in TASKS[: len(rows)]]
        if [str(row.get("task_id")) for row in rows] != expected_prefix:
            raise RuntimeError("existing census rows are not the canonical task prefix")
        for row in rows:
            task_id = str(row.get("task_id"))
            if task_id in seen or task_id not in TASKS_BY_ID:
                raise RuntimeError("existing census rows duplicate or escape task list")
            seen.add(task_id)
            task = TASKS_BY_ID[task_id]
            validate_attempt_row(
                row, source=sources_by_id[task.cad_id], task=task,
                run_signature=signature, expected_binding=bindings[task.cad_id],
                expected_runtime_abi_sentinel=expected_runtime_abi_sentinel,
            )
            validate_local_step_artifact(row, output)
        if terminal_reopen:
            if (
                len(rows) != len(TASKS)
                or run.get("attempts") != len(TASKS)
                or not summary_path.is_file()
            ):
                raise RuntimeError("terminal census has incomplete artifacts")
            terminal_binding_failures = current_source_binding_failures(
                sources, bindings
            )
            if terminal_binding_failures:
                raise RuntimeError(
                    "terminal census source binding drifted: "
                    + ",".join(terminal_binding_failures)
                )
            archived = validate_terminal_artifact_hashes(
                run, rows_path=rows_path, summary_path=summary_path,
            )
            by_task = {str(row["task_id"]): row for row in rows}
            derived = summarize([by_task[task.task_id] for task in TASKS])
            if archived != derived:
                raise RuntimeError("terminal census summary drifted")
            expected_terminal_status = (
                "COMPLETED" if derived["census_conclusive"] else "INCONCLUSIVE"
            )
            if run.get("status") != expected_terminal_status:
                raise RuntimeError("terminal census status disagrees with summary")
            for row in rows:
                validate_local_step_artifact(row, output)
            print(json.dumps(derived, indent=2, sort_keys=True))
            return 0 if derived["census_conclusive"] else 2
        for task in TASKS:
            if task.task_id in seen:
                continue
            row = run_isolated(
                sources_by_id[task.cad_id], task, args=args,
                run_signature=signature, expected_binding=bindings[task.cad_id],
                expected_runtime_abi_sentinel=expected_runtime_abi_sentinel,
            )
            append_row(rows_path, row)
            rows.append(row)
            seen.add(task.task_id)
            print(json.dumps({"task_id": task.task_id, "status": row.get("status"),
                              "first_bad_stage": (row.get("assessment") or {}).get("first_bad_stage")},
                             sort_keys=True), flush=True)
        terminal_binding_failures = current_source_binding_failures(
            sources, bindings
        )
        if terminal_binding_failures:
            # Source drift is not encoded in an immutable attempt row, so it
            # cannot be added to a terminal summary that must be derivable
            # exactly from the ledger.  Leave the manifest RUNNING and all
            # rows untouched; after the source is restored, an identical
            # invocation can safely finish and sign the run.
            raise RuntimeError(
                "terminal source binding drifted before finalization: "
                + ",".join(terminal_binding_failures)
            )
        by_task = {str(row["task_id"]): row for row in rows}
        ordered = [by_task[task.task_id] for task in TASKS]
        for row in ordered:
            validate_local_step_artifact(row, output)
        summary = summarize(ordered)
        atomic_json(summary_path, summary)
        terminal = dict(run)
        terminal.update(
            status="COMPLETED" if summary["census_conclusive"] else "INCONCLUSIVE",
            attempts=len(TASKS), rows_sha256=sha256_file(rows_path),
            summary_sha256=sha256_file(summary_path),
        )
        atomic_json(output / RUN_NAME, terminal)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["census_conclusive"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
