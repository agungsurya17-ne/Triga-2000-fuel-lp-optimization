"""Immutable SDM campaign contracts, provenance, private I/O, and process locks."""

from contextlib import contextmanager
from datetime import datetime
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import shutil
import sys
import tempfile
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET

import openmc

from modules.public_inputs import select_front
from .io_utils import ROOT, fingerprint, json_default
from .model import rod_cases

DEFAULT_CONFIG = Path(__file__).with_name("sdm_final.json")


def timestamp():
    """Return current ISO timestamp in Asia/Jakarta; no parameters."""
    return datetime.now(ZoneInfo("Asia/Jakarta")).isoformat()


def digest(value):
    """Return stable SHA256 of JSON-compatible value using canonical key ordering."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=json_default
        ).encode()
    ).hexdigest()


def sha256(path):
    """Return streaming SHA256 of file path without modifying the file."""
    return fingerprint(Path(path))["sha256"]


def read_json(path):
    """Return decoded JSON from path; propagate malformed/missing-file errors."""
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    """Atomically replace private JSON path with value, fsyncing file and directory."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(
                value, stream, indent=2, ensure_ascii=False, allow_nan=False, default=json_default
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_csv(path, rows):
    """Write nonempty public-safe dictionary rows exclusively to a new CSV path."""
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    Path(path).chmod(0o600)


@contextmanager
def lock(path):
    """Hold a nonblocking advisory exclusive lock at path; yield until context exits."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another SDM operation holds the requested lock.") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def load_config(path=DEFAULT_CONFIG):
    """Validate SDM settings at ``path`` and return the public demonstration contract.

    The dollar conversion uses an illustrative fixed beta; it is not a calculated
    beta for this fresh core. Batch budgets stay fixed so completion and precision
    diagnostics retain the same meaning across runs.
    """
    config = read_json(path)
    if config["materials"] != "config/materials.xml":
        raise ValueError("Use the public material file bound to the selected checkpoint.")
    required = dict(
        particles=100000,
        inactive=45,
        active=155,
        batches=200,
        generations_per_batch=1,
        trigger=False,
        parallel_cases=1,
    )
    transport = config["transport"]
    if any(transport.get(k) != v for k, v in required.items()):
        raise ValueError("Expected 100000 particles and 45 inactive/200 total batches, no trigger.")
    if type(transport["threads"]) is not int or transport["threads"] < 1:
        raise ValueError("threads must be a positive integer.")
    for key, expected in dict(
        openmc_version="0.15.3",
        beta_eff=0.007294,
        z=1.645,
        sdm_min_dollar=0.5,
        sigma_target=0.0003,
        temperature_K=300.0,
    ).items():
        if config[key] != expected:
            raise ValueError(f"Unsupported SDM demonstration value: {key}.")
    snapshots = config["source_snapshots"]
    if (
        not snapshots
        or len(set(snapshots)) != len(snapshots)
        or any(type(b) is not int or not 1 <= b <= transport["batches"] for b in snapshots)
    ):
        raise ValueError("Require unique source snapshot batches within the transport budget.")
    if type(config["seed_base"]) is not int or not 1 <= config["seed_base"] < 2**63:
        raise ValueError("Require a positive signed 64-bit seed base.")
    return config


def cohort(config):
    """Return public final-front IDs, complete genomes and evidence for ``config``."""
    path = Path(config["checkpoint"])
    if not path.is_absolute():
        path = ROOT / path
    ids, genomes, _, audit = select_front(path)
    return ids, genomes, audit


def planned_jobs(ids, genomes, config):
    """Return candidate/case jobs with deterministic unique seeds and genome hashes."""
    jobs = []
    for candidate_id in ids:
        for case, states in rod_cases().items():
            jobs.append(
                dict(
                    candidate_id=candidate_id,
                    case=case,
                    seed=config["seed_base"] + len(jobs),
                    genome_sha256=digest(genomes[candidate_id]),
                    rod_states=states,
                    transport=config["transport"],
                    directory=f"jobs/{candidate_id}/{case}",
                )
            )
    expected_count = len(ids) * len(rod_cases())
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Require unique nonempty candidate IDs.")
    if (
        len(jobs) != expected_count
        or len({job["seed"] for job in jobs}) != expected_count
        or jobs[-1]["seed"] >= 2**63
    ):
        raise ValueError("Invalid candidate/case count or signed 64-bit seeds.")
    return jobs


def file_identity(path):
    """Return resolved path, content hash, size and mtime for a provenance file."""
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), **fingerprint(path))


def environment(config):
    """Verify installed API/executable and hash used data/shared libraries; return context."""
    executable = shutil.which(config["executable"]) or config["executable"]
    interpreter = shutil.which(config["python"]) or config["python"]
    if Path(sys.executable).resolve() != Path(interpreter).resolve():
        raise RuntimeError("Run with the configured OpenMC Python interpreter.")
    version = subprocess.check_output([executable, "--version"], text=True)
    if openmc.__version__ != "0.15.3" or not re.search(r"OpenMC version 0\.15\.3\b", version):
        raise RuntimeError("Both actual executable and API must be OpenMC 0.15.3.")
    materials_path = ROOT / config["materials"]
    materials = ET.parse(materials_path).getroot()
    if any(float(m.get("temperature", "nan")) != 300 for m in materials.findall("material")):
        raise RuntimeError("Static material temperatures must remain 300 K.")
    xs_path = Path(materials.findtext("cross_sections")).resolve(strict=True)
    index = ET.parse(xs_path).getroot()
    base = Path(index.findtext("directory") or xs_path.parent)
    if not base.is_absolute():
        base = xs_path.parent / base
    required = {
        n.get("name")
        for m in materials.findall("material")
        for n in [*m.findall("nuclide"), *m.findall("sab")]
    }
    libraries, covered = set(), set()
    for entry in index.findall("library"):
        names = set(entry.get("materials", "").split())
        if names & required:
            data_path = Path(entry.get("path"))
            libraries.add(data_path if data_path.is_absolute() else base / data_path)
            covered |= names
    if required - covered:
        raise RuntimeError("Cross-section index does not cover required model data.")
    linked = subprocess.check_output(["ldd", executable], text=True)
    linked_paths = set(re.findall(r"=> (/\S+)", linked))
    api_root = Path(openmc.__file__).parent
    api_files = sorted(api_root.rglob("*.py"))
    implementation = sorted((ROOT / "src/final_evaluation").glob("*.py")) + [
        ROOT / "src/modules/openmc_model_builder.py",
        ROOT / "src/modules/genome_sorter.py",
        ROOT / "src/modules/OpenMCResultEvaluator.py",
        ROOT / "src/modules/public_inputs.py",
    ]
    return dict(
        api_version=openmc.__version__,
        executable_version=version,
        dynamic_library_environment={
            key: os.environ.get(key) for key in ("LD_LIBRARY_PATH", "LD_PRELOAD")
        },
        executable=file_identity(executable),
        python=file_identity(sys.executable),
        api_source_sha256=digest({str(p): sha256(p) for p in api_files}),
        implementation_sha256={str(p): sha256(p) for p in implementation},
        materials=file_identity(materials_path),
        cross_sections=file_identity(xs_path),
        nuclear_data=[file_identity(p) for p in sorted(libraries)],
        linked_libraries=[file_identity(p) for p in sorted(linked_paths)],
    )


def unchanged_context(context):
    """Reject changed provenance file size/mtime during a serial run; return None."""
    files = [context[key] for key in ("executable", "python", "materials", "cross_sections")]
    files += context["nuclear_data"] + context["linked_libraries"]
    for item in files:
        stat = Path(item["path"]).stat()
        if stat.st_size != item["bytes"] or stat.st_mtime_ns != item["mtime_ns"]:
            raise RuntimeError("Provenance changed during execution; transport stopped.")


def running_openmc():
    """Return process IDs of existing OpenMC transports without signaling any process."""
    found = []
    for path in Path("/proc").glob("[0-9]*/comm"):
        try:
            if path.read_text().strip() in ("openmc", "openmc.exe"):
                found.append(int(path.parent.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found
