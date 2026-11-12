"""Atomic storage and volume-weighted nominal response arithmetic."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import fcntl
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_OUTPUT = HERE / "production"
NOMINAL = dict(particles=100_000, inactive=45, batches=500, generations_per_batch=1)
TALLY_NAME = "nominal_all_fuel_q"
BASELINE_ID = "PUBLIC_FRESH_REFERENCE"


def digest(path):
    """Return the SHA256 of file path, streaming its bytes to limit memory use."""
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def object_digest(value):
    """Return a stable SHA256 for a JSON-compatible configuration value."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@contextmanager
def atomic_path(path):
    """Yield a temporary sibling of path; fsync and atomically replace it on success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.stem + "-", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def json_default(value):
    """Convert NumPy scalars/arrays and paths for finite JSON storage; return plain Python value."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def write_json(path, value):
    """Atomically write finite JSON-compatible value to path; return None."""
    with atomic_path(path) as temporary:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=json_default)
            + "\n"
        )


def read_json(path):
    """Return the parsed UTF-8 JSON document at path."""
    return json.loads(Path(path).read_text())


@contextmanager
def campaign_lock(output):
    """Exclusively lock output for this process; yield None or reject concurrent writers."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is using {output}") from exc
        yield


def power_metrics(scores, records):
    """Return PPFs/profiles from integrated Q (eV/source) and fuel volumes (cm³)."""
    q = np.asarray(scores, dtype=float)
    v = np.asarray([r["volume_cm3"] for r in records], dtype=float)
    if q.ndim != 1 or len(q) != len(v) or not len(q):
        raise ValueError("Scores and records must be nonempty and aligned.")
    if (
        not np.isfinite(q).all()
        or not np.isfinite(v).all()
        or (q < 0).any()
        or (v <= 0).any()
        or q.sum() <= 0
    ):
        raise ValueError("Require finite nonnegative Q, positive total Q and positive fuel V.")
    density = q / v
    mean = float(q.sum() / v.sum())
    profiles = []
    peaks = {}
    for axis, field in [("radial", "slot"), ("axial", "axial")]:
        groups = defaultdict(list)
        for index, record in enumerate(records):
            groups[record[field]].append(index)
        axis_profiles = []
        for key, indices in groups.items():
            total_q, total_v = float(q[indices].sum()), float(v[indices].sum())
            first = records[indices[0]]
            axis_profiles.append(
                dict(
                    axis=axis,
                    group=str(key),
                    q_eV_per_source=total_q,
                    volume_cm3=total_v,
                    density_eV_per_source_cm3=total_q / total_v,
                    relative_density=total_q / total_v / mean,
                    fuel_id=first.get("fuel_id", "") if axis == "radial" else "",
                    category=first.get("category", "") if axis == "radial" else "",
                    z_low_cm=min(records[i].get("z_low_cm", 0) for i in indices),
                    z_high_cm=max(records[i].get("z_high_cm", 0) for i in indices),
                )
            )
        peak = max(axis_profiles, key=lambda row: row["relative_density"])
        peaks[axis] = peak
        profiles.extend(axis_profiles)
    maximum = int(np.argmax(density))
    hotspot = dict(
        records[maximum],
        q_eV_per_source=float(q[maximum]),
        density_eV_per_source_cm3=float(density[maximum]),
        relative_density=float(density[maximum] / mean),
    )
    return dict(
        ppf_3d=float(density[maximum] / mean),
        ppf_radial=peaks["radial"]["relative_density"],
        ppf_axial=peaks["axial"]["relative_density"],
        radial_peak=peaks["radial"]["group"],
        axial_peak=peaks["axial"]["group"],
        mean_density_eV_per_source_cm3=mean,
        fuel_volume_cm3=float(v.sum()),
        segment_count=len(records),
        element_count=len({r["slot"] for r in records}),
        hotspot=hotspot,
        tied_maximum_segments=[
            dict(records[i]) for i in np.flatnonzero(density == density[maximum])
        ],
        profiles=profiles,
        ppf_uncertainty=None,
    )


def dominates(a, b):
    """Return whether (k, PPF) point a strictly Pareto-dominates b without rounding."""
    return a[0] >= b[0] and a[1] <= b[1] and (a[0] > b[0] or a[1] < b[1])


def classify(point, reference):
    """Return nominal comparison class for (k, PPF); A increases both, B decreases both."""
    if point == reference:
        return "setara"
    if dominates(point, reference):
        return "mendominasi acuan"
    if dominates(reference, point):
        return "didominasi acuan"
    return "Trade-off A" if point[0] > reference[0] else "Trade-off B"
