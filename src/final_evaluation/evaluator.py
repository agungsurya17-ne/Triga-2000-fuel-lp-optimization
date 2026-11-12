"""Volume-weighted PPF and fixed-beta SDM; no changes to historical objectives."""

from collections import defaultdict
import math
import xml.etree.ElementTree as ET
import numpy as np
import openmc

BETA_EFF = 0.007294
UPPER_FACTOR = 1.645
MIN_SDM_DOLLAR = 0.5
K_LIMIT = 1.0 / (1.0 + MIN_SDM_DOLLAR * BETA_EFF)


def power_metrics(scores, records):
    """Return three PPFs from integrated scores and fuel volumes (cm³) in records."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    volumes = np.asarray([record["volume_cm3"] for record in records])
    if (
        len(scores) != len(records)
        or not np.isfinite(scores).all()
        or not np.isfinite(volumes).all()
        or (scores < 0).any()
        or (volumes <= 0).any()
        or scores.sum() <= 0
    ):
        raise ValueError("Require matching finite nonnegative scores and positive fuel volumes.")
    mean_density = scores.sum() / volumes.sum()
    radial, axial = defaultdict(lambda: [0.0, 0.0]), defaultdict(lambda: [0.0, 0.0])
    for score, volume, record in zip(scores, volumes, records, strict=True):
        for groups, key in ((radial, record["slot"]), (axial, record["axial"])):
            groups[key][0] += score
            groups[key][1] += volume
    return dict(
        ppf_3d=float(np.max(scores / volumes) / mean_density),
        ppf_radial=max(q / v for q, v in radial.values()) / mean_density,
        ppf_axial=max(q / v for q, v in axial.values()) / mean_density,
        element_count=len(radial),
        segment_count=len(records),
        fuel_volume_cm3=float(volumes.sum()),
        ppf_uncertainty=None,
        uncertainty_note="Joint bin covariance and maximum-selection uncertainty not estimated.",
    )


def evaluate_nominal(statepoint_path, records, tally_name="final_fuel_segment_q"):
    """Read ``tally_name`` from ``statepoint_path`` using ordered segment ``records``.

    Return keff and volume-weighted power metrics for the full 111-element domain
    and the separate 108-element regular-fuel domain. PPF values are dimensionless."""
    with openmc.StatePoint(statepoint_path, autolink=False) as statepoint:
        tally = statepoint.get_tally(name=tally_name)
        if tally.scores != ["fission-q-recoverable"]:
            raise ValueError("Unexpected tally score.")
        bin_ids = tally.find_filter(openmc.CellFilter).bins.reshape(-1).tolist()
        by_id = {record["cell_id"]: record for record in records}
        if len(by_id) != 1665 or set(bin_ids) != set(by_id) or len(bin_ids) != 1665:
            raise ValueError("Nominal final tally must contain each of 1665 fuel segments once.")
        ordered = [by_id[cell_id] for cell_id in bin_ids]
        scores = tally.mean.reshape(-1)
        ordinary = np.array([record["kind"] == "fuel" for record in ordered])
        return dict(
            source=str(statepoint_path),
            openmc_version=statepoint.version,
            current_batch=statepoint.current_batch,
            inactive=statepoint.n_inactive,
            historical_108=power_metrics(
                scores[ordinary], [r for r in ordered if r["kind"] == "fuel"]
            ),
            final_111=power_metrics(scores, ordered),
        )


def baseline_records(geometry_path):
    """Read baseline XML fuel geometry without exporting identities; return 1665 records."""
    xml = ET.parse(geometry_path).getroot()
    surfaces = {item.attrib["id"]: item for item in xml.findall("surface")}
    records = []
    for cell in xml.findall("cell"):
        name = cell.get("name", "")
        if "fuel" not in name.lower() and "uzrh" not in name.lower():
            continue
        # Exact classification is provided by annular fuel region and axial layer bounds.
        tokens = cell.get("region", "").split()
        if len(tokens) != 4 or not all(token.lstrip("-").isdigit() for token in tokens):
            continue
        z, radial = {}, {}
        valid = True
        for token in tokens:
            sign = "-" if token.startswith("-") else "+"
            surface = surfaces[token.lstrip("-")]
            coeffs = list(map(float, surface.get("coeffs").split()))
            if surface.get("type") == "z-plane":
                z[sign] = coeffs[0]
            elif surface.get("type") == "z-cylinder":
                radial[sign] = coeffs[2]
            elif surface.get("type") == "quadric":
                a, b, c, d, e, f, g, h, j, k = coeffs
                if not (
                    math.isclose(a, b, abs_tol=1e-12)
                    and a > 0
                    and all(abs(v) < 1e-12 for v in (c, d, e, f, j))
                ):
                    valid = False
                    continue
                radial[sign] = math.sqrt((g / (2 * a)) ** 2 + (h / (2 * a)) ** 2 - k / a)
            else:
                valid = False
        if (
            not valid
            or len(z) != 2
            or len(radial) != 2
            or not math.isclose(z["-"] - z["+"], 2.54, abs_tol=1e-8)
        ):
            continue
        volume = math.pi * (radial["-"] ** 2 - radial["+"] ** 2) * (z["-"] - z["+"])
        # Cell names encode position/segment in the baseline, but never fuel identity.
        import re

        match = re.search(r"([A-G]\d+)", name)
        if match is None:
            raise ValueError("Baseline cell has no geometric position.")
        slot = match.group(1)
        records.append(
            dict(
                cell_id=int(cell.get("id")),
                slot=slot,
                axial=round((z["+"] - 17.653) / 2.54) + 1,
                volume_cm3=volume,
                kind="ffcr" if slot in ("D2", "D13", "D16") else "fuel",
            )
        )
    if len(records) != 1665:
        raise ValueError(f"Baseline extraction found {len(records)} fuel segments, expected 1665.")
    return records


def sdm_metrics(mean_k, sigma_k):
    """Return fixed-beta SDM dollars and conditional lower bound from k mean/SEM."""
    if not math.isfinite(mean_k) or mean_k <= 0 or not math.isfinite(sigma_k) or sigma_k < 0:
        raise ValueError("Require finite positive k and nonnegative SEM.")
    upper_k = mean_k + UPPER_FACTOR * sigma_k
    return dict(
        k_upper=upper_k,
        sdm_dollar=(1 / mean_k - 1) / BETA_EFF,
        sdm_lower_dollar=(1 / upper_k - 1) / BETA_EFF,
        sigma_k_times_1e5=sigma_k * 1e5,
        sigma_rho_pcm=1e5 * sigma_k / mean_k**2,
        k_limit=K_LIMIT,
        passes_conditional_bound=upper_k <= K_LIMIT,
    )


def budget_scenarios(sigma_k, active_batches, active_seconds, inactive_seconds, inactive=50):
    """Return estimated batch budgets from pilot ``sigma_k`` and measured timings.

    ``active_batches`` is the pilot sample count. ``active_seconds`` and
    ``inactive_seconds`` separate runtime costs, while ``inactive`` fixes the
    discarded-batch count for each estimate. Scaling assumes comparable stationary
    correlation and does not replace a convergence check."""
    rows = []
    for target in (0.0002, 0.0003, 0.0004):
        active = max(2, math.ceil(active_batches * (sigma_k / target) ** 2))
        rows.append(
            dict(
                target_sigma_k=target,
                estimated_active_batches=active,
                inactive_batches=inactive,
                estimated_total_batches=active + inactive,
                estimated_active_histories=active * 100000,
                estimated_total_seconds=inactive_seconds + active_seconds * active / active_batches,
            )
        )
    return rows
