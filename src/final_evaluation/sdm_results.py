"""Read combined keff and report conditional SDM separately from source assessment."""

from collections import Counter
import math
from pathlib import Path
import re

import h5py
import numpy as np
import openmc

from modules.OpenMCResultEvaluator import OpenMCResultEvaluator
from .evaluator import sdm_metrics, BETA_EFF
from .sdm_contract import atomic_json, read_json, sha256, timestamp, write_csv


def metrics(mean_k, sigma_k):
    """Return exact reactivity and SDM for dimensionless combined mean k and SEM."""
    result = sdm_metrics(mean_k, sigma_k)
    return dict(
        rho=(mean_k - 1) / mean_k,
        k_U=result["k_upper"],
        SDM_nominal_dollar=result["sdm_dollar"],
        SDM_L_dollar=result["sdm_lower_dollar"],
        SDM_nominal_pcm=result["sdm_dollar"] * BETA_EFF * 1e5,
        precision_pass=sigma_k <= 0.0003,
        conditional_sdm_pass=result["passes_conditional_bound"],
    )


def read_statepoint(path):
    """Read actual OpenMC StatePoint path; return combined statistics and raw histories."""
    with openmc.StatePoint(path, autolink=False) as statepoint:
        try:
            combined = OpenMCResultEvaluator._extract_keff(statepoint)
            mean, sigma = combined["mean"], combined["std_dev"]
        except ValueError:
            mean, sigma = float("nan"), float("nan")
        measured = dict(
            mean_k=mean,
            sigma_k=sigma,
            particles=int(statepoint.n_particles),
            inactive=int(statepoint.n_inactive),
            batches=int(statepoint.n_batches),
            current_batch=int(statepoint.current_batch),
            generations_per_batch=int(statepoint.generations_per_batch),
            active=int(statepoint.current_batch - statepoint.n_inactive),
            n_realizations=int(statepoint.n_realizations),
            seed=int(statepoint.seed),
            version=[int(v) for v in statepoint.version],
            run_mode=statepoint.run_mode,
            runtime={key: float(value) for key, value in statepoint.runtime.items()},
        )
        try:
            generations = np.asarray(statepoint.k_generation, dtype=float).reshape(-1)
        except (KeyError, ValueError, TypeError):
            generations = np.array([])
        try:
            entropy = np.asarray(statepoint.entropy, dtype=float).reshape(-1)
        except (KeyError, ValueError, TypeError):
            entropy = np.array([])
    measured["numeric_valid"] = bool(
        math.isfinite(mean) and mean > 0 and math.isfinite(sigma) and sigma >= 0
    )
    if not math.isfinite(mean):
        measured["mean_k"] = None
    if not math.isfinite(sigma):
        measured["sigma_k"] = None
    return measured, generations, entropy


def warnings_from_log(log):
    """Return private warning/error contexts and public category counts from log text."""
    lines, contexts, categories = log.splitlines(), [], Counter()
    for index, line in enumerate(lines):
        if re.search(r"warning|error|lost particle|negative.*probability", line, re.I):
            contexts.append(dict(line=index + 1, text="\n".join(lines[index : index + 4])))
            if re.search("negative.*probability", "\n".join(lines[index : index + 3]), re.I):
                categories["negative_probability_tables"] += 1
            elif re.search("warning", line, re.I):
                categories["other_warning"] += 1
            else:
                categories["error_or_lost_particle"] += 1
    return contexts, dict(categories)


def snapshot_evidence(directory, config):
    """Validate scheduled source banks in directory; return aggregate evidence and completeness."""
    rows = []
    for batch in config["source_snapshots"]:
        path = directory / f"source.{batch:03d}.h5"
        row = dict(batch=batch, path=str(path), valid=False)
        if path.is_file():
            try:
                with h5py.File(path, "r") as source:
                    bank = source["source_bank"][()]
                    xyz = np.column_stack([bank["r"][axis] for axis in ("x", "y", "z")])
                    row.update(
                        count=len(bank),
                        sha256=sha256(path),
                        z_min_cm=float(xyz[:, 2].min()) if np.isfinite(xyz).all() else None,
                        z_max_cm=float(xyz[:, 2].max()) if np.isfinite(xyz).all() else None,
                        fraction_below_ordinary_fuel=float(np.mean(xyz[:, 2] < 17.653)),
                        valid=bool(
                            source.attrs.get("filetype") in (b"source", "source")
                            and len(bank) == config["transport"]["particles"]
                            and np.isfinite(xyz).all()
                        ),
                    )
            except (OSError, KeyError, ValueError):
                row["read_error"] = True
        rows.append(row)
    hashes = [row.get("sha256") for row in rows if row["valid"]]
    # Identical snapshots at distinct batches merit review; never certify convergence here.
    return dict(
        snapshots=rows,
        complete=all(row["valid"] for row in rows),
        repeated_bank_hashes=len(hashes) != len(set(hashes)),
    )


def inspect_attempt(directory, job, config, manifest_sha256, context_sha256):
    """Inspect one immutable attempt directory against job/context; return evidence and histories."""
    execution_path = directory / "execution.json"
    execution = read_json(execution_path) if execution_path.exists() else {}
    binding = read_json(directory / "binding.json")
    expected = dict(
        job_sha256=job["job_sha256"],
        manifest_sha256=manifest_sha256,
        context_sha256=context_sha256,
        candidate_id=job["candidate_id"],
        genome_sha256=job["genome_sha256"],
        seed=job["seed"],
        case=job["case"],
        input_sha256=job["input_sha256"],
    )
    if binding != expected or sha256(directory / "model.xml") != job["input_sha256"]:
        raise ValueError("Attempt candidate/configuration/seed/input/context binding mismatch.")
    log_path = directory / "openmc.log"
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    warning_contexts, warning_counts = warnings_from_log(log)
    threads = re.search(r"OpenMP Threads\s*\|\s*(\d+)", log)
    mpi = re.search(r"MPI Processes\s*\|\s*(\d+)", log)
    version = re.search(r"Version\s*\|\s*(\S+)", log)
    launched = execution.get("command") == [config["executable"], "-s", "32"]
    row = dict(
        transport_status=(
            "completed" if execution.get("returncode") == 0 else "failed_or_incomplete"
        ),
        transport_success=execution.get("returncode") == 0,
        statepoint_complete=False,
        configuration_match=False,
        numeric_valid=False,
        final_valid=False,
        diagnostics_complete=False,
        mean_k=None,
        sigma_k=None,
        k_U=None,
        rho=None,
        SDM_nominal_dollar=None,
        SDM_L_dollar=None,
        SDM_nominal_pcm=None,
        precision_pass=None,
        conditional_sdm_pass=None,
        source_review_status="pending_manual_review",
        wall_seconds=execution.get("wall_seconds"),
        warnings=warning_counts,
        actual_threads=int(threads[1]) if threads else None,
        actual_mpi_processes=int(mpi[1]) if mpi else None,
        evidence=str(directory),
        warning_contexts=warning_contexts,
    )
    path = directory / "statepoint.200.h5"
    if not path.exists():
        return row, np.array([]), np.array([])
    try:
        measured, generations, entropy = read_statepoint(path)
    except (OSError, ValueError, KeyError, RuntimeError, AttributeError):
        row["statepoint_read_error"] = True
        return row, np.array([]), np.array([])
    t = config["transport"]
    config_match = all(
        measured[key] == t[key]
        for key in ("particles", "inactive", "active", "batches", "generations_per_batch")
    )
    complete = measured["current_batch"] == 200 and measured["n_realizations"] == 155
    row.update(
        actual_settings=measured,
        mean_k=measured["mean_k"],
        sigma_k=measured["sigma_k"],
        statepoint_complete=complete,
        statepoint_sha256=sha256(path),
        numeric_valid=measured["numeric_valid"],
        configuration_match=bool(
            config_match
            and launched
            and threads
            and threads[1] == "32"
            and mpi
            and mpi[1] == "1"
            and version
            and version[1] == "0.15.3"
            and measured["seed"] == job["seed"]
            and measured["version"] == [0, 15, 3]
            and measured["run_mode"] == "eigenvalue"
        ),
    )
    if row["numeric_valid"]:
        row.update(metrics(measured["mean_k"], measured["sigma_k"]))
    row["final_valid"] = bool(
        row["transport_success"]
        and complete
        and row["configuration_match"]
        and row["numeric_valid"]
    )
    snapshots = snapshot_evidence(directory, config)
    histories_ok = (
        len(generations) == len(entropy) == 200
        and np.isfinite(generations).all()
        and np.isfinite(entropy).all()
    )
    row.update(
        source_snapshots=snapshots,
        diagnostics_complete=bool(histories_ok and snapshots["complete"]),
    )
    review = directory / "source_review.json"
    if review.exists():
        decision = read_json(review)
        required = ("reviewer", "basis", "uncertainty_limitations", "reviewed_at")
        if (
            decision.get("statepoint_sha256") == row["statepoint_sha256"]
            and all(decision.get(key) for key in required)
            and decision.get("status") in ("reviewed_with_limitations", "requires_further_review")
        ):
            row["source_review_status"] = decision["status"]
    if not row["diagnostics_complete"]:
        row["source_review_status"] = "diagnostics_incomplete"
    return row, generations, entropy


def summarize(candidate_id, rows):
    """Summarize exactly five required cases; return distinct numerical/precision/review statuses."""
    cases = {row["case"] for row in rows}
    required_cases = {"STUCK-D2", "STUCK-D5", "STUCK-D9", "STUCK-D13", "STUCK-D16"}
    complete = (
        len(rows) == 5 and cases == required_cases and all(row["final_valid"] for row in rows)
    )
    valid = [row for row in rows if row["final_valid"]]
    limiting = min(valid, key=lambda row: row["SDM_L_dollar"]) if valid else None
    precision = all(row["precision_pass"] for row in rows) if complete else None
    bound = all(row["conditional_sdm_pass"] for row in rows) if complete else None
    review = (
        "reviewed_with_limitations"
        if complete
        and all(row["source_review_status"] == "reviewed_with_limitations" for row in rows)
        else "requires_review_or_data"
    )
    if not complete:
        status = "incomplete_or_invalid_cases"
    elif not bound:
        status = "conditional_numeric_bound_not_met"
    else:
        status = "conditional_numeric_bound_met"
    if complete and not precision:
        status += ";precision_target_not_met"
    if review != "reviewed_with_limitations":
        status += ";source_review_required"
    return dict(
        candidate_id=candidate_id,
        completed_cases=sum(row["transport_success"] for row in rows),
        valid_cases=len(valid),
        five_cases_complete=complete,
        SDM_L_min_dollar=limiting["SDM_L_dollar"] if complete else None,
        limiting_case=limiting["case"] if complete else None,
        observed_partial_min_dollar=limiting["SDM_L_dollar"] if limiting else None,
        precision_pass=precision,
        conditional_sdm_pass=bound,
        diagnostics_complete=bool(complete and all(row["diagnostics_complete"] for row in rows)),
        source_review_status=review,
        evaluation_status=status,
    )


def diagnostic_plot(directory, generations, entropy, inactive):
    """Save raw track-length/entropy histories and PNG at directory; return None."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = max(len(generations), len(entropy))
    write_csv(
        directory / "diagnostics.csv",
        [
            dict(
                batch=i + 1,
                inactive=i < inactive,
                k_generation_tracklength=float(generations[i]) if i < len(generations) else None,
                entropy=float(entropy[i]) if i < len(entropy) else None,
            )
            for i in range(count)
        ],
    )
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(np.arange(1, len(generations) + 1), generations, lw=0.8)
    axes[1].plot(np.arange(1, len(entropy) + 1), entropy, lw=0.8)
    for axis in axes:
        axis.axvline(inactive + 0.5, color="red", ls="--", label=f"inactive={inactive}")
        axis.legend()
    axes[0].set_ylabel("k_generation (track-length)")
    axes[1].set_ylabel("Entropy (bit)")
    axes[1].set_xlabel("Batch")
    figure.suptitle("Source diagnostics — manual review required")
    figure.tight_layout()
    figure.savefig(directory / "diagnostics.png", dpi=140)
    plt.close(figure)


def report(root, manifest):
    """Create a new timestamped report beneath root from all manifest jobs; return report path."""
    output = root / "reports" / timestamp().replace(":", "").replace("+", "_")
    output.mkdir(parents=True, mode=0o700)
    rows = []
    for job in manifest["jobs"]:
        directory = root / job["directory"]
        state = read_json(directory / "status.json")
        row = dict(
            transport_status=state["state"],
            transport_success=False,
            statepoint_complete=False,
            configuration_match=False,
            numeric_valid=False,
            final_valid=False,
            diagnostics_complete=False,
            mean_k=None,
            sigma_k=None,
            rho=None,
            k_U=None,
            SDM_nominal_dollar=None,
            SDM_L_dollar=None,
            SDM_nominal_pcm=None,
            precision_pass=None,
            conditional_sdm_pass=None,
            source_review_status="not_available",
            wall_seconds=None,
            warnings={},
            actual_threads=None,
            actual_mpi_processes=None,
            actual_settings={},
            evidence=str(directory),
        )
        if state.get("attempt"):
            attempt = directory / "attempts" / state["attempt"]
            try:
                evidence, generations, entropy = inspect_attempt(
                    attempt,
                    job,
                    manifest["config"],
                    manifest["manifest_sha256"],
                    manifest["context_sha256"],
                )
                atomic_json(output / f"{job['candidate_id']}_{job['case']}_internal.json", evidence)
                row.update(
                    {
                        key: value
                        for key, value in evidence.items()
                        if key not in ("warning_contexts", "source_snapshots")
                    }
                )
                if len(generations) or len(entropy):
                    plot_dir = output / "diagnostics" / job["candidate_id"] / job["case"]
                    plot_dir.mkdir(parents=True)
                    diagnostic_plot(
                        plot_dir, generations, entropy, evidence["actual_settings"]["inactive"]
                    )
            except (ValueError, OSError, KeyError):
                row["transport_status"] = "verification_conflict"
        row.update(
            candidate_id=job["candidate_id"],
            case=job["case"],
            seed=job["seed"],
            planned_settings=job["transport"],
            actual_settings=row.get("actual_settings", {}),
        )
        rows.append(row)
    # Uniform column set covers unstarted, failed and completed cases in one table.
    keys = list(dict.fromkeys(key for row in rows for key in row))
    public = []
    for row in rows:
        public.append(
            {
                key: (
                    __import__("json").dumps(row.get(key), sort_keys=True)
                    if isinstance(row.get(key), (dict, list))
                    else row.get(key)
                )
                for key in keys
            }
        )
    write_csv(output / "case_results.csv", public)
    summary = [
        summarize(candidate_id, [r for r in rows if r["candidate_id"] == candidate_id])
        for candidate_id in manifest["cohort"]
    ]
    write_csv(output / "candidate_summary.csv", summary)
    counts = Counter(row["transport_status"] for row in rows)
    text = f"""# Conditional SDM of the public fresh-fuel front

Dibuat {timestamp()}. Manifest: `{manifest['manifest_sha256']}`.
Jumlah pekerjaan: {len(rows)}; status transport: {dict(counts)}.
Kandidat dengan lima kasus valid: {sum(r['five_cases_complete'] for r in summary)}/{len(summary)}.

Public demonstration inventory at 300 K; OpenMC executable/API 0.15.3.
beta_eff=0.007294 is an illustrative fixed conversion, not calculated for this fresh core.
No facility safety qualification or independent MCNP benchmark is claimed.
Anggaran: 100000 partikel, 45 inactive + 155 active = 200 batch, gpb=1,
trigger nonaktif, satu kasus dengan {manifest['config']['transport']['threads']} thread setiap waktu. Tidak ada perpanjangan
atau pengulangan otomatis untuk mencapai sigma_k <= 0.0003.
45 inactive menjaga konsistensi optimisasi; belum membuktikan konvergensi sumber.

Combined keff dibaca melalui StatePoint.keff (mean dan standard deviation).
rho=(k-1)/k; k_U=k+1.645*sigma_k;
SDM_nominal=(1/k-1)/0.007294; SDM_L=(1/k_U-1)/0.007294.
SDM kandidat adalah minimum lima SDM_L; keadaan pembatas argmin.
0.5 dollar = 0.003647 reaktivitas = 364.7 pcm. Nilai negatif dipertahankan.
Tidak memakai selisih terhadap nominal all-OUT atau baseline.

Faktor 1.645 merupakan pendekatan batas satu sisi 95% per keadaan dengan
asumsi estimasi ketidakpastian memadai; bukan jaminan simultan 95% untuk seluruh kasus yang dievaluasi.
`k_generation` adalah estimator track-length pada build ini; bukan combined keff.
Riwayat itu tidak digunakan untuk merekonstruksi combined setelah pemotongan batch.

`case_results.csv` memisahkan transport_success, statepoint_complete,
configuration_match, numeric_valid, final_valid, precision_pass,
conditional_sdm_pass, diagnostics_complete, dan source_review_status.
SDM numerik tetap ditampilkan bila valid walaupun konfigurasi atau review bermasalah;
kelayakan hasil final mensyaratkan final_valid. Nilai kosong bukan nol/lulus/gagal.
`candidate_summary.csv` hanya mengisi SDM_L_min dan status batas kandidat bila
kelima kasus final_valid. observed_partial_min hanya informasi parsial.
Kegagalan presisi ditampilkan terpisah dari batas SDM. Status numerik terpenuhi
belum berarti diagnostik memadai atau kualifikasi keselamatan instalasi.

Grafik dan CSV diagnostik berisi nomor batch, batas inactive, entropy dan
k_generation. Tidak ada ambang slope/p-value atau sertifikat konvergensi otomatis.
Keberadaan bank di domain valid tidak membuktikan konvergensi distribusi sumber.
Snapshot terjadwal diperiksa keberadaan, jumlah, koordinat finite dan hash;
penilaian distribusi spasial serta kelayakan SEM masih memerlukan telaah.

Kolom warnings hanya kategori/jumlah; seluruh isi dan konteks peringatan,
termasuk negative probability tables, ada pada log dan evidence internal privat.
Dampak peringatan data nuklir tidak dianggap nol. Input, log, HDF5 dan
*_internal.json tidak boleh dipublikasikan: mengandung data penelitian privat.
Laporan ini tidak mengekspor identitas bahan bakar/komposisi/pemetaan rahasia.

Rujukan API primer: [OpenMC 0.15.3 StatePoint](https://docs.openmc.org/en/v0.15.3/pythonapi/generated/openmc.StatePoint.html).
"""
    (output / "report.md").write_text(text)
    atomic_json(
        output / "summary.json",
        dict(
            counts=dict(counts),
            candidates=summary,
            final_transport_completed=sum(r["final_valid"] for r in rows),
        ),
    )
    return output
