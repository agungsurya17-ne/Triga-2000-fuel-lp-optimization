#!/usr/bin/env python3
"""Analyze saved reevaluation artifacts without launching or importing OpenMC."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import (
    BASELINE_ID,
    DEFAULT_OUTPUT,
    HERE,
    NOMINAL,
    atomic_path,
    campaign_lock,
    classify,
    digest,
    dominates,
    object_digest,
    power_metrics,
    read_json,
    write_json,
)

SEGMENT_COLUMNS = [
    "candidate_id",
    "cell_id",
    "slot",
    "axial",
    "fuel_id",
    "genome_index",
    "category",
    "x_cm",
    "y_cm",
    "z_low_cm",
    "z_high_cm",
    "inner_radius_cm",
    "outer_radius_cm",
    "volume_cm3",
    "material_id",
    "material_name",
    "q_eV_per_source",
    "q_std_dev_eV_per_source",
    "density_eV_per_source_cm3",
    "density_std_dev_eV_per_source_cm3",
    "relative_density",
    "tally_relative_std_dev",
]
PROFILE_COLUMNS = [
    "candidate_id",
    "domain",
    "axis",
    "group",
    "q_eV_per_source",
    "volume_cm3",
    "density_eV_per_source_cm3",
    "relative_density",
    "fuel_id",
    "category",
    "z_low_cm",
    "z_high_cm",
    "delta_reference_relative_density",
]
HOTSPOT_COLUMNS = [
    "candidate_id",
    "domain",
    "fuel_id",
    "slot",
    "axial",
    "category",
    "x_cm",
    "y_cm",
    "z_low_cm",
    "z_high_cm",
    "q_eV_per_source",
    "density_eV_per_source_cm3",
    "relative_density",
    "volume_cm3",
    "tied_maximum_count",
    "radial_peak",
    "axial_peak",
]


def save_csv(path, frame):
    """Atomically write DataFrame frame as UTF-8 CSV at path; return None."""
    with atomic_path(path) as temporary:
        frame.to_csv(temporary, index=False, float_format="%.17g")


def verified_result(output, entry):
    """Verify completed artifact hashes/configuration and saved arithmetic; return result dictionary."""
    if entry["configuration_sha256"] != object_digest(entry["configuration"]):
        raise ValueError("Manifest configuration hash mismatch.")
    directory = output / entry["attempt_path"]
    required = [
        directory / name for name in ["model.xml", "segments.json", "loading.json", "result.json"]
    ]
    required.append(output / entry["statepoint_path"])
    if set(entry["artifacts"]) != {str(p.relative_to(output)) for p in required}:
        raise ValueError("Missing required artifact verification.")
    for path in required:
        if (
            not path.resolve().is_relative_to(output.resolve())
            or digest(path) != entry["artifacts"][str(path.relative_to(output))]
        ):
            raise ValueError(f"Missing or changed completed artifact: {path}")
    result = read_json(directory / "result.json")
    if (
        result["configuration_sha256"] != entry["configuration_sha256"]
        or result["seed"] != entry["seed"]
        or result["candidate_id"] != entry["candidate_id"]
    ):
        raise ValueError("Result identity mismatch.")
    metadata = read_json(directory / "segments.json")
    if [{k: row[k] for k in metadata[0]} for row in result["segments"]] != metadata:
        raise ValueError("Result segment identities differ from exported model metadata.")
    if (
        result["current_batch"] != entry["configuration"]["parameters"]["batches"]
        or result["active_realizations"]
        != entry["configuration"]["parameters"]["batches"]
        - entry["configuration"]["parameters"]["inactive"]
    ):
        raise ValueError("Result is not a final nominal statepoint.")
    for domain, selected in [
        ("all_fuel", result["segments"]),
        ("regular", [s for s in result["segments"] if s["category"] == "REGULAR"]),
    ]:
        records = [{key: row[key] for key in metadata[0]} for row in selected]
        actual = power_metrics([row["q_eV_per_source"] for row in selected], records)
        if result[domain] != actual:
            raise ValueError(f"Saved PPF/profile arithmetic mismatch: {domain}")
    return result


def make_tables(manifest, output):
    """Verify all completed cases and retain pending/failed rows; return results/profiles/relations/errors."""
    rows, segments, profiles, hotspots, errors = [], [], [], [], {}
    for candidate_id, entry in manifest["cases"].items():
        historical = {
            k: v for k, v in entry["configuration"]["case"].items() if k.startswith("historical_")
        }
        row = dict(
            candidate_id=candidate_id,
            status=entry["status"],
            seed=entry["seed"],
            purpose=manifest["purpose"],
            domain="all_fuel_111_elements_1665_segments",
            particles=manifest["parameters"]["particles"],
            batches=manifest["parameters"]["batches"],
            inactive=manifest["parameters"]["inactive"],
            threads=manifest["threads"],
            elapsed_seconds=entry.get("elapsed_seconds"),
            transport_seconds=entry.get("transport_seconds"),
            statepoint_path=entry.get("statepoint_path"),
            error=entry.get("error"),
            **historical,
        )
        for key in [
            "keff",
            "keff_std_dev",
            "ppf_3d_all_fuel",
            "ppf_radial_all_fuel",
            "ppf_axial_all_fuel",
            "ppf_3d_regular",
            "ppf_radial_regular",
            "ppf_axial_regular",
            "delta_keff_historical",
            "delta_ppf_regular_historical",
            "domain_ppf_difference",
            "delta_keff_reference",
            "delta_keff_reference_pcm",
            "delta_ppf_3d_reference",
            "comparison_class",
            "nondominated_within_evaluated_candidates",
            "hotspot_fuel_id",
            "hotspot_slot",
            "hotspot_axial",
            "hotspot_category",
            "maximum_density_eV_per_source_cm3",
            "total_fuel_volume_cm3",
            "zero_mean_bins",
            "ppf_uncertainty",
            "max_abs_radial_profile_delta_reference",
            "max_abs_axial_profile_delta_reference",
        ]:
            row[key] = None
        if entry["status"] == "completed":
            try:
                result = verified_result(output, entry)
                row.update(
                    keff=result["keff"],
                    keff_std_dev=result["keff_std_dev"],
                    zero_mean_bins=result["zero_mean_bins"],
                )
                for domain, suffix in [("all_fuel", "all_fuel"), ("regular", "regular")]:
                    metrics = result[domain]
                    for key in ["ppf_3d", "ppf_radial", "ppf_axial"]:
                        row[f"{key}_{suffix}"] = metrics[key]
                    profiles.extend(
                        dict(candidate_id=candidate_id, domain=domain, **p)
                        for p in metrics["profiles"]
                    )
                    hotspot = metrics["hotspot"]
                    hotspots.append(
                        dict(
                            candidate_id=candidate_id,
                            domain=domain,
                            **{k: v for k, v in hotspot.items() if k in HOTSPOT_COLUMNS},
                            tied_maximum_count=len(metrics["tied_maximum_segments"]),
                            radial_peak=metrics["radial_peak"],
                            axial_peak=metrics["axial_peak"],
                        )
                    )
                segments.extend(
                    dict(candidate_id=candidate_id, **segment) for segment in result["segments"]
                )
                hotspot = result["all_fuel"]["hotspot"]
                row.update(
                    hotspot_fuel_id=hotspot["fuel_id"],
                    hotspot_slot=hotspot["slot"],
                    hotspot_axial=hotspot["axial"],
                    hotspot_category=hotspot["category"],
                    maximum_density_eV_per_source_cm3=hotspot["density_eV_per_source_cm3"],
                    total_fuel_volume_cm3=result["all_fuel"]["fuel_volume_cm3"],
                )
                if historical["historical_keff"] is not None:
                    row["delta_keff_historical"] = result["keff"] - historical["historical_keff"]
                    row["delta_ppf_regular_historical"] = (
                        result["regular"]["ppf_3d"] - historical["historical_ppf_regular"]
                    )
                row["domain_ppf_difference"] = (
                    result["all_fuel"]["ppf_3d"] - result["regular"]["ppf_3d"]
                )
            except Exception as exc:
                row.update(status="verification_failed", error=str(exc))
                errors[candidate_id] = str(exc)
        rows.append(row)
    reference = next(
        (r for r in rows if r["candidate_id"] == BASELINE_ID and r["status"] == "completed"), None
    )
    valid = [r for r in rows if r["status"] == "completed" and r["candidate_id"] != BASELINE_ID]
    relations = []
    profile_reference = {
        (p["domain"], p["axis"], p["group"]): p["relative_density"]
        for p in profiles
        if p["candidate_id"] == BASELINE_ID
    }
    for profile in profiles:
        baseline_value = profile_reference.get(
            (profile["domain"], profile["axis"], profile["group"])
        )
        profile["delta_reference_relative_density"] = (
            None if baseline_value is None else profile["relative_density"] - baseline_value
        )
    for row in valid:
        point = (row["keff"], row["ppf_3d_all_fuel"])
        dominators = []
        for other in valid:
            if other["candidate_id"] == row["candidate_id"]:
                continue
            relation = dominates((other["keff"], other["ppf_3d_all_fuel"]), point)
            relations.append(
                dict(
                    candidate_id=other["candidate_id"],
                    target_id=row["candidate_id"],
                    dominates=relation,
                )
            )
            if relation:
                dominators.append(other["candidate_id"])
        row["nondominated_within_evaluated_candidates"] = not dominators
        row["dominated_by"] = ",".join(dominators)
        if reference:
            row["delta_keff_reference"] = row["keff"] - reference["keff"]
            row["delta_keff_reference_pcm"] = 1e5 * row["delta_keff_reference"]
            row["delta_ppf_3d_reference"] = row["ppf_3d_all_fuel"] - reference["ppf_3d_all_fuel"]
            for axis in ["radial", "axial"]:
                values = [
                    p["delta_reference_relative_density"]
                    for p in profiles
                    if p["candidate_id"] == row["candidate_id"]
                    and p["domain"] == "all_fuel"
                    and p["axis"] == axis
                ]
                row[f"max_abs_{axis}_profile_delta_reference"] = max(abs(v) for v in values)
            row["comparison_class"] = classify(
                point, (reference["keff"], reference["ppf_3d_all_fuel"])
            )
    return (
        pd.DataFrame(rows),
        pd.DataFrame(segments, columns=SEGMENT_COLUMNS),
        pd.DataFrame(profiles, columns=PROFILE_COLUMNS),
        pd.DataFrame(hotspots, columns=HOTSPOT_COLUMNS),
        pd.DataFrame(relations, columns=["candidate_id", "target_id", "dominates"]),
        errors,
    )


def make_plots(output, results, purpose):
    """Save Pareto and reference-difference plots with explicit domains/purpose; return None."""
    valid = results[results.status == "completed"]
    candidates = valid[valid.candidate_id != BASELINE_ID]
    reference = valid[valid.candidate_id == BASELINE_ID]
    fig, ax = plt.subplots(figsize=(10, 7))
    if len(candidates):
        nd = candidates.nondominated_within_evaluated_candidates.fillna(False).astype(bool)
        ax.scatter(
            candidates.loc[~nd, "keff"],
            candidates.loc[~nd, "ppf_3d_all_fuel"],
            c="0.65",
            label="Dominated within evaluated cohort",
        )
        front = candidates.loc[nd].sort_values("keff")
        ax.plot(
            front.keff,
            front.ppf_3d_all_fuel,
            "o-",
            color="tab:blue",
            label="Nondominated within evaluated cohort",
        )
        for row in candidates.itertuples():
            ax.annotate(
                row.candidate_id,
                (row.keff, row.ppf_3d_all_fuel),
                fontsize=5,
                xytext=(3, 3),
                textcoords="offset points",
            )
    if len(reference):
        ax.scatter(
            reference.keff,
            reference.ppf_3d_all_fuel,
            marker="*",
            s=200,
            color="tab:red",
            label="Public fresh reference, reevaluated",
        )
    ax.set(
        xlabel=r"$k_{eff}$ (combined estimator, dimensionless; maximize)",
        ylabel="PPF 3D (dimensionless; minimize)",
        title=f"{purpose}: 111 fuel elements / 1665 segments\nVerified {len(valid)}/{len(results)} calculations",
    )
    if len(valid):
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No verified completed simulations", transform=ax.transAxes, ha="center")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    with atomic_path(output / "pareto_reevaluation.png") as path:
        fig.savefig(path, dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    colors = {
        "mendominasi acuan": "tab:green",
        "didominasi acuan": "tab:red",
        "Trade-off A": "tab:orange",
        "Trade-off B": "tab:blue",
        "setara": "tab:purple",
    }
    if len(reference):
        for label, color in colors.items():
            selected = candidates[candidates.comparison_class == label]
            if len(selected):
                ax.scatter(
                    selected.delta_keff_reference_pcm,
                    selected.delta_ppf_3d_reference,
                    label=label,
                    c=color,
                )
        if len(candidates):
            ax.legend(fontsize=8)
    else:
        ax.text(
            0.5,
            0.6,
            "Reference not yet verified: comparison unavailable",
            transform=ax.transAxes,
            ha="center",
        )
    ax.set(
        xlabel=r"$10^5 (k_{candidate}-k_{reference})$ (pcm in k; not reactivity)",
        ylabel=r"$PPF_{3D,candidate}-PPF_{3D,reference}$ (dimensionless)",
        title=f"{purpose}: same domain, 111 fuel elements / 1665 segments",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    with atomic_path(output / "comparison_nominal.png") as path:
        fig.savefig(path, dpi=180)
    plt.close(fig)


def summary_text(manifest, results, profiles, hotspots, errors):
    """Build an Indonesian numeric report with explicit incomplete/smoke limits; return Markdown."""
    valid = results[results.status == "completed"]
    candidates = valid[valid.candidate_id != BASELINE_ID]
    reference = valid[valid.candidate_id == BASELINE_ID]
    complete = len(valid) == len(results)
    lines = [
        f"# Evaluasi ulang IV.4.5.2 — {manifest['purpose']}",
        "",
        f"Status: {'LENGKAP' if complete else 'BELUM LENGKAP'}. Hasil terverifikasi {len(valid)}/{len(results)} perhitungan.",
        f"Public final front: {manifest['cohort_audit']['actual_front_count']} candidates; population {manifest['cohort_audit']['population_size']}, {manifest['cohort_audit']['evolution_cycles']} saved generations. Membership remains fixed.",
        f"Status tersimpan: {dict(Counter(results.status))}.",
        f"Parameter: {manifest['parameters']}; thread {manifest['threads']}, satu proses OpenMC berurutan.",
        "",
        "Domain utama: 108 elemen REGULAR + 3 follower FFCR, masing-masing 15 lapisan = 1665 segmen. Domain reguler terpisah: 1620 segmen.",
        "Q: eV per partikel sumber; volume hanya fuel meat dalam cm³; d=Q/V dalam eV/(partikel sumber·cm³). PPF tanpa dimensi, tidak dikonversi menjadi watt.",
        "d̄=ΣQ/ΣV; PPF3D=max(Q/V)/d̄; PPFr=max_i(Σ_j Q/Σ_j V)/d̄; PPFz=max_j(Σ_i Q/Σ_i V)/d̄.",
        "Indeks aksial 1–15 dari bawah ke atas. Identitas material/elemen serta koordinat cm disimpan dalam CSV segmen.",
        "",
        "The public reference is shuffled independently using the fictional 108-pin inventory and three fixed followers. Every case uses the same geometry-v2 builder, fresh recipes and nominal rod states.",
        "Real serials, irradiated compositions and the facility reference loading are excluded. These results describe the public demonstration and do not reproduce the confidential thesis results.",
        "",
        "Perbandingan historis: delta PPF reguler hanya membandingkan dua domain 1620 segmen yang sama. Selisih PPF seluruh bahan bakar dan PPF reguler dari satu simulasi adalah perubahan cakupan domain; selisih lintas domain terhadap historis tidak ditafsirkan sebagai efek ketelitian Monte Carlo.",
        "Klasifikasi memakai angka sebelum pembulatan: maksimalkan k dan minimalkan PPF. Trade-off A: k lebih tinggi dan PPF lebih tinggi; Trade-off B: k lebih rendah dan PPF lebih rendah. Kesamaan satu objektif mengikuti dominasi lemah dengan minimal satu perbaikan ketat.",
        "Nondominated baru hanya berlaku di antara kandidat yang berhasil diverifikasi dari himpunan tetap ini; bukan optimisasi ulang atau pencarian seluruh populasi. Jika belum lengkap, hubungan dominasi masih provisional.",
        "Standar deviasi tally dilaporkan sebagai kualitas statistik per segmen. Ketidakpastian PPF tidak dihitung; tidak ada klaim signifikansi statistik, kriteria penerimaan optimisasi, atau perhitungan shutdown margin.",
        "",
    ]
    if manifest["purpose"] == "SMOKE_ONLY":
        lines.extend(
            [
                "**SMOKE TEST: seluruh nilai berikut hanya bukti fungsi perangkat lunak; bukan data produksi atau kesimpulan fisika.**",
                "",
            ]
        )
    if len(valid):
        lines.append(
            f"Rentang k terverifikasi: {valid.keff.min():.9f}–{valid.keff.max():.9f}; PPF3D seluruh bahan bakar: {valid.ppf_3d_all_fuel.min():.8f}–{valid.ppf_3d_all_fuel.max():.8f}."
        )
        lines.append(
            f"Volume domain seluruh bahan bakar: {valid.total_fuel_volume_cm3.iloc[0]:.9f} cm³."
        )
        if len(reference):
            row = reference.iloc[0]
            lines.append(
                f"Acuan ulang: k={row.keff:.9f} ± {row.keff_std_dev:.9f} (galat baku 1σ estimator gabungan); PPF3D={row.ppf_3d_all_fuel:.8f}."
            )
        lines.append(
            f"Klasifikasi kandidat terhadap acuan: {dict(Counter(candidates.comparison_class.dropna()))}."
        )
        if len(candidates):
            lines.append(
                f"Kandidat nondominated dari {len(candidates)} hasil kandidat terverifikasi: {', '.join(candidates.loc[candidates.nondominated_within_evaluated_candidates == True, 'candidate_id'])}."
            )
            lines.append(
                f"Delta PPF cakupan domain (seluruh bahan bakar minus reguler dari simulasi identik): {candidates.domain_ppf_difference.min():.8f} sampai {candidates.domain_ppf_difference.max():.8f}."
            )
        lines.append(
            f"Kategori hotspot domain seluruh bahan bakar: {dict(Counter(valid.hotspot_category))}."
        )
        for row in valid.itertuples():
            lines.append(
                f"- {row.candidate_id}: puncak {row.hotspot_fuel_id} di {row.hotspot_slot}, lapisan {row.hotspot_axial}, {row.hotspot_category}; PPFr={row.ppf_radial_all_fuel:.6f}, PPFz={row.ppf_axial_all_fuel:.6f}, PPF3D={row.ppf_3d_all_fuel:.6f}; {row.zero_mean_bins} bin rerata nol."
            )
        radial = profiles[(profiles.domain == "all_fuel") & (profiles.axis == "radial")]
        for candidate_id, group in radial.groupby("candidate_id", sort=False):
            ffcr = group[group.category == "FFCR"]
            fraction = ffcr.q_eV_per_source.sum() / group.q_eV_per_source.sum()
            axial = profiles[
                (profiles.candidate_id == candidate_id)
                & (profiles.domain == "all_fuel")
                & (profiles.axis == "axial")
            ]
            layers = axial.set_index("group").relative_density
            if candidate_id != BASELINE_ID and group.delta_reference_relative_density.notna().all():
                peak = group.loc[group.delta_reference_relative_density.abs().idxmax()]
                axial_peak = axial.loc[axial.delta_reference_relative_density.abs().idxmax()]
                lines.append(
                    f"- Perbedaan profil {candidate_id} terhadap acuan, normalisasi rerata domain identik: deviasi radial absolut terbesar di {peak['group']} (Δ={peak.delta_reference_relative_density:.6f}); deviasi aksial absolut terbesar pada lapisan {axial_peak['group']} (Δ={axial_peak.delta_reference_relative_density:.6f})."
                )
            lines.append(
                f"- Profil {candidate_id}: fraksi respons terintegrasi FFCR={fraction:.6%}; maksimum radial FFCR={ffcr.relative_density.max():.6f}; maksimum aksial lapisan {axial.loc[axial.relative_density.idxmax(), 'group']}; rasio rapat lapisan 15/1={float(layers['15'] / layers['1']) if layers['1'] > 0 else float('nan'):.6f}."
            )
        seconds = valid.transport_seconds.sum()
        lines.append(
            f"Waktu transport yang telah tersimpan: {seconds:.1f} s; hasil lengkap mencatat waktu per kasus."
        )
    else:
        lines.append(
            "Belum tersedia hasil neutronik; tabel berisi seluruh ID dan status, grafik menandai ketiadaan hasil."
        )
    if errors:
        lines.extend(
            ["", f"Gagal verifikasi artefak: {errors}. Tidak dihitung sebagai hasil selesai."]
        )
    return "\n".join(lines) + "\n"


def analyze(output=DEFAULT_OUTPUT, locked=False):
    """Generate all tables/figures from output's saved artifacts, without transport; return report status."""
    output = Path(output).resolve()
    if not output.is_relative_to(HERE) or output == HERE:
        raise ValueError("Analysis output must remain in the isolated reevaluation directory.")
    with nullcontext() if locked else campaign_lock(output):
        manifest = read_json(output / "manifest.json")
        if object_digest(manifest["identity"]) != manifest["identity_sha256"]:
            raise ValueError("Manifest immutable identity changed.")
        expected_cases = manifest["identity"]["cases"]
        if list(manifest["cases"]) != [c["candidate_id"] for c in expected_cases]:
            raise ValueError("Manifest cohort membership/order changed.")
        for offset, case in enumerate(expected_cases):
            entry = manifest["cases"][case["candidate_id"]]
            config = entry["configuration"]
            if (
                config["case"] != case
                or config["parameters"] != manifest["identity"]["parameters"]
                or entry["seed"] != manifest["identity"]["seed_base"] + offset
                or config["seed"] != entry["seed"]
                or entry["configuration_sha256"] != object_digest(config)
            ):
                raise ValueError(
                    "Manifest per-case loading, budget, seed, or configuration changed."
                )
        tables = make_tables(manifest, output)
        results, segments, profiles, hotspots, relations, errors = tables
        for name, frame in [
            ("reevaluation_results", results),
            ("segment_responses", segments),
            ("radial_axial_profiles", profiles),
            ("hotspot_summary", hotspots),
            ("dominance_relations", relations),
        ]:
            save_csv(output / (name + ".csv"), frame)
        definitions = pd.DataFrame(
            [
                ["main_domain", "111 fuel elements / 1665 fuel-meat segments"],
                ["regular_domain", "108 regular fuel elements / 1620 fuel-meat segments"],
                ["Q", "eV/source particle, integrated recoverable fission response"],
                ["density", "Q/V in eV/(source particle cm^3); not watts"],
                ["PPF", "Dimensionless, volume-weighted domain mean; based on tally means"],
                ["keff_std_dev", "Combined-estimator standard error, 1 sigma, dimensionless"],
                ["delta_keff_reference_pcm", "1e5 delta k, not delta reactivity"],
                ["Trade-off A", "Higher k and higher PPF than reference"],
                ["Trade-off B", "Lower k and lower PPF than reference"],
                ["ppf_uncertainty", "Not estimated; segment SEM is not PPF uncertainty"],
                [
                    "scope",
                    "Saved public first front; new nondominated set only within verified candidates",
                ],
                ["purpose", manifest["purpose"]],
            ],
            columns=["field", "definition"],
        )
        with atomic_path(output / "comparison.xlsx") as path:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                results.to_excel(writer, sheet_name="Reevaluation", index=False)
                hist_columns = ["candidate_id", "status"] + [
                    c
                    for c in results
                    if "historical" in c or c.endswith("_regular") or c == "domain_ppf_difference"
                ]
                results[list(dict.fromkeys(hist_columns))].to_excel(
                    writer, sheet_name="Historical_regular", index=False
                )
                columns = [
                    "candidate_id",
                    "status",
                    "keff",
                    "keff_std_dev",
                    "ppf_3d_all_fuel",
                    "delta_keff_reference",
                    "delta_keff_reference_pcm",
                    "delta_ppf_3d_reference",
                    "comparison_class",
                ]
                results[columns].to_excel(writer, sheet_name="Reference_comparison", index=False)
                relations.to_excel(writer, sheet_name="Dominance_pairs", index=False)
                hotspots.to_excel(writer, sheet_name="Hotspots", index=False)
                profiles.to_excel(writer, sheet_name="Radial_axial_profiles", index=False)
                definitions.to_excel(writer, sheet_name="Definitions", index=False)
        make_plots(output, results, manifest["purpose"])
        with atomic_path(output / "summary.md") as path:
            path.write_text(summary_text(manifest, results, profiles, hotspots, errors))
        report = dict(
            purpose=manifest["purpose"],
            verified_completed=int((results.status == "completed").sum()),
            expected=len(results),
            complete=bool((results.status == "completed").all()),
            verification_errors=errors,
            manifest_sha256=digest(output / "manifest.json"),
        )
        report["output_sha256"] = {
            p.name: digest(p)
            for p in output.iterdir()
            if p.name
            in [
                "reevaluation_results.csv",
                "segment_responses.csv",
                "radial_axial_profiles.csv",
                "hotspot_summary.csv",
                "dominance_relations.csv",
                "comparison.xlsx",
                "pareto_reevaluation.png",
                "comparison_nominal.png",
                "summary.md",
            ]
        }
        write_json(output / "analysis_status.json", report)
        print(
            f"Analysis {manifest['purpose']}: {report['verified_completed']}/{len(results)} verified; complete={report['complete']}",
            flush=True,
        )
        return report


def main():
    """Parse saved campaign directory and regenerate all analysis outputs; return CLI exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = analyze(args.output)
    return 1 if report["verification_errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
