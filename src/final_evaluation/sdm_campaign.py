"""Prepare/preflight/run/resume/analyze a public final-front SDM campaign.

Only run/resume without --dry-run can launch transport. Preparation and analysis
never call transport. Every attempt is retained; failed attempts require explicit retry."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import warnings

from lxml import etree as ET
import openmc

from .geometry_checks import check_geometry
from .model import FinalModelBuilder
from .sdm_contract import (
    ROOT,
    DEFAULT_CONFIG,
    atomic_json,
    cohort,
    digest,
    environment,
    load_config,
    lock,
    planned_jobs,
    read_json,
    running_openmc,
    sha256,
    timestamp,
    unchanged_context,
    write_csv,
)
from .sdm_results import inspect_attempt, report


def verify_input(path, job, config):
    """Return verified transport settings from the XML file at ``path``.

    Compare the actual input with the planned candidate/case ``job`` and frozen
    ``config``. Reject mismatched batch budgets, source/entropy domains, or an
    unexpected nominal PPF tally."""
    xml = ET.parse(str(path)).getroot()
    element = xml.find("settings")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", openmc.IDWarning)
        settings = openmc.Settings.from_xml_element(element)
    t = config["transport"]
    if (
        settings.particles != t["particles"]
        or settings.batches != t["batches"]
        or settings.inactive != t["inactive"]
        or settings.seed != job["seed"]
        or settings.generations_per_batch != 1
        or settings.trigger_active is not False
        or settings.keff_trigger
        or xml.findall(".//tally")
        or element.find(".//overwrite_latest") is not None
        or any(
            settings.sourcepoint[key] != value
            for key, value in dict(
                batches=config["source_snapshots"], separate=True, write=True
            ).items()
        )
        or settings.statepoint != dict(batches=[200])
    ):
        raise ValueError("Actual input differs from final fixed transport/sourcepoint contract.")
    materials = xml.find("materials")
    if (
        Path(materials.findtext("cross_sections")).resolve()
        != Path(
            ET.parse(str(ROOT / config["materials"])).getroot().findtext("cross_sections")
        ).resolve()
        or any(float(m.get("temperature", "nan")) != 300 for m in materials.findall("material"))
        or any(c.get("temperature") is not None for c in xml.findall(".//cell"))
    ):
        raise ValueError("Material/library/temperature contract changed.")
    return dict(
        particles=settings.particles,
        batches=settings.batches,
        inactive=settings.inactive,
        active=settings.batches - settings.inactive,
        generations_per_batch=settings.generations_per_batch,
        seed=settings.seed,
        trigger=False,
        source_snapshots=list(settings.sourcepoint["batches"]),
        entropy_lower_cm=list(settings.entropy_mesh.lower_left),
        entropy_upper_cm=list(settings.entropy_mesh.upper_right),
    )


def prepare(output, config_path):
    """Build and gate public final-front inputs at new output using config_path; return manifest."""
    config = load_config(config_path)
    ids, genomes, checkpoint = cohort(config)
    context = environment(config)
    # Confidential MCNP reference decks are deliberately absent from this example.
    source_check = dict(scope="Public CSG checks only", independent_mcnp_benchmark=False)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    (output / "internal").mkdir(mode=0o700)
    atomic_json(output / "preparation_status.json", dict(state="preparing", started_at=timestamp()))
    atomic_json(output / "internal/complete_genomes.json", genomes)
    atomic_json(output / "internal/reference_sources.json", source_check)
    atomic_json(output / "configuration.json", config)
    jobs = planned_jobs(ids, genomes, config)
    for index, job in enumerate(jobs, 1):
        directory = output / job["directory"]
        (directory / "input").mkdir(parents=True, mode=0o700)
        (directory / "attempts").mkdir(mode=0o700)
        with warnings.catch_warnings():
            # OpenMC object IDs repeat between independent model constructions.
            warnings.simplefilter("ignore", openmc.IDWarning)
            openmc.reset_auto_ids()
            builder = FinalModelBuilder(
                genomes[job["candidate_id"]], ROOT / config["materials"], job["rod_states"]
            )
            model, data = builder.build_final(
                shutdown=True, particles=100000, total_batches=200, inactive=45, seed=job["seed"]
            )
            model.settings.sourcepoint = dict(
                batches=config["source_snapshots"], separate=True, write=True
            )
            geometry = check_geometry(builder, model, data)
            geometry["validation_scope"] = source_check
            if not geometry["passed"]:
                atomic_json(directory / "geometry.json", geometry)
                raise RuntimeError("Targeted geometry checks failed: preparation stopped.")
            low = min(r["z_low"] for r in data["records"])
            high = max(r["z_high"] for r in data["records"])
            mesh = model.settings.entropy_mesh
            if not (mesh.lower_left[2] < low and mesh.upper_right[2] > high):
                raise RuntimeError("Entropy domain does not cover moving fissile domain.")
            path = directory / "input/model.xml"
            model.export_to_model_xml(path)
        actual = verify_input(path, job, config)
        job.update(
            input_sha256=sha256(path),
            actual_input_settings=actual,
            geometry_sha256=digest(geometry),
        )
        job["job_sha256"] = digest(job)
        atomic_json(directory / "geometry.json", geometry)
        atomic_json(directory / "metadata.json", job)
        atomic_json(
            directory / "status.json", dict(state="prepared", attempt=None, updated_at=timestamp())
        )
        if index % 5 == 0:
            print(f"Prepared {index}/{len(jobs)} inputs; geometry checks passed.", flush=True)
    manifest = dict(
        schema="final_sdm_campaign/v1",
        created_at=timestamp(),
        config=config,
        checkpoint=checkpoint,
        cohort=ids,
        context=context,
        context_sha256=digest(context),
        jobs=jobs,
        validation_scope=source_check,
        dataset="public-fresh-fuel/v1",
        transport_launched=False,
    )
    manifest["manifest_sha256"] = digest(manifest)
    atomic_json(output / "manifest.json", manifest)
    write_csv(
        output / "manifest.csv",
        [
            dict(
                candidate_id=j["candidate_id"],
                case=j["case"],
                seed=j["seed"],
                genome_sha256=j["genome_sha256"],
                input_sha256=j["input_sha256"],
                job_sha256=j["job_sha256"],
                directory=j["directory"],
                **j["transport"],
            )
            for j in jobs
        ],
    )
    write_csv(
        output / "kohort_tetap.csv",
        [dict(candidate_id=key, genome_sha256=digest(genomes[key])) for key in ids],
    )
    atomic_json(
        output / "preparation_status.json",
        dict(state="prepared", finished_at=timestamp(), jobs=len(jobs), transport_launched=False),
    )
    return manifest


def load_manifest(output):
    """Read and verify self-hashed manifest in output; return immutable campaign definition."""
    manifest = read_json(output / "manifest.json")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if digest(unsigned) != manifest["manifest_sha256"]:
        raise ValueError("Manifest digest mismatch.")
    load_config(output / "configuration.json")
    if manifest["config"] != read_json(output / "configuration.json"):
        raise ValueError("Manifest configuration mismatch.")
    return manifest


def preflight(output, verify_environment=True):
    """Validate all source/input/job identities in output without transport; return manifest."""
    manifest = load_manifest(output)
    config = manifest["config"]
    ids, genomes, checkpoint = cohort(config)
    if ids != manifest["cohort"] or checkpoint != manifest["checkpoint"]:
        raise ValueError("Checkpoint/cohort provenance changed; refusing launch or reuse.")
    expected_jobs = planned_jobs(ids, genomes, config)
    if len(manifest["jobs"]) != len(expected_jobs):
        raise ValueError("Manifest job count differs from the selected public cohort.")
    for job, expected in zip(manifest["jobs"], expected_jobs, strict=True):
        if any(job[key] != value for key, value in expected.items()):
            raise ValueError("Job candidate, genome, seed, order or configuration changed.")
        unsigned = {key: value for key, value in job.items() if key != "job_sha256"}
        directory = output / job["directory"]
        if (
            digest(unsigned) != job["job_sha256"]
            or read_json(directory / "metadata.json") != job
            or sha256(directory / "input/model.xml") != job["input_sha256"]
            or digest(read_json(directory / "geometry.json")) != job["geometry_sha256"]
        ):
            raise ValueError("Job input/metadata/geometry evidence changed.")
        if verify_input(directory / "input/model.xml", job, config) != job["actual_input_settings"]:
            raise ValueError("Actual XML configuration changed.")
    if verify_environment and environment(config) != manifest["context"]:
        raise ValueError("Executable/API/data/shared library/implementation context changed.")
    return manifest


def attempt_binding(job, manifest):
    """Return provenance binding for job under manifest, excluding private genomes."""
    return dict(
        job_sha256=job["job_sha256"],
        manifest_sha256=manifest["manifest_sha256"],
        context_sha256=manifest["context_sha256"],
        candidate_id=job["candidate_id"],
        genome_sha256=job["genome_sha256"],
        seed=job["seed"],
        case=job["case"],
        input_sha256=job["input_sha256"],
    )


def verify_completed(output, job, manifest, state):
    """Verify completed result/status against job and manifest; return inspection evidence."""
    directory = output / job["directory"] / "attempts" / state["attempt"]
    evidence, _, _ = inspect_attempt(
        directory, job, manifest["config"], manifest["manifest_sha256"], manifest["context_sha256"]
    )
    if not evidence["final_valid"]:
        raise ValueError("Completed result is incomplete or does not match final configuration.")
    if (
        state.get("statepoint_sha256")
        and state["statepoint_sha256"] != evidence["statepoint_sha256"]
    ):
        raise ValueError("Completed statepoint changed after verification.")
    return evidence


def execution_plan(output, manifest, resume=False, retry_failed=False):
    """Classify verified results/pending attempts without mutation; return actions for all jobs."""
    actions = []
    for job in manifest["jobs"]:
        directory = output / job["directory"]
        state = read_json(directory / "status.json")
        attempts = sorted((directory / "attempts").iterdir())
        if state["state"] == "completed":
            if not resume:
                raise RuntimeError(
                    "Existing results require resume; run never overwrites attempts."
                )
            verify_completed(output, job, manifest, state)
            action = "skip_verified"
        elif state["state"] == "prepared" and not attempts:
            action = "launch"
        else:
            if not resume:
                raise RuntimeError("Existing attempt found: use resume and inspect evidence.")
            # A driver may have stopped after recording successful execution but before status commit.
            try:
                if state.get("attempt"):
                    verify_completed(output, job, manifest, state)
                    action = "recover_completed"
                else:
                    action = "retry" if retry_failed else "blocked_explicit_retry_required"
            except (ValueError, OSError, KeyError):
                action = "retry" if retry_failed else "blocked_explicit_retry_required"
        actions.append(dict(job=job, action=action))
    return actions


def execute_job(output, job, manifest):
    """Execute one numbered attempt for ``job`` below ``output``; return None.

    ``manifest`` binds the candidate, seed, configuration, and input hashes. Retain
    attempt artifacts and save execution status atomically for later verification."""
    directory = output / job["directory"]
    with lock(directory / ".job.lock"):
        if sha256(directory / "input/model.xml") != job["input_sha256"]:
            raise ValueError("Prepared input changed before launch; no transport started.")
        attempts = directory / "attempts"
        number = max([int(p.name) for p in attempts.iterdir()] + [0]) + 1
        name = f"{number:03d}"
        attempt = attempts / name
        attempt.mkdir(mode=0o700)
        shutil.copyfile(directory / "input/model.xml", attempt / "model.xml")
        if sha256(attempt / "model.xml") != job["input_sha256"]:
            raise ValueError("Actual attempt input changed before launch; no transport started.")
        atomic_json(attempt / "binding.json", attempt_binding(job, manifest))
        command = [manifest["config"]["executable"], "-s", "32"]
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="32", OMP_DYNAMIC="FALSE", OPENBLAS_NUM_THREADS="1")
        # Explicit XML cross_sections remains authoritative; inherited overrides are recorded/removed.
        env.pop("OPENMC_CROSS_SECTIONS", None)
        started = timestamp()
        atomic_json(
            attempt / "launch.json",
            dict(
                command=command,
                started_at=started,
                threads=32,
                mpi_processes=1,
                driver_pid=os.getpid(),
                OMP_NUM_THREADS="32",
                OMP_DYNAMIC="FALSE",
                OPENBLAS_NUM_THREADS="1",
                LD_LIBRARY_PATH=env.get("LD_LIBRARY_PATH"),
                LD_PRELOAD=env.get("LD_PRELOAD"),
            ),
        )
        atomic_json(
            directory / "status.json", dict(state="running", attempt=name, started_at=started)
        )
        start = time.monotonic()
        try:
            with (attempt / "openmc.log").open("x") as log:
                process = subprocess.run(
                    command, cwd=attempt, env=env, stdout=log, stderr=subprocess.STDOUT
                )
            atomic_json(
                attempt / "execution.json",
                dict(
                    command=command,
                    returncode=process.returncode,
                    wall_seconds=time.monotonic() - start,
                    started_at=started,
                    finished_at=timestamp(),
                ),
            )
            evidence, _, _ = inspect_attempt(
                attempt,
                job,
                manifest["config"],
                manifest["manifest_sha256"],
                manifest["context_sha256"],
            )
            atomic_json(attempt / "result_internal.json", evidence)
            state = dict(
                state="completed" if evidence["final_valid"] else "failed",
                attempt=name,
                updated_at=timestamp(),
                statepoint_sha256=evidence.get("statepoint_sha256"),
                diagnostics_complete=evidence["diagnostics_complete"],
            )
            atomic_json(directory / "status.json", state)
            if not evidence["final_valid"]:
                raise RuntimeError(
                    "Attempt failed or configuration/output invalid; inspect private evidence."
                )
        except BaseException:
            if read_json(directory / "status.json")["state"] == "running":
                atomic_json(
                    directory / "status.json",
                    dict(state="interrupted_or_failed", attempt=name, updated_at=timestamp()),
                )
            raise


def run_campaign(output, resume=False, retry_failed=False, dry_run=False):
    """Run the SDM campaign below ``output`` and return no value.

    ``resume`` recovers existing work, ``retry_failed`` permits failed attempts to be
    retried, and ``dry_run`` reports the execution plan without launching transport.
    Previously successful attempts are reused only after verification."""
    with lock(ROOT / "results/sdm_final/.transport.lock"), lock(output / ".campaign.lock"):
        manifest = preflight(output)
        actions = execution_plan(output, manifest, resume=resume, retry_failed=retry_failed)
        counts = {
            key: sum(a["action"] == key for a in actions)
            for key in (
                "launch",
                "skip_verified",
                "recover_completed",
                "retry",
                "blocked_explicit_retry_required",
            )
        }
        if dry_run:
            print(
                json.dumps(dict(dry_run=True, transport_launched=False, actions=counts)), flush=True
            )
            return counts
        if counts["blocked_explicit_retry_required"]:
            raise RuntimeError(
                "Failed/incomplete attempts require explicit --retry-failed; no transport started."
            )
        for item in actions:
            job, action = item["job"], item["action"]
            if action == "skip_verified":
                continue
            if action == "recover_completed":
                directory = output / job["directory"]
                state = read_json(directory / "status.json")
                evidence = verify_completed(output, job, manifest, state)
                state.update(
                    state="completed",
                    statepoint_sha256=evidence["statepoint_sha256"],
                    recovered_at=timestamp(),
                )
                atomic_json(directory / "status.json", state)
                continue
            if running_openmc():
                raise RuntimeError(
                    "Existing OpenMC process detected; no process stopped. Resume when idle."
                )
            unchanged_context(manifest["context"])
            print(f"START {job['candidate_id']} {job['case']} seed={job['seed']}", flush=True)
            execute_job(output, job, manifest)
            print(f"FINISH {job['candidate_id']} {job['case']}", flush=True)
        return counts


def main():
    """Parse CLI operation and new/prepared output path; return None after requested operation."""
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation", choices=("prepare", "preflight", "run", "resume", "analyze", "report")
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.retry_failed and args.operation != "resume":
        parser.error("--retry-failed is only available for resume")
    if args.dry_run and args.operation not in ("run", "resume"):
        parser.error("--dry-run is only available for run/resume")
    output = args.output.resolve()
    if args.operation == "prepare":
        manifest = prepare(output, args.config)
        print(f"Prepared {len(manifest['jobs'])} jobs; no transport launched.")
    elif args.operation == "preflight":
        with lock(output / ".campaign.lock"):
            manifest = preflight(output)
            atomic_json(
                output / "preflight.json",
                dict(
                    passed=True,
                    checked_at=timestamp(),
                    manifest_sha256=manifest["manifest_sha256"],
                    jobs=len(manifest["jobs"]),
                    existing_openmc_pids=running_openmc(),
                    transport_launched=False,
                ),
            )
        print(
            f"Preflight passed: {len(manifest['jobs'])} inputs, versions, provenance; no transport launched."
        )
    elif args.operation in ("run", "resume"):
        run_campaign(
            output,
            resume=args.operation == "resume",
            retry_failed=args.retry_failed,
            dry_run=args.dry_run,
        )
    else:
        with lock(output / ".campaign.lock"):
            manifest = load_manifest(output)
            print(report(output, manifest))


if __name__ == "__main__":
    main()
