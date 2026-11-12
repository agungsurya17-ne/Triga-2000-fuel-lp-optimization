#!/usr/bin/env python3
"""Sequential, checkpointed IV.4.5.2 reevaluation; production requires --run/--resume."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import logging
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

from common import (
    HERE,
    ROOT,
    NOMINAL,
    DEFAULT_OUTPUT,
    BASELINE_ID,
    digest,
    object_digest,
    read_json,
    write_json,
    campaign_lock,
)
from inputs import load_cases, CHECKPOINT

SMOKE = dict(particles=1000, inactive=2, batches=6, generations_per_batch=1)
DEFAULT_EXECUTABLE = shutil.which("openmc") or "openmc"
LOGGER = logging.getLogger("reevaluation")


def now():
    """Return an ISO-8601 UTC timestamp for durable campaign metadata."""
    return datetime.now(timezone.utc).isoformat()


def provenance(executable, checkpoint_path=CHECKPOINT):
    """Hash executable, public checkpoint/inputs and used nuclear data; return evidence."""
    import openmc

    original = [
        checkpoint_path,
        ROOT / "config/materials.xml",
        ROOT / "config/fuel_inventory.json",
        ROOT / "tests/fixtures/geometry/reference_genome.json",
        ROOT / "tests/fixtures/geometry/cell_mapping_verified.csv",
        ROOT / "src/modules/public_inputs.py",
        ROOT / "src/modules/openmc_model_builder.py",
        ROOT / "src/modules/genome_sorter.py",
    ]
    code = sorted(HERE.glob("*.py"))
    material_root = ET.parse(ROOT / "config/materials.xml").getroot()
    index = Path(material_root.findtext("cross_sections")).expanduser().resolve()
    nuclear = ET.parse(index).getroot()
    directory = Path(nuclear.findtext("directory", str(index.parent)))
    if not directory.is_absolute():
        directory = index.parent / directory
    required = {n.get("name") for n in material_root.iter("nuclide")} | {
        n.get("name") for n in material_root.iter("sab")
    }
    nuclear_files = set()
    available = set()
    for library in nuclear.findall("library"):
        names = set(library.get("materials", "").split())
        if names & required:
            path = Path(library.get("path"))
            nuclear_files.add(path if path.is_absolute() else directory / path)
            available.update(names)
    if required - available:
        raise ValueError(f"Missing nuclear libraries: {sorted(required - available)}")
    sources = {str(path.resolve()): digest(path) for path in original + code + [index, executable]}
    LOGGER.info("Hashing %d nuclear data files for reproducibility", len(nuclear_files))
    nuclear_hashes = {
        str(path.resolve()): dict(sha256=digest(path), bytes=path.stat().st_size)
        for path in sorted(nuclear_files)
    }
    version = subprocess.check_output(
        [str(executable), "--version"], text=True, stderr=subprocess.STDOUT
    ).strip()
    if openmc.__version__ not in version:
        raise ValueError(f"OpenMC API/executable version mismatch: {openmc.__version__}, {version}")
    return dict(
        sources=sources,
        nuclear_files=nuclear_hashes,
        openmc_api=openmc.__version__,
        executable_version=version,
        python=sys.version,
        packages={
            key: importlib.metadata.version(key)
            for key in ["numpy", "pandas", "openpyxl", "matplotlib", "h5py"]
        },
    )


def case_configuration(case, seed, parameters, context_hash):
    """Return immutable identity for case loading, seed, transport settings and source context."""
    return dict(case=case, seed=seed, parameters=parameters, context_sha256=context_hash)


def prepare_case(case, entry, output, parameters):
    """Build and export a fresh input under entry's new attempt; return attempt directory/records."""
    from model import build_model, validate_model, common_model_signature

    directory = output / entry["attempt_path"]
    directory.mkdir(parents=True, exist_ok=False)
    model, records, audit = build_model(case, entry["seed"], parameters)
    validate_model(model, entry["seed"], parameters)
    model.export_to_model_xml(directory / "model.xml")
    entry["common_model_sha256"] = common_model_signature(directory / "model.xml", records)
    write_json(directory / "segments.json", records)
    write_json(directory / "loading.json", case["genome"])
    entry["geometry_audit"] = audit
    entry["model_sha256"] = digest(directory / "model.xml")
    entry["records_sha256"] = digest(directory / "segments.json")
    return directory, records


def verify_completed(output, entry):
    """Verify hashes, exact config, numeric results and final statepoint; return result or raise."""
    from model import extract_statepoint, validate_records
    from analyze_reevaluation import verified_result

    if entry["status"] != "completed":
        raise ValueError("Entry is not completed.")
    verified_result(output, entry)
    for key, checksum in entry["artifacts"].items():
        if digest(output / key) != checksum:
            raise ValueError(f"Completed artifact changed: {key}")
    directory = output / entry["attempt_path"]
    saved = read_json(directory / "result.json")
    if (
        saved["configuration_sha256"] != entry["configuration_sha256"]
        or saved["seed"] != entry["seed"]
        or saved["candidate_id"] != entry["candidate_id"]
    ):
        raise ValueError("Completed result configuration mismatch.")
    records = read_json(directory / "segments.json")
    validate_records(records, entry["configuration"]["case"]["genome"])
    actual = extract_statepoint(
        output / entry["statepoint_path"],
        records,
        entry["seed"],
        entry["configuration"]["parameters"],
    )
    for key in [
        "keff",
        "keff_std_dev",
        "all_fuel",
        "regular",
        "segments",
        "zero_mean_bins",
        "current_batch",
        "active_realizations",
        "openmc_version",
    ]:
        if actual[key] != saved[key]:
            raise ValueError(f"Saved numeric result differs from statepoint: {key}")
    return saved


def run_transport(executable, directory, threads):
    """Run one OpenMC child in ``directory`` and return elapsed wall time in seconds.

    ``executable`` identifies the binary and ``threads`` sets its OpenMP thread
    count. Stream output to the attempt log; this function does not launch MPI."""
    started = time.monotonic()
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(threads)
    with (directory / "openmc.log").open("w") as stream:
        process = subprocess.Popen(
            [str(executable), "-s", str(threads)],
            cwd=directory,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        try:
            returncode = process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if returncode:
        raise RuntimeError(f'OpenMC exited {returncode}; inspect {directory / "openmc.log"}')
    return time.monotonic() - started


def execute_campaign(output, cases, manifest, executable):
    """Process ordered ``cases`` below ``output`` and return the campaign exit code.

    ``manifest`` stores seeds, attempt status, and artifact hashes. Use ``executable``
    for transport, checkpoint each outcome, and retain failed attempts for audit."""
    from model import extract_statepoint

    manifest["finished"] = False
    write_json(output / "manifest.json", manifest)
    for case in cases:
        entry = manifest["cases"][case["candidate_id"]]
        if entry["status"] == "completed":
            try:
                verify_completed(output, entry)
                LOGGER.info("Verified completed; skipping %s", case["candidate_id"])
                continue
            except Exception as exc:
                entry["verification_error"] = str(exc)
                LOGGER.warning(
                    "Invalid completed result, restarting %s: %s", case["candidate_id"], exc
                )
        if entry.get("attempt_path"):
            entry.setdefault("previous_attempts", []).append(
                dict(
                    path=entry["attempt_path"],
                    status=entry["status"],
                    error=entry.get("error"),
                    verification_error=entry.get("verification_error"),
                )
            )
        entry["attempt_number"] = entry.get("attempt_number", 0) + 1
        entry["attempt_path"] = f"runs/{case['candidate_id']}/attempt_{entry['attempt_number']:03d}"
        entry.update(status="running", started_at=now(), error=None, completed_at=None)
        write_json(output / "manifest.json", manifest)
        LOGGER.info(
            "Starting %s, seed %s, attempt %s",
            case["candidate_id"],
            entry["seed"],
            entry["attempt_number"],
        )
        start = time.monotonic()
        try:
            directory, records = prepare_case(case, entry, output, manifest["parameters"])
            if manifest.get("common_model_sha256") not in (None, entry["common_model_sha256"]):
                raise ValueError("Common nominal model differs beyond loading and seed.")
            manifest["common_model_sha256"] = entry["common_model_sha256"]
            write_json(output / "manifest.json", manifest)
            # Recheck serialized input against its hash immediately before the child starts.
            if digest(directory / "model.xml") != entry["model_sha256"]:
                raise ValueError("Input changed before transport.")
            wall = run_transport(executable, directory, manifest["threads"])
            statepoint = directory / f"statepoint.{manifest['parameters']['batches']}.h5"
            result = extract_statepoint(statepoint, records, entry["seed"], manifest["parameters"])
            entry["statepoint_path"] = str(statepoint.relative_to(output))
            result.update(
                candidate_id=case["candidate_id"],
                seed=entry["seed"],
                configuration_sha256=entry["configuration_sha256"],
                statepoint_path=str(statepoint),
                transport_seconds=wall,
                elapsed_seconds=time.monotonic() - start,
                historical={
                    key: value for key, value in case.items() if key.startswith("historical_")
                },
            )
            write_json(directory / "result.json", result)
            entry["artifacts"] = {
                str(path.relative_to(output)): digest(path)
                for path in [
                    directory / "model.xml",
                    directory / "segments.json",
                    directory / "loading.json",
                    directory / "result.json",
                    statepoint,
                ]
            }
            entry.update(
                status="completed",
                completed_at=now(),
                elapsed_seconds=result["elapsed_seconds"],
                transport_seconds=wall,
            )
            write_json(output / "manifest.json", manifest)
            LOGGER.info(
                "Completed %s: k=%.8f, all-fuel PPF=%.6f",
                case["candidate_id"],
                result["keff"],
                result["all_fuel"]["ppf_3d"],
            )
        except (KeyboardInterrupt, SystemExit):
            entry.update(
                status="failed",
                error="Interrupted; next resume restarts this candidate from the beginning.",
                elapsed_seconds=time.monotonic() - start,
            )
            write_json(output / "manifest.json", manifest)
            LOGGER.warning("Interrupted %s; checkpoint saved", case["candidate_id"])
            raise
        except Exception:
            entry.update(
                status="failed",
                error=traceback.format_exc(),
                elapsed_seconds=time.monotonic() - start,
            )
            write_json(output / "manifest.json", manifest)
            LOGGER.exception("Failed %s; continuing remaining cohort", case["candidate_id"])
    manifest["finished"] = all(
        entry["status"] == "completed" for entry in manifest["cases"].values()
    )
    manifest["updated_at"] = now()
    write_json(output / "manifest.json", manifest)
    from analyze_reevaluation import analyze

    report = analyze(output, locked=True)
    if report["verification_errors"]:
        for key, error in report["verification_errors"].items():
            manifest["cases"][key].update(status="failed", error=error)
        manifest["finished"] = False
        write_json(output / "manifest.json", manifest)
        analyze(output, locked=True)
    return 0 if manifest["finished"] else 1


def parse_arguments():
    """Parse independent production/dry-run/smoke modes; return namespace with validated paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run", action="store_true", help="Validate/export every case; never execute transport"
    )
    mode.add_argument(
        "--run", action="store_true", help="Start nominal production, or start after dry-run"
    )
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Verify completed cases; restart unfinished cases from batch 1",
    )
    mode.add_argument(
        "--smoke-test",
        action="store_true",
        help="Only 2 cases at 1000 particles, 2 inactive/6 total batches",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed-base", type=int, default=2026102601)
    parser.add_argument("--openmc-executable", type=Path, default=Path(DEFAULT_EXECUTABLE))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT,
        help="Finished public optimization checkpoint to reevaluate.",
    )
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.threads < 1 or args.seed_base < 1 or args.seed_base + 100 >= 2**63:
        parser.error("threads and seed-base must be positive; seed-base must fit signed 64-bit.")
    args.output = (args.output or (HERE / "smoke" if args.smoke_test else DEFAULT_OUTPUT)).resolve()
    args.openmc_executable = (
        Path(shutil.which(str(args.openmc_executable)) or args.openmc_executable)
        .expanduser()
        .resolve()
    )
    # All writes must remain in the isolated reevaluation tree.
    if not args.output.is_relative_to(HERE) or args.output == HERE:
        parser.error("--output must be a subdirectory of reevaluation_iv_4_5_2")
    return args


def handle_termination(signum, frame):
    """Translate SIGTERM into interruption so child transport is stopped and checkpointed; return never."""
    raise KeyboardInterrupt(f"Received signal {signum}")


def main():
    """Prepare verified cohort and execute selected mode with a process lock; return CLI exit code."""
    args = parse_arguments()
    signal.signal(signal.SIGTERM, handle_termination)
    with campaign_lock(args.output):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler(args.output / "pipeline.log")],
        )
        cases, audit = load_cases(args.checkpoint)
        LOGGER.info("Verified public front: %d candidates + reference", audit["actual_front_count"])
        existing_path = args.output / "manifest.json"
        is_smoke = args.smoke_test or (
            args.resume
            and existing_path.exists()
            and read_json(existing_path).get("purpose") == "SMOKE_ONLY"
        )
        parameters = dict(SMOKE if is_smoke else NOMINAL)
        if is_smoke:
            cases = [cases[0], cases[-1]]
        context = provenance(args.openmc_executable, args.checkpoint)
        identity = dict(
            schema_version=1,
            purpose="SMOKE_ONLY" if is_smoke else "NOMINAL_IV_4_5_2",
            parameters=parameters,
            threads=args.threads,
            seed_base=args.seed_base,
            provenance=context,
            cases=cases,
            domain="111 fuel elements / 1665 fuel-meat segments; separate 108/1620 metrics",
        )
        identity_hash = object_digest(identity)
        path = args.output / "manifest.json"
        if path.exists():
            manifest = read_json(path)
            if manifest["identity_sha256"] != identity_hash or manifest["identity"] != identity:
                raise ValueError(
                    "Configuration, source, code, environment or cohort changed; use a new output directory."
                )
            if args.run and any(e["status"] != "pending" for e in manifest["cases"].values()):
                raise ValueError("Existing attempted campaign: use --resume.")
        else:
            if args.resume:
                raise FileNotFoundError("Cannot resume without manifest.json")
            manifest = dict(
                identity=identity,
                identity_sha256=identity_hash,
                parameters=parameters,
                purpose=identity["purpose"],
                threads=args.threads,
                cohort_audit=audit,
                created_at=now(),
                finished=False,
                cases={},
            )
            for offset, case in enumerate(cases):
                seed = args.seed_base + offset
                config = case_configuration(case, seed, parameters, object_digest(context))
                manifest["cases"][case["candidate_id"]] = dict(
                    candidate_id=case["candidate_id"],
                    status="pending",
                    seed=seed,
                    configuration=config,
                    configuration_sha256=object_digest(config),
                )
            write_json(path, manifest)
        if list(manifest["cases"]) != [case["candidate_id"] for case in cases]:
            raise ValueError("Manifest case membership/order differs from the frozen cohort.")
        # Protect even seeds/config hashes inside the mutable checkpoint against accidental edits.
        for offset, case in enumerate(cases):
            entry = manifest["cases"][case["candidate_id"]]
            config = case_configuration(
                case, args.seed_base + offset, parameters, object_digest(context)
            )
            if (
                entry["configuration"] != config
                or entry["configuration_sha256"] != object_digest(config)
                or entry["seed"] != config["seed"]
            ):
                raise ValueError("Manifest per-case configuration/seed changed.")
        if args.dry_run:
            from model import build_model, common_model_signature

            audits = []
            for case in cases:
                entry = manifest["cases"][case["candidate_id"]]
                model, records, geometry_audit = build_model(case, entry["seed"], parameters)
                directory = args.output / "prepared" / case["candidate_id"]
                directory.mkdir(parents=True, exist_ok=True)
                model.export_to_model_xml(directory / "model.xml")
                signature = common_model_signature(directory / "model.xml", records)
                if manifest.get("common_model_sha256") not in (None, signature):
                    raise ValueError("Common nominal model differs beyond loading and seed.")
                manifest["common_model_sha256"] = signature
                write_json(directory / "segments.json", records)
                write_json(directory / "loading.json", case["genome"])
                audits.append(
                    dict(candidate_id=case["candidate_id"], seed=entry["seed"], **geometry_audit)
                )
                LOGGER.info(
                    "Prepared and validated %s (%d segments)", case["candidate_id"], len(records)
                )
            write_json(path, manifest)
            write_json(
                args.output / "dry_run_validation.json",
                dict(
                    passed=True,
                    transport_launched=False,
                    cohort_audit=audit,
                    cases=audits,
                    parameters=parameters,
                ),
            )
            from analyze_reevaluation import analyze

            analyze(args.output, locked=True)
            LOGGER.info("Dry-run passed: %d inputs prepared; no transport launched", len(cases))
            return 0
        return execute_campaign(args.output, cases, manifest, args.openmc_executable)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        logging.exception("Pipeline stopped")
        sys.exit(1)
