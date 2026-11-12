"""Launch or preflight a TRIGA NSGA-II campaign from the command line.

The launcher parses paths and campaign options, checks the runtime environment,
and delegates execution to MainOptimizer. Preflight checks do not run neutron
transport or optimizer generations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
import xml.etree.ElementTree as ET


RUNTIME_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = RUNTIME_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from main import DEFAULT_CONFIG_PATH, PROJECT_ROOT, MainOptimizer  # noqa: E402
from modules.campaign_identity import validate_campaign_id  # noqa: E402
from modules.checkpoint import CheckpointManager  # noqa: E402


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments and return the selected runtime options.

    ``argv`` is an optional argument list used by tests; ``None`` reads the
    process arguments. The returned namespace contains paths, campaign identity,
    dry-run mode, and logging level."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path config.ini; relative paths are resolved from this runtime folder.",
    )
    parser.add_argument(
        "--campaign-id",
        type=validate_campaign_id,
        default=None,
        help="Use a unique portable campaign ID without editing config.ini.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Override the results root without editing config.ini.",
    )
    parser.add_argument(
        "--openmc-executable",
        type=Path,
        default=None,
        help="Override the OpenMC executable; preflight validates it before campaign mutation.",
    )
    parser.add_argument(
        "--dry-run",
        "--preflight",
        dest="dry_run",
        action="store_true",
        help="Validate runtime composition without OpenMC transport or optimizer generations.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Set stderr logging detail.",
    )
    return parser.parse_args(argv)


def _resolve_runtime_path(path: Path) -> Path:
    """Resolve ``path`` against the runtime root and return an absolute path."""
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _active_membership_counts(state: dict[str, Any]) -> dict[str, int]:
    """Count memberships in checkpoint ``state`` and return parent/offspring/survivor totals."""
    generation = state["progress"]["generation"]
    generation_record = next(
        (record for record in state["generations"] if record["generation"] == generation), None
    )
    if generation_record is None:
        return {"parents": 0, "offspring": 0, "survivors": 0}
    return {
        "parents": len(generation_record["parent_ids"]),
        "offspring": len(generation_record["offspring_ids"]),
        "survivors": len(generation_record["survivor_ids"]),
    }


def _build_run_summary(optimizer: MainOptimizer) -> dict[str, Any]:
    """Read ``optimizer`` durable state and return its JSON-safe campaign summary."""
    state = optimizer.state
    status_counter = Counter(record["status"] for record in state["individuals"].values())
    membership_counts = _active_membership_counts(state)
    return {
        "campaign_id": optimizer.campaign_id,
        "checkpoint_path": str(optimizer.checkpoint_manager.checkpoint_path.resolve()),
        "generation": state["progress"]["generation"],
        "phase": state["progress"]["phase"],
        "membership_counts": membership_counts,
        "status_counts": {
            status: status_counter.get(status, 0)
            for status in sorted(CheckpointManager.INDIVIDUAL_STATUSES)
        },
        "configured_population_size": optimizer.config["ga"]["population_size"],
        "configured_generation_limit": optimizer.config["ga"]["num_generations"],
        "objectives": ["-keff", "ppf_3d"],
        "transport_enabled": True,
        "seed": optimizer.config["ga"]["random_seed"],
        "run_root": str(optimizer.run_root.resolve()),
        "report_root": str(optimizer.report_root.resolve()),
        "history_plot": str((optimizer.report_root / "optimization_history.png").resolve()),
        "openmc_executable": str(optimizer.openmc_executable),
        "transport_parameters": _build_transport_summary(optimizer),
        "evaluation_cache": optimizer.evaluation_cache.summary(),
    }


def _build_transport_summary(optimizer: MainOptimizer) -> dict[str, Any]:
    """Return the effective GA, transport, convergence, and stopping parameters."""
    trigger_enabled = optimizer.keff_trigger_enabled or optimizer.tally_trigger_enabled
    stopping_rule = (
        f"minimum {optimizer.inactive_batches + optimizer.active_batches} batches; trigger cap {optimizer.trigger_max_batches}"
        if trigger_enabled
        else f"exactly {optimizer.inactive_batches + optimizer.active_batches} batches"
    )
    return {
        "population_size": optimizer.config["ga"]["population_size"],
        "num_generations": optimizer.config["ga"]["num_generations"],
        "particles_per_batch": optimizer.particles,
        "inactive_batches": optimizer.inactive_batches,
        "active_batches": optimizer.active_batches,
        "openmc_seed_base": optimizer.openmc_seed,
        "threads": optimizer.threads,
        "geometry_debug": optimizer.geometry_debug,
        "keff_trigger_enabled": optimizer.keff_trigger_enabled,
        "tally_trigger_enabled": optimizer.tally_trigger_enabled,
        "trigger_batch_interval": optimizer.trigger_batch_interval,
        "trigger_max_batches": optimizer.trigger_max_batches,
        "convergence_gate_enabled": optimizer.convergence_enabled,
        "max_attempts": optimizer.config["retry"]["max_attempts"],
        "on_max_attempts": optimizer.config["retry"]["on_max_attempts"],
        "stopping_rule": stopping_rule,
    }


def _read_cross_sections_path(materials_path: Path) -> Path:
    """Read ``materials_path`` XML and return its referenced cross-section index path."""
    try:
        cross_sections_text = ET.parse(materials_path).getroot().findtext("cross_sections")
    except ET.ParseError as error:
        raise ValueError(f"materials.xml tidak valid: {materials_path}.") from error
    if not cross_sections_text or not cross_sections_text.strip():
        raise ValueError("materials.xml tidak memiliki elemen cross_sections yang terisi.")
    cross_sections_path = Path(cross_sections_text.strip())
    return (
        cross_sections_path
        if cross_sections_path.is_absolute()
        else (materials_path.parent / cross_sections_path).resolve()
    )


def _require_writable_results_root(results_root: Path) -> None:
    """Verify ``results_root`` is writable and return ``None`` without creating campaign data."""
    if not results_root.is_dir():
        raise FileNotFoundError(
            f"Results root tidak ditemukan atau bukan directory: {results_root}."
        )
    if not os.access(results_root, os.W_OK | os.X_OK):
        raise PermissionError(f"Results root tidak dapat ditulis: {results_root}.")


def _require_import_independence() -> dict[str, str]:
    """Verify loaded production modules originate here and return their module-to-path mapping."""
    source_root = SOURCE_ROOT.resolve()
    loaded_paths: dict[str, str] = {}
    for module_name, module in sorted(sys.modules.items()):
        if (
            module_name != "main"
            and module_name != "modules"
            and not module_name.startswith("modules.")
        ):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        resolved_path = Path(module_file).resolve()
        if not resolved_path.is_relative_to(source_root):
            raise RuntimeError(
                f"Import produksi keluar dari runtime: {module_name} -> {resolved_path}."
            )
        loaded_paths[module_name] = str(resolved_path)
    return loaded_paths


def _run_preflight(arguments: argparse.Namespace, config_path: Path) -> dict[str, Any]:
    """Validate ``config_path`` and CLI ``arguments`` without transport, returning a check summary."""
    with tempfile.TemporaryDirectory(prefix="ta_nsga2_preflight_") as temporary_directory:
        optimizer = MainOptimizer(
            config_path=config_path,
            campaign_id_override="preflight",
            run_root_override=temporary_directory,
            openmc_executable_override=arguments.openmc_executable,
        )
        selected_campaign_id = (
            optimizer.config["campaign"]["campaign_id"]
            if arguments.campaign_id is None
            else arguments.campaign_id
        )
        chromosome = optimizer.config["chromosome"]
        mutable_genome = optimizer.population_initializer.mutable_genome_contract.validate(
            chromosome["mutable_genes"], "Genome mutable preflight"
        )
        complete_genome = optimizer.genome_sorter.construct_complete_genome(mutable_genome)
        round_trip_genome = optimizer.genome_sorter.extract_included_genome(complete_genome)
        if round_trip_genome != mutable_genome:
            raise RuntimeError("Round trip mutable-complete-mutable genome tidak identik.")
        complete_mapping = dict(zip(optimizer.genome_sorter.slot_order, complete_genome))
        fixed_mapping_valid = all(
            complete_mapping[position] == gene
            for position, gene in optimizer.genome_sorter.excluded_genes.items()
        )
        if not fixed_mapping_valid:
            raise RuntimeError("Pemetaan fixed gene berubah saat konstruksi genome lengkap.")
        strategy_counts = optimizer.population_initializer._allocate_strategy_counts()
        configured_run_root = Path(
            optimizer.config["campaign"]["run_root"]
            if arguments.run_root is None
            else arguments.run_root
        )
        results_root = (
            configured_run_root
            if configured_run_root.is_absolute()
            else PROJECT_ROOT / configured_run_root
        )
        _require_writable_results_root(results_root)
        cross_sections_path = _read_cross_sections_path(optimizer.materials_path)
        if not cross_sections_path.is_file():
            raise FileNotFoundError(f"cross_sections.xml tidak ditemukan: {cross_sections_path}.")
        imported_modules = _require_import_independence()
        return {
            "mode": "dry-run",
            "status": "passed",
            "transport_enabled": False,
            "transport_call_count": 0,
            "optimizer_generation_count": 0,
            "checkpoint_retained": False,
            "campaign_id": selected_campaign_id,
            "config_path": str(config_path),
            "results_root": str(results_root.resolve()),
            "fuel_inventory_path": str(optimizer.fuel_inventory_path.resolve()),
            "materials_path": str(optimizer.materials_path.resolve()),
            "cross_sections_path": str(cross_sections_path.resolve()),
            "openmc_executable": str(optimizer.openmc_executable),
            "openmc_threads": optimizer.openmc_runner.threads,
            "geometry_debug": optimizer.openmc_runner.geometry_debug,
            "genome": {
                "mutable_genes": len(mutable_genome),
                "fixed_positions": len(optimizer.genome_sorter.excluded_genes),
                "physical_positions": len(complete_genome),
                "round_trip_valid": True,
                "fixed_mapping_valid": True,
            },
            "initialization_strategy_counts": strategy_counts,
            "operators": {
                "selection": optimizer.config["selection"]["method"],
                "crossover": optimizer.crossover_dispatcher.method,
                "mutation": optimizer.mutation_dispatcher.method,
                "replacement": optimizer.config["replacement"]["method"],
            },
            "objectives": ["-keff", "ppf_3d"],
            "ppf_3d_aggregation": "max_segment_score_over_mean_segment_score",
            "openmc_settings": _build_transport_summary(optimizer),
            "imported_runtime_module_count": len(imported_modules),
            "import_independence": True,
            "evaluation_cache": optimizer.evaluation_cache.summary(),
        }


def main(argv: list[str] | None = None) -> int:
    """Run preflight or optimization from optional ``argv`` and return the process exit code."""
    arguments = _parse_arguments(argv)
    logging.basicConfig(
        level=getattr(logging, arguments.log_level), format="%(asctime)s %(levelname)s %(message)s"
    )
    config_path = _resolve_runtime_path(arguments.config)
    if arguments.dry_run:
        print(
            json.dumps(
                _run_preflight(arguments, config_path),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 0
    try:
        optimizer = MainOptimizer(
            config_path=config_path,
            campaign_id_override=arguments.campaign_id,
            run_root_override=arguments.run_root,
            openmc_executable_override=arguments.openmc_executable,
        )
        transport_summary = _build_transport_summary(optimizer)
        logging.warning(
            "Parameter efektif sebelum transport: campaign_id=%s, campaign_root=%s, openmc_work_root=%s, report_root=%s, openmc_executable=%s, settings=%s.",
            optimizer.campaign_id,
            optimizer.campaign_root.resolve(),
            optimizer.openmc_work_root.resolve(),
            optimizer.report_root.resolve(),
            optimizer.openmc_executable,
            json.dumps(transport_summary, ensure_ascii=False, allow_nan=False, sort_keys=True),
        )
        optimizer.run_optimization()
    except KeyboardInterrupt:
        logging.warning(
            "Optimisasi dihentikan oleh pengguna. Checkpoint durable terakhir dipertahankan; jalankan kembali command yang sama untuk resume."
        )
        return 130
    print(
        json.dumps(
            _build_run_summary(optimizer), ensure_ascii=False, allow_nan=False, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
