"""Evaluate one mutable genome through the checkpointed OpenMC pipeline.

SingleCandidatePipeline constructs the complete core loading, builds and runs
the model, reads its statepoint, and applies the convergence gate. It persists
completed, not-converged, or failed outcomes and manages retry accounting.
Initial parents and offspring use the same evaluation path."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

from modules.checkpoint import COMPLETED_STATUS, NOT_CONVERGED_STATUS
from modules.mutable_genome import MutableGenomeContract


class SingleCandidateEvaluationError(RuntimeError):
    """Report a terminal candidate failure while preserving its original cause."""


@dataclass(frozen=True)
class SingleCandidateOutcome:
    """Store terminal artifacts and metadata from one candidate evaluation."""

    results: Mapping[str, Any]
    status: str
    complete_genome: tuple[Any, ...]
    tally_records: tuple[Mapping[str, Any], ...]
    statepoint_path: Path
    run_id: str
    work_dir: Path
    openmc_seed: int


class SingleCandidatePipeline:
    """Connect one mutable candidate to injected build, run, evaluate, and checkpoint components."""

    ON_MAX_ATTEMPT_POLICIES = frozenset({"halt_campaign", "continue_campaign"})

    def __init__(
        self,
        genome_sorter: Any,
        builder_factory: Callable[..., Any],
        runner: Any,
        result_evaluator: Any,
        checkpoint_manager: Any,
        material_file: str | Path,
        work_root: str | Path,
        tally_name: str,
        tally_score: str,
        particles: int,
        inactive_batches: int,
        active_batches: int,
        keff_trigger_enabled: bool,
        keff_trigger_stdev_pcm: float,
        trigger_batch_interval: int,
        trigger_max_batches: int,
        tally_trigger_enabled: bool,
        tally_trigger_type: str,
        tally_trigger_threshold: float,
        tally_trigger_ignore_zeros: bool,
        convergence_gate: Any,
        openmc_seed: int,
        max_attempts: int,
        interrupted_counts_as_failed_attempt: bool,
        on_max_attempts: str,
    ) -> None:
        """Store injected components and explicit transport/retry parameters.

        Parameters are the sorter, builder factory, runner, evaluator, checkpoint
        manager, artifact paths, OpenMC settings, seed, and retry policy. The
        constructor returns no value."""
        if not callable(builder_factory):
            raise TypeError("builder_factory harus dapat dipanggil.")
        if not isinstance(material_file, (str, Path)) or not str(material_file).strip():
            raise TypeError("material_file harus berupa path non-kosong.")
        if not isinstance(work_root, (str, Path)) or not str(work_root).strip():
            raise TypeError("work_root harus berupa path non-kosong.")
        if not isinstance(keff_trigger_enabled, bool):
            raise TypeError("keff_trigger_enabled harus berupa bool.")
        if not isinstance(tally_trigger_enabled, bool):
            raise TypeError("tally_trigger_enabled harus berupa bool.")
        if not isinstance(tally_trigger_ignore_zeros, bool):
            raise TypeError("tally_trigger_ignore_zeros harus berupa bool.")
        if tally_trigger_type not in {"rel_err", "std_dev", "variance"}:
            raise ValueError("tally_trigger_type harus rel_err, std_dev, atau variance.")
        if not callable(getattr(convergence_gate, "evaluate", None)):
            raise TypeError("convergence_gate harus menyediakan method evaluate().")
        if not isinstance(interrupted_counts_as_failed_attempt, bool):
            raise TypeError("interrupted_counts_as_failed_attempt harus berupa bool.")
        if (
            not isinstance(on_max_attempts, str)
            or on_max_attempts not in self.ON_MAX_ATTEMPT_POLICIES
        ):
            raise ValueError(
                f"on_max_attempts harus salah satu {sorted(self.ON_MAX_ATTEMPT_POLICIES)}."
            )

        self.genome_sorter = genome_sorter
        self.builder_factory = builder_factory
        self.runner = runner
        self.result_evaluator = result_evaluator
        self.checkpoint_manager = checkpoint_manager
        self.material_file = Path(material_file)
        self.work_root = Path(work_root)
        self.tally_name = self._require_non_empty_string("tally_name", tally_name)
        self.tally_score = self._require_non_empty_string("tally_score", tally_score)
        if (
            getattr(self.result_evaluator, "tally_name", None) != self.tally_name
            or getattr(self.result_evaluator, "tally_score", None) != self.tally_score
        ):
            raise ValueError("Identity tally evaluator tidak sama dengan identity tally pipeline.")
        self.particles = self._require_integer("particles", particles, minimum=1)
        self.inactive_batches = self._require_integer(
            "inactive_batches", inactive_batches, minimum=0
        )
        self.active_batches = self._require_integer("active_batches", active_batches, minimum=1)
        self.keff_trigger_enabled = keff_trigger_enabled
        self.keff_trigger_stdev_pcm = self._require_positive_finite_number(
            "keff_trigger_stdev_pcm", keff_trigger_stdev_pcm
        )
        self.trigger_batch_interval = self._require_integer(
            "trigger_batch_interval", trigger_batch_interval, minimum=1
        )
        self.trigger_max_batches = self._require_integer(
            "trigger_max_batches", trigger_max_batches, minimum=1
        )
        self.tally_trigger_enabled = tally_trigger_enabled
        self.tally_trigger_type = tally_trigger_type
        self.tally_trigger_threshold = self._require_positive_finite_number(
            "tally_trigger_threshold", tally_trigger_threshold
        )
        self.tally_trigger_ignore_zeros = tally_trigger_ignore_zeros
        self.convergence_gate = convergence_gate
        self.openmc_seed = self._require_integer("openmc_seed", openmc_seed, minimum=1)
        self.max_attempts = self._require_integer("max_attempts", max_attempts, minimum=1)
        self.interrupted_counts_as_failed_attempt = interrupted_counts_as_failed_attempt
        self.on_max_attempts = on_max_attempts

    def evaluation_context(self, openmc_version: str | None = None) -> dict[str, Any]:
        """Return the model, evaluator, input, convergence, and OpenMC settings
        that determine whether a cached parent result is compatible."""
        provenance = getattr(self.checkpoint_manager, "input_provenance", {})
        inputs = provenance.get("inputs", {}) if isinstance(provenance, Mapping) else {}
        scientific_inputs = {}
        for name in (
            "fuel_inventory.json",
            "materials.xml",
            "canonical_genome_contract",
        ):
            record = inputs.get(name)
            scientific_inputs[name] = record.get("sha256") if isinstance(record, Mapping) else None
        model_version = getattr(self.builder_factory, "MODEL_VERSION", None)
        if model_version is None:
            model_version = f"{getattr(self.builder_factory, '__module__', 'unknown')}.{getattr(self.builder_factory, '__qualname__', type(self.builder_factory).__qualname__)}"
        expected_openmc_version = getattr(self.builder_factory, "OPENMC_VERSION", "unknown")
        selected_openmc_version = (
            expected_openmc_version if openmc_version is None else openmc_version
        )
        convergence = {
            name: getattr(self.convergence_gate, name, None)
            for name in (
                "enabled",
                "max_keff_std_dev_pcm",
                "max_tally_mean_relative_std_dev",
                "max_tally_p95_relative_std_dev",
                "max_tally_max_relative_std_dev",
                "allow_zero_mean_bins",
            )
        }
        constraint = getattr(self.checkpoint_manager, "constraint_contract", None)
        constraint_identity = (
            constraint.identity() if callable(getattr(constraint, "identity", None)) else None
        )
        return {
            "schema_version": 2,
            "model_version": str(model_version),
            "evaluator_version": str(
                getattr(
                    self.result_evaluator,
                    "EVALUATION_VERSION",
                    type(self.result_evaluator).__qualname__,
                )
            ),
            "openmc_version": str(selected_openmc_version),
            "scientific_inputs": scientific_inputs,
            "evaluation_config": {
                "tally_name": self.tally_name,
                "tally_score": self.tally_score,
                "constraint": constraint_identity,
                "convergence": convergence,
            },
            "openmc_parameters": {
                "particles": self.particles,
                "inactive_batches": self.inactive_batches,
                "active_batches": self.active_batches,
                "keff_trigger_enabled": self.keff_trigger_enabled,
                "keff_trigger_stdev_pcm": self.keff_trigger_stdev_pcm,
                "trigger_batch_interval": self.trigger_batch_interval,
                "trigger_max_batches": self.trigger_max_batches,
                "tally_trigger_enabled": self.tally_trigger_enabled,
                "tally_trigger_type": self.tally_trigger_type,
                "tally_trigger_threshold": self.tally_trigger_threshold,
                "tally_trigger_ignore_zeros": self.tally_trigger_ignore_zeros,
                "threads": getattr(self.runner, "threads", None),
                "geometry_debug": getattr(self.runner, "geometry_debug", None),
                "openmc_executable": str(getattr(self.runner, "openmc_exec", "unknown")),
            },
        }

    @staticmethod
    def _openmc_version_from_results(results: Mapping[str, Any]) -> str | None:
        """Return the evaluated statepoint version, or ``None`` when an evaluator
        does not expose it."""
        statistics = results.get("statistics")
        statepoint = statistics.get("statepoint") if isinstance(statistics, Mapping) else None
        version = statepoint.get("openmc_version") if isinstance(statepoint, Mapping) else None
        return str(version) if version is not None else None

    @staticmethod
    def _require_non_empty_string(name: str, value: Any) -> str:
        """Validate one string parameter and return its stripped value."""
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"{name} harus berupa string non-kosong.")
        return value.strip()

    @staticmethod
    def _require_integer(name: str, value: Any, minimum: int) -> int:
        """Validate an integer parameter and return its built-in integer value."""
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} harus berupa integer dan bukan bool.")
        if value < minimum:
            raise ValueError(f"{name} harus lebih besar atau sama dengan {minimum}.")
        return value

    @staticmethod
    def _require_positive_finite_number(name: str, value: Any) -> float:
        """Validate a positive finite real parameter and return a built-in float."""
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} harus numerik dan bukan bool.")
        normalized_value = float(value)
        if not math.isfinite(normalized_value) or normalized_value <= 0.0:
            raise ValueError(f"{name} harus finite dan positif.")
        return normalized_value

    @staticmethod
    def _validate_results(results: Any) -> Mapping[str, Any]:
        """Require a mapping that can be serialized as strict JSON and return it."""
        if not isinstance(results, Mapping):
            raise TypeError("Hasil evaluator harus berupa mapping.")
        json.dumps(results, allow_nan=False)
        return results

    def _count_consumed_attempts(self, individual: Mapping[str, Any]) -> int:
        """Count failed attempts and optionally interrupted attempts for retry limits."""
        consumed_statuses = {"failed", "running"}
        if self.interrupted_counts_as_failed_attempt:
            consumed_statuses.add("interrupted")
        return sum(attempt.get("status") in consumed_statuses for attempt in individual["attempts"])

    def evaluate(
        self,
        candidate: Any,
        state: MutableMapping[str, Any],
        individual_id: str,
        run_id_prefix: str,
        openmc_seed_base: int | None = None,
    ) -> SingleCandidateOutcome:
        """Evaluate one registered mutable candidate until success or retry exhaustion.

        ``candidate`` supplies the included genome, ``state`` and ``individual_id``
        identify the pending checkpoint record, ``run_id_prefix`` names all
        attempts, and ``openmc_seed_base`` optionally assigns an individual seed
        block. A successful call returns paths, metadata, and evaluator results."""
        if not isinstance(run_id_prefix, str) or not run_id_prefix.strip():
            raise ValueError("run_id_prefix harus berupa string non-kosong.")
        selected_seed_base = (
            self.openmc_seed
            if openmc_seed_base is None
            else self._require_integer("openmc_seed_base", openmc_seed_base, minimum=1)
        )

        individual = state["individuals"][individual_id]
        mutable_contract = MutableGenomeContract.from_checkpoint_contract(state["genome_contract"])
        candidate_genome = mutable_contract.validate(candidate.genome, "Genome candidate")
        recorded_genome = mutable_contract.validate(
            individual["genome"], "Genome mutable pada checkpoint"
        )
        if recorded_genome != candidate_genome:
            raise ValueError("Genome candidate tidak sama dengan genome mutable pada checkpoint.")
        if self.genome_sorter.included_length != mutable_contract.included_length:
            raise ValueError(
                "Cardinality GenomeSorter tidak sama dengan genome_contract checkpoint."
            )

        operator = individual.get("operator")
        if not isinstance(operator, MutableMapping):
            raise ValueError("Metadata operator individual harus berupa mutable mapping.")
        current_context = self.evaluation_context()
        stored_context = operator.get("evaluation_context")
        if stored_context is not None and stored_context != current_context:
            raise ValueError(
                "Evaluation context individual tidak sama dengan konfigurasi pipeline saat ini."
            )
        operator["evaluation_context"] = deepcopy(current_context)

        # GenomeSorter is the only boundary that inserts fixed genes into a complete physical genome.
        complete_genome = self.genome_sorter.construct_complete_genome(candidate_genome)

        while True:
            attempt_number = len(individual["attempts"]) + 1
            run_id = f"{run_id_prefix}-A{attempt_number:02d}"
            work_dir = self.work_root / run_id
            consumed_attempts = self._count_consumed_attempts(individual)
            openmc_seed = selected_seed_base + consumed_attempts
            builder = self.builder_factory(
                genome=complete_genome,
                material_file=self.material_file,
                tally_name=self.tally_name,
                tally_score=self.tally_score,
            )
            model, build_data = builder.build_model(
                particles=self.particles,
                inactive_batches=self.inactive_batches,
                active_batches=self.active_batches,
                keff_trigger_enabled=self.keff_trigger_enabled,
                keff_trigger_stdev_pcm=self.keff_trigger_stdev_pcm,
                openmc_seed=openmc_seed,
                trigger_batch_interval=self.trigger_batch_interval,
                trigger_max_batches=self.trigger_max_batches,
                tally_trigger_enabled=self.tally_trigger_enabled,
                tally_trigger_type=self.tally_trigger_type,
                tally_trigger_threshold=self.tally_trigger_threshold,
                tally_trigger_ignore_zeros=self.tally_trigger_ignore_zeros,
            )

            # Pass only ordered tally metadata to the evaluator, rather than the complete build_data.
            if (
                build_data.get("tally_name") != self.tally_name
                or build_data.get("tally_score") != self.tally_score
            ):
                raise ValueError(
                    "Identity tally builder tidak sama dengan identity tally pipeline."
                )
            tally_records = tuple(build_data["tally_records"])
            self.checkpoint_manager.mark_individual_running(
                state, individual_id, run_id, work_dir, openmc_seed
            )

            # Persist the running checkpoint before starting neutron transport.
            self.checkpoint_manager.save(state)
            try:
                run_result = self.runner.run(model, run_id)
                statepoint_path = Path(run_result["statepoint_path"])
                evaluated_results = self.result_evaluator.evaluate(statepoint_path, tally_records)
                if not isinstance(evaluated_results, Mapping):
                    raise TypeError("Hasil evaluator harus berupa mapping.")
                results = dict(evaluated_results)
                results["convergence"] = self.convergence_gate.evaluate(results)
                results = self._validate_results(results)
            except Exception as error:
                consumed_attempts = self._count_consumed_attempts(individual)
                will_retry = consumed_attempts < self.max_attempts

                # CheckpointManager owns transitions from running to pending or failed.
                self.checkpoint_manager.mark_individual_failed(
                    state, individual_id, error, will_retry
                )
                self.checkpoint_manager.save(state)
                if will_retry:
                    continue
                raise SingleCandidateEvaluationError(
                    f"Evaluasi {individual_id} gagal setelah {consumed_attempts} attempt; policy={self.on_max_attempts}."
                ) from error

            convergence = results["convergence"]
            actual_openmc_version = self._openmc_version_from_results(results)
            operator["evaluation_context"] = self.evaluation_context(actual_openmc_version)
            if convergence["enabled"] and convergence["passed"] is not True:
                terminal_status = NOT_CONVERGED_STATUS
                self.checkpoint_manager.mark_individual_not_converged(
                    state, individual_id, results, statepoint_path
                )
            else:
                terminal_status = COMPLETED_STATUS
                self.checkpoint_manager.mark_individual_completed(
                    state, individual_id, results, statepoint_path
                )
            self.checkpoint_manager.save(state)
            return SingleCandidateOutcome(
                results=state["individuals"][individual_id]["results"],
                status=terminal_status,
                complete_genome=tuple(complete_genome),
                tally_records=tally_records,
                statepoint_path=statepoint_path,
                run_id=run_id,
                work_dir=Path(run_result.get("work_dir", work_dir)),
                openmc_seed=openmc_seed,
            )
