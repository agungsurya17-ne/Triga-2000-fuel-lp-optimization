"""Evaluate registered offspring without repeating reproduction.

OffspringEvaluationScheduler recovers interrupted attempts, validates duplicate
handling, assigns deterministic seeds, and reuses compatible parent fitness or
invokes SingleCandidatePipeline. Candidates are reconstructed from checkpoints."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from modules.campaign_identity import validate_campaign_id
from modules.checkpoint import TERMINAL_EVALUATION_STATUSES
from modules.mutable_genome import MutableGenomeContract
from modules.single_candidate_pipeline import SingleCandidateEvaluationError


class OffspringEvaluationError(RuntimeError):
    """Report an unsafe offspring-evaluation lifecycle boundary."""


@dataclass(frozen=True)
class RegisteredOffspringCandidate:
    """Represent one mutable offspring genome reconstructed from checkpoint data."""

    genome: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class OffspringEvaluationScheduleResult:
    """Summarize one resumable offspring-evaluation scheduler call."""

    generation: int
    phase: str
    status_counts: Mapping[str, int]
    evaluated_ids: tuple[str, ...]
    recovered_ids: tuple[str, ...]


class OffspringEvaluationScheduler:
    """Evaluate only a durable generation's offspring with the shared lifecycle pipeline."""

    def __init__(
        self,
        candidate_pipeline: Any,
        checkpoint_manager: Any,
        state: MutableMapping[str, Any],
        campaign_id: str,
        resample_duplicate_offspring: bool = True,
        evaluation_cache: Any | None = None,
    ) -> None:
        """Store the injected pipeline, checkpoint state, and campaign identity.

        ``candidate_pipeline`` supplies the existing build/run/evaluate retry
        lifecycle. ``checkpoint_manager`` owns phase and status transitions;
        ``state`` is its mutable campaign state. The constructor returns no value."""
        required_methods = (
            "begin_offspring_evaluation",
            "mark_offspring_evaluation_complete",
            "get_pending_individual_ids",
            "recover_interrupted_individuals",
            "reuse_parent_evaluation",
            "save",
        )
        if not callable(getattr(candidate_pipeline, "evaluate", None)):
            raise TypeError("candidate_pipeline harus menyediakan evaluate().")
        if any(
            not callable(getattr(checkpoint_manager, method, None)) for method in required_methods
        ):
            raise TypeError(
                "checkpoint_manager tidak menyediakan lifecycle evaluasi offspring lengkap."
            )
        if not isinstance(state, MutableMapping):
            raise TypeError("state harus berupa mutable mapping.")
        campaign_id = validate_campaign_id(campaign_id)
        max_attempts = getattr(candidate_pipeline, "max_attempts", None)
        campaign_seed = getattr(candidate_pipeline, "openmc_seed", None)
        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, Integral)
            or max_attempts <= 0
        ):
            raise ValueError("candidate_pipeline.max_attempts harus integer positif.")
        if (
            isinstance(campaign_seed, bool)
            or not isinstance(campaign_seed, Integral)
            or campaign_seed <= 0
        ):
            raise ValueError("candidate_pipeline.openmc_seed harus integer positif.")
        if not isinstance(resample_duplicate_offspring, bool):
            raise TypeError("resample_duplicate_offspring harus berupa bool.")
        if not callable(getattr(candidate_pipeline, "evaluation_context", None)):
            raise TypeError("candidate_pipeline harus menyediakan evaluation_context().")

        self.candidate_pipeline = candidate_pipeline
        self.checkpoint_manager = checkpoint_manager
        self.state = state
        self.campaign_id = campaign_id
        self.max_attempts = int(max_attempts)
        self.campaign_seed = int(campaign_seed)
        self.resample_duplicate_offspring = resample_duplicate_offspring
        if evaluation_cache is not None and not callable(
            getattr(evaluation_cache, "try_reuse", None)
        ):
            raise TypeError("evaluation_cache harus menyediakan try_reuse().")
        self.evaluation_cache = evaluation_cache

    def run(self) -> OffspringEvaluationScheduleResult:
        """Resume pending offspring attempts and close the terminal evaluation boundary.

        The method uses only ``generation_record.offspring_ids`` and their saved
        mutable genomes. It returns an ordered evaluation summary and never
        selects survivors, reevaluates parents, or creates a new generation."""
        generation, offspring_ids, phase = self._active_offspring_generation()
        if phase == "offspring_evaluated":
            return self._build_result(generation, offspring_ids, (), ())

        self._validate_unevaluated_offspring_duplicates(generation, offspring_ids)
        active_id = self.state["progress"].get("active_individual_id")
        if active_id is not None and active_id not in offspring_ids:
            raise OffspringEvaluationError(
                "Individual running bukan anggota offspring generasi aktif."
            )
        recovered_ids = tuple(self.checkpoint_manager.recover_interrupted_individuals(self.state))
        if recovered_ids:
            self.checkpoint_manager.save(self.state)

        self._ensure_seed_bases(generation, offspring_ids)
        self.checkpoint_manager.begin_offspring_evaluation(self.state, generation)
        self.checkpoint_manager.save(self.state)
        evaluated_ids = []
        pending_ids = self.checkpoint_manager.get_pending_individual_ids(
            self.state, generation, "offspring"
        )
        for offspring_index, individual_id in enumerate(offspring_ids):
            if individual_id not in pending_ids:
                continue
            if self._try_reuse_parent_evaluation(generation, individual_id):
                self.checkpoint_manager.save(self.state)
                evaluated_ids.append(individual_id)
                continue
            if self.evaluation_cache is not None and self.evaluation_cache.try_reuse(
                self.state, individual_id
            ):
                self.checkpoint_manager.save(self.state)
                evaluated_ids.append(individual_id)
                continue
            candidate = self._candidate_from_checkpoint(individual_id)
            seed_base = self.state["individuals"][individual_id]["operator"]["openmc_seed_base"]
            run_id_prefix = f"{self.campaign_id}-{individual_id}"
            try:
                self.candidate_pipeline.evaluate(
                    candidate, self.state, individual_id, run_id_prefix, openmc_seed_base=seed_base
                )
            except SingleCandidateEvaluationError:
                evaluated_ids.append(individual_id)
                if self.candidate_pipeline.on_max_attempts == "halt_campaign":
                    raise
            else:
                evaluated_ids.append(individual_id)

        self.checkpoint_manager.mark_offspring_evaluation_complete(self.state, generation)
        self.checkpoint_manager.save(self.state)
        return self._build_result(generation, offspring_ids, evaluated_ids, recovered_ids)

    def _validate_unevaluated_offspring_duplicates(
        self, generation: int, offspring_ids: tuple[str, ...]
    ) -> None:
        """Reject unapproved historical/sibling duplicates while allowing only an
        explicitly annotated exact clone of an active parent in reuse mode."""
        contract = MutableGenomeContract.from_checkpoint_contract(self.state["genome_contract"])
        generation_record = self.state["generations"][-1]
        if generation_record.get("generation") != generation:
            raise OffspringEvaluationError("Generation berubah saat validasi duplicate offspring.")
        parent_by_genome = {}
        for parent_id in generation_record["parent_ids"]:
            parent_by_genome.setdefault(
                contract.validate(self.state["individuals"][parent_id]["genome"]), parent_id
            )
        offspring_by_genome = {}
        for individual_id in offspring_ids:
            key = contract.validate(self.state["individuals"][individual_id]["genome"])
            offspring_by_genome.setdefault(key, []).append(individual_id)
        active_ids = set(generation_record["parent_ids"]) | set(offspring_ids)
        historical_by_genome = {}
        for individual_id, record in self.state["individuals"].items():
            if individual_id in active_ids:
                continue
            key = contract.validate(record["genome"])
            historical_by_genome.setdefault(key, []).append(individual_id)
        for individual_id in offspring_ids:
            individual = self.state["individuals"][individual_id]
            if individual["status"] in TERMINAL_EVALUATION_STATUSES:
                continue
            genome = contract.validate(individual["genome"])
            duplicate_parent_id = parent_by_genome.get(genome)
            duplicate_metadata = individual.get("operator", {}).get("duplicate_handling")
            approved_parent_clone = (
                not self.resample_duplicate_offspring
                and duplicate_parent_id is not None
                and isinstance(duplicate_metadata, Mapping)
                and duplicate_metadata.get("mode") == "reuse_parent_fitness"
                and duplicate_metadata.get("duplicate_parent_id") == duplicate_parent_id
            )
            sibling_duplicate = len(offspring_by_genome[genome]) > 1
            if duplicate_parent_id is not None and not approved_parent_clone:
                raise OffspringEvaluationError(
                    f"Offspring {individual_id} duplicate parent {duplicate_parent_id}; evaluasi dihentikan sebelum OpenMC karena metadata/policy tidak mengizinkan reuse."
                )
            if sibling_duplicate and duplicate_parent_id is None:
                raise OffspringEvaluationError(
                    f"Offspring {individual_id} duplicate dengan sibling {offspring_by_genome[genome]}; evaluasi dihentikan sebelum OpenMC."
                )
            historical_matches = historical_by_genome.get(genome, [])
            if historical_matches and duplicate_parent_id is None:
                raise OffspringEvaluationError(
                    f"Offspring {individual_id} duplicate dengan riwayat {historical_matches}; evaluasi dihentikan sebelum OpenMC."
                )

    def _try_reuse_parent_evaluation(self, generation: int, individual_id: str) -> bool:
        """Return ``True`` after a durable-ready in-memory reuse transition. A
        context mismatch is recorded and returns ``False`` so OpenMC evaluation
        proceeds normally."""
        individual = self.state["individuals"][individual_id]
        operator = individual.get("operator")
        duplicate_metadata = (
            operator.get("duplicate_handling") if isinstance(operator, MutableMapping) else None
        )
        if self.resample_duplicate_offspring or not isinstance(duplicate_metadata, MutableMapping):
            return False
        source_parent_id = duplicate_metadata.get("duplicate_parent_id")
        if not isinstance(source_parent_id, str):
            return False
        if (
            self.state["generations"][-1].get("generation") != generation
            or source_parent_id not in self.state["generations"][-1]["parent_ids"]
        ):
            raise OffspringEvaluationError(
                "Sumber reuse fitness bukan parent aktif pada generation ini."
            )
        source = self.state["individuals"].get(source_parent_id)
        current_context = self.candidate_pipeline.evaluation_context()
        source_context = (
            source.get("operator", {}).get("evaluation_context")
            if isinstance(source, Mapping)
            else None
        )
        if source_context != current_context:
            duplicate_metadata["fitness_source_parent_id"] = None
            duplicate_metadata["fitness_reused"] = False
            duplicate_metadata["reuse_reason"] = "evaluation_context_mismatch_or_missing"
            return False
        self.checkpoint_manager.reuse_parent_evaluation(
            self.state,
            individual_id,
            source_parent_id,
            current_context,
        )
        return True

    def _active_offspring_generation(self) -> tuple[int, tuple[str, ...], str]:
        """Return the active registered offspring record or fail closed by phase."""
        progress = self.state.get("progress")
        generations = self.state.get("generations")
        if (
            not isinstance(progress, Mapping)
            or not isinstance(generations, list)
            or not generations
        ):
            raise OffspringEvaluationError("Checkpoint tidak memiliki generation offspring aktif.")
        generation = progress.get("generation")
        if isinstance(generation, bool) or not isinstance(generation, Integral) or generation < 0:
            raise OffspringEvaluationError(
                "Progress tidak menunjuk generation aktif untuk evaluasi offspring."
            )
        record = generations[-1]
        if (
            not isinstance(record, Mapping)
            or record.get("generation") != generation
            or record.get("status") != "running"
        ):
            raise OffspringEvaluationError(
                "Generation aktif untuk evaluasi offspring tidak valid atau tidak running."
            )
        phase = progress.get("phase")
        if phase not in {"offspring_registered", "evaluating_offspring", "offspring_evaluated"}:
            raise OffspringEvaluationError(
                "Evaluasi offspring memerlukan phase offspring_registered atau evaluating_offspring."
            )
        offspring_ids = tuple(record.get("offspring_ids", ()))
        if not offspring_ids:
            raise OffspringEvaluationError(
                "Generation aktif tidak memiliki offspring_ids terdaftar."
            )
        if any(individual_id not in self.state["individuals"] for individual_id in offspring_ids):
            raise OffspringEvaluationError(
                "offspring_ids menunjuk individual yang tidak terdaftar."
            )
        return int(generation), offspring_ids, phase

    def _candidate_from_checkpoint(self, individual_id: str) -> RegisteredOffspringCandidate:
        """Reconstruct one child only from its durable mutable genome record."""
        individual = self.state["individuals"][individual_id]
        if individual.get("origin") != "offspring":
            raise OffspringEvaluationError(
                f"Individual {individual_id!r} bukan offspring terdaftar."
            )
        return RegisteredOffspringCandidate(
            genome=tuple((gene_type, gene_id) for gene_type, gene_id in individual["genome"])
        )

    def _seed_base(self, generation: int, offspring_index: int, parent_count: int) -> int:
        """Return a non-overlapping offspring seed block for generation and index.

        The Cantor pairing makes ``(generation, offspring_index)`` injective.
        ``parent_count`` reserves the generation-zero parent seed range already
        allocated by the initial-population scheduler. The pipeline advances
        inside this block only for attempts consumed by the retry policy."""
        paired_index = (generation + offspring_index) * (
            generation + offspring_index + 1
        ) // 2 + offspring_index
        return self.campaign_seed + (parent_count + paired_index + 1) * self.max_attempts

    def _ensure_seed_bases(self, generation: int, offspring_ids: tuple[str, ...]) -> None:
        """Durably allocate or verify every child's deterministic retry seed block.

        ``generation`` and ordered ``offspring_ids`` determine the allocation.
        Existing bases must match exactly, so a resumed campaign cannot silently
        change seeds after a configuration override or interrupted process."""
        parent_count = len(self.state["generations"][-1]["parent_ids"])
        for offspring_index, individual_id in enumerate(offspring_ids):
            operator = self.state["individuals"][individual_id].get("operator")
            if not isinstance(operator, MutableMapping):
                raise OffspringEvaluationError(
                    f"Metadata operator offspring {individual_id!r} tidak mutable."
                )
            expected_seed_base = self._seed_base(generation, offspring_index, parent_count)
            stored_seed_base = operator.get("openmc_seed_base")
            if stored_seed_base is None:
                operator["openmc_seed_base"] = expected_seed_base
            elif stored_seed_base != expected_seed_base:
                raise OffspringEvaluationError(
                    f"openmc_seed_base offspring {individual_id!r} tidak konsisten dengan campaign seed dan urutan generation."
                )

    def _build_result(
        self,
        generation: int,
        offspring_ids: tuple[str, ...],
        evaluated_ids: tuple[str, ...] | list[str],
        recovered_ids: tuple[str, ...],
    ) -> OffspringEvaluationScheduleResult:
        """Build an ordered terminal or resumable status summary from checkpoint state."""
        status_counts = dict(
            sorted(
                Counter(
                    self.state["individuals"][individual_id]["status"]
                    for individual_id in offspring_ids
                ).items()
            )
        )
        return OffspringEvaluationScheduleResult(
            generation=generation,
            phase=self.state["progress"]["phase"],
            status_counts=status_counts,
            evaluated_ids=tuple(evaluated_ids),
            recovered_ids=recovered_ids,
        )
