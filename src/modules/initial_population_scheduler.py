"""Register generation-zero parents and evaluate pending individuals.

InitialPopulationScheduler obtains initial genomes from PopulationInitializer,
persists the registry and seeds, and closes parent evaluation at a durable
checkpoint boundary."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

from modules.campaign_identity import validate_campaign_id
from modules.checkpoint import TERMINAL_EVALUATION_STATUSES
from modules.single_candidate_pipeline import SingleCandidateEvaluationError


@dataclass(frozen=True)
class RegisteredInitialCandidate:
    """Represent one initial candidate reconstructed from its checkpoint record."""

    genome: tuple[tuple[str, str], ...]
    strategy: str


@dataclass(frozen=True)
class InitialPopulationScheduleResult:
    """Summarize one registration-only or generation-zero scheduling call."""

    generation: int
    phase: str
    status_counts: Mapping[str, int]
    evaluated_ids: tuple[str, ...]
    registration_only: bool


class InitialPopulationScheduler:
    """Register and evaluate generation-zero parents without creating offspring."""

    GENERATION = 0

    def __init__(
        self,
        population_initializer: Any,
        candidate_pipeline: Any,
        checkpoint_manager: Any,
        state: MutableMapping[str, Any],
        campaign_id: str,
        evaluation_cache: Any | None = None,
    ) -> None:
        """Connect initialization and evaluation services to a campaign; return None.

        ``population_initializer`` creates genomes; ``candidate_pipeline`` evaluates
        them. ``checkpoint_manager`` persists the shared ``state`` for ``campaign_id``.
        An optional ``evaluation_cache`` reuses results only after compatibility checks."""
        if not callable(getattr(population_initializer, "initialize_population", None)):
            raise TypeError("population_initializer harus menyediakan initialize_population().")
        if not callable(getattr(candidate_pipeline, "evaluate", None)):
            raise TypeError("candidate_pipeline harus menyediakan evaluate().")
        if not callable(getattr(checkpoint_manager, "register_individual", None)):
            raise TypeError("checkpoint_manager tidak menyediakan lifecycle individual.")
        if not isinstance(state, MutableMapping):
            raise TypeError("state harus berupa mutable mapping.")
        campaign_id = validate_campaign_id(campaign_id)

        population_size = getattr(population_initializer, "population_size", None)
        if (
            isinstance(population_size, bool)
            or not isinstance(population_size, int)
            or population_size <= 0
        ):
            raise ValueError("population_initializer.population_size harus integer positif.")
        max_attempts = getattr(candidate_pipeline, "max_attempts", None)
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts <= 0:
            raise ValueError("candidate_pipeline.max_attempts harus integer positif.")
        openmc_seed = getattr(candidate_pipeline, "openmc_seed", None)
        if isinstance(openmc_seed, bool) or not isinstance(openmc_seed, int) or openmc_seed <= 0:
            raise ValueError("candidate_pipeline.openmc_seed harus integer positif.")

        self.population_initializer = population_initializer
        self.candidate_pipeline = candidate_pipeline
        self.checkpoint_manager = checkpoint_manager
        self.state = state
        self.campaign_id = campaign_id
        self.population_size = population_size
        self.max_attempts = max_attempts
        self.openmc_seed = openmc_seed
        if evaluation_cache is not None and not callable(
            getattr(evaluation_cache, "try_reuse", None)
        ):
            raise TypeError("evaluation_cache harus menyediakan try_reuse().")
        self.evaluation_cache = evaluation_cache

    def run(self, registration_only: bool = False) -> InitialPopulationScheduleResult:
        """Register generation zero, evaluate pending parents, and return its checkpoint summary."""
        if not isinstance(registration_only, bool):
            raise TypeError("registration_only harus berupa bool.")

        parent_ids = self._ensure_registered_generation()
        if registration_only:
            return self._build_result((), registration_only=True)

        evaluated_ids = []
        pending_ids = self.checkpoint_manager.get_pending_individual_ids(
            self.state, self.GENERATION, "parents"
        )
        for individual_id in pending_ids:
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

        self._finish_parent_evaluation(parent_ids)
        return self._build_result(evaluated_ids, registration_only=False)

    def _ensure_registered_generation(self) -> tuple[str, ...]:
        """Create or validate the complete deterministic generation-zero registry."""
        expected_ids = self._expected_parent_ids()
        individuals = self.state["individuals"]
        generations = self.state["generations"]

        if not individuals and not generations:
            candidates = tuple(self.population_initializer.initialize_population())
            if len(candidates) != self.population_size:
                raise ValueError(
                    f"Initializer harus menghasilkan {self.population_size} kandidat, menerima {len(candidates)}."
                )
            for index, (individual_id, candidate) in enumerate(
                zip(expected_ids, candidates, strict=True)
            ):
                strategy = getattr(candidate, "strategy", None)
                if not isinstance(strategy, str) or not strategy.strip():
                    raise ValueError(f"Strategi candidate indeks {index} harus string non-kosong.")
                operator = {
                    "initialization_strategy": strategy.strip(),
                    "candidate_index": index,
                    "openmc_seed_base": self._seed_base(index),
                }
                self.checkpoint_manager.register_individual(
                    self.state,
                    individual_id,
                    candidate.genome,
                    self.GENERATION,
                    "initial",
                    operator=operator,
                )
            self.checkpoint_manager.save(self.state)
            individuals = self.state["individuals"]

        self._validate_registered_population(expected_ids)
        if not generations:
            self.checkpoint_manager.begin_generation(self.state, self.GENERATION, expected_ids)
            self.checkpoint_manager.save(self.state)
        else:
            generation = generations[0]
            if len(generations) != 1 or generation["generation"] != self.GENERATION:
                raise ValueError("Scheduler P0-5 hanya menerima tepat satu generation 0.")
            if generation["status"] != "running":
                raise ValueError(
                    "Scheduler P0-5 tidak menerima generation 0 yang sudah diselesaikan NSGA-II."
                )
            if tuple(generation["parent_ids"]) != expected_ids:
                raise ValueError(
                    "Urutan parent generation 0 tidak sama dengan ID initial deterministik."
                )
            if generation["offspring_ids"]:
                raise ValueError("Scheduler P0-5 tidak menerima offspring pada generation 0.")
            if self.state["progress"]["phase"] not in {"evaluating_parents", "parents_evaluated"}:
                raise ValueError(
                    "Scheduler P0-5 hanya dapat berjalan pada phase evaluasi parent generation 0."
                )
        return expected_ids

    def _validate_registered_population(self, expected_ids: tuple[str, ...]) -> None:
        """Require exact initial IDs, candidate indices, strategies, genomes, and seed bases."""
        individuals = self.state["individuals"]
        if tuple(individuals) != expected_ids:
            raise ValueError(
                "Registry initial population harus lengkap dan mengikuti urutan ID deterministik."
            )
        for index, individual_id in enumerate(expected_ids):
            individual = individuals[individual_id]
            operator = individual["operator"]
            if (
                individual["origin"] != "initial"
                or individual["generation_created"] != self.GENERATION
            ):
                raise ValueError(f"Record {individual_id} bukan initial parent generation 0.")
            if operator.get("candidate_index") != index:
                raise ValueError(f"candidate_index checkpoint {individual_id} tidak konsisten.")
            if (
                not isinstance(operator.get("initialization_strategy"), str)
                or not operator["initialization_strategy"].strip()
            ):
                raise ValueError(f"initialization_strategy checkpoint {individual_id} tidak valid.")
            if operator.get("openmc_seed_base") != self._seed_base(index):
                raise ValueError(f"openmc_seed_base checkpoint {individual_id} tidak konsisten.")

    def _finish_parent_evaluation(self, parent_ids: tuple[str, ...]) -> None:
        """Persist the parents-evaluated boundary after every parent becomes terminal."""
        nonterminal_ids = [
            individual_id
            for individual_id in parent_ids
            if self.state["individuals"][individual_id]["status"]
            not in TERMINAL_EVALUATION_STATUSES
        ]
        if nonterminal_ids:
            raise RuntimeError(
                "Scheduler selesai tanpa status terminal untuk: " + ", ".join(nonterminal_ids) + "."
            )
        if self.state["progress"]["phase"] != "parents_evaluated":
            self.checkpoint_manager.mark_parent_evaluation_complete(self.state, self.GENERATION)
            self.checkpoint_manager.save(self.state)

    def _candidate_from_checkpoint(self, individual_id: str) -> RegisteredInitialCandidate:
        """Reconstruct one pending candidate solely from its durable checkpoint record."""
        individual = self.state["individuals"][individual_id]
        genome = tuple((gene_type, gene_id) for gene_type, gene_id in individual["genome"])
        strategy = individual["operator"]["initialization_strategy"]
        return RegisteredInitialCandidate(genome=genome, strategy=strategy)

    def _expected_parent_ids(self) -> tuple[str, ...]:
        """Return stable generation-zero IDs in deterministic population order."""
        width = max(3, len(str(self.population_size - 1)))
        return tuple(f"G000-I{index:0{width}d}" for index in range(self.population_size))

    def _seed_base(self, candidate_index: int) -> int:
        """Return a unique seed block start for one candidate index."""
        return self.openmc_seed + candidate_index * self.max_attempts

    def _build_result(
        self, evaluated_ids: tuple[str, ...] | list[str], registration_only: bool
    ) -> InitialPopulationScheduleResult:
        """Build a deterministic status-count summary from current checkpoint state."""
        parent_ids = self.state["generations"][0]["parent_ids"]
        status_counts = dict(
            sorted(
                Counter(
                    self.state["individuals"][individual_id]["status"]
                    for individual_id in parent_ids
                ).items()
            )
        )
        return InitialPopulationScheduleResult(
            generation=self.GENERATION,
            phase=self.state["progress"]["phase"],
            status_counts=status_counts,
            evaluated_ids=tuple(evaluated_ids),
            registration_only=registration_only,
        )
