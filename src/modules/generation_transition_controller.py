"""Coordinate durable reproduction, evaluation, and replacement boundaries.

GenerationTransitionController invokes injected schedulers and callbacks for one
generation, validates its population and phase, and resumes from saved boundaries
without implementing NSGA-II operators itself."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any


class GenerationTransitionError(RuntimeError):
    """Report an invalid or unsafe one-generation controller state."""


class GenerationLimitReachedError(GenerationTransitionError):
    """Report a requested generation outside the configured generation budget."""


@dataclass(frozen=True)
class GenerationTransitionResult:
    """Summarize one completed or idempotently resumed generation transition."""

    generation: int
    initial_phase: str
    final_phase: str
    parent_ids: tuple[str, ...]
    offspring_ids: tuple[str, ...]
    survivor_ids: tuple[str, ...]
    offspring_status_counts: Mapping[str, int]
    evaluated_offspring_ids: tuple[str, ...]
    recovered_offspring_ids: tuple[str, ...]
    durable_boundaries: tuple[str, ...]
    offspring_registry_reused: bool
    already_completed: bool


class GenerationTransitionController:
    """Advance one explicit generation target across durable phase boundaries."""

    RESUMABLE_PHASES = frozenset(
        {
            "evaluating_parents",
            "parents_evaluated",
            "creating_offspring",
            "offspring_registered",
            "evaluating_offspring",
            "offspring_evaluated",
            "environmental_selection",
            "generation_completed",
        }
    )

    def __init__(
        self,
        checkpoint_manager: Any,
        state: MutableMapping[str, Any],
        population_size: int,
        num_generations: int,
        offspring_registration_scheduler: Any,
        offspring_evaluation_scheduler: Any,
        complete_generation: Callable[[], Any],
        rng: Any,
        boundary_callback: Callable[[str], None] | None = None,
    ) -> None:
        """Store lifecycle dependencies and configured population/generation limits.

        ``population_size`` is required for parents, offspring, and survivors.
        ``num_generations`` permits target indices from zero through one less
        than that value. ``boundary_callback`` observes a phase only after its
        checkpoint is durable and is primarily useful for interruption tests."""
        required_checkpoint_methods = (
            "begin_generation",
            "mark_parent_evaluation_complete",
            "begin_offspring_creation",
            "begin_offspring_evaluation",
            "begin_environmental_selection",
            "save",
        )
        if any(
            not callable(getattr(checkpoint_manager, method, None))
            for method in required_checkpoint_methods
        ):
            raise TypeError(
                "checkpoint_manager tidak menyediakan lifecycle transisi generasi lengkap."
            )
        if not isinstance(state, MutableMapping):
            raise TypeError("state harus berupa mutable mapping.")
        if (
            isinstance(population_size, bool)
            or not isinstance(population_size, Integral)
            or population_size <= 0
        ):
            raise ValueError("population_size harus berupa integer positif.")
        if (
            isinstance(num_generations, bool)
            or not isinstance(num_generations, Integral)
            or num_generations <= 0
        ):
            raise ValueError("num_generations harus berupa integer positif.")
        if not callable(getattr(offspring_registration_scheduler, "run", None)):
            raise TypeError("offspring_registration_scheduler harus menyediakan run().")
        if not callable(getattr(offspring_evaluation_scheduler, "run", None)):
            raise TypeError("offspring_evaluation_scheduler harus menyediakan run().")
        if not callable(complete_generation):
            raise TypeError("complete_generation harus callable.")
        if not callable(getattr(rng, "getstate", None)) or not callable(
            getattr(rng, "setstate", None)
        ):
            raise TypeError("rng harus menyediakan getstate() dan setstate().")
        if boundary_callback is not None and not callable(boundary_callback):
            raise TypeError("boundary_callback harus callable atau None.")
        self.checkpoint_manager = checkpoint_manager
        self.state = state
        self.population_size = int(population_size)
        self.num_generations = int(num_generations)
        self.offspring_registration_scheduler = offspring_registration_scheduler
        self.offspring_evaluation_scheduler = offspring_evaluation_scheduler
        self.complete_generation = complete_generation
        self.rng = rng
        self.boundary_callback = boundary_callback

    def run(self, target_generation: int) -> GenerationTransitionResult:
        """Complete exactly ``target_generation`` or return its durable result.

        The explicit target is the idempotency key: retrying after the final
        checkpoint returns the completed generation instead of starting the
        next one. Intermediate phases resume only toward the same target."""
        target_generation = self._normalize_target_generation(target_generation)
        initial_phase, current_generation = self._read_progress()
        if initial_phase == "generation_completed" and current_generation == target_generation:
            return self._build_result(target_generation, initial_phase, (), (), (), True, True)
        self._require_target_matches_progress(target_generation, current_generation, initial_phase)

        durable_boundaries: list[str] = []
        evaluated_ids: tuple[str, ...] = ()
        recovered_ids: tuple[str, ...] = ()
        registry_reused = initial_phase in {
            "offspring_registered",
            "evaluating_offspring",
            "offspring_evaluated",
            "environmental_selection",
        }

        if initial_phase == "generation_completed":
            previous_generation = self._generation_record(current_generation)
            parent_ids = tuple(previous_generation["survivor_ids"])
            self._require_population_size(parent_ids, "survivor generasi sebelumnya")
            self.checkpoint_manager.begin_generation(self.state, target_generation, parent_ids)
            self._persist_boundary("evaluating_parents", durable_boundaries)

        if self._phase() == "evaluating_parents":
            self._require_completed_survivor_parents(target_generation)
            self.checkpoint_manager.mark_parent_evaluation_complete(self.state, target_generation)
            self._persist_boundary("parents_evaluated", durable_boundaries)

        if self._phase() == "parents_evaluated":
            self._require_active_population(target_generation, offspring_required=False)
            self.checkpoint_manager.begin_offspring_creation(self.state, target_generation)
            self._persist_boundary("creating_offspring", durable_boundaries)

        if self._phase() == "creating_offspring":
            registration_result = self.offspring_registration_scheduler.run()
            if registration_result.generation != target_generation:
                raise GenerationTransitionError(
                    "Scheduler registrasi mengembalikan generation yang berbeda dari target."
                )
            registry_reused = registry_reused or registration_result.resumed
            self._require_population_size(tuple(registration_result.offspring_ids), "offspring")
            self._record_existing_boundary("offspring_registered", durable_boundaries)

        if self._phase() == "offspring_registered":
            self._require_active_population(target_generation, offspring_required=True)
            self.checkpoint_manager.begin_offspring_evaluation(self.state, target_generation)
            self._persist_boundary("evaluating_offspring", durable_boundaries)

        if self._phase() == "evaluating_offspring":
            evaluation_result = self.offspring_evaluation_scheduler.run()
            if evaluation_result.generation != target_generation:
                raise GenerationTransitionError(
                    "Scheduler evaluasi mengembalikan generation yang berbeda dari target."
                )
            evaluated_ids = tuple(evaluation_result.evaluated_ids)
            recovered_ids = tuple(evaluation_result.recovered_ids)
            self._record_existing_boundary("offspring_evaluated", durable_boundaries)

        if self._phase() == "offspring_evaluated":
            self._require_active_population(target_generation, offspring_required=True)
            self.checkpoint_manager.begin_environmental_selection(self.state, target_generation)
            self._persist_boundary("environmental_selection", durable_boundaries)

        if self._phase() == "environmental_selection":
            selection_result = self.complete_generation()
            if tuple(selection_result.survivor_ids) != tuple(
                self._generation_record(target_generation)["survivor_ids"]
            ):
                raise GenerationTransitionError(
                    "Hasil replacement tidak sama dengan survivor checkpoint."
                )
            self._require_population_size(tuple(selection_result.survivor_ids), "survivor")
            self._persist_boundary("generation_completed", durable_boundaries)

        if self._phase() != "generation_completed":
            raise GenerationTransitionError(
                f"Controller berhenti pada phase tidak terminal: {self._phase()!r}."
            )
        return self._build_result(
            target_generation,
            initial_phase,
            evaluated_ids,
            recovered_ids,
            durable_boundaries,
            registry_reused,
            False,
        )

    def _normalize_target_generation(self, target_generation: int) -> int:
        """Validate the explicit idempotency target against ``num_generations``."""
        if (
            isinstance(target_generation, bool)
            or not isinstance(target_generation, Integral)
            or target_generation < 0
        ):
            raise ValueError("target_generation harus berupa integer non-negatif.")
        normalized = int(target_generation)
        if normalized >= self.num_generations:
            raise GenerationLimitReachedError(
                f"Generation {normalized} berada di luar ga.num_generations={self.num_generations}; indeks maksimum adalah {self.num_generations - 1}."
            )
        return normalized

    def _read_progress(self) -> tuple[str, int]:
        """Return the resumable phase and current generation from checkpoint state."""
        progress = self.state.get("progress")
        if not isinstance(progress, Mapping):
            raise GenerationTransitionError("Checkpoint tidak memiliki progress mapping.")
        phase = progress.get("phase")
        generation = progress.get("generation")
        if phase not in self.RESUMABLE_PHASES:
            raise GenerationTransitionError(f"Controller tidak dapat dimulai dari phase {phase!r}.")
        if isinstance(generation, bool) or not isinstance(generation, Integral) or generation < 0:
            raise GenerationTransitionError("Progress tidak menunjuk generation non-negatif.")
        return str(phase), int(generation)

    def _require_target_matches_progress(
        self, target_generation: int, current_generation: int, phase: str
    ) -> None:
        """Require one consecutive target or the active in-progress generation."""
        expected_target = (
            current_generation + 1 if phase == "generation_completed" else current_generation
        )
        if target_generation != expected_target:
            raise GenerationTransitionError(
                f"target_generation harus {expected_target} untuk resume dari phase {phase}, menerima {target_generation}."
            )

    def _require_completed_survivor_parents(self, generation: int) -> None:
        """Verify resumed next-generation parents are the prior completed survivors."""
        generation_record = self._generation_record(generation)
        parent_ids = tuple(generation_record["parent_ids"])
        self._require_population_size(parent_ids, "parent")
        if generation == 0:
            raise GenerationTransitionError(
                "Controller tidak mengambil alih evaluasi parent awal generation 0."
            )
        previous_survivors = tuple(self._generation_record(generation - 1)["survivor_ids"])
        if parent_ids != previous_survivors:
            raise GenerationTransitionError(
                "Parent generasi lanjutan tidak sama dan seurut dengan survivor generasi sebelumnya."
            )
        incomplete_ids = [
            individual_id
            for individual_id in parent_ids
            if self.state["individuals"][individual_id]["status"] != "completed"
        ]
        if incomplete_ids:
            raise GenerationTransitionError(
                "Parent survivor generasi lanjutan belum completed: "
                + ", ".join(incomplete_ids)
                + "."
            )

    def _require_active_population(self, generation: int, offspring_required: bool) -> None:
        """Verify exact configured parent and optional offspring cardinality."""
        generation_record = self._generation_record(generation)
        if generation_record.get("status") != "running":
            raise GenerationTransitionError(f"Generation {generation} tidak berstatus running.")
        self._require_population_size(tuple(generation_record["parent_ids"]), "parent")
        offspring_ids = tuple(generation_record["offspring_ids"])
        if offspring_required:
            self._require_population_size(offspring_ids, "offspring")
        elif offspring_ids:
            raise GenerationTransitionError(
                "Phase sebelum reproduksi tidak boleh sudah memiliki offspring."
            )

    def _require_population_size(
        self, individual_ids: tuple[str, ...], population_name: str
    ) -> None:
        """Require one ordered population to match configured ``population_size``."""
        if (
            len(individual_ids) != self.population_size
            or len(set(individual_ids)) != self.population_size
        ):
            raise GenerationTransitionError(
                f"Populasi {population_name} harus berisi tepat {self.population_size} ID unik."
            )

    def _generation_record(self, generation: int) -> Mapping[str, Any]:
        """Return one registered generation record by its numeric identifier."""
        generations = self.state.get("generations")
        if not isinstance(generations, list):
            raise GenerationTransitionError("Checkpoint generations harus berupa list.")
        for generation_record in generations:
            if (
                isinstance(generation_record, Mapping)
                and generation_record.get("generation") == generation
            ):
                return generation_record
        raise GenerationTransitionError(f"Generation {generation} tidak ditemukan pada checkpoint.")

    def _phase(self) -> str:
        """Return the current lifecycle phase after a controller operation."""
        return str(self.state["progress"]["phase"])

    def _persist_boundary(self, expected_phase: str, durable_boundaries: list[str]) -> None:
        """Save RNG and state before announcing one newly reached phase boundary."""
        if self._phase() != expected_phase:
            raise GenerationTransitionError(
                f"Phase setelah side effect harus {expected_phase!r}, menerima {self._phase()!r}."
            )
        self.checkpoint_manager.save(self.state, python_rng=self.rng)
        self._record_boundary(expected_phase, durable_boundaries)

    def _record_existing_boundary(self, expected_phase: str, durable_boundaries: list[str]) -> None:
        """Announce a phase already saved atomically by a delegated scheduler."""
        if self._phase() != expected_phase:
            raise GenerationTransitionError(
                f"Scheduler harus berhenti pada phase {expected_phase!r}, menerima {self._phase()!r}."
            )
        self._record_boundary(expected_phase, durable_boundaries)

    def _record_boundary(self, phase: str, durable_boundaries: list[str]) -> None:
        """Record and optionally expose one durable boundary to an observer."""
        durable_boundaries.append(phase)
        if self.boundary_callback is not None:
            self.boundary_callback(phase)

    def _build_result(
        self,
        generation: int,
        initial_phase: str,
        evaluated_ids: tuple[str, ...],
        recovered_ids: tuple[str, ...],
        durable_boundaries: tuple[str, ...] | list[str],
        registry_reused: bool,
        already_completed: bool,
    ) -> GenerationTransitionResult:
        """Build a deterministic generation summary from durable checkpoint data."""
        generation_record = self._generation_record(generation)
        parent_ids = tuple(generation_record["parent_ids"])
        offspring_ids = tuple(generation_record["offspring_ids"])
        survivor_ids = tuple(generation_record["survivor_ids"])
        self._require_population_size(parent_ids, "parent")
        self._require_population_size(offspring_ids, "offspring")
        self._require_population_size(survivor_ids, "survivor")
        status_counts = dict(
            sorted(
                Counter(
                    self.state["individuals"][individual_id]["status"]
                    for individual_id in offspring_ids
                ).items()
            )
        )
        return GenerationTransitionResult(
            generation=generation,
            initial_phase=initial_phase,
            final_phase=self._phase(),
            parent_ids=parent_ids,
            offspring_ids=offspring_ids,
            survivor_ids=survivor_ids,
            offspring_status_counts=status_counts,
            evaluated_offspring_ids=evaluated_ids,
            recovered_offspring_ids=recovered_ids,
            durable_boundaries=tuple(durable_boundaries),
            offspring_registry_reused=registry_reused,
            already_completed=already_completed,
        )
