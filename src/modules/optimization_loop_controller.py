"""Drive the campaign lifecycle from durable checkpoint phases.

OptimizationLoopController schedules initial parents or the next generation and
marks the campaign finished when the generation target is reached. Helpers
validate phase boundaries and summarize each invocation."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from modules.generation_transition_controller import GenerationTransitionResult


class OptimizationLoopError(RuntimeError):
    """Report an invalid checkpoint boundary for the multi-generation loop."""


@dataclass(frozen=True)
class OptimizationRunResult:
    """Summarize one uninterrupted or resumed optimization-loop invocation."""

    configured_num_generations: int
    initial_phase: str
    final_phase: str
    final_generation: int
    initial_population_result: Any | None
    transition_results: tuple[GenerationTransitionResult, ...]
    newly_finished: bool
    already_finished: bool


class OptimizationLoopController:
    """Advance durable GA phases until exactly ``num_generations`` are complete."""

    INITIAL_POPULATION_PHASES = frozenset({"initialized", "evaluating_parents"})
    TRANSITION_PHASES = frozenset(
        {
            "parents_evaluated",
            "creating_offspring",
            "offspring_registered",
            "evaluating_offspring",
            "offspring_evaluated",
            "environmental_selection",
        }
    )

    def __init__(
        self,
        checkpoint_manager: Any,
        state: MutableMapping[str, Any],
        num_generations: int,
        initial_population_scheduler: Any,
        generation_transition_controller: Any,
        rng: Any,
    ) -> None:
        """Store checkpoint state, configured limit, schedulers, and shared RNG.

        ``num_generations`` counts reproduction/replacement transitions whose
        zero-based generation indices are ``0`` through
        ``num_generations - 1``. The method returns no value."""
        required_checkpoint_methods = ("mark_finished", "save")
        if any(
            not callable(getattr(checkpoint_manager, method, None))
            for method in required_checkpoint_methods
        ):
            raise TypeError("checkpoint_manager tidak menyediakan lifecycle akhir campaign.")
        if not isinstance(state, MutableMapping):
            raise TypeError("state harus berupa mutable mapping.")
        if (
            isinstance(num_generations, bool)
            or not isinstance(num_generations, Integral)
            or num_generations <= 0
        ):
            raise ValueError("num_generations harus berupa integer positif.")
        if not callable(getattr(initial_population_scheduler, "run", None)):
            raise TypeError("initial_population_scheduler harus menyediakan run().")
        if not callable(getattr(generation_transition_controller, "run", None)):
            raise TypeError("generation_transition_controller harus menyediakan run().")
        if not callable(getattr(rng, "getstate", None)) or not callable(
            getattr(rng, "setstate", None)
        ):
            raise TypeError("rng harus menyediakan getstate() dan setstate().")
        self.checkpoint_manager = checkpoint_manager
        self.state = state
        self.num_generations = int(num_generations)
        self.initial_population_scheduler = initial_population_scheduler
        self.generation_transition_controller = generation_transition_controller
        self.rng = rng
        self._require_configured_generation_budget()

    def run(self) -> OptimizationRunResult:
        """Resume pending durable work and stop after the configured final generation.

        The checkpoint phase selects the only next operation. A ``finished``
        checkpoint is validated and returned without invoking initialization,
        reproduction, evaluation, or replacement."""
        initial_phase = self._phase()
        if initial_phase == "finished":
            final_generation = self._require_finished_campaign()
            return self._build_result(initial_phase, final_generation, None, (), False, True)

        initial_population_result = None
        transition_results: list[GenerationTransitionResult] = []
        while self._phase() != "finished":
            phase = self._phase()
            generation = self._generation()
            if phase in self.INITIAL_POPULATION_PHASES and generation in {None, 0}:
                initial_population_result = self.initial_population_scheduler.run(
                    registration_only=False
                )
                self._require_phase("parents_evaluated")
                continue
            if phase in self.TRANSITION_PHASES:
                target_generation = self._require_active_generation(generation)
                transition_results.append(
                    self.generation_transition_controller.run(target_generation)
                )
                self._require_phase("generation_completed")
                continue
            if phase == "evaluating_parents":
                target_generation = self._require_active_generation(generation)
                if target_generation == 0:
                    raise OptimizationLoopError(
                        "Generation 0 evaluating_parents harus ditangani scheduler initial population."
                    )
                transition_results.append(
                    self.generation_transition_controller.run(target_generation)
                )
                self._require_phase("generation_completed")
                continue
            if phase == "generation_completed":
                completed_generation = self._require_active_generation(generation)
                if completed_generation == self.num_generations - 1:
                    self._mark_finished_durably()
                    continue
                if completed_generation >= self.num_generations:
                    raise OptimizationLoopError(
                        "Checkpoint telah melewati ga.num_generations sebelum ditandai finished."
                    )
                next_generation = completed_generation + 1
                transition_results.append(
                    self.generation_transition_controller.run(next_generation)
                )
                self._require_phase("generation_completed")
                continue
            raise OptimizationLoopError(f"Loop optimisasi tidak dapat melanjutkan phase {phase!r}.")

        final_generation = self._require_finished_campaign()
        return self._build_result(
            initial_phase,
            final_generation,
            initial_population_result,
            transition_results,
            True,
            False,
        )

    def _phase(self) -> str:
        """Return the current checkpoint phase after validating progress shape."""
        progress = self.state.get("progress")
        if not isinstance(progress, Mapping) or not isinstance(progress.get("phase"), str):
            raise OptimizationLoopError("Checkpoint tidak memiliki progress phase yang valid.")
        return progress["phase"]

    def _require_configured_generation_budget(self) -> None:
        """Require runtime and checkpoint generation limits to have one owner value."""
        ga_settings = self.state.get("ga_settings")
        stored_num_generations = (
            ga_settings.get("num_generations") if isinstance(ga_settings, Mapping) else None
        )
        if (
            isinstance(stored_num_generations, bool)
            or not isinstance(stored_num_generations, Integral)
            or int(stored_num_generations) != self.num_generations
        ):
            raise OptimizationLoopError(
                "num_generations runtime tidak sama dengan ga_settings checkpoint."
            )
        controller_limit = getattr(
            self.generation_transition_controller, "num_generations", self.num_generations
        )
        if (
            isinstance(controller_limit, bool)
            or not isinstance(controller_limit, Integral)
            or int(controller_limit) != self.num_generations
        ):
            raise OptimizationLoopError(
                "num_generations controller satu-generasi tidak sama dengan runtime loop."
            )

    def _generation(self) -> int | None:
        """Return the current zero-based generation or ``None`` before generation 0."""
        generation = self.state["progress"].get("generation")
        if generation is None:
            return None
        if isinstance(generation, bool) or not isinstance(generation, Integral) or generation < 0:
            raise OptimizationLoopError(
                "Checkpoint progress tidak memiliki generation non-negatif."
            )
        return int(generation)

    def _require_active_generation(self, generation: int | None) -> int:
        """Return one in-budget active generation index or fail closed."""
        if generation is None:
            raise OptimizationLoopError("Phase aktif tidak menunjuk generation.")
        if generation >= self.num_generations:
            raise OptimizationLoopError(
                f"Generation aktif {generation} berada di luar ga.num_generations={self.num_generations}."
            )
        return generation

    def _require_phase(self, expected_phase: str) -> None:
        """Require a delegated scheduler to stop at its documented durable phase."""
        actual_phase = self._phase()
        if actual_phase != expected_phase:
            raise OptimizationLoopError(
                f"Scheduler harus berhenti pada phase {expected_phase!r}, menerima {actual_phase!r}."
            )

    def _mark_finished_durably(self) -> None:
        """Mark and save ``finished``, restoring memory if persistence fails."""
        state_before = deepcopy(self.state)
        try:
            self.checkpoint_manager.mark_finished(self.state)
            self.checkpoint_manager.save(self.state, python_rng=self.rng)
        except Exception:
            self.state.clear()
            self.state.update(state_before)
            raise

    def _require_finished_campaign(self) -> int:
        """Validate exact generation count and final membership for a finished no-op."""
        if self._phase() != "finished":
            raise OptimizationLoopError("Campaign belum berada pada phase finished.")
        generation = self._generation()
        expected_final_generation = self.num_generations - 1
        if generation != expected_final_generation:
            raise OptimizationLoopError(
                f"Campaign finished harus menunjuk generation {expected_final_generation}, menerima {generation}."
            )
        generations = self.state.get("generations")
        if not isinstance(generations, list) or len(generations) != self.num_generations:
            raise OptimizationLoopError(
                f"Campaign finished harus memiliki tepat {self.num_generations} generation record."
            )
        final_record = generations[-1]
        if (
            not isinstance(final_record, Mapping)
            or final_record.get("generation") != expected_final_generation
            or final_record.get("status") != "completed"
        ):
            raise OptimizationLoopError(
                "Campaign finished harus mempertahankan generation terakhir yang completed."
            )
        return expected_final_generation

    def _build_result(
        self,
        initial_phase: str,
        final_generation: int,
        initial_population_result: Any | None,
        transition_results: (
            tuple[GenerationTransitionResult, ...] | list[GenerationTransitionResult]
        ),
        newly_finished: bool,
        already_finished: bool,
    ) -> OptimizationRunResult:
        """Build the immutable summary returned by one optimization-loop call."""
        return OptimizationRunResult(
            configured_num_generations=self.num_generations,
            initial_phase=initial_phase,
            final_phase=self._phase(),
            final_generation=final_generation,
            initial_population_result=initial_population_result,
            transition_results=tuple(transition_results),
            newly_finished=newly_finished,
            already_finished=already_finished,
        )
