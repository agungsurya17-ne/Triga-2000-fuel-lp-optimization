"""Create and persist one complete offspring batch before transport.

OffspringCreationScheduler requests parents through a callback, applies core
crossover and mutation dispatchers, handles duplicates according to configuration,
and atomically registers the batch. Registry validation supports safe resume."""

# Check both children before drawing another pair; rejected children never reach transport.
# Duplicates require identical ordered genomes, rather than merely equal allele inventories.
from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from copy import deepcopy
from dataclasses import dataclass
import logging
from numbers import Integral
from typing import Any

from modules.mutable_genome import MutableGenomeContract


logger = logging.getLogger(__name__)


class OffspringRegistrationError(RuntimeError):
    """Report an unsafe offspring registration or resume boundary."""


@dataclass(frozen=True)
class OffspringRegistrationResult:
    """Summarize a complete registered offspring population."""

    generation: int
    offspring_ids: tuple[str, ...]
    resumed: bool


OffspringCreationResult = OffspringRegistrationResult


class OffspringCreationScheduler:
    """Create a complete offspring registry before any offspring evaluation."""

    # max_reproduction_attempts counts selected pairs, not rejected children.
    def __init__(
        self,
        checkpoint_manager: Any,
        state: MutableMapping[str, Any],
        population_size: int,
        select_parent_ids: Callable[[int], tuple[str, ...]],
        crossover_dispatcher: Any,
        mutation_dispatcher: Any,
        rng: Any,
        max_reproduction_attempts: int = 10000,
        resample_duplicate_offspring: bool = True,
    ) -> None:
        """Store checkpoint, pure operators, deterministic selector, and shared RNG.

        ``population_size`` is the exact required offspring cardinality.
        ``select_parent_ids`` receives an even parent-ID count and returns ordered
        parent IDs. The dispatchers operate on mutable genomes only. ``rng`` is
        captured by checkpoint persistence after the entire batch is formed.
        ``max_reproduction_attempts`` caps selected pairs per batch; exhaustion
        raises without registering partial offspring. When
        ``resample_duplicate_offspring`` is false, exact clones of active
        parents are registered for conditional evaluation reuse. The constructor
        returns None."""
        if (
            not callable(getattr(checkpoint_manager, "begin_offspring_creation", None))
            or not callable(getattr(checkpoint_manager, "register_generation_offspring", None))
            or not callable(getattr(checkpoint_manager, "save", None))
        ):
            raise TypeError(
                "checkpoint_manager tidak menyediakan API registrasi offspring lengkap."
            )
        if not isinstance(state, MutableMapping):
            raise TypeError("state harus berupa mutable mapping.")
        if (
            isinstance(population_size, bool)
            or not isinstance(population_size, Integral)
            or population_size <= 0
        ):
            raise ValueError("population_size harus berupa integer positif.")
        if not callable(select_parent_ids):
            raise TypeError("select_parent_ids harus callable.")
        if not callable(getattr(crossover_dispatcher, "crossover", None)):
            raise TypeError("crossover_dispatcher harus menyediakan crossover().")
        if not callable(getattr(mutation_dispatcher, "mutate", None)):
            raise TypeError("mutation_dispatcher harus menyediakan mutate().")
        if not callable(getattr(rng, "getstate", None)) or not callable(
            getattr(rng, "setstate", None)
        ):
            raise TypeError(
                "rng harus menyediakan getstate() dan setstate() untuk checkpoint resumable."
            )
        # Selection, crossover, and mutation must share one restorable RNG object.
        if (
            getattr(crossover_dispatcher, "rng", rng) is not rng
            or getattr(mutation_dispatcher, "rng", rng) is not rng
        ):
            raise ValueError(
                "Seleksi, crossover, dan mutation harus memakai satu injected RNG yang sama."
            )
        if (
            isinstance(max_reproduction_attempts, bool)
            or not isinstance(max_reproduction_attempts, Integral)
            or max_reproduction_attempts <= 0
        ):
            raise ValueError("max_reproduction_attempts harus berupa integer positif.")
        if not isinstance(resample_duplicate_offspring, bool):
            raise TypeError("resample_duplicate_offspring harus berupa bool.")
        self.max_reproduction_attempts = int(max_reproduction_attempts)
        self.resample_duplicate_offspring = resample_duplicate_offspring
        self.checkpoint_manager = checkpoint_manager
        self.state = state
        self.population_size = int(population_size)
        self.select_parent_ids = select_parent_ids
        self.crossover_dispatcher = crossover_dispatcher
        self.mutation_dispatcher = mutation_dispatcher
        self.rng = rng

    def run(self) -> OffspringRegistrationResult:
        """Register the exact offspring batch or resume it without consuming RNG.

        The active generation must have finished parent evaluation. The return
        value marks whether a prior complete registry was reused. This method
        does not call an evaluator, construct OpenMC objects, or start another
        generation."""
        generation = self.state.get("progress", {}).get("generation")
        if isinstance(generation, bool) or not isinstance(generation, Integral) or generation < 0:
            raise OffspringRegistrationError(
                "Progress tidak menunjuk generation aktif untuk offspring."
            )
        generation = int(generation)
        generation_record = self._active_generation(generation)
        expected_ids = tuple(
            self.checkpoint_manager.format_offspring_id(generation, index)
            for index in range(self.population_size)
        )
        if generation_record["offspring_ids"]:
            self._require_complete_registry(generation, generation_record, expected_ids)
            return OffspringRegistrationResult(
                generation=generation, offspring_ids=expected_ids, resumed=True
            )

        # Snapshot state and RNG before reproduction so a failed batch can be retried identically.
        state_before = deepcopy(self.state)
        rng_state_before = self.rng.getstate()
        try:
            self._reject_partial_registry(generation)
            phase = self.state["progress"].get("phase")
            if phase == "parents_evaluated":
                self.checkpoint_manager.begin_offspring_creation(self.state, generation)
            elif phase != "creating_offspring":
                raise OffspringRegistrationError(
                    "Registrasi offspring memerlukan phase parents_evaluated atau creating_offspring."
                )

            contract = MutableGenomeContract.from_checkpoint_contract(self.state["genome_contract"])
            seen = {}
            for individual_id, individual in self.state["individuals"].items():
                seen.setdefault(contract.validate(individual["genome"]), individual_id)
            active_parent_by_genome = {}
            for parent_id in generation_record["parent_ids"]:
                parent_genome = contract.validate(self.state["individuals"][parent_id]["genome"])
                active_parent_by_genome.setdefault(parent_genome, parent_id)
            offspring_records = []
            attempts = 0
            rejections = []
            # Resampling preserves configured Pc/Pm and does not force a changed child.
            while len(offspring_records) < self.population_size:
                if attempts >= self.max_reproduction_attempts:
                    logger.error(
                        "Reproduction generation=%s exhausted: attempts=%s duplicate_rejections=%s accepted=%s target=%s; state/RNG rolled back.",
                        generation,
                        attempts,
                        len(rejections),
                        len(offspring_records),
                        self.population_size,
                    )
                    raise OffspringRegistrationError(
                        f"Batas reproduction {attempts} tercapai: duplicate={len(rejections)}, offspring unik={len(offspring_records)}/{self.population_size}; batch dibatalkan tanpa evaluasi."
                    )
                parent_pair = self._select_parent_ids(2)
                # Count a selected pair even if both children are rejected.
                attempts += 1
                crossover_result = self.crossover_dispatcher.crossover(
                    parent_pair, self.state["individuals"]
                )
                children = tuple(crossover_result.children)
                if len(children) != 2:
                    raise OffspringRegistrationError(
                        "Crossover harus mengembalikan tepat dua child."
                    )
                for child_index, child in enumerate(children):
                    if len(offspring_records) == self.population_size:
                        break
                    mutation_result = self.mutation_dispatcher.mutate(child)
                    genome = contract.validate(mutation_result.genome, "Genome kandidat offspring")
                    duplicate_parent_id = active_parent_by_genome.get(genome)
                    # With resampling disabled, only active-parent clones receive the duplicate exception.
                    accept_parent_duplicate = (
                        duplicate_parent_id is not None and not self.resample_duplicate_offspring
                    )
                    if genome in seen and not accept_parent_duplicate:
                        duplicate_of = seen[genome]
                        rejections.append(
                            {
                                "attempt": attempts,
                                "child_index": child_index,
                                "reason": "duplicate",
                                "duplicate_of": duplicate_of,
                                "parent_ids": list(parent_pair),
                                "crossover_applied": crossover_result.metadata.get("applied"),
                                "mutation_applied": mutation_result.metadata.get("applied"),
                            }
                        )
                        logger.info(
                            "Rejected duplicate offspring generation=%s attempt=%s child=%s duplicate_of=%s parents=%s.",
                            generation,
                            attempts,
                            child_index,
                            duplicate_of,
                            parent_pair,
                        )
                        continue
                    if duplicate_parent_id is None:
                        seen[genome] = expected_ids[len(offspring_records)]
                    duplicate_handling = {
                        "mode": (
                            "resample_duplicate_offspring"
                            if self.resample_duplicate_offspring
                            else "reuse_parent_fitness"
                        ),
                        "duplicate_parent_id": duplicate_parent_id,
                        # Fitness reuse is decided during evaluation after checking the full evaluation context.
                        "fitness_source_parent_id": None,
                        "fitness_reused": False,
                    }
                    offspring_records.append(
                        {
                            "genome": genome,
                            "parent_ids": parent_pair,
                            "crossover": crossover_result.metadata,
                            "mutation": mutation_result.metadata,
                            "duplicate_handling": duplicate_handling,
                        }
                    )

            policy = (
                "reject_duplicate_repeat_reproduction"
                if self.resample_duplicate_offspring
                else "accept_parent_duplicate_conditional_fitness_reuse"
            )
            audit = {
                "policy": policy,
                "resample_duplicate_offspring": self.resample_duplicate_offspring,
                "max_attempts": self.max_reproduction_attempts,
                "attempts": attempts,
                "accepted": len(offspring_records),
                "duplicate_rejections": rejections,
            }
            self.state["metadata"].setdefault("reproduction", {})[str(generation)] = audit
            offspring_ids = self.checkpoint_manager.register_generation_offspring(
                self.state, generation, offspring_records
            )
            # Commit the complete registry with final RNG state so resume does not redraw offspring.
            self.checkpoint_manager.save(self.state, python_rng=self.rng)
        except Exception:
            # Restore the existing state mapping in place to preserve references held by other components.
            self.state.clear()
            self.state.update(state_before)
            # Restore the pre-batch RNG stream before propagating the original failure.
            self.rng.setstate(rng_state_before)
            raise
        return OffspringRegistrationResult(
            generation=generation, offspring_ids=offspring_ids, resumed=False
        )

    def _active_generation(self, generation: int) -> Mapping[str, Any]:
        """Return the running final generation record for ``generation``."""
        generations = self.state.get("generations")
        if not isinstance(generations, list) or not generations:
            raise OffspringRegistrationError(
                "Checkpoint tidak memiliki generation aktif untuk offspring."
            )
        record = generations[-1]
        if (
            not isinstance(record, Mapping)
            or record.get("generation") != generation
            or record.get("status") != "running"
        ):
            raise OffspringRegistrationError(
                "Generation aktif untuk offspring tidak valid atau tidak running."
            )
        return record

    def _select_parent_ids(self, expected_count: int) -> tuple[str, ...]:
        """Select and validate ordered, distinct parent pairs for this batch."""
        parent_ids = self.select_parent_ids(expected_count)
        if not isinstance(parent_ids, tuple) or len(parent_ids) != expected_count:
            raise OffspringRegistrationError(
                f"Seleksi parent harus mengembalikan tuple berisi {expected_count} ID."
            )
        for pair_index in range(0, expected_count, 2):
            first_parent_id, second_parent_id = parent_ids[pair_index : pair_index + 2]
            if (
                not isinstance(first_parent_id, str)
                or not isinstance(second_parent_id, str)
                or first_parent_id == second_parent_id
            ):
                raise OffspringRegistrationError(
                    f"Pasangan parent indeks {pair_index // 2} harus berisi dua ID berbeda."
                )
        return parent_ids

    def _reject_partial_registry(self, generation: int) -> None:
        """Reject any unlinked offspring records instead of regenerating them."""
        partial_ids = [
            individual_id
            for individual_id, record in self.state["individuals"].items()
            if record.get("origin") == "offspring"
            and record.get("generation_created") == generation
        ]
        if partial_ids:
            # Refuse partial registries rather than mixing saved offspring with fresh random draws.
            raise OffspringRegistrationError(
                f"Registry offspring generasi {generation} hanya terisi sebagian: "
                + ", ".join(partial_ids)
                + ". Resume dihentikan tanpa regenerasi."
            )

    def _require_complete_registry(
        self, generation: int, generation_record: Mapping[str, Any], expected_ids: tuple[str, ...]
    ) -> None:
        """Require an exact linked registry before taking the RNG-free resume path."""
        if self.state["progress"].get("phase") != "offspring_registered":
            raise OffspringRegistrationError(
                "Registry offspring durable harus berada pada phase offspring_registered untuk resume aman."
            )
        registered_ids = tuple(generation_record["offspring_ids"])
        if registered_ids != expected_ids:
            raise OffspringRegistrationError(
                f"Registry offspring generasi {generation} tidak lengkap atau urutannya berubah; resume dihentikan tanpa regenerasi."
            )
        for individual_id in expected_ids:
            individual = self.state["individuals"].get(individual_id)
            if (
                not isinstance(individual, Mapping)
                or individual.get("origin") != "offspring"
                or individual.get("generation_created") != generation
                or individual.get("status") != "pending"
            ):
                raise OffspringRegistrationError(
                    f"Record offspring {individual_id!r} tidak lengkap untuk resume aman."
                )
            operator = individual.get("operator")
            if (
                not isinstance(operator, Mapping)
                or not isinstance(operator.get("crossover"), Mapping)
                or not isinstance(operator.get("mutation"), Mapping)
            ):
                raise OffspringRegistrationError(
                    f"Provenance crossover/mutation offspring {individual_id!r} tidak lengkap."
                )


OffspringRegistrationScheduler = OffspringCreationScheduler
