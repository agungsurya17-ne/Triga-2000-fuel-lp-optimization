"""Dispatch configured swap mutation on mutable fuel permutations.

swap_mutation exchanges loci without changing allele inventory. MutationDispatcher
draws once per individual, applies the registered operator, and validates the
result and its metadata. This module performs no I/O or neutron transport."""

# Mutation changes placement, not fuel IDs, burnup, or allele inventory.
# Each child receives its mutation draw even when crossover was skipped.
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import json
import math
import random
from typing import Any, TypeAlias

from modules.genome_sorter import GenomeSorter
from modules.mutable_genome import Gene, MutableGenomeContract, MutableGenomeValidationError


MutableGenome: TypeAlias = tuple[Gene, ...]


class MutationError(ValueError):
    """Report invalid mutation configuration, input genome, or operator output."""


class UnknownMutationMethodError(MutationError):
    """Report a configured mutation method that is absent from the registry."""


@dataclass(frozen=True)
class MutationOperatorResult:
    """Contain one raw mutable child and operator-specific JSON-safe metadata."""

    genome: MutableGenome
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class MutationResult:
    """Contain one validated immutable child and complete dispatcher metadata."""

    genome: MutableGenome
    metadata: Mapping[str, Any]


MutationMethod: TypeAlias = Callable[..., MutationOperatorResult]


def _draw_unit_interval(rng: random.Random, value_name: str) -> float:
    """Draw one finite injected-RNG value in the half-open unit interval."""
    draw = float(rng.random())
    if not math.isfinite(draw) or not 0.0 <= draw < 1.0:
        raise MutationError(f"{value_name} dari injected RNG harus berada dalam [0, 1).")
    return draw


def swap_mutation(
    genome: MutableGenome, *, rng: random.Random, swaps_per_mutated_individual: int
) -> MutationOperatorResult:
    """Swap distinct mutable loci the configured number of times using ``rng``.

    ``genome`` is an already validated mutable genome and
    ``swaps_per_mutated_individual`` specifies how many independent swaps to
    perform. The return value is an immutable mutated genome plus swapped loci."""
    if (
        isinstance(swaps_per_mutated_individual, bool)
        or not isinstance(swaps_per_mutated_individual, int)
        or swaps_per_mutated_individual <= 0
    ):
        raise MutationError("swaps_per_mutated_individual harus berupa integer positif.")
    if len(genome) < 2:
        raise MutationError("swap_mutation memerlukan sedikitnya dua mutable gene.")

    mutated_genome = list(genome)
    swapped_indices = []
    for _ in range(swaps_per_mutated_individual):
        # Each swap samples two distinct loci; later swaps may revisit the same loci.
        first_index, second_index = rng.sample(range(len(mutated_genome)), 2)
        if first_index == second_index:
            raise MutationError("Injected RNG menghasilkan dua indeks swap yang sama.")
        mutated_genome[first_index], mutated_genome[second_index] = (
            mutated_genome[second_index],
            mutated_genome[first_index],
        )
        swapped_indices.append([first_index, second_index])
    return MutationOperatorResult(
        genome=tuple(mutated_genome),
        metadata={"swap_count": swaps_per_mutated_individual, "swapped_indices": swapped_indices},
    )


DEFAULT_MUTATION_METHODS: dict[str, MutationMethod] = {"swap_mutation": swap_mutation}
IMPLEMENTED_MUTATION_METHODS = frozenset(DEFAULT_MUTATION_METHODS)


class MutationRegistry:
    """Resolve dedicated mutation callables by stable configured method name."""

    def __init__(self, methods: Mapping[str, MutationMethod] | None = None) -> None:
        """Copy optional ``methods`` into an independent mutation registry; return None."""
        if methods is not None and not isinstance(methods, Mapping):
            raise TypeError("methods mutation harus berupa mapping nama ke callable.")
        self._methods: dict[str, MutationMethod] = {}
        for name, method in (methods or {}).items():
            self.register(name, method)

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered method names in deterministic lexicographic order."""
        return tuple(sorted(self._methods))

    def register(self, name: str, method: MutationMethod) -> None:
        """Register one dedicated mutation method under a unique stable name."""
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ValueError(
                "Nama method mutation harus berupa string non-kosong tanpa whitespace tepi."
            )
        if not callable(method):
            raise TypeError(f"Operator mutation {name!r} harus callable.")
        if name in self._methods:
            raise ValueError(f"Method mutation {name!r} sudah terdaftar.")
        self._methods[name] = method

    def resolve(self, name: str) -> MutationMethod:
        """Return the mutation method named by ``name`` or fail closed."""
        if name not in self._methods:
            raise UnknownMutationMethodError(
                f"Method mutation {name!r} belum diimplementasikan; tersedia={list(self.names)}."
            )
        return self._methods[name]


def build_default_mutation_registry() -> MutationRegistry:
    """Return the production registry containing all implemented mutation methods."""
    return MutationRegistry(DEFAULT_MUTATION_METHODS)


class MutationDispatcher:
    """Apply one configured mutation method to a validated mutable genome only."""

    def __init__(
        self,
        method: str,
        probability_per_individual: float,
        swaps_per_mutated_individual: int,
        registry: MutationRegistry,
        mutable_genome_contract: MutableGenomeContract,
        genome_sorter: GenomeSorter[Gene],
        rng: random.Random,
    ) -> None:
        """Configure ``method`` with per-individual mutation probability; return None.

        ``swaps_per_mutated_individual`` is the number of swaps after a successful draw.
        ``registry`` resolves the operator, the genome contract and sorter validate its
        result, and ``rng`` shares the campaign's reproducible random-number stream."""
        if not isinstance(registry, MutationRegistry):
            raise TypeError("registry harus berupa MutationRegistry.")
        if not isinstance(mutable_genome_contract, MutableGenomeContract):
            raise TypeError("mutable_genome_contract harus berupa MutableGenomeContract.")
        if not isinstance(genome_sorter, GenomeSorter):
            raise TypeError("genome_sorter harus berupa GenomeSorter.")
        if mutable_genome_contract.included_length != genome_sorter.included_length:
            raise MutationError(
                "Panjang MutableGenomeContract tidak sama dengan included positions GenomeSorter."
            )
        # Pm applies per individual, rather than per locus.
        if (
            isinstance(probability_per_individual, bool)
            or not isinstance(probability_per_individual, (int, float))
            or not math.isfinite(float(probability_per_individual))
            or not 0.0 <= float(probability_per_individual) <= 1.0
        ):
            raise ValueError(
                "probability_per_individual mutation harus finite dalam rentang [0, 1]."
            )
        if (
            isinstance(swaps_per_mutated_individual, bool)
            or not isinstance(swaps_per_mutated_individual, int)
            or swaps_per_mutated_individual <= 0
        ):
            raise ValueError("swaps_per_mutated_individual harus berupa integer positif.")
        if mutable_genome_contract.included_length < 2:
            raise MutationError("Mutation swap memerlukan sedikitnya dua mutable gene.")
        required_rng_methods = ("random", "sample", "getstate", "setstate")
        if not isinstance(rng, random.Random) or any(
            not callable(getattr(rng, method_name, None)) for method_name in required_rng_methods
        ):
            raise TypeError(
                "rng harus random.Random checkpoint-compatible dan menyediakan random(), sample(), getstate(), serta setstate()."
            )

        self.method = method
        self.probability_per_individual = float(probability_per_individual)
        self.swaps_per_mutated_individual = swaps_per_mutated_individual
        self.registry = registry
        self.mutable_genome_contract = mutable_genome_contract
        self.genome_sorter = genome_sorter
        # Selection, crossover, and mutation use the same restorable RNG.
        self.rng = rng
        self._operator = self.registry.resolve(method)

    def mutate(self, genome: Iterable[Iterable[str]]) -> MutationResult:
        """Validate and mutate one mutable genome, returning immutable JSON-safe results.

        ``genome`` must match the configured mutable allele inventory. The return
        value records one probability decision and all swaps without mutating the
        caller's genome or constructing a transport model."""
        try:
            source_genome = self.mutable_genome_contract.validate(genome, "Genome mutation input")
        except MutableGenomeValidationError as error:
            raise MutationError(str(error)) from error

        probability_draw = _draw_unit_interval(self.rng, "Mutation probability draw")
        # Apply when u < Pm; Pm=0 always skips and Pm=1 always applies for valid draws.
        applied = probability_draw < self.probability_per_individual
        if applied:
            operator_result = self._operator(
                source_genome,
                rng=self.rng,
                swaps_per_mutated_individual=self.swaps_per_mutated_individual,
            )
            if not isinstance(operator_result, MutationOperatorResult):
                raise MutationError(
                    f"Operator {self.method!r} harus mengembalikan MutationOperatorResult."
                )
            if not isinstance(operator_result.metadata, Mapping):
                raise MutationError(f"Metadata operator {self.method!r} harus berupa mapping.")
            raw_genome = operator_result.genome
            operator_metadata = dict(operator_result.metadata)
        else:
            raw_genome = tuple(gene for gene in source_genome)
            operator_metadata = {"swap_count": 0, "swapped_indices": []}

        # Validate the resulting inventory and slot mapping even when mutation is skipped.
        normalized_genome = self._validate_mutated_genome(raw_genome)
        reserved_keys = {
            "method",
            "applied",
            "probability",
            "probability_draw",
            "swap_count",
            "swapped_indices",
        }
        conflicting_keys = reserved_keys.intersection(operator_metadata).difference(
            {"swap_count", "swapped_indices"}
        )
        if conflicting_keys:
            raise MutationError(
                f"Metadata operator {self.method!r} memakai key milik dispatcher: {sorted(conflicting_keys)}."
            )
        metadata = {
            "method": self.method,
            # Applied records the probability decision; multiple swaps may restore the original genome.
            "applied": applied,
            "probability": self.probability_per_individual,
            "probability_draw": probability_draw,
            "swap_count": operator_metadata.pop("swap_count", 0),
            "swapped_indices": operator_metadata.pop("swapped_indices", []),
            **operator_metadata,
        }
        self._validate_metadata(metadata, applied)
        return MutationResult(genome=normalized_genome, metadata=metadata)

    def _validate_mutated_genome(self, genome: Iterable[Iterable[str]]) -> MutableGenome:
        """Validate inventory and fixed-position round trip for one child genome."""
        try:
            normalized = self.mutable_genome_contract.validate(genome, "Genome hasil mutation")
        except MutableGenomeValidationError as error:
            raise MutationError(str(error)) from error
        complete_genome = self.genome_sorter.construct_complete_genome(normalized)
        if self.genome_sorter.extract_included_genome(complete_genome) != normalized:
            raise MutationError("Genome hasil mutation gagal construct/extract round trip.")
        return normalized

    def _validate_metadata(self, metadata: Mapping[str, Any], applied: bool) -> None:
        """Reject malformed swap evidence and non-JSON-safe operator metadata."""
        swap_count = metadata["swap_count"]
        swapped_indices = metadata["swapped_indices"]
        if isinstance(swap_count, bool) or not isinstance(swap_count, int) or swap_count < 0:
            raise MutationError("Metadata mutation swap_count harus integer non-negatif.")
        if not isinstance(swapped_indices, list) or len(swapped_indices) != swap_count:
            raise MutationError(
                "Metadata mutation swapped_indices harus list yang panjangnya sama dengan swap_count."
            )
        if applied and swap_count != self.swaps_per_mutated_individual:
            raise MutationError(
                "Operator mutation tidak menjalankan jumlah swap yang dikonfigurasi."
            )
        if not applied and (swap_count != 0 or swapped_indices):
            raise MutationError("Metadata mutation no-op harus mencatat nol swap.")
        for index_pair in swapped_indices:
            if (
                not isinstance(index_pair, list)
                or len(index_pair) != 2
                or any(
                    isinstance(index, bool) or not isinstance(index, int) for index in index_pair
                )
            ):
                raise MutationError("Setiap swapped_indices harus berupa dua integer indeks.")
            first_index, second_index = index_pair
            if (
                not 0 <= first_index < self.mutable_genome_contract.included_length
                or not 0 <= second_index < self.mutable_genome_contract.included_length
                or first_index == second_index
            ):
                raise MutationError(
                    "Setiap swapped_indices harus berbeda dan berada dalam mutable genome."
                )
        try:
            # Strict JSON rejects NaN, infinity, and nonserializable metadata.
            json.dumps(metadata, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise MutationError(f"Metadata operator {self.method!r} harus JSON-safe.") from error
