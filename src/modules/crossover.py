"""Provide permutation crossover operators and their dispatch registry.

pmx_crossover creates two permutation children. triga_hcmx_power_mask_crossover
uses parent power tallies as an alternative. CrossoverDispatcher applies the
configured probability and validates each child's inventory and metadata."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import math
from numbers import Integral, Real
import random
from types import MappingProxyType
from typing import Any, TypeAlias

from modules.genome_sorter import GenomeSorter
from modules.mutable_genome import (
    Gene,
    MutableGenomeContract,
    MutableGenomeValidationError,
    normalize_gene_pairs,
)


MutableGenome: TypeAlias = tuple[Gene, ...]


class CrossoverError(ValueError):
    """Report invalid crossover configuration, parents, or operator output."""


class UnknownCrossoverMethodError(CrossoverError):
    """Report a crossover method that is absent from the runtime registry."""


@dataclass(frozen=True)
class CrossoverOperatorContext:
    """Provide stable parent and position context to one dedicated operator."""

    parent_ids: tuple[str, str]
    # Mutable locus indices follow included_positions, not the complete physical genome.
    included_positions: tuple[str, ...]
    parent_results: tuple[Any, Any]
    operator_options: Mapping[str, Any]
    prepared_data: Any = None


@dataclass(frozen=True)
class CrossoverOperatorResult:
    """Contain two mutable children and operator-specific JSON-safe metadata."""

    children: tuple[MutableGenome, MutableGenome]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class CrossoverResult:
    """Contain validated immutable children and complete dispatcher metadata."""

    children: tuple[MutableGenome, MutableGenome]
    metadata: Mapping[str, Any]


CrossoverMethod: TypeAlias = Callable[..., CrossoverOperatorResult]
CrossoverPreparation: TypeAlias = Callable[
    [MutableGenome, MutableGenome, CrossoverOperatorContext], Any
]
CrossoverSkippedMetadata: TypeAlias = Callable[[CrossoverOperatorContext], Mapping[str, Any]]


@dataclass(frozen=True)
class CrossoverMethodDefinition:
    """Bind one operator to optional deterministic preflight and no-op metadata hooks."""

    operator: CrossoverMethod
    # Run deterministic operator preflight before consuming RNG state.
    prepare: CrossoverPreparation | None = None
    skipped_metadata: CrossoverSkippedMetadata | None = None


def _validate_unique_permutation_parents(
    parent_1: Iterable[Iterable[str]], parent_2: Iterable[Iterable[str]], operator_name: str
) -> tuple[MutableGenome, MutableGenome]:
    """Validate unique equal-inventory parents for a named permutation operator."""
    first = normalize_gene_pairs(parent_1, f"{operator_name} parent_1")
    second = normalize_gene_pairs(parent_2, f"{operator_name} parent_2")
    if len(first) < 2 or len(second) < 2:
        raise CrossoverError(f"{operator_name} memerlukan sedikitnya dua gene pada setiap parent.")
    if len(first) != len(second):
        raise CrossoverError(f"Parent {operator_name} harus memiliki panjang yang sama.")
    for label, parent in (("parent_1", first), ("parent_2", second)):
        duplicate_alleles = sorted(allele for allele, count in Counter(parent).items() if count > 1)
        if duplicate_alleles:
            raise CrossoverError(
                f"{operator_name} {label} tidak boleh memuat allele internal duplikat: {duplicate_alleles}."
            )
    if Counter(first) != Counter(second):
        raise CrossoverError(
            f"Kedua parent {operator_name} harus memiliki exact allele multiset yang sama."
        )
    return first, second


def _validate_pmx_parents(
    parent_1: Iterable[Iterable[str]], parent_2: Iterable[Iterable[str]]
) -> tuple[MutableGenome, MutableGenome]:
    """Validate unique equal-inventory PMX parents and return normalized tuples."""
    return _validate_unique_permutation_parents(parent_1, parent_2, "PMX")


# PMX inherits segment_parent[left:right] and fills remaining alleles from fill_parent.
# Cut bounds are half-open locus boundaries, not fuel IDs.
def _build_pmx_child(
    segment_parent: MutableGenome, fill_parent: MutableGenome, left: int, right: int
) -> MutableGenome:
    """Return one PMX child, taking ``segment_parent[left:right]`` unchanged.

    Follow allele mappings in ``fill_parent`` to place the remaining genes without
    duplicates. The half-open bounds refer to mutable loci, not physical core slots."""
    gene_count = len(segment_parent)
    if not 0 <= left < right <= gene_count:
        raise CrossoverError("Cut PMX harus memenuhi 0 <= left < right <= mutable length.")

    child: list[Gene | None] = [None] * gene_count
    child[left:right] = segment_parent[left:right]
    copied_segment = set(segment_parent[left:right])
    position_in_fill = {gene: index for index, gene in enumerate(fill_parent)}

    for index in range(left, right):
        gene = fill_parent[index]
        if gene in copied_segment:
            continue
        position = index
        while left <= position < right:
            displaced_gene = segment_parent[position]
            position = position_in_fill[displaced_gene]
        if child[position] is not None:
            raise CrossoverError("PMX menghasilkan mapping collision yang tidak diharapkan.")
        child[position] = gene

    for index, gene in enumerate(fill_parent):
        if child[index] is None:
            child[index] = gene

    if any(gene is None for gene in child):
        raise CrossoverError("PMX meninggalkan locus child yang belum terisi.")
    normalized_child = tuple(gene for gene in child if gene is not None)
    if Counter(normalized_child) != Counter(segment_parent):
        raise CrossoverError("PMX gagal mempertahankan exact allele multiset.")
    return normalized_child


def pmx_crossover(
    parent_1: MutableGenome,
    parent_2: MutableGenome,
    *,
    rng: random.Random,
    context: CrossoverOperatorContext,
) -> CrossoverOperatorResult:
    """Return two reciprocal PMX children and their shared cut-bound metadata.

    ``parent_1`` and ``parent_2`` are equal-inventory mutable permutations. ``rng``
    draws two cut boundaries; ``context`` carries the dispatcher contract even
    though PMX does not use parent power tallies."""
    first, second = _validate_pmx_parents(parent_1, parent_2)
    if not isinstance(context, CrossoverOperatorContext):
        raise TypeError("context PMX harus berupa CrossoverOperatorContext.")
    # A full-length PMX segment is valid; an applied crossover need not produce a new genome.
    left, right = sorted(rng.sample(range(len(first) + 1), 2))
    child_1 = _build_pmx_child(first, second, left, right)
    child_2 = _build_pmx_child(second, first, left, right)
    return CrossoverOperatorResult(
        children=(child_1, child_2), metadata={"cut_boundaries": [left, right]}
    )


TRIGA_HCMX_POWER_MASK_METHOD = "triga_hcmx_power_mask"
TRIGA_HCMX_POWER_MASK_SCHEMA_VERSION = "triga_hcmx_power_mask/v1"
# This power-mask operator is a TRIGA heuristic, not an exact reproduction of published HCMX.
TRIGA_HCMX_POWER_MASK_SCIENTIFIC_STATUS = "TRIGA-specific heuristic inspired by secondary HCMX concepts; not an exact reproduction of Poon and Parks HCMX."
TRIGA_HCMX_POWER_NORMALIZATION = "relative_to_mean_fission_q_recoverable"
TRIGA_HCMX_TALLY_QUANTITY = "integrated_fission_q_recoverable_per_segment"
# Radial aggregation sums segment scores per fuel slot, rather than averaging power density.
TRIGA_HCMX_RADIAL_AGGREGATION = "sum_segment_scores_per_fuel_slot"
# Mutable nonfuel positions retain the donor layout; fixed positions are outside the genome.
TRIGA_HCMX_NONFUEL_POLICY = "inherit_donor_layout"
TRIGA_HCMX_REQUIRED_OPTIONS = frozenset(
    {
        "mask_size_min",
        "mask_size_max",
        "selection_alpha",
        "epsilon",
        "nonfuel_policy",
        "require_parent_power_tally",
        "require_convergence_passed",
        "expected_tally_name",
        "expected_tally_score",
    }
)


@dataclass(frozen=True)
class _TrigaHCMXParentData:
    """Contain one validated parent's canonical fuel loci and relative powers."""

    parent_id: str
    genome: MutableGenome
    fuel_loci: tuple[int, ...]
    fuel_slots: tuple[str, ...]
    fuel_powers: tuple[float, ...]


@dataclass(frozen=True)
class _TrigaHCMXPreparedData:
    """Contain validated HCMX inputs shared by applied and no-op paths."""

    parents: tuple[_TrigaHCMXParentData, _TrigaHCMXParentData]
    mask_size_min: int
    mask_size_max: int
    selection_alpha: float
    # Positive epsilon regularizes the weight when relative power equals one.
    epsilon: float
    expected_tally_name: str
    expected_tally_score: str


def _require_hcmx_positive_integer(name: str, value: Any) -> int:
    """Validate one positive HCMX integer option and return a built-in integer."""
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise CrossoverError(f"{name} harus berupa integer positif.")
    return int(value)


def _require_hcmx_positive_float(name: str, value: Any) -> float:
    """Validate one positive finite HCMX float option and return a built-in float."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise CrossoverError(f"{name} harus berupa angka finite positif.")
    return float(value)


def _require_hcmx_string(name: str, value: Any) -> str:
    """Validate one non-empty HCMX string option and return its unchanged value."""
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise CrossoverError(f"{name} harus berupa string non-kosong tanpa whitespace tepi.")
    return value


def _validate_hcmx_options(
    options: Mapping[str, Any], fuel_count: int
) -> tuple[int, int, float, float, str, str]:
    """Validate the complete strategy-specific HCMX option set for a fuel count."""
    if not isinstance(options, Mapping):
        raise CrossoverError("operator_options triga_hcmx_power_mask harus berupa mapping.")
    missing_options = TRIGA_HCMX_REQUIRED_OPTIONS.difference(options)
    unknown_options = set(options).difference(TRIGA_HCMX_REQUIRED_OPTIONS)
    if missing_options:
        raise CrossoverError(
            f"Option triga_hcmx_power_mask belum lengkap: {sorted(missing_options)}."
        )
    if unknown_options:
        raise CrossoverError(
            f"Option triga_hcmx_power_mask tidak dikenal: {sorted(unknown_options)}."
        )

    mask_size_min = _require_hcmx_positive_integer("mask_size_min", options["mask_size_min"])
    mask_size_max = _require_hcmx_positive_integer("mask_size_max", options["mask_size_max"])
    selection_alpha = _require_hcmx_positive_float("selection_alpha", options["selection_alpha"])
    epsilon = _require_hcmx_positive_float("epsilon", options["epsilon"])
    nonfuel_policy = _require_hcmx_string("nonfuel_policy", options["nonfuel_policy"])
    expected_tally_name = _require_hcmx_string(
        "expected_tally_name", options["expected_tally_name"]
    )
    expected_tally_score = _require_hcmx_string(
        "expected_tally_score", options["expected_tally_score"]
    )
    if mask_size_min > mask_size_max:
        raise CrossoverError("mask_size_min tidak boleh melebihi mask_size_max.")
    if mask_size_max > fuel_count:
        raise CrossoverError(
            f"mask_size_max tidak boleh melebihi jumlah mutable fuel loci ({fuel_count})."
        )
    if nonfuel_policy != TRIGA_HCMX_NONFUEL_POLICY:
        raise CrossoverError(
            f"nonfuel_policy triga_hcmx_power_mask harus {TRIGA_HCMX_NONFUEL_POLICY!r}."
        )
    if options["require_parent_power_tally"] is not True:
        raise CrossoverError("require_parent_power_tally harus true untuk triga_hcmx_power_mask.")
    if options["require_convergence_passed"] is not True:
        raise CrossoverError("require_convergence_passed harus true untuk triga_hcmx_power_mask.")
    return (
        mask_size_min,
        mask_size_max,
        selection_alpha,
        epsilon,
        expected_tally_name,
        expected_tally_score,
    )


def _validate_hcmx_parent_result(
    parent_id: str,
    genome: MutableGenome,
    included_positions: tuple[str, ...],
    results: Any,
    expected_tally_name: str,
    expected_tally_score: str,
) -> _TrigaHCMXParentData:
    """Validate one parent's convergence, tally provenance, slot map, and powers."""
    if not isinstance(results, Mapping):
        raise CrossoverError(f"Parent {parent_id!r} tidak memiliki heuristic results mapping.")
    convergence = results.get("convergence")
    if not isinstance(convergence, Mapping) or convergence.get("passed") is not True:
        raise CrossoverError(f"Parent {parent_id!r} harus memiliki convergence.passed is True.")
    metadata = results.get("metadata")
    if not isinstance(metadata, Mapping):
        raise CrossoverError(f"Parent {parent_id!r} tidak memiliki metadata tally yang valid.")
    if metadata.get("tally_name") != expected_tally_name:
        raise CrossoverError(
            f"Tally name parent {parent_id!r} tidak sama dengan expected campaign tally {expected_tally_name!r}."
        )
    if metadata.get("tally_score") != expected_tally_score:
        raise CrossoverError(
            f"Tally score parent {parent_id!r} tidak sama dengan expected campaign score {expected_tally_score!r}."
        )
    if metadata.get("power_normalization") != TRIGA_HCMX_POWER_NORMALIZATION:
        raise CrossoverError(
            f"Power normalization parent {parent_id!r} tidak kompatibel dengan triga_hcmx_power_mask."
        )
    if (
        metadata.get("tally_quantity") != TRIGA_HCMX_TALLY_QUANTITY
        or metadata.get("radial_ppf_aggregation") != TRIGA_HCMX_RADIAL_AGGREGATION
    ):
        raise CrossoverError(
            f"Tally quantity atau radial aggregation parent {parent_id!r} tidak kompatibel."
        )
    if (
        metadata.get("included_slot_types") != ["fuel"]
        or metadata.get("ffcr_included") is not False
    ):
        raise CrossoverError(
            f"Tally parent {parent_id!r} harus hanya memuat mutable fuel dan mengecualikan FFCR."
        )

    fuel_loci = tuple(index for index, gene in enumerate(genome) if gene[0] == "fuel")
    fuel_slots = tuple(included_positions[index] for index in fuel_loci)
    expected_fuel_id_by_slot = {included_positions[index]: genome[index][1] for index in fuel_loci}
    power_by_slot = results.get("power_by_slot")
    fuel_id_by_slot = results.get("fuel_id_by_slot")
    if not isinstance(power_by_slot, Mapping) or not isinstance(fuel_id_by_slot, Mapping):
        raise CrossoverError(
            f"Parent {parent_id!r} harus memiliki power_by_slot dan fuel_id_by_slot mapping."
        )
    expected_slots = set(fuel_slots)
    if set(power_by_slot) != expected_slots:
        missing = sorted(expected_slots.difference(power_by_slot))
        extra = sorted(map(str, set(power_by_slot).difference(expected_slots)))
        raise CrossoverError(
            f"power_by_slot parent {parent_id!r} tidak sejajar dengan mutable fuel slots; missing={missing}, extra={extra}."
        )
    if set(fuel_id_by_slot) != expected_slots:
        missing = sorted(expected_slots.difference(fuel_id_by_slot))
        extra = sorted(map(str, set(fuel_id_by_slot).difference(expected_slots)))
        raise CrossoverError(
            f"fuel_id_by_slot parent {parent_id!r} tidak sejajar dengan mutable fuel slots; missing={missing}, extra={extra}."
        )
    if dict(fuel_id_by_slot) != expected_fuel_id_by_slot:
        # Reject parent tally data from a different loading even when physical slot names match.
        raise CrossoverError(
            f"fuel_id_by_slot parent {parent_id!r} tidak sesuai dengan actual parent mutable genome."
        )
    fuel_slot_count = metadata.get("fuel_slot_count")
    if (
        isinstance(fuel_slot_count, bool)
        or not isinstance(fuel_slot_count, Integral)
        or int(fuel_slot_count) != len(fuel_slots)
    ):
        raise CrossoverError(
            f"fuel_slot_count parent {parent_id!r} tidak sesuai dengan mutable fuel loci."
        )

    fuel_powers = []
    for slot_name in fuel_slots:
        power_value = power_by_slot[slot_name]
        if isinstance(power_value, bool) or not isinstance(power_value, Real):
            raise CrossoverError(
                f"power_by_slot[{slot_name!r}] parent {parent_id!r} harus numerik dan bukan bool."
            )
        normalized_power = float(power_value)
        if not math.isfinite(normalized_power) or normalized_power < 0.0:
            raise CrossoverError(
                f"power_by_slot[{slot_name!r}] parent {parent_id!r} harus finite dan non-negatif."
            )
        fuel_powers.append(normalized_power)
    return _TrigaHCMXParentData(
        parent_id=parent_id,
        genome=genome,
        fuel_loci=fuel_loci,
        fuel_slots=fuel_slots,
        fuel_powers=tuple(fuel_powers),
    )


# Prepare and validate both parents before consuming any random draws.
def _prepare_triga_hcmx_power_mask(
    parent_1: MutableGenome, parent_2: MutableGenome, context: CrossoverOperatorContext
) -> _TrigaHCMXPreparedData:
    """Validate all deterministic HCMX inputs before any mating RNG is consumed."""
    first, second = _validate_unique_permutation_parents(
        parent_1, parent_2, TRIGA_HCMX_POWER_MASK_METHOD
    )
    if not isinstance(context, CrossoverOperatorContext):
        raise TypeError("context triga_hcmx_power_mask harus berupa CrossoverOperatorContext.")
    if len(context.parent_ids) != 2 or context.parent_ids[0] == context.parent_ids[1]:
        raise CrossoverError("triga_hcmx_power_mask memerlukan dua parent ID berbeda.")
    if len(context.included_positions) != len(first) or len(set(context.included_positions)) != len(
        first
    ):
        raise CrossoverError(
            "included_positions HCMX harus unik dan sejajar dengan mutable genome."
        )
    if any(
        not isinstance(position, str) or not position for position in context.included_positions
    ):
        raise CrossoverError("included_positions HCMX harus berisi nama posisi non-kosong.")
    if not isinstance(context.parent_results, tuple) or len(context.parent_results) != 2:
        raise CrossoverError("parent_results HCMX harus berisi tepat dua hasil evaluator.")

    fuel_count = sum(gene[0] == "fuel" for gene in first)
    if fuel_count <= 0:
        raise CrossoverError(
            "triga_hcmx_power_mask memerlukan sedikitnya satu mutable fuel allele."
        )
    (
        mask_size_min,
        mask_size_max,
        selection_alpha,
        epsilon,
        expected_tally_name,
        expected_tally_score,
    ) = _validate_hcmx_options(context.operator_options, fuel_count)
    first_data = _validate_hcmx_parent_result(
        context.parent_ids[0],
        first,
        context.included_positions,
        context.parent_results[0],
        expected_tally_name,
        expected_tally_score,
    )
    second_data = _validate_hcmx_parent_result(
        context.parent_ids[1],
        second,
        context.included_positions,
        context.parent_results[1],
        expected_tally_name,
        expected_tally_score,
    )
    return _TrigaHCMXPreparedData(
        parents=(first_data, second_data),
        mask_size_min=mask_size_min,
        mask_size_max=mask_size_max,
        selection_alpha=selection_alpha,
        epsilon=epsilon,
        expected_tally_name=expected_tally_name,
        expected_tally_score=expected_tally_score,
    )


def _draw_unit_interval(rng: random.Random, value_name: str) -> float:
    """Draw and validate one injected-RNG value in the half-open unit interval."""
    draw = float(rng.random())
    if not math.isfinite(draw) or not 0.0 <= draw < 1.0:
        raise CrossoverError(f"{value_name} dari injected RNG harus berada dalam [0, 1).")
    return draw


# Weights favor near-unity donor power: w_i = (epsilon + abs(power_i - 1))**(-alpha).
# Relative power is dimensionless; inheritance weights do not predict offspring fitness.
def _weighted_sample_hcmx_loci(
    parent_data: _TrigaHCMXParentData,
    mask_size: int,
    selection_alpha: float,
    epsilon: float,
    rng: random.Random,
) -> tuple[int, ...]:
    """Return ``mask_size`` distinct donor loci from ``parent_data`` using ``rng``.

    ``selection_alpha`` controls the preference for near-unity relative power;
    positive ``epsilon`` keeps the power-based weights finite."""
    remaining = [
        (locus, -selection_alpha * math.log(epsilon + abs(power - 1.0)))
        for locus, power in zip(parent_data.fuel_loci, parent_data.fuel_powers, strict=True)
    ]
    selected = []
    for selection_index in range(mask_size):
        largest_log_weight = max(log_weight for _, log_weight in remaining)
        # Subtract the maximum log-weight before exponentiation to preserve ratios without overflow.
        scaled_weights = [math.exp(log_weight - largest_log_weight) for _, log_weight in remaining]
        total_weight = math.fsum(scaled_weights)
        threshold = _draw_unit_interval(rng, f"HCMX weighted draw {selection_index}") * total_weight
        cumulative_weight = 0.0
        selected_remaining_index = len(remaining) - 1
        for remaining_index, scaled_weight in enumerate(scaled_weights):
            cumulative_weight += scaled_weight
            if threshold < cumulative_weight:
                selected_remaining_index = remaining_index
                break
        # Remove each chosen locus so the power mask is sampled without replacement.
        selected_locus, _ = remaining.pop(selected_remaining_index)
        selected.append(selected_locus)
    return tuple(selected)


def _build_triga_hcmx_child(
    donor: _TrigaHCMXParentData,
    recipient: _TrigaHCMXParentData,
    mask_size: int,
    selection_alpha: float,
    epsilon: float,
    included_positions: tuple[str, ...],
    rng: random.Random,
    child_index: int,
) -> tuple[MutableGenome, dict[str, Any]]:
    """Build one donor-layout HCMX child and return its auditable mask metadata."""
    selected_loci_in_draw_order = _weighted_sample_hcmx_loci(
        donor, mask_size, selection_alpha, epsilon, rng
    )
    selected_loci = tuple(sorted(selected_loci_in_draw_order))
    selected_locus_set = set(selected_loci)
    child: list[Gene | None] = [None] * len(donor.genome)
    used_alleles = set()

    for locus, allele in enumerate(donor.genome):
        if allele[0] != "fuel" or locus in selected_locus_set:
            child[locus] = allele
            used_alleles.add(allele)

    remaining_fuel_alleles = tuple(
        allele for allele in recipient.genome if allele[0] == "fuel" and allele not in used_alleles
    )
    remaining_fuel_loci = tuple(
        locus for locus in donor.fuel_loci if locus not in selected_locus_set
    )
    if len(remaining_fuel_alleles) != len(remaining_fuel_loci):
        raise CrossoverError(
            "HCMX deterministic fuel fill memiliki cardinality yang tidak konsisten."
        )
    for locus, allele in zip(remaining_fuel_loci, remaining_fuel_alleles, strict=True):
        child[locus] = allele
        used_alleles.add(allele)

    if any(allele is None for allele in child):
        raise CrossoverError("HCMX meninggalkan locus child yang belum terisi.")
    normalized_child = tuple(allele for allele in child if allele is not None)
    if Counter(normalized_child) != Counter(donor.genome):
        raise CrossoverError("HCMX gagal mempertahankan exact allele multiset by construction.")
    power_by_locus = dict(zip(donor.fuel_loci, donor.fuel_powers, strict=True))
    child_metadata = {
        "child_index": child_index,
        "donor_parent_id": donor.parent_id,
        "recipient_parent_id": recipient.parent_id,
        "child_source": TRIGA_HCMX_POWER_MASK_METHOD,
        "selected_mask_size": mask_size,
        "selected_donor_fuel_loci": list(selected_loci),
        "selected_donor_fuel_slots": [included_positions[locus] for locus in selected_loci],
        # Record donor power at inherited loci; these are parent values, not simulated child power.
        "selected_donor_relative_powers": [power_by_locus[locus] for locus in selected_loci],
    }
    return normalized_child, child_metadata


def _triga_hcmx_metadata(
    prepared: _TrigaHCMXPreparedData, child_metadata: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build common JSON-safe scientific and heuristic metadata for HCMX results."""
    return {
        "cut_boundaries": None,
        "operator_schema_version": TRIGA_HCMX_POWER_MASK_SCHEMA_VERSION,
        "scientific_status": TRIGA_HCMX_POWER_MASK_SCIENTIFIC_STATUS,
        "selection_alpha": prepared.selection_alpha,
        "epsilon": prepared.epsilon,
        "heuristic": {
            "name": "parent_integrated_relative_radial_power_near_unity",
            "quantity": "integrated_relative_radial_power",
            # Relative power is dimensionless, not watts or volumetric power density.
            "units": "dimensionless_relative_power",
            "tally_name": prepared.expected_tally_name,
            "tally_score": prepared.expected_tally_score,
            "weight_equation": "(epsilon + abs(power_i - 1.0)) ** (-selection_alpha)",
        },
        "convergence_required": True,
        "parent_convergence_results": [
            {"parent_id": parent.parent_id, "passed": True} for parent in prepared.parents
        ],
        "nonfuel_policy": TRIGA_HCMX_NONFUEL_POLICY,
        "inventory_validation_result": "passed_exact_configured_allele_multiset",
        "children": child_metadata,
    }


def triga_hcmx_power_mask_crossover(
    parent_1: MutableGenome,
    parent_2: MutableGenome,
    *,
    rng: random.Random,
    context: CrossoverOperatorContext,
) -> CrossoverOperatorResult:
    """Return reciprocal children from the approved TRIGA power-mask heuristic."""
    prepared = context.prepared_data if isinstance(context, CrossoverOperatorContext) else None
    if not isinstance(prepared, _TrigaHCMXPreparedData):
        prepared = _prepare_triga_hcmx_power_mask(parent_1, parent_2, context)
    # A fixed mask size consumes no integer draw; randint includes both configured bounds.
    mask_size_1 = (
        prepared.mask_size_min
        if prepared.mask_size_min == prepared.mask_size_max
        else rng.randint(prepared.mask_size_min, prepared.mask_size_max)
    )
    mask_size_2 = (
        prepared.mask_size_min
        if prepared.mask_size_min == prepared.mask_size_max
        else rng.randint(prepared.mask_size_min, prepared.mask_size_max)
    )
    if (
        isinstance(mask_size_1, bool)
        or not isinstance(mask_size_1, Integral)
        or isinstance(mask_size_2, bool)
        or not isinstance(mask_size_2, Integral)
    ):
        raise CrossoverError("Injected RNG harus menghasilkan integer untuk HCMX mask size.")
    mask_size_1 = int(mask_size_1)
    mask_size_2 = int(mask_size_2)
    if (
        not prepared.mask_size_min <= mask_size_1 <= prepared.mask_size_max
        or not prepared.mask_size_min <= mask_size_2 <= prepared.mask_size_max
    ):
        raise CrossoverError("Injected RNG menghasilkan HCMX mask size di luar configured range.")
    child_1, metadata_1 = _build_triga_hcmx_child(
        prepared.parents[0],
        prepared.parents[1],
        mask_size_1,
        prepared.selection_alpha,
        prepared.epsilon,
        context.included_positions,
        rng,
        0,
    )
    child_2, metadata_2 = _build_triga_hcmx_child(
        prepared.parents[1],
        prepared.parents[0],
        mask_size_2,
        prepared.selection_alpha,
        prepared.epsilon,
        context.included_positions,
        rng,
        1,
    )
    return CrossoverOperatorResult(
        children=(child_1, child_2),
        metadata=_triga_hcmx_metadata(prepared, [metadata_1, metadata_2]),
    )


def _triga_hcmx_skipped_metadata(context: CrossoverOperatorContext) -> Mapping[str, Any]:
    """Return HCMX audit metadata for validated immutable parent clones."""
    prepared = context.prepared_data
    # Even skipped HCMX validates the complete parent tally context.
    if not isinstance(prepared, _TrigaHCMXPreparedData):
        raise CrossoverError("Prepared HCMX data tidak tersedia pada no-op path.")
    child_metadata = []
    for child_index, (donor, recipient) in enumerate(
        ((prepared.parents[0], prepared.parents[1]), (prepared.parents[1], prepared.parents[0]))
    ):
        # Skipped crossover creates parent clones and consumes no mask draws.
        child_metadata.append(
            {
                "child_index": child_index,
                "donor_parent_id": donor.parent_id,
                "recipient_parent_id": recipient.parent_id,
                "child_source": "unchanged_parent_clone",
                "selected_mask_size": 0,
                "selected_donor_fuel_loci": [],
                "selected_donor_fuel_slots": [],
                "selected_donor_relative_powers": [],
            }
        )
    return _triga_hcmx_metadata(prepared, child_metadata)


DEFAULT_CROSSOVER_METHODS: Mapping[str, CrossoverMethodDefinition] = MappingProxyType(
    {
        "pmx": CrossoverMethodDefinition(operator=pmx_crossover),
        TRIGA_HCMX_POWER_MASK_METHOD: CrossoverMethodDefinition(
            operator=triga_hcmx_power_mask_crossover,
            prepare=_prepare_triga_hcmx_power_mask,
            skipped_metadata=_triga_hcmx_skipped_metadata,
        ),
    }
)
IMPLEMENTED_CROSSOVER_METHODS = frozenset(DEFAULT_CROSSOVER_METHODS)


class CrossoverRegistry:
    """Resolve dedicated crossover callables by stable configured method name."""

    def __init__(
        self, methods: Mapping[str, CrossoverMethod | CrossoverMethodDefinition] | None = None
    ) -> None:
        """Copy optional ``methods`` into an independent crossover registry; return None."""
        if methods is not None and not isinstance(methods, Mapping):
            raise TypeError("methods crossover harus berupa mapping nama ke callable.")
        self._methods: dict[str, CrossoverMethodDefinition] = {}
        for name, method_or_definition in (methods or {}).items():
            if isinstance(method_or_definition, CrossoverMethodDefinition):
                self.register(
                    name,
                    method_or_definition.operator,
                    prepare=method_or_definition.prepare,
                    skipped_metadata=method_or_definition.skipped_metadata,
                )
            else:
                self.register(name, method_or_definition)

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered method names in deterministic lexicographic order."""
        return tuple(sorted(self._methods))

    def register(
        self,
        name: str,
        method: CrossoverMethod,
        *,
        prepare: CrossoverPreparation | None = None,
        skipped_metadata: CrossoverSkippedMetadata | None = None,
    ) -> None:
        """Register one dedicated operator under a unique non-empty strategy name."""
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ValueError(
                "Nama method crossover harus berupa string non-kosong tanpa whitespace tepi."
            )
        if not callable(method):
            raise TypeError(f"Operator crossover {name!r} harus callable.")
        if prepare is not None and not callable(prepare):
            raise TypeError(f"Preflight crossover {name!r} harus callable atau None.")
        if skipped_metadata is not None and not callable(skipped_metadata):
            raise TypeError(f"Skipped metadata crossover {name!r} harus callable atau None.")
        if name in self._methods:
            raise ValueError(f"Method crossover {name!r} sudah terdaftar.")
        self._methods[name] = CrossoverMethodDefinition(
            operator=method, prepare=prepare, skipped_metadata=skipped_metadata
        )

    def resolve(self, name: str) -> CrossoverMethod:
        """Return the dedicated operator for ``name`` or reject it explicitly."""
        return self.resolve_definition(name).operator

    def resolve_definition(self, name: str) -> CrossoverMethodDefinition:
        """Return the operator definition and its optional lifecycle hooks."""
        if name not in self._methods:
            raise UnknownCrossoverMethodError(
                f"Method crossover {name!r} belum diimplementasikan; tersedia={list(self.names)}."
            )
        return self._methods[name]


def build_default_crossover_registry() -> CrossoverRegistry:
    """Return the production registry containing all implemented operators."""
    return CrossoverRegistry(DEFAULT_CROSSOVER_METHODS)


class CrossoverDispatcher:
    """Validate checkpoint parents, dispatch one operator, and validate children."""

    def __init__(
        self,
        method: str,
        probability: float,
        registry: CrossoverRegistry,
        mutable_genome_contract: MutableGenomeContract,
        genome_sorter: GenomeSorter[Gene],
        rng: random.Random,
        operator_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Configure crossover ``method`` and per-pair ``probability``; return None.

        ``registry`` resolves the operator. ``mutable_genome_contract`` validates allele
        inventory, ``genome_sorter`` checks fixed-position insertion, and ``rng`` is the
        shared campaign generator. ``operator_options`` holds method-specific settings."""
        if not isinstance(registry, CrossoverRegistry):
            raise TypeError("registry harus berupa CrossoverRegistry.")
        if not isinstance(mutable_genome_contract, MutableGenomeContract):
            raise TypeError("mutable_genome_contract harus berupa MutableGenomeContract.")
        if not isinstance(genome_sorter, GenomeSorter):
            raise TypeError("genome_sorter harus berupa GenomeSorter.")
        if mutable_genome_contract.included_length != genome_sorter.included_length:
            raise CrossoverError(
                "Panjang MutableGenomeContract tidak sama dengan included positions GenomeSorter."
            )
        # Pc is dimensionless and applies per parent pair.
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(float(probability))
            or not 0.0 <= float(probability) <= 1.0
        ):
            raise ValueError("probability crossover harus finite dalam rentang [0, 1].")
        required_rng_methods = ("random", "sample", "randint", "getstate", "setstate")
        if any(
            not callable(getattr(rng, method_name, None)) for method_name in required_rng_methods
        ):
            raise TypeError(
                "rng harus checkpoint-compatible dan menyediakan random(), sample(), randint(), getstate(), serta setstate()."
            )
        if operator_options is not None and not isinstance(operator_options, Mapping):
            raise TypeError("operator_options harus berupa mapping.")

        self.method = method
        self.probability = float(probability)
        self.registry = registry
        self.mutable_genome_contract = mutable_genome_contract
        self.genome_sorter = genome_sorter
        self.rng = rng
        self.operator_options = deepcopy(dict(operator_options or {}))
        self._definition = self.registry.resolve_definition(method)
        self._operator = self._definition.operator

    def crossover(
        self, parent_ids: Iterable[str], checkpoint_records: Mapping[str, Mapping[str, Any]]
    ) -> CrossoverResult:
        """Resolve two completed checkpoint parents and return validated mutable children."""
        normalized_parent_ids = self._normalize_parent_ids(parent_ids)
        if not isinstance(checkpoint_records, Mapping):
            raise TypeError("checkpoint_records harus berupa mapping individual ID ke record.")

        parent_records = []
        parents = []
        for parent_id in normalized_parent_ids:
            if parent_id not in checkpoint_records:
                raise CrossoverError(
                    f"Parent ID {parent_id!r} tidak ditemukan pada checkpoint records."
                )
            record = checkpoint_records[parent_id]
            if not isinstance(record, Mapping):
                raise TypeError(f"Record parent {parent_id!r} harus berupa mapping.")
            if record.get("status") != "completed":
                raise CrossoverError(
                    f"Parent {parent_id!r} harus berstatus completed untuk crossover."
                )
            if "genome" not in record:
                raise CrossoverError(f"Record parent {parent_id!r} tidak memiliki mutable genome.")
            try:
                parent = self.mutable_genome_contract.validate(
                    record["genome"], f"Genome parent {parent_id!r}"
                )
            except MutableGenomeValidationError as error:
                raise CrossoverError(str(error)) from error
            parent_records.append(record)
            parents.append(parent)

        context = CrossoverOperatorContext(
            parent_ids=normalized_parent_ids,
            included_positions=tuple(self.genome_sorter.included_positions),
            parent_results=(
                deepcopy(parent_records[0].get("results")),
                deepcopy(parent_records[1].get("results")),
            ),
            operator_options=deepcopy(self.operator_options),
        )
        if self._definition.prepare is not None:
            context = replace(
                context, prepared_data=self._definition.prepare(parents[0], parents[1], context)
            )

        probability_draw = _draw_unit_interval(self.rng, "Crossover probability draw")
        # Apply when u < Pc; valid RNG draws make Pc=0 always skip and Pc=1 always apply.
        applied = probability_draw < self.probability
        if applied:
            operator_result = self._operator(parents[0], parents[1], rng=self.rng, context=context)
            if not isinstance(operator_result, CrossoverOperatorResult):
                raise CrossoverError(
                    f"Operator {self.method!r} harus mengembalikan CrossoverOperatorResult."
                )
            raw_children = operator_result.children
            if not isinstance(operator_result.metadata, Mapping):
                raise CrossoverError(f"Metadata operator {self.method!r} harus berupa mapping.")
            operator_metadata = dict(operator_result.metadata)
        else:
            # Skipped crossover clones both parents; the scheduler still applies mutation to each child.
            raw_children = (tuple(gene for gene in parents[0]), tuple(gene for gene in parents[1]))
            operator_metadata = (
                dict(self._definition.skipped_metadata(context))
                if self._definition.skipped_metadata is not None
                else {"cut_boundaries": None}
            )

        if not isinstance(raw_children, tuple) or len(raw_children) != 2:
            raise CrossoverError(f"Operator {self.method!r} harus menghasilkan tepat dua child.")
        # Validate inventory and complete/mutable round trips for both children.
        children = (
            self._validate_child(raw_children[0], 0),
            self._validate_child(raw_children[1], 1),
        )
        reserved_metadata_keys = {
            "method",
            "applied",
            "parent_ids",
            "probability",
            "probability_draw",
        }
        conflicting_metadata_keys = reserved_metadata_keys.intersection(operator_metadata)
        if conflicting_metadata_keys:
            raise CrossoverError(
                f"Metadata operator {self.method!r} memakai key milik dispatcher: {sorted(conflicting_metadata_keys)}."
            )
        metadata = {
            "method": self.method,
            "applied": applied,
            "cut_boundaries": operator_metadata.pop("cut_boundaries", None),
            "parent_ids": list(normalized_parent_ids),
            "probability": self.probability,
            "probability_draw": probability_draw,
            **operator_metadata,
        }
        try:
            # Strict JSON rejects NaN, infinity, and nonserializable operator metadata.
            json.dumps(metadata, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise CrossoverError(f"Metadata operator {self.method!r} harus JSON-safe.") from error
        return CrossoverResult(children=children, metadata=metadata)

    @staticmethod
    def _normalize_parent_ids(parent_ids: Iterable[str]) -> tuple[str, str]:
        """Validate exactly two distinct parent IDs and return an immutable pair."""
        if isinstance(parent_ids, (str, bytes, Mapping)):
            raise TypeError("parent_ids harus berupa iterable berisi tepat dua ID.")
        try:
            normalized = tuple(parent_ids)
        except TypeError as error:
            raise TypeError("parent_ids harus berupa iterable berisi tepat dua ID.") from error
        if len(normalized) != 2:
            raise CrossoverError(
                f"Crossover memerlukan tepat dua parent ID, menerima {len(normalized)}."
            )
        if any(
            not isinstance(parent_id, str)
            or not parent_id.strip()
            or parent_id != parent_id.strip()
            for parent_id in normalized
        ):
            raise CrossoverError(
                "Setiap parent ID harus berupa string non-kosong tanpa whitespace tepi."
            )
        # Parent IDs must differ, although their genomes may match.
        if normalized[0] == normalized[1]:
            raise CrossoverError("Crossover memerlukan dua parent ID yang berbeda.")
        return normalized[0], normalized[1]

    def _validate_child(self, child: Iterable[Iterable[str]], child_index: int) -> MutableGenome:
        """Validate one child inventory and its fixed-position construct/extract round trip."""
        try:
            normalized = self.mutable_genome_contract.validate(
                child, f"Child crossover indeks {child_index}"
            )
        except MutableGenomeValidationError as error:
            raise CrossoverError(str(error)) from error
        complete = self.genome_sorter.construct_complete_genome(normalized)
        extracted = self.genome_sorter.extract_included_genome(complete)
        if extracted != normalized:
            raise CrossoverError(
                f"Child crossover indeks {child_index} gagal construct/extract round trip."
            )
        return normalized
