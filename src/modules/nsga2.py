"""Rank candidates and select survivors with constrained NSGA-II.

Completed candidates are ranked by feasibility, nondominated sorting, and
crowding distance. Environmental selection retains exactly N survivors from
parents and offspring. Both objectives are minimized as (-keff, ppf_3d).
All operations use in-memory data without transport or file I/O."""

# keff, PPF, and normalized crowding distance are dimensionless.
# Compare keff feasibility before Pareto objectives.
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

from modules.keff_constraint import KeffConstraint


OBJECTIVE_COUNT = 2
ELIGIBLE_STATUS = "completed"


class NSGA2Error(ValueError):
    """Base error for invalid NSGA-II inputs or selection requirements."""


class InvalidObjectivesError(NSGA2Error):
    """Raised when an objective vector is not exactly two finite numbers."""


class InsufficientEligibleCandidatesError(NSGA2Error):
    """Raised when eligible parents and offspring cannot fill the target population."""


@dataclass(frozen=True)
class NSGA2Candidate:
    """Represent one eligible individual using its stable ID and raw objectives."""

    individual_id: str
    objectives: tuple[float, float]
    constraint_contract: KeffConstraint
    feasible: bool = field(init=False)
    keff_violation: float = field(init=False)

    def __post_init__(self) -> None:
        """Validate the ID and normalize the two minimization objectives in place."""
        object.__setattr__(self, "individual_id", _validate_individual_id(self.individual_id))
        object.__setattr__(
            self,
            "objectives",
            validate_objectives(self.objectives, f"objectives {self.individual_id}"),
        )
        constraint = self.constraint_contract.evaluate(-self.objectives[0])
        object.__setattr__(self, "feasible", constraint["feasible"])
        object.__setattr__(self, "keff_violation", constraint["keff_violation"])

    @property
    def constraint(self) -> dict[str, Any]:
        """Return metadata from the mean-keff objective under the injected contract."""
        return self.constraint_contract.evaluate(-self.objectives[0])


@dataclass(frozen=True)
# Keep boundary flags separate so strict JSON need not store infinity.
class CrowdingMetric:
    """Store one JSON-safe crowding result for a candidate within one front."""

    individual_id: str
    crowding_distance: float | None
    is_boundary: bool


@dataclass(frozen=True)
class RankedCandidate:
    """Store one candidate together with its nondominated rank and crowding result."""

    individual_id: str
    # Crowding normalization never changes the raw objectives (-keff, ppf_3d).
    objectives: tuple[float, float]
    rank: int
    crowding_distance: float | None
    is_boundary: bool


@dataclass(frozen=True)
class NSGA2Ranking:
    """Contain deterministic fronts and metrics without mutating source records."""

    fronts: tuple[tuple[str, ...], ...]
    ranked_candidates: tuple[RankedCandidate, ...]

    def metrics_by_id(self) -> dict[str, RankedCandidate]:
        """Return a new ID-to-metrics mapping for lookup or checkpoint persistence."""
        return {candidate.individual_id: candidate for candidate in self.ranked_candidates}


@dataclass(frozen=True)
class EnvironmentalSelectionResult:
    """Contain the exact survivors and combined parent-offspring ranking."""

    survivor_ids: tuple[str, ...]
    eligible_parent_ids: tuple[str, ...]
    eligible_offspring_ids: tuple[str, ...]
    ranking: NSGA2Ranking


def _validate_individual_id(individual_id: Any) -> str:
    """Validate and return a non-empty stable individual identifier."""
    if not isinstance(individual_id, str) or not individual_id.strip():
        raise NSGA2Error("individual_id harus berupa string non-kosong.")
    if individual_id != individual_id.strip():
        raise NSGA2Error("individual_id tidak boleh memiliki whitespace di tepi.")
    return individual_id


def validate_objectives(
    objectives: Iterable[Real], value_name: str = "objectives"
) -> tuple[float, float]:
    """Validate exactly two finite raw minimization objectives and return floats.

    Parameters
    ----------
    objectives : iterable of real
        Expected raw objective vector ``(-keff, ppf_3d)``.
    value_name : str, optional
        Field label included in validation errors.

    Returns
    -------
    tuple[float, float]
        The unchanged objective values normalized to built-in floats."""
    if isinstance(objectives, (str, bytes, Mapping)):
        raise InvalidObjectivesError(
            f"{value_name} harus berupa iterable berisi tepat dua angka finite."
        )
    try:
        objective_values = tuple(objectives)
    except TypeError as error:
        raise InvalidObjectivesError(
            f"{value_name} harus berupa iterable berisi tepat dua angka finite."
        ) from error
    if len(objective_values) != OBJECTIVE_COUNT:
        raise InvalidObjectivesError(
            f"{value_name} harus berisi tepat dua objective, menerima {len(objective_values)}."
        )

    normalized = []
    for index, value in enumerate(objective_values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise InvalidObjectivesError(
                f"{value_name}[{index}] harus berupa angka real dan bukan bool."
            )
        # Convert numeric types without weighting or rescaling the objectives.
        normalized_value = float(value)
        if not math.isfinite(normalized_value):
            raise InvalidObjectivesError(f"{value_name}[{index}] harus finite.")
        normalized.append(normalized_value)
    return normalized[0], normalized[1]


def dominates(first: NSGA2Candidate, second: NSGA2Candidate) -> bool:
    """Return constrained dominance for candidates ``first`` and ``second`` under one contract."""
    if first.constraint_contract != second.constraint_contract:
        # Mixed feasibility contracts would make ranks depend on inconsistent acceptance rules.
        raise NSGA2Error("Kandidat memakai kontrak constraint yang berbeda.")
    if first.feasible != second.feasible:
        return first.feasible
    if not first.feasible:
        return first.keff_violation < second.keff_violation
    # Use ordinary two-objective Pareto dominance only after feasibility comparison.
    return _dominates_validated(first.objectives, second.objectives)


def _dominates_validated(first: tuple[float, float], second: tuple[float, float]) -> bool:
    """Compare two already validated minimization vectors and return dominance."""
    no_worse = first[0] <= second[0] and first[1] <= second[1]
    strictly_better = first[0] < second[0] or first[1] < second[1]
    return no_worse and strictly_better


def _materialize_candidates(
    candidates: Iterable[NSGA2Candidate], value_name: str
) -> tuple[NSGA2Candidate, ...]:
    """Validate an iterable of unique candidates and return canonical ID order."""
    if isinstance(candidates, (str, bytes, Mapping)):
        raise TypeError(f"{value_name} harus berupa iterable NSGA2Candidate.")
    try:
        materialized = tuple(candidates)
    except TypeError as error:
        raise TypeError(f"{value_name} harus berupa iterable NSGA2Candidate.") from error
    if any(not isinstance(candidate, NSGA2Candidate) for candidate in materialized):
        raise TypeError(f"Setiap item {value_name} harus berupa NSGA2Candidate.")

    if len({candidate.constraint_contract for candidate in materialized}) > 1:
        raise NSGA2Error("Kandidat memakai kontrak constraint yang berbeda.")
    individual_ids = [candidate.individual_id for candidate in materialized]
    duplicate_ids = sorted(
        individual_id for individual_id, count in Counter(individual_ids).items() if count > 1
    )
    if duplicate_ids:
        raise NSGA2Error("individual_id duplikat: " + ", ".join(duplicate_ids) + ".")
    return tuple(sorted(materialized, key=lambda candidate: candidate.individual_id))


def extract_eligible_candidates(
    individuals: Mapping[str, Mapping[str, Any]], population_name: str = "individuals"
) -> tuple[NSGA2Candidate, ...]:
    """Extract completed records with valid raw objectives without changing input.

    Parameters
    ----------
    individuals : mapping
        Checkpoint-like mapping from individual ID to individual record.
    population_name : str, optional
        Population label included in validation errors.

    Returns
    -------
    tuple[NSGA2Candidate, ...]
        Eligible candidates in stable individual-ID order; all other statuses are excluded."""
    if not isinstance(individuals, Mapping):
        raise TypeError(f"{population_name} harus berupa mapping individual.")
    validated_ids = [_validate_individual_id(individual_id) for individual_id in individuals]
    candidates = []
    for individual_id in sorted(validated_ids):
        individual = individuals[individual_id]
        if not isinstance(individual, Mapping):
            raise TypeError(f"Record {population_name}[{individual_id!r}] harus berupa mapping.")
        if individual.get("status") != ELIGIBLE_STATUS:
            continue
        results = individual.get("results")
        if not isinstance(results, Mapping):
            raise InvalidObjectivesError(
                f"Individual completed {individual_id!r} harus memiliki results mapping."
            )
        if "objectives" not in results:
            raise InvalidObjectivesError(
                f"Individual completed {individual_id!r} tidak memiliki results.objectives."
            )
        objectives = validate_objectives(
            results["objectives"], f"results.objectives {individual_id}"
        )
        contract = KeffConstraint.from_identity(results.get("constraint"))
        contract.validate_results(results)
        # The first objective is negative mean keff, without a Monte Carlo uncertainty adjustment.
        if objectives[0] != -results["keff"]["mean"]:
            raise InvalidObjectivesError(
                f"Objective negative_keff {individual_id} tidak konsisten dengan mean keff BOC."
            )
        # Completed infeasible candidates remain in the constrained-ranking pool.
        candidates.append(NSGA2Candidate(individual_id, objectives, contract))
    return tuple(candidates)


def fast_non_dominated_sort(candidates: Iterable[NSGA2Candidate]) -> tuple[tuple[str, ...], ...]:
    """Sort candidates into deterministic Pareto fronts for two-objective minimization.

    Parameters
    ----------
    candidates : iterable of NSGA2Candidate
        Eligible candidates with unique stable IDs and finite objectives.

    Returns
    -------
    tuple[tuple[str, ...], ...]
        Fronts from rank zero onward, each ordered lexicographically by individual ID."""
    materialized = _materialize_candidates(candidates, "candidates")
    return _fast_non_dominated_sort_materialized(materialized)


def _fast_non_dominated_sort_materialized(
    candidates: tuple[NSGA2Candidate, ...]
) -> tuple[tuple[str, ...], ...]:
    """Return deterministic fronts for already validated, canonical candidates."""
    if not candidates:
        return ()

    dominated_ids = {candidate.individual_id: [] for candidate in candidates}
    domination_counts = {candidate.individual_id: 0 for candidate in candidates}
    for first_index, first in enumerate(candidates):
        for second in candidates[first_index + 1 :]:
            if dominates(first, second):
                dominated_ids[first.individual_id].append(second.individual_id)
                domination_counts[second.individual_id] += 1
            elif dominates(second, first):
                dominated_ids[second.individual_id].append(first.individual_id)
                domination_counts[first.individual_id] += 1

    current_front = sorted(
        individual_id for individual_id, count in domination_counts.items() if count == 0
    )
    fronts = []
    while current_front:
        fronts.append(tuple(current_front))
        next_front = []
        for individual_id in current_front:
            for dominated_id in dominated_ids[individual_id]:
                domination_counts[dominated_id] -= 1
                if domination_counts[dominated_id] == 0:
                    next_front.append(dominated_id)
        current_front = sorted(next_front)
    return tuple(fronts)


# Crowding measures density in objective space, not physical fuel-position distance.
def calculate_crowding_distance(front: Iterable[NSGA2Candidate]) -> tuple[CrowdingMetric, ...]:
    """Calculate JSON-safe NSGA-II crowding metrics for one nondominated front.

    Parameters
    ----------
    front : iterable of NSGA2Candidate
        Members of exactly one nondominated front.

    Returns
    -------
    tuple[CrowdingMetric, ...]
        Metrics in stable ID order; informative boundaries use ``None`` and ``is_boundary=True``."""
    candidates = _materialize_candidates(front, "front")
    if not candidates:
        return ()

    distances = {candidate.individual_id: 0.0 for candidate in candidates}
    boundaries = set()
    for objective_index in range(OBJECTIVE_COUNT):
        ordered = sorted(
            candidates,
            key=lambda candidate: (candidate.objectives[objective_index], candidate.individual_id),
        )
        minimum = ordered[0].objectives[objective_index]
        maximum = ordered[-1].objectives[objective_index]
        objective_range = maximum - minimum
        if objective_range == 0.0:
            # If all objectives are constant, assign zero distance and no boundary, even for a singleton.
            continue
        # Extreme objective values receive effective infinite distance to preserve front endpoints.
        boundaries.update((ordered[0].individual_id, ordered[-1].individual_id))
        for index in range(1, len(ordered) - 1):
            individual_id = ordered[index].individual_id
            previous_value = ordered[index - 1].objectives[objective_index]
            next_value = ordered[index + 1].objectives[objective_index]
            distances[individual_id] += (next_value - previous_value) / objective_range

    return tuple(
        CrowdingMetric(
            candidate.individual_id,
            None if candidate.individual_id in boundaries else distances[candidate.individual_id],
            candidate.individual_id in boundaries,
        )
        for candidate in candidates
    )


def rank_candidates(candidates: Iterable[NSGA2Candidate]) -> NSGA2Ranking:
    """Calculate nondominated rank and crowding for all eligible candidates.

    Parameters
    ----------
    candidates : iterable of NSGA2Candidate
        Unique eligible individuals to rank together.

    Returns
    -------
    NSGA2Ranking
        Immutable front membership and JSON-safe metrics in rank then ID order."""
    materialized = _materialize_candidates(candidates, "candidates")
    candidate_by_id = {candidate.individual_id: candidate for candidate in materialized}
    fronts = _fast_non_dominated_sort_materialized(materialized)
    ranked_candidates = []
    for rank, front_ids in enumerate(fronts):
        front_candidates = tuple(candidate_by_id[individual_id] for individual_id in front_ids)
        crowding_by_id = {
            metric.individual_id: metric for metric in calculate_crowding_distance(front_candidates)
        }
        for individual_id in front_ids:
            candidate = candidate_by_id[individual_id]
            crowding = crowding_by_id[individual_id]
            ranked_candidates.append(
                RankedCandidate(
                    individual_id,
                    candidate.objectives,
                    rank,
                    crowding.crowding_distance,
                    crowding.is_boundary,
                )
            )
    return NSGA2Ranking(fronts=fronts, ranked_candidates=tuple(ranked_candidates))


def _validate_target_population_size(target_population_size: Any) -> int:
    """Validate and return a positive integer environmental-selection target."""
    if isinstance(target_population_size, bool) or not isinstance(target_population_size, Integral):
        raise NSGA2Error("target_population_size harus berupa integer positif.")
    normalized = int(target_population_size)
    if normalized <= 0:
        raise NSGA2Error("target_population_size harus berupa integer positif.")
    return normalized


def _crowded_comparison_key(candidate: RankedCandidate) -> tuple[int, float, str]:
    """Return rank-local ordering by boundary, descending distance, then stable ID."""
    boundary_order = 0 if candidate.is_boundary else 1
    distance_order = 0.0 if candidate.crowding_distance is None else -candidate.crowding_distance
    # Survivor ties use boundary priority, decreasing distance, then ID; no random draw occurs.
    return boundary_order, distance_order, candidate.individual_id


def environmental_selection_candidates(
    parent_candidates: Iterable[NSGA2Candidate],
    offspring_candidates: Iterable[NSGA2Candidate],
    target_population_size: int,
) -> EnvironmentalSelectionResult:
    """Select an exact survivor population from eligible parents plus offspring.

    Parameters
    ----------
    parent_candidates, offspring_candidates : iterable of NSGA2Candidate
        Already eligible candidate populations with globally unique IDs.
    target_population_size : int
        Exact number of survivors required.

    Returns
    -------
    EnvironmentalSelectionResult
        Survivors ordered by rank, descending crowding, and stable ID plus the full ranking."""
    target = _validate_target_population_size(target_population_size)
    parents = _materialize_candidates(parent_candidates, "parent_candidates")
    offspring = _materialize_candidates(offspring_candidates, "offspring_candidates")
    parent_ids = tuple(candidate.individual_id for candidate in parents)
    offspring_ids = tuple(candidate.individual_id for candidate in offspring)
    overlapping_ids = sorted(set(parent_ids).intersection(offspring_ids))
    if overlapping_ids:
        raise NSGA2Error(
            "ID parent dan offspring tidak boleh overlap: " + ", ".join(overlapping_ids) + "."
        )

    combined = tuple(sorted(parents + offspring, key=lambda candidate: candidate.individual_id))
    if len(combined) < target:
        raise InsufficientEligibleCandidatesError(
            f"Kandidat eligible tidak cukup untuk environmental selection: target={target}, eligible={len(combined)}, parents={len(parents)}, offspring={len(offspring)}."
        )

    # Recompute ranks on the combined parent-offspring pool for environmental selection.
    ranking = rank_candidates(combined)
    metrics_by_id = ranking.metrics_by_id()
    survivors = []
    for front_ids in ranking.fronts:
        ordered_front = sorted(
            (metrics_by_id[individual_id] for individual_id in front_ids),
            key=_crowded_comparison_key,
        )
        remaining = target - len(survivors)
        survivors.extend(candidate.individual_id for candidate in ordered_front[:remaining])
        if len(survivors) == target:
            break
    return EnvironmentalSelectionResult(tuple(survivors), parent_ids, offspring_ids, ranking)


def environmental_selection(
    parent_individuals: Mapping[str, Mapping[str, Any]],
    offspring_individuals: Mapping[str, Mapping[str, Any]],
    target_population_size: int,
) -> EnvironmentalSelectionResult:
    """Filter checkpoint-like records and perform pure parent-offspring replacement.

    Parameters
    ----------
    parent_individuals, offspring_individuals : mapping
        Individual records; only ``completed`` with two finite objectives enter ranking.
    target_population_size : int
        Exact survivor population size required.

    Returns
    -------
    EnvironmentalSelectionResult
        Exact survivors and combined ranking without mutating either input mapping."""
    parents = extract_eligible_candidates(parent_individuals, "parent_individuals")
    offspring = extract_eligible_candidates(offspring_individuals, "offspring_individuals")
    return environmental_selection_candidates(parents, offspring, target_population_size)
