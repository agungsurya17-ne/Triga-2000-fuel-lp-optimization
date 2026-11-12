"""Select parents by constrained NSGA-II crowded tournament.

CrowdedTournamentSelector uses existing rank and crowding values, preferring
lower rank and then larger crowding distance. Self-pairing is avoided when
possible. Selection does not calculate objectives or modify checkpoints."""

# Prefer lower rank, then larger crowding distance; break remaining ties with the shared RNG.
# Recompute ranking on the supplied pool rather than trusting metrics from another pool.
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

from modules.nsga2 import extract_eligible_candidates, rank_candidates


ELIGIBLE_STATUS = "completed"


class ParentSelectionError(ValueError):
    """Base error for invalid crowded-tournament selection inputs."""


class InvalidSelectionMetricError(ParentSelectionError):
    """Raised when an eligible individual lacks valid rank or crowding data."""


class InsufficientEligibleParentsError(ParentSelectionError):
    """Raised when eligible individuals cannot satisfy one tournament or pair."""


@dataclass(frozen=True)
class TournamentCandidate:
    """Represent one eligible individual using only selection metrics and its ID."""

    individual_id: str
    rank: int
    crowding_distance: float | None
    is_boundary: bool

    @property
    def effective_crowding_distance(self) -> float:
        """Return infinity for a boundary and finite crowding otherwise."""
        return math.inf if self.is_boundary else float(self.crowding_distance)


class CrowdedTournamentSelector:
    """Perform rank-and-crowding tournaments without reading or changing genomes.

    Consecutive outputs form parent pairs. By default, the second member cannot
    equal the first when at least two eligible individuals exist. If the normal
    tournament winner would self-pair, the best different contestant from that
    tournament is used; a size-one tournament draws from the remaining eligible
    pool. This keeps the configured tournament size while avoiding self-pairs."""

    def __init__(self, tournament_size: int, rng: Any, avoid_self_pairing: bool = True) -> None:
        """Store the configured tournament size, injected RNG, and pairing policy.

        Parameters
        ----------
        tournament_size : int
            Number of distinct eligible candidates sampled per tournament.
        rng : object
            Injected deterministic generator providing ``sample`` and ``choice``.
        avoid_self_pairing : bool, optional
            Prevent equal IDs within each consecutive pair when alternatives exist.

        Returns
        -------
        None
            The selector retains these dependencies without using global RNG state."""
        if (
            isinstance(tournament_size, bool)
            or not isinstance(tournament_size, Integral)
            or tournament_size <= 0
        ):
            raise ValueError("tournament_size harus berupa integer positif.")
        if not callable(getattr(rng, "sample", None)) or not callable(getattr(rng, "choice", None)):
            raise TypeError("rng harus menyediakan method sample() dan choice().")
        if not isinstance(avoid_self_pairing, bool):
            raise TypeError("avoid_self_pairing harus berupa bool.")
        self.tournament_size = int(tournament_size)
        self.rng = rng
        self.avoid_self_pairing = avoid_self_pairing

    def select_parent_ids(
        self, individuals: Mapping[str, Mapping[str, Any]], parent_count: int
    ) -> tuple[str, ...]:
        """Select parent IDs from completed records with persisted NSGA-II metrics.

        Parameters
        ----------
        individuals : mapping
            Scoped checkpoint records; non-completed records are excluded.
        parent_count : int
            Number of parent IDs to return, paired in consecutive positions.

        Returns
        -------
        tuple[str, ...]
            Selected stable IDs; no genome or inventory content is returned or changed."""
        if not isinstance(individuals, Mapping):
            raise TypeError("individuals harus berupa mapping record checkpoint.")
        if (
            isinstance(parent_count, bool)
            or not isinstance(parent_count, Integral)
            or parent_count <= 0
        ):
            raise ValueError("parent_count harus berupa integer positif.")

        candidates = self._extract_eligible_candidates(individuals)
        if len(candidates) < self.tournament_size:
            raise InsufficientEligibleParentsError(
                f"Eligible parent tidak cukup untuk tournament_size={self.tournament_size}: tersedia {len(candidates)} completed."
            )
        if self.avoid_self_pairing and parent_count >= 2 and len(candidates) < 2:
            raise InsufficientEligibleParentsError(
                "Sedikitnya dua eligible parent diperlukan ketika self-pairing dihindari."
            )

        selected_ids = []
        for selection_index in range(int(parent_count)):
            # Sample without replacement within a tournament; reuse candidates in later tournaments.
            contestants = tuple(self.rng.sample(candidates, self.tournament_size))
            winner = self._choose_winner(contestants)
            if (
                self.avoid_self_pairing
                and selection_index % 2 == 1
                and winner.individual_id == selected_ids[-1]
            ):
                winner = self._choose_non_self_winner(contestants, candidates, selected_ids[-1])
            selected_ids.append(winner.individual_id)
        # Consecutive returned IDs form pairs; an odd final ID remains unpaired.
        return tuple(selected_ids)

    def _extract_eligible_candidates(
        self, individuals: Mapping[str, Mapping[str, Any]]
    ) -> tuple[TournamentCandidate, ...]:
        """Return completed candidates in stable ID order after metric validation."""
        ranking = rank_candidates(extract_eligible_candidates(individuals))
        metrics_by_id = ranking.metrics_by_id()
        return tuple(
            TournamentCandidate(
                item.individual_id, item.rank, item.crowding_distance, item.is_boundary
            )
            for item in sorted(metrics_by_id.values(), key=lambda item: item.individual_id)
        )

    @staticmethod
    def _build_candidate(individual_id: str, metrics: Any) -> TournamentCandidate:
        """Validate checkpoint rank and crowding fields and return one candidate."""
        if not isinstance(metrics, Mapping):
            raise InvalidSelectionMetricError(
                f"Individual completed {individual_id!r} harus memiliki nsga2 mapping."
            )
        rank = metrics.get("rank")
        crowding_distance = metrics.get("crowding_distance")
        is_boundary = metrics.get("is_boundary")
        if isinstance(rank, bool) or not isinstance(rank, Integral) or rank < 0:
            raise InvalidSelectionMetricError(
                f"nsga2.rank {individual_id!r} harus berupa integer non-negatif."
            )
        if not isinstance(is_boundary, bool):
            raise InvalidSelectionMetricError(
                f"nsga2.is_boundary {individual_id!r} harus berupa bool."
            )
        if is_boundary:
            # Store a boundary distance as None for strict JSON; compare it as positive infinity.
            if crowding_distance is not None:
                raise InvalidSelectionMetricError(
                    f"Boundary {individual_id!r} harus menyimpan crowding_distance=None sebagai infinity JSON-safe."
                )
            return TournamentCandidate(individual_id, int(rank), None, True)
        if isinstance(crowding_distance, bool) or not isinstance(crowding_distance, Real):
            raise InvalidSelectionMetricError(
                f"nsga2.crowding_distance {individual_id!r} harus berupa angka finite non-negatif."
            )
        normalized_distance = float(crowding_distance)
        if not math.isfinite(normalized_distance) or normalized_distance < 0.0:
            raise InvalidSelectionMetricError(
                f"nsga2.crowding_distance {individual_id!r} harus berupa angka finite non-negatif."
            )
        return TournamentCandidate(individual_id, int(rank), normalized_distance, False)

    def _choose_winner(self, contestants: tuple[TournamentCandidate, ...]) -> TournamentCandidate:
        """Choose by lower rank, larger effective crowding, then injected RNG."""
        best_rank = min(candidate.rank for candidate in contestants)
        rank_finalists = tuple(
            candidate for candidate in contestants if candidate.rank == best_rank
        )
        best_distance = max(candidate.effective_crowding_distance for candidate in rank_finalists)
        # Sort tied IDs before drawing so the tie-breaking population is reproducible.
        finalists = tuple(
            sorted(
                (
                    candidate
                    for candidate in rank_finalists
                    if candidate.effective_crowding_distance == best_distance
                ),
                key=lambda candidate: candidate.individual_id,
            )
        )
        return finalists[0] if len(finalists) == 1 else self.rng.choice(finalists)

    def _choose_non_self_winner(
        self,
        contestants: tuple[TournamentCandidate, ...],
        candidates: tuple[TournamentCandidate, ...],
        first_parent_id: str,
    ) -> TournamentCandidate:
        """Return the best tournament alternative to the first parent in a pair."""
        alternatives = tuple(
            candidate for candidate in contestants if candidate.individual_id != first_parent_id
        )
        if alternatives:
            return self._choose_winner(alternatives)
        remaining_pool = tuple(
            candidate for candidate in candidates if candidate.individual_id != first_parent_id
        )
        return self._choose_winner(tuple(self.rng.sample(remaining_pool, 1)))
