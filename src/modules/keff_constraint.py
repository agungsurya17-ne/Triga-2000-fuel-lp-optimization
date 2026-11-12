"""Apply the keff feasibility constraint before objective comparison.

KeffConstraint validates the lower bound and tolerance, then computes g_keff,
constraint violation, and feasibility from keff.mean. No transport or I/O occurs."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any


CONSTRAINT_ID = "boc_mean_keff_constraint_domination_v1"


def finite_number(value: Any, name: str) -> float:
    """Validate real ``value`` identified by ``name`` and return a finite float."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} harus berupa angka finite.")
    return float(value)


@dataclass(frozen=True)
class KeffConstraint:
    """Store configured dimensionless criticality threshold and feasibility tolerance."""

    keff_min: float
    feasibility_tolerance: float

    def __post_init__(self) -> None:
        """Validate injected parameters and normalize finite floats; return no value."""
        object.__setattr__(self, "keff_min", finite_number(self.keff_min, "keff_min"))
        object.__setattr__(
            self,
            "feasibility_tolerance",
            finite_number(self.feasibility_tolerance, "feasibility_tolerance"),
        )
        if self.feasibility_tolerance < 0.0:
            raise ValueError("feasibility_tolerance harus non-negatif.")

    def identity(self) -> dict[str, Any]:
        """Return the versioned contract and configured parameters for checkpoint compatibility."""
        return {
            "contract_id": CONSTRAINT_ID,
            "keff_min": self.keff_min,
            "feasibility_tolerance": self.feasibility_tolerance,
        }

    @classmethod
    def from_identity(cls, value: Any) -> "KeffConstraint":
        """Validate stored contract ``value`` and return its constraint; reject missing metadata."""
        if not isinstance(value, Mapping) or value.get("contract_id") != CONSTRAINT_ID:
            raise ValueError(
                "Kontrak constraint keff hilang atau tidak kompatibel; mulai campaign baru."
            )
        return cls(value.get("keff_min"), value.get("feasibility_tolerance"))

    # NSGA-II compares feasibility and violation before objective dominance.
    def evaluate(self, keff: float) -> dict[str, Any]:
        """Return signed constraint, violation, and feasibility from finite BOC mean ``keff``."""
        # Use dimensionless mean BOC keff without subtracting Monte Carlo uncertainty.
        mean = finite_number(keff, "keff")
        # g_keff = keff_min - mean; g <= 0 satisfies the unrelaxed bound.
        g_keff = finite_number(self.keff_min - mean, "g_keff")
        violation = max(0.0, g_keff)
        # Positive tolerance relaxes the bound to mean >= keff_min - tolerance; objectives have no penalty.
        return {
            **self.identity(),
            "g_keff": g_keff,
            "keff_violation": violation,
            "feasible": violation <= self.feasibility_tolerance,
        }

    def from_results(self, results: Mapping[str, Any]) -> dict[str, Any]:
        """Read evaluator ``results.keff.mean`` and return constraint metadata without changing results."""
        keff = results.get("keff")
        if not isinstance(keff, Mapping):
            raise ValueError("Hasil evaluasi memerlukan results.keff.mean BOC.")
        return self.evaluate(keff.get("mean"))

    def validate_results(self, results: Mapping[str, Any]) -> dict[str, Any]:
        """Validate persisted constraint against ``results.keff.mean`` and return the expected metadata."""
        expected = self.from_results(results)
        stored = results.get("constraint")
        if (
            not isinstance(stored, Mapping)
            or type(stored.get("feasible")) is not bool
            or stored != expected
        ):
            raise ValueError(
                "Metadata constraint hilang atau tidak konsisten dengan mean keff/kontrak; resume ditolak."
            )
        return expected
