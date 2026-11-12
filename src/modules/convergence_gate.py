"""Evaluate statistical acceptance criteria after neutron transport.

ConvergenceGate returns pass/fail criteria and diagnostics for keff uncertainty
in pcm, relative tally uncertainty, and unsampled bins. The pipeline stores
these diagnostics in the checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Integral, Real
from typing import Any


class ConvergenceGate:
    """Decide whether evaluated OpenMC statistics satisfy configured limits."""

    def __init__(
        self,
        enabled: bool,
        max_keff_std_dev_pcm: float,
        max_tally_mean_relative_std_dev: float,
        max_tally_p95_relative_std_dev: float,
        max_tally_max_relative_std_dev: float,
        allow_zero_mean_bins: bool,
    ) -> None:
        """Store validated convergence limits and zero-bin policy.

        Parameters are the enable flag, keff uncertainty limit in pcm, three
        dimensionless tally relative-uncertainty limits, and the zero-bin policy.
        The constructor returns no value."""
        if not isinstance(enabled, bool):
            raise TypeError("enabled harus berupa bool.")
        if not isinstance(allow_zero_mean_bins, bool):
            raise TypeError("allow_zero_mean_bins harus berupa bool.")
        self.enabled = enabled
        self.max_keff_std_dev_pcm = self._validate_limit(
            "max_keff_std_dev_pcm", max_keff_std_dev_pcm
        )
        self.max_tally_mean_relative_std_dev = self._validate_limit(
            "max_tally_mean_relative_std_dev", max_tally_mean_relative_std_dev
        )
        self.max_tally_p95_relative_std_dev = self._validate_limit(
            "max_tally_p95_relative_std_dev", max_tally_p95_relative_std_dev
        )
        self.max_tally_max_relative_std_dev = self._validate_limit(
            "max_tally_max_relative_std_dev", max_tally_max_relative_std_dev
        )
        self.allow_zero_mean_bins = allow_zero_mean_bins

    @staticmethod
    def _validate_limit(name: str, value: Any) -> float:
        """Validate one finite non-negative criterion limit and return a float."""
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} harus numerik dan bukan bool.")
        normalized_value = float(value)
        if not math.isfinite(normalized_value) or normalized_value < 0.0:
            raise ValueError(f"{name} harus finite dan non-negatif.")
        return normalized_value

    @staticmethod
    def _read_nested(mapping: Mapping[str, Any], *keys: str) -> Any:
        """Return one nested mapping value, or None when its path is missing."""
        current: Any = mapping
        for key in keys:
            if not isinstance(current, Mapping) or key not in current:
                return None
            current = current[key]
        return current

    @staticmethod
    def _numeric_criterion(observed: Any, limit: float, unit: str) -> dict[str, Any]:
        """Return a criterion comparing finite ``observed`` with ``limit``.

        ``unit`` labels the saved diagnostic. Missing or non-finite observations fail
        the criterion; valid observations pass when observed <= limit."""
        valid = (
            isinstance(observed, Real)
            and not isinstance(observed, bool)
            and math.isfinite(float(observed))
        )
        normalized_observed = float(observed) if valid else None
        criterion = {
            "observed": normalized_observed,
            "limit": float(limit),
            "unit": unit,
            "passed": bool(valid and normalized_observed <= limit),
        }
        if not valid:
            criterion["reason"] = "missing_or_non_finite_observation"
        return criterion

    def _zero_bin_criterion(self, observed: Any) -> dict[str, Any]:
        """Build the zero-mean-bin criterion using the configured allow policy."""
        valid = (
            isinstance(observed, Integral) and not isinstance(observed, bool) and int(observed) >= 0
        )
        normalized_observed = int(observed) if valid else None
        passed = bool(valid and (self.allow_zero_mean_bins or normalized_observed == 0))
        criterion = {
            "observed": normalized_observed,
            "limit": None if self.allow_zero_mean_bins else 0,
            "unit": "bins",
            "passed": passed,
            "allowed": self.allow_zero_mean_bins,
        }
        if not valid:
            criterion["reason"] = "missing_or_invalid_observation"
        return criterion

    def evaluate(self, results: Mapping[str, Any]) -> dict[str, Any]:
        """Evaluate result statistics and return a structured gate decision.

        Parameters
        ----------
        results : mapping
            Evaluator result containing keff and tally uncertainty statistics.

        Returns
        -------
        dict
            JSON-safe enabled/evaluated state, criteria, failures, and decision."""
        if not isinstance(results, Mapping):
            raise TypeError("results harus berupa mapping.")
        if not self.enabled:
            return {
                "enabled": False,
                "evaluated": False,
                "passed": None,
                "criteria": {},
                "failed_criteria": [],
                "reason": "convergence_gate_disabled",
            }

        tally_statistics = self._read_nested(results, "statistics", "tally_uncertainty")
        if not isinstance(tally_statistics, Mapping):
            tally_statistics = results.get("tally_uncertainty", {})
        criteria = {
            "keff_std_dev_pcm": self._numeric_criterion(
                self._read_nested(results, "keff", "std_dev_pcm"), self.max_keff_std_dev_pcm, "pcm"
            ),
            "tally_mean_relative_std_dev": self._numeric_criterion(
                (
                    tally_statistics.get("mean_relative_std_dev")
                    if isinstance(tally_statistics, Mapping)
                    else None
                ),
                self.max_tally_mean_relative_std_dev,
                "relative",
            ),
            "tally_p95_relative_std_dev": self._numeric_criterion(
                (
                    tally_statistics.get("p95_relative_std_dev")
                    if isinstance(tally_statistics, Mapping)
                    else None
                ),
                self.max_tally_p95_relative_std_dev,
                "relative",
            ),
            "tally_max_relative_std_dev": self._numeric_criterion(
                (
                    tally_statistics.get("max_relative_std_dev")
                    if isinstance(tally_statistics, Mapping)
                    else None
                ),
                self.max_tally_max_relative_std_dev,
                "relative",
            ),
            "zero_mean_bins": self._zero_bin_criterion(
                tally_statistics.get("zero_mean_bin_count")
                if isinstance(tally_statistics, Mapping)
                else None
            ),
        }
        failed_criteria = [name for name, criterion in criteria.items() if not criterion["passed"]]
        return {
            "enabled": True,
            "evaluated": True,
            "passed": not failed_criteria,
            "criteria": criteria,
            "failed_criteria": failed_criteria,
        }
