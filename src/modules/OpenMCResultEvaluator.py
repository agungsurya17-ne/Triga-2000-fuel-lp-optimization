"""Extract eigenvalue statistics and relative-power metrics from statepoints.

OpenMCResultEvaluator matches tally bins to fuel-segment records and calculates
dimensionless 3-D, radial, and axial PPF values. The minimization objectives are
(-keff, ppf_3d); the 3-D metric uses the complete configured fuel-segment domain."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np


class OpenMCResultEvaluator:
    """Read one statepoint and calculate physical metrics and NSGA-II objectives."""

    EVALUATION_VERSION = "triga2000_keff_ppf3d/v1"
    REQUIRED_TALLY_RECORD_KEYS = frozenset(
        {"cell_id", "slot_name", "genome_index", "gene_id", "axial_index"}
    )

    def __init__(
        self,
        tally_name: str,
        tally_score: str,
    ) -> None:
        """Initialize the evaluator from the required tally identity.

        Parameters
        ----------
        tally_name, tally_score : str
            OpenMC tally name and score used as the relative-power proxy.

        Returns
        -------
        None
            Validated parameters are stored on the evaluator."""
        self.tally_name = self._require_non_empty_string("tally_name", tally_name)
        self.tally_score = self._require_non_empty_string("tally_score", tally_score)

    @staticmethod
    def _require_non_empty_string(name: str, value: Any) -> str:
        """Validate and return a stripped non-empty string.

        Parameters
        ----------
        name : str
            Parameter name used in validation messages.
        value : object
            Candidate string value.

        Returns
        -------
        str
            Stripped validated string."""
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"{name} harus berupa string non-kosong.")
        return value.strip()

    @staticmethod
    def _require_finite_number(name: str, value: Any) -> float:
        """Validate a real non-boolean number and return a Python float.

        Parameters
        ----------
        name : str
            Parameter name used in validation messages.
        value : object
            Candidate numeric value.

        Returns
        -------
        float
            Finite built-in float."""
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} harus berupa angka dan bukan bool.")
        normalized_value = float(value)
        if not math.isfinite(normalized_value):
            raise ValueError(f"{name} harus finite.")
        return normalized_value

    @staticmethod
    def _require_integer(name: str, value: Any, minimum: int) -> int:
        """Validate an integer field and return a built-in integer.

        Parameters
        ----------
        name : str
            Field name used in validation messages.
        value : object
            Candidate integer value.
        minimum : int
            Smallest accepted value.

        Returns
        -------
        int
            Validated built-in integer."""
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} harus berupa integer.")
        normalized_value = int(value)
        if normalized_value < minimum:
            raise ValueError(f"{name} harus lebih besar atau sama dengan {minimum}.")
        return normalized_value

    @staticmethod
    def _validate_statepoint_path(statepoint_path: str | Path) -> Path:
        """Validate that a statepoint path identifies an available file.

        Parameters
        ----------
        statepoint_path : str or pathlib.Path
            Statepoint path returned by the OpenMC runner.

        Returns
        -------
        pathlib.Path
            Existing statepoint path."""
        if not isinstance(statepoint_path, (str, Path)) or not str(statepoint_path).strip():
            raise TypeError("statepoint_path harus berupa path non-kosong.")
        normalized_path = Path(statepoint_path)
        if not normalized_path.is_file():
            raise FileNotFoundError(f"Statepoint tidak ditemukan: {normalized_path}")
        return normalized_path

    @classmethod
    def _validate_tally_records(
        cls, tally_records: Iterable[Mapping[str, Any]]
    ) -> tuple[dict[str, Any], ...]:
        """Validate and normalize ordered metadata for every fuel tally bin.

        Parameters
        ----------
        tally_records : iterable of mappings
            Ordered builder metadata connecting cells, slots, genes, and axial indices.

        Returns
        -------
        tuple of dict
            Validated records with Python-native scalar values."""
        if isinstance(tally_records, (str, bytes, Mapping)):
            raise TypeError("tally_records harus berupa iterable record mapping.")
        try:
            raw_records = tuple(tally_records)
        except TypeError as error:
            raise TypeError("tally_records harus berupa iterable record mapping.") from error
        if not raw_records:
            raise ValueError("tally_records tidak boleh kosong.")

        normalized_records = []
        seen_cell_ids = set()
        seen_segment_keys = set()
        slot_identity = {}
        axial_indices_by_slot = {}

        for record_index, record in enumerate(raw_records):
            if not isinstance(record, Mapping):
                raise TypeError(f"tally_records[{record_index}] harus berupa mapping.")
            missing_keys = cls.REQUIRED_TALLY_RECORD_KEYS.difference(record)
            if missing_keys:
                missing = ", ".join(sorted(missing_keys))
                raise ValueError(f"tally_records[{record_index}] kehilangan key: {missing}.")

            cell_id = cls._require_integer(
                f"tally_records[{record_index}].cell_id", record["cell_id"], 1
            )
            slot_name = cls._require_non_empty_string(
                f"tally_records[{record_index}].slot_name", record["slot_name"]
            )
            genome_index = cls._require_integer(
                f"tally_records[{record_index}].genome_index", record["genome_index"], 0
            )
            gene_id = cls._require_non_empty_string(
                f"tally_records[{record_index}].gene_id", record["gene_id"]
            )
            axial_index = cls._require_integer(
                f"tally_records[{record_index}].axial_index", record["axial_index"], 1
            )

            if cell_id in seen_cell_ids:
                raise ValueError(f"cell_id duplikat pada tally_records: {cell_id}.")
            segment_key = (slot_name, axial_index)
            if segment_key in seen_segment_keys:
                raise ValueError(f"Pasangan slot_name dan axial_index duplikat: {segment_key}.")
            identity = (gene_id, genome_index)
            if slot_name in slot_identity and slot_identity[slot_name] != identity:
                raise ValueError(
                    f"Slot {slot_name!r} memiliki gene_id atau genome_index yang tidak konsisten."
                )

            seen_cell_ids.add(cell_id)
            seen_segment_keys.add(segment_key)
            slot_identity.setdefault(slot_name, identity)
            axial_indices_by_slot.setdefault(slot_name, []).append(axial_index)
            normalized_records.append(
                {
                    "cell_id": cell_id,
                    "slot_name": slot_name,
                    "genome_index": genome_index,
                    "gene_id": gene_id,
                    "axial_index": axial_index,
                }
            )

        expected_axial_indices = None
        for slot_name, axial_indices in axial_indices_by_slot.items():
            sequential_indices = list(range(1, len(axial_indices) + 1))
            if axial_indices != sequential_indices:
                raise ValueError(f"Indeks aksial slot {slot_name!r} harus berurutan mulai dari 1.")
            if expected_axial_indices is None:
                expected_axial_indices = tuple(axial_indices)
            elif tuple(axial_indices) != expected_axial_indices:
                raise ValueError("Seluruh fuel slot harus memiliki indeks aksial yang konsisten.")
        return tuple(normalized_records)

    @staticmethod
    def _open_statepoint(statepoint_path: Path) -> Any:
        """Open an OpenMC statepoint through a lazy dependency import.

        Parameters
        ----------
        statepoint_path : pathlib.Path
            Existing statepoint file to open.

        Returns
        -------
        openmc.StatePoint
            Open statepoint object for result extraction."""
        try:
            import openmc
        except ImportError as error:
            raise RuntimeError("Paket OpenMC diperlukan untuk membuka statepoint.") from error
        return openmc.StatePoint(statepoint_path)

    def evaluate(
        self, statepoint_path: str | Path, tally_records: Iterable[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Evaluate one statepoint using ordered fuel-segment tally metadata.

        Parameters
        ----------
        statepoint_path : str or pathlib.Path
            Statepoint file returned by the OpenMC runner.
        tally_records : iterable of mappings
            Ordered metadata returned as ``build_data["tally_records"]``.

        Returns
        -------
        dict
            JSON-compatible keff, power, PPF, objectives, and metadata."""
        normalized_path = self._validate_statepoint_path(statepoint_path)
        normalized_records = self._validate_tally_records(tally_records)
        statepoint = self._open_statepoint(normalized_path)
        try:
            return self._evaluate_statepoint(statepoint, normalized_records)
        finally:
            close_statepoint = getattr(statepoint, "close", None)
            if callable(close_statepoint):
                close_statepoint()

    def _evaluate_statepoint(
        self, statepoint: Any, tally_records: tuple[dict[str, Any], ...]
    ) -> dict[str, Any]:
        """Extract validated statepoint values and assemble the evaluation result.

        Parameters
        ----------
        statepoint : openmc.StatePoint-compatible object
            Open statepoint or a fake object used by unit tests.
        tally_records : tuple of dict
            Already validated ordered tally metadata.

        Returns
        -------
        dict
            Complete Python-native evaluation result."""
        keff = self._extract_keff(statepoint)
        tally = self._get_tally(statepoint)
        tally_means, tally_std_devs = self._extract_tally_values(tally, tally_records)
        power_results = self._calculate_power_results(tally_means, tally_records)
        tally_uncertainty = self._summarize_tally_uncertainty(
            tally_means, tally_std_devs, tally_records
        )
        statepoint_statistics = self._extract_statepoint_statistics(statepoint, tally)

        ppf_3d = power_results["ppf_3d"]
        radial_ppf = power_results["radial_ppf"]
        axial_ppf = power_results["axial_ppf"]
        objectives = self.build_objectives(keff["mean"], ppf_3d)

        return {
            "keff": keff,
            "ppf_3d": ppf_3d,
            "radial_ppf": radial_ppf,
            "axial_ppf": axial_ppf,
            "objectives": [objectives[0], objectives[1]],
            "objective_1": objectives[0],
            "objective_2": objectives[1],
            "power_by_slot": power_results["power_by_slot"],
            "power_by_axial_index": power_results["power_by_axial_index"],
            "power_by_segment": power_results["power_by_segment"],
            "fuel_id_by_slot": power_results["fuel_id_by_slot"],
            "tally_uncertainty": tally_uncertainty,
            "statistics": {
                "statepoint": statepoint_statistics,
                "tally_uncertainty": tally_uncertainty,
            },
            "metadata": {
                "tally_name": self.tally_name,
                "tally_score": self.tally_score,
                "fuel_slot_count": power_results["fuel_slot_count"],
                "axial_segment_count": power_results["axial_segment_count"],
                "tally_bin_count": len(tally_records),
                "power_normalization": "relative_to_mean_fission_q_recoverable",
                "tally_quantity": "integrated_fission_q_recoverable_per_segment",
                "ppf_3d_aggregation": "max_segment_score_over_mean_segment_score",
                "ppf_3d_volume_assumption": "equal_fuel_segment_volumes",
                "radial_ppf_aggregation": "sum_segment_scores_per_fuel_slot",
                "included_slot_types": ["fuel"],
                "ffcr_included": False,
                "objective_names": ["negative_keff", "ppf_3d"],
                "objective_orientation": "minimize",
                "radial_ppf_role": "diagnostic_only",
                "axial_ppf_role": "diagnostic_only",
            },
        }

    @classmethod
    def build_objectives(cls, keff: float, ppf_3d: float) -> tuple[float, float]:
        """Build the two raw minimization objectives from physical metrics.

        Parameters
        ----------
        keff : float
            Effective multiplication factor; a larger value is preferred.
        ppf_3d : float
            Three-dimensional segment power peaking factor; a smaller value is preferred.

        Returns
        -------
        tuple[float, float]
            ``(-keff, ppf_3d)`` without clipping, normalization, or weighting."""
        validated_keff = cls._require_finite_number("keff", keff)
        validated_ppf_3d = cls._require_finite_number("ppf_3d", ppf_3d)
        if validated_keff <= 0.0:
            raise ValueError("keff harus positif.")
        if validated_ppf_3d < 1.0:
            raise ValueError("ppf_3d tidak boleh lebih kecil dari 1.0.")
        return (-validated_keff, validated_ppf_3d)

    @staticmethod
    def _extract_keff(statepoint: Any) -> dict[str, float]:
        """Extract and validate keff mean and standard deviation.

        Parameters
        ----------
        statepoint : openmc.StatePoint-compatible object
            Statepoint exposing a keff uncertainty value.

        Returns
        -------
        dict
            Python floats for mean, standard deviation, and pcm uncertainty."""
        try:
            mean = float(statepoint.keff.nominal_value)
            std_dev = float(statepoint.keff.std_dev)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("Statepoint tidak menyediakan keff yang valid.") from error
        if not math.isfinite(mean) or mean <= 0.0:
            raise ValueError("Mean keff harus finite dan positif.")
        if not math.isfinite(std_dev) or std_dev < 0.0:
            raise ValueError("Standard deviation keff harus finite dan non-negatif.")
        return {"mean": mean, "std_dev": std_dev, "std_dev_pcm": float(std_dev * 1.0e5)}

    def _get_tally(self, statepoint: Any) -> Any:
        """Return the configured tally or raise a domain-specific error.

        Parameters
        ----------
        statepoint : openmc.StatePoint-compatible object
            Statepoint exposing ``get_tally``.

        Returns
        -------
        openmc.Tally-compatible object
            Tally with the configured name and score."""
        try:
            tally = statepoint.get_tally(name=self.tally_name)
        except (LookupError, ValueError) as error:
            raise ValueError(
                f"Tally {self.tally_name!r} tidak ditemukan dalam statepoint."
            ) from error
        scores = tuple(getattr(tally, "scores", ()))
        if self.tally_score not in scores:
            raise ValueError(
                f"Score {self.tally_score!r} tidak tersedia pada tally {self.tally_name!r}."
            )
        return tally

    @staticmethod
    def _extract_cell_filter_ids(tally: Any) -> tuple[int, ...]:
        """Extract ordered cell IDs from the tally's single CellFilter.

        Parameters
        ----------
        tally : openmc.Tally-compatible object
            Tally whose filters define the fuel-segment bin order.

        Returns
        -------
        tuple of int
            Cell IDs in exact tally-bin order."""
        filters = tuple(getattr(tally, "filters", ()))
        cell_filters = [
            tally_filter
            for tally_filter in filters
            if tally_filter.__class__.__name__ == "CellFilter"
        ]
        if len(cell_filters) != 1:
            raise ValueError("Tally harus memiliki tepat satu CellFilter.")
        bins = tuple(getattr(cell_filters[0], "bins", ()))
        return tuple(
            OpenMCResultEvaluator._require_integer("CellFilter cell_id", cell_id, 1)
            for cell_id in bins
        )

    def _extract_tally_values(
        self, tally: Any, tally_records: tuple[dict[str, Any], ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract validated tally mean and uncertainty arrays in record order.

        Parameters
        ----------
        tally : openmc.Tally-compatible object
            Configured fission-q tally.
        tally_records : tuple of dict
            Ordered record metadata used to verify CellFilter alignment.

        Returns
        -------
        tuple of numpy.ndarray
            One-dimensional mean and standard-deviation arrays."""
        # Match tally bins to positions only after verifying the exact CellFilter ID order.
        cell_filter_ids = self._extract_cell_filter_ids(tally)
        record_cell_ids = tuple(record["cell_id"] for record in tally_records)
        if cell_filter_ids != record_cell_ids:
            raise ValueError("Urutan cell ID pada CellFilter tidak sama dengan tally_records.")
        try:
            means = np.asarray(
                tally.get_values(scores=[self.tally_score], value="mean"), dtype=float
            ).reshape(-1)
            std_devs = np.asarray(
                tally.get_values(scores=[self.tally_score], value="std_dev"), dtype=float
            ).reshape(-1)
        except (TypeError, ValueError) as error:
            raise ValueError("Nilai mean atau std_dev tally tidak dapat dibaca.") from error
        expected_bin_count = len(tally_records)
        if means.size != expected_bin_count or std_devs.size != expected_bin_count:
            raise ValueError(
                f"Jumlah tally bin harus {expected_bin_count}, diperoleh mean={means.size} dan std_dev={std_devs.size}."
            )
        if not np.all(np.isfinite(means)) or not np.all(np.isfinite(std_devs)):
            raise ValueError("Seluruh nilai mean dan std_dev tally harus finite.")
        if np.any(means < 0.0) or np.any(std_devs < 0.0):
            raise ValueError("Nilai mean dan std_dev tally harus non-negatif.")
        if math.fsum(float(value) for value in means) <= 0.0:
            raise ValueError("Jumlah total tally mean harus lebih besar dari nol.")
        return means, std_devs

    @staticmethod
    def _calculate_power_results(
        tally_means: np.ndarray, tally_records: tuple[dict[str, Any], ...]
    ) -> dict[str, Any]:
        """Reconstruct relative segment, slot, and axial power distributions.

        Parameters
        ----------
        tally_means : numpy.ndarray
            Ordered fission-q-recoverable means used as relative-power proxies.
        tally_records : tuple of dict
            Metadata aligned one-to-one with ``tally_means``.

        Returns
        -------
        dict
            Relative power maps, PPF values, identities, and dynamic counts."""
        slot_totals = {}
        axial_totals = {}
        segment_scores = {}
        fuel_id_by_slot = {}

        for record, score_value in zip(tally_records, tally_means, strict=True):
            slot_name = record["slot_name"]
            axial_index = record["axial_index"]
            segment_score = float(score_value)
            slot_totals.setdefault(slot_name, 0.0)
            segment_scores.setdefault(slot_name, {})
            fuel_id_by_slot.setdefault(slot_name, record["gene_id"])
            axial_totals.setdefault(axial_index, 0.0)
            slot_totals[slot_name] += segment_score
            axial_totals[axial_index] += segment_score
            segment_scores[slot_name][axial_index] = segment_score

        total_score = math.fsum(float(value) for value in tally_means)
        mean_slot_power = total_score / len(slot_totals)
        mean_axial_power = total_score / len(axial_totals)
        mean_segment_power = total_score / len(tally_records)
        axial_key_width = max(2, len(str(max(axial_totals))))

        # Normalize each aggregation by its own mean; the result is not absolute power in watts.
        power_by_slot = {
            slot_name: float(slot_total / mean_slot_power)
            for slot_name, slot_total in slot_totals.items()
        }
        power_by_axial_index = {
            f"{axial_index:0{axial_key_width}d}": float(axial_total / mean_axial_power)
            for axial_index, axial_total in axial_totals.items()
        }
        power_by_segment = {
            slot_name: {
                f"{axial_index:0{axial_key_width}d}": float(segment_score / mean_segment_power)
                for axial_index, segment_score in slot_segments.items()
            }
            for slot_name, slot_segments in segment_scores.items()
        }

        # Equal segment volumes make integrated-score ratios equal segment power-density ratios.
        ppf_3d = float(max(float(value) for value in tally_means) / mean_segment_power)
        # Retain radial and axial PPF as diagnostics for plots and operators.
        radial_ppf = float(max(slot_totals.values()) / mean_slot_power)
        axial_ppf = float(max(axial_totals.values()) / mean_axial_power)
        return {
            "power_by_slot": power_by_slot,
            "power_by_axial_index": power_by_axial_index,
            "power_by_segment": power_by_segment,
            "fuel_id_by_slot": dict(fuel_id_by_slot),
            "ppf_3d": ppf_3d,
            "radial_ppf": radial_ppf,
            "axial_ppf": axial_ppf,
            "fuel_slot_count": len(slot_totals),
            "axial_segment_count": len(axial_totals),
        }

    @classmethod
    def _summarize_tally_uncertainty(
        cls,
        tally_means: np.ndarray,
        tally_std_devs: np.ndarray,
        tally_records: tuple[dict[str, Any], ...],
    ) -> dict[str, Any]:
        """Summarize tally uncertainty and identify the worst positive-mean bin.

        Parameters
        ----------
        tally_means, tally_std_devs : numpy.ndarray
            Validated mean and standard-deviation arrays.
        tally_records : tuple of dict
            Ordered physical metadata aligned with the tally arrays.

        Returns
        -------
        dict
            JSON-safe counts, aggregate relative uncertainties, and worst bin."""
        positive_mask = tally_means > 0.0
        relative_std_devs = tally_std_devs[positive_mask] / tally_means[positive_mask]
        positive_indices = np.flatnonzero(positive_mask)
        worst_relative_index = int(np.argmax(relative_std_devs))
        worst_bin_index = int(positive_indices[worst_relative_index])
        worst_record = tally_records[worst_bin_index]
        return {
            "total_bin_count": int(tally_means.size),
            "positive_mean_bin_count": int(np.count_nonzero(positive_mask)),
            "zero_mean_bin_count": int(np.count_nonzero(~positive_mask)),
            "mean_relative_std_dev": float(np.mean(relative_std_devs)),
            "median_relative_std_dev": float(np.median(relative_std_devs)),
            "p95_relative_std_dev": float(np.percentile(relative_std_devs, 95.0)),
            "max_relative_std_dev": float(relative_std_devs[worst_relative_index]),
            "worst_bin": {
                "fuel_slot": worst_record["slot_name"],
                "axial_segment": int(worst_record["axial_index"]),
                "cell_id": int(worst_record["cell_id"]),
                "relative_std_dev": float(relative_std_devs[worst_relative_index]),
            },
        }

    @classmethod
    def _extract_statepoint_statistics(cls, statepoint: Any, tally: Any) -> dict[str, Any]:
        """Extract statepoint run metadata and derived histories as JSON-safe values.

        Parameters
        ----------
        statepoint : openmc.StatePoint-compatible object
            Source of run metadata stored by OpenMC.
        tally : openmc.Tally-compatible object
            Evaluated tally whose realization count is reported.

        Returns
        -------
        dict
            Statepoint metadata and history counts using built-in JSON types."""
        current_batch = cls._optional_integer_attribute(statepoint, "current_batch")
        configured_batches = cls._optional_integer_attribute(statepoint, "n_batches")
        inactive_batches = cls._optional_integer_attribute(statepoint, "n_inactive")
        particles = cls._optional_integer_attribute(statepoint, "n_particles")
        generations_per_batch = cls._optional_integer_attribute(statepoint, "generations_per_batch")
        statepoint_realizations = cls._optional_integer_attribute(statepoint, "n_realizations")
        tally_realizations = cls._optional_integer_attribute(tally, "num_realizations")
        seed = cls._optional_integer_attribute(statepoint, "seed")
        active_batches_completed = None
        total_histories = None
        active_histories = None

        if current_batch is not None and inactive_batches is not None:
            active_batches_completed = max(current_batch - inactive_batches, 0)
        if (
            current_batch is not None
            and particles is not None
            and generations_per_batch is not None
        ):
            total_histories = current_batch * particles * generations_per_batch
        if (
            active_batches_completed is not None
            and particles is not None
            and generations_per_batch is not None
        ):
            active_histories = active_batches_completed * particles * generations_per_batch

        version = cls._optional_attribute(statepoint, "version")
        if isinstance(version, (tuple, list, np.ndarray)):
            version = ".".join(str(int(item)) for item in version)
        run_mode = cls._optional_attribute(statepoint, "run_mode")
        if isinstance(run_mode, bytes):
            run_mode = run_mode.decode("utf-8")

        return {
            "openmc_version": cls._to_json_safe(version),
            "run_mode": cls._to_json_safe(run_mode),
            "current_batch": current_batch,
            "configured_batches": configured_batches,
            "inactive_batches": inactive_batches,
            "active_batches_completed": active_batches_completed,
            "particles_per_generation": particles,
            "generations_per_batch": generations_per_batch,
            "statepoint_realizations": statepoint_realizations,
            "tally_realizations": tally_realizations,
            "total_histories": total_histories,
            "active_histories": active_histories,
            "seed": seed,
            "date_and_time": cls._to_json_safe(
                cls._optional_attribute(statepoint, "date_and_time")
            ),
            "runtime_seconds": cls._to_json_safe(cls._optional_attribute(statepoint, "runtime")),
        }

    @staticmethod
    def _optional_attribute(source: Any, name: str) -> Any:
        """Return an optional object attribute, or None when it is unavailable."""
        try:
            return getattr(source, name)
        except (AttributeError, KeyError):
            return None

    @classmethod
    def _optional_integer_attribute(cls, source: Any, name: str) -> int | None:
        """Return one optional integral metadata attribute as a built-in int."""
        value = cls._optional_attribute(source, name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"Metadata {name} harus berupa integer ketika tersedia.")
        return int(value)

    @classmethod
    def _to_json_safe(cls, value: Any) -> Any:
        """Recursively convert NumPy, datetime, bytes, and containers to JSON types."""
        if value is None or isinstance(value, (bool, str, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, np.generic):
            return cls._to_json_safe(value.item())
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, Mapping):
            return {str(key): cls._to_json_safe(item) for key, item in value.items()}
        if isinstance(value, np.ndarray):
            return [cls._to_json_safe(item) for item in value.tolist()]
        if isinstance(value, (list, tuple)):
            return [cls._to_json_safe(item) for item in value]
        if isinstance(value, Integral):
            return int(value)
        if isinstance(value, Real):
            return float(value)
        raise TypeError(f"Nilai metadata {type(value).__name__} tidak dapat diserialisasi ke JSON.")
