"""Write optimization CSV and PNG reports from completed checkpoint data.

OptimizationReporter gathers survivor histories and feasible Pareto fronts,
extracts (-keff, ppf_3d), and writes reports atomically without changing campaign
state or candidate ranking."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
from dataclasses import dataclass
import math
from numbers import Real
import os
from pathlib import Path
import tempfile
from typing import Any

from modules.keff_constraint import KeffConstraint
from modules.nsga2 import extract_eligible_candidates, rank_candidates

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


class OptimizationReportingError(RuntimeError):
    """Report invalid checkpoint data or a failed report write."""


@dataclass(frozen=True)
class ObjectivePoint:
    """Store one completed individual's physical objectives for plotting."""

    individual_id: str
    keff: float
    ppf_3d: float
    feasible: bool
    keff_violation: float

    @property
    def negative_keff(self) -> float:
        """Return the raw minimization objective corresponding to physical ``keff``."""
        return -self.keff


@dataclass(frozen=True)
class GenerationReportData:
    """Store one completed generation's eligible population, front, and survivors."""

    generation: int
    eligible_points: tuple[ObjectivePoint, ...]
    fronts: tuple[tuple[str, ...], ...]
    survivor_ids: tuple[str, ...]


@dataclass(frozen=True)
class OptimizationReportingResult:
    """Summarize the report files generated from completed checkpoint generations."""

    completed_generations: tuple[int, ...]
    history_plot: Path | None
    history_csv: Path | None
    generation_summary_csv: Path | None
    pareto_plots: tuple[Path, ...]
    pareto_csv_files: tuple[Path, ...]


class OptimizationReporter:
    """Render objective history and per-generation Pareto reports from durable state."""

    def __init__(self, output_directory: str | Path) -> None:
        """Store ``output_directory`` as the destination and return no value."""
        self.output_directory = Path(output_directory)

    def generate(self, state: Mapping[str, Any]) -> OptimizationReportingResult:
        """Validate checkpoint ``state``, write all completed reports, and return their paths."""
        generation_data = self._collect_completed_generations(state)
        if not generation_data:
            return OptimizationReportingResult((), None, None, None, (), ())
        self.output_directory.mkdir(parents=True, exist_ok=True)
        history_plot = self.output_directory / "optimization_history.png"
        history_csv = self.output_directory / "optimization_history.csv"
        generation_summary_csv = self.output_directory / "generation_summary.csv"
        self._write_history_csv(generation_data, history_csv)
        self._write_generation_summary_csv(generation_data, generation_summary_csv)
        self._plot_history(generation_data, history_plot)
        pareto_plots: list[Path] = []
        pareto_csv_files: list[Path] = []
        for generation in generation_data:
            pareto_plot = (
                self.output_directory / f"pareto_generation_{generation.generation:03d}.png"
            )
            pareto_csv = (
                self.output_directory / f"pareto_generation_{generation.generation:03d}.csv"
            )
            self._write_pareto_csv(generation, pareto_csv)
            self._plot_pareto(generation, pareto_plot)
            pareto_plots.append(pareto_plot)
            pareto_csv_files.append(pareto_csv)
        return OptimizationReportingResult(
            tuple(item.generation for item in generation_data),
            history_plot,
            history_csv,
            generation_summary_csv,
            tuple(pareto_plots),
            tuple(pareto_csv_files),
        )

    def _collect_completed_generations(
        self, state: Mapping[str, Any]
    ) -> tuple[GenerationReportData, ...]:
        """Extract completed generation data from ``state`` and return it in numeric order."""
        if not isinstance(state, Mapping):
            raise OptimizationReportingError(
                "Checkpoint state untuk reporting harus berupa mapping."
            )
        individuals = state.get("individuals")
        generations = state.get("generations")
        if not isinstance(individuals, Mapping) or not isinstance(generations, list):
            raise OptimizationReportingError(
                "Checkpoint reporting memerlukan individuals mapping dan generations list."
            )
        completed_records = [
            record
            for record in generations
            if isinstance(record, Mapping) and record.get("status") == "completed"
        ]
        generation_numbers = [record.get("generation") for record in completed_records]
        if any(
            isinstance(number, bool) or not isinstance(number, int) or number < 0
            for number in generation_numbers
        ):
            raise OptimizationReportingError(
                "Generation completed memiliki identifier non-negatif yang tidak valid."
            )
        if len(set(generation_numbers)) != len(generation_numbers):
            raise OptimizationReportingError("Generation completed memiliki identifier duplikat.")
        try:
            contract = KeffConstraint.from_identity(state.get("constraint_contract"))
            return tuple(
                self._build_generation_data(record, individuals, contract)
                for record in sorted(completed_records, key=lambda item: item["generation"])
            )
        except ValueError as error:
            raise OptimizationReportingError(str(error)) from error

    def _build_generation_data(
        self,
        generation_record: Mapping[str, Any],
        individuals: Mapping[str, Any],
        contract: KeffConstraint,
    ) -> GenerationReportData:
        """Build and return validated report data for one completed ``generation_record``."""
        generation = int(generation_record["generation"])
        summary = generation_record.get("summary")
        replacement = summary.get("replacement") if isinstance(summary, Mapping) else None
        if not isinstance(replacement, Mapping):
            raise OptimizationReportingError(
                f"Generation {generation} tidak memiliki summary replacement."
            )
        if replacement.get("objectives") != ["-keff", "ppf_3d"]:
            raise OptimizationReportingError(
                f"Generation {generation} tidak memakai kontrak objective (-keff, ppf_3d)."
            )
        if replacement.get("constraint_contract") != contract.identity():
            raise OptimizationReportingError(
                "Kontrak constraint replacement tidak konsisten dengan checkpoint."
            )
        fronts_value = replacement.get("fronts")
        if (
            not isinstance(fronts_value, list)
            or not fronts_value
            or any(not isinstance(front, list) or not front for front in fronts_value)
        ):
            raise OptimizationReportingError(
                f"Generation {generation} tidak memiliki daftar Pareto front yang valid."
            )
        fronts = tuple(tuple(front) for front in fronts_value)
        front_ids = tuple(individual_id for front in fronts for individual_id in front)
        if any(
            not isinstance(individual_id, str) or not individual_id for individual_id in front_ids
        ) or len(set(front_ids)) != len(front_ids):
            raise OptimizationReportingError(
                f"Generation {generation} memiliki ID Pareto tidak valid atau duplikat."
            )
        eligible_parent_ids = self._require_id_sequence(
            replacement.get("eligible_parent_ids"), generation, "eligible_parent_ids"
        )
        eligible_offspring_ids = self._require_id_sequence(
            replacement.get("eligible_offspring_ids"), generation, "eligible_offspring_ids"
        )
        eligible_ids = eligible_parent_ids + eligible_offspring_ids
        if (
            len(set(eligible_ids)) != len(eligible_ids)
            or set(eligible_ids) != set(front_ids)
            or len(eligible_ids) != len(front_ids)
        ):
            raise OptimizationReportingError(
                f"Generation {generation} memiliki membership eligible dan Pareto front yang berbeda."
            )
        survivor_ids = self._require_id_sequence(
            generation_record.get("survivor_ids"), generation, "survivor_ids"
        )
        if not survivor_ids:
            raise OptimizationReportingError(
                f"Generation {generation} tidak memiliki survivor untuk history objective."
            )
        if not set(survivor_ids).issubset(front_ids):
            raise OptimizationReportingError(
                f"Generation {generation} memiliki survivor di luar kandidat eligible."
            )
        eligible_points = tuple(
            self._extract_objective_point(individual_id, individuals, generation)
            for individual_id in front_ids
        )
        for individual_id in front_ids:
            contract.validate_results(individuals[individual_id]["results"])
        ranking = rank_candidates(
            extract_eligible_candidates({key: individuals[key] for key in front_ids})
        )
        if tuple(tuple(sorted(front)) for front in fronts) != ranking.fronts:
            raise OptimizationReportingError(
                "Front tersimpan tidak konsisten dengan constrained ranking; report ditolak."
            )
        return GenerationReportData(generation, eligible_points, fronts, survivor_ids)

    @staticmethod
    def _require_id_sequence(value: Any, generation: int, field_name: str) -> tuple[str, ...]:
        """Validate an ID sequence ``value`` for ``field_name`` and return an immutable tuple."""
        if not isinstance(value, list) or any(
            not isinstance(individual_id, str) or not individual_id for individual_id in value
        ):
            raise OptimizationReportingError(
                f"Generation {generation} memiliki {field_name} yang tidak valid."
            )
        identifiers = tuple(value)
        if len(set(identifiers)) != len(identifiers):
            raise OptimizationReportingError(
                f"Generation {generation} memiliki ID duplikat pada {field_name}."
            )
        return identifiers

    def _extract_objective_point(
        self, individual_id: str, individuals: Mapping[str, Any], generation: int
    ) -> ObjectivePoint:
        """Extract and return one completed ``individual_id`` objective point for ``generation``."""
        record = individuals.get(individual_id)
        if not isinstance(record, Mapping) or record.get("status") != "completed":
            raise OptimizationReportingError(
                f"Kandidat eligible {individual_id!r} generation {generation} tidak completed."
            )
        results = record.get("results")
        keff = results.get("keff") if isinstance(results, Mapping) else None
        if not isinstance(keff, Mapping):
            raise OptimizationReportingError(
                f"Kandidat {individual_id!r} tidak memiliki results.keff."
            )
        keff_mean = self._finite_float(keff.get("mean"), f"keff {individual_id}")
        ppf_3d = self._finite_float(results.get("ppf_3d"), f"ppf_3d {individual_id}")
        objectives = results.get("objectives")
        if not isinstance(objectives, (list, tuple)) or len(objectives) != 2:
            raise OptimizationReportingError(
                f"Kandidat {individual_id!r} tidak memiliki dua objective."
            )
        negative_keff = self._finite_float(
            objectives[0], f"objective negative_keff {individual_id}"
        )
        objective_ppf_3d = self._finite_float(objectives[1], f"objective ppf_3d {individual_id}")
        if negative_keff != -keff_mean or objective_ppf_3d != ppf_3d:
            raise OptimizationReportingError(
                f"Objective kandidat {individual_id!r} tidak konsisten dengan keff dan ppf_3d."
            )
        contract = KeffConstraint.from_identity(results.get("constraint"))
        constraint = contract.validate_results(results)
        return ObjectivePoint(
            individual_id, keff_mean, ppf_3d, constraint["feasible"], constraint["keff_violation"]
        )

    @staticmethod
    def _finite_float(value: Any, field_name: str) -> float:
        """Validate numeric ``value`` for ``field_name`` and return a finite float."""
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise OptimizationReportingError(f"Nilai {field_name} harus numerik finite.")
        return float(value)

    @staticmethod
    def _survivor_points(generation: GenerationReportData) -> tuple[ObjectivePoint, ...]:
        """Return points whose IDs belong to the survivor population of ``generation``."""
        points_by_id = {point.individual_id: point for point in generation.eligible_points}
        return tuple(points_by_id[individual_id] for individual_id in generation.survivor_ids)

    def _write_history_csv(
        self, generation_data: Sequence[GenerationReportData], destination: Path
    ) -> None:
        """Write survivor objective rows from ``generation_data`` atomically to ``destination``."""
        rows = []
        for generation in generation_data:
            for point in self._survivor_points(generation):
                rows.append(
                    (
                        generation.generation,
                        point.individual_id,
                        point.keff,
                        point.ppf_3d,
                        point.feasible,
                        point.keff_violation,
                    )
                )
        self._atomic_write_csv(
            destination,
            ("generation", "individual_id", "keff", "ppf_3d", "feasible", "keff_violation"),
            rows,
        )

    def _write_generation_summary_csv(
        self, generation_data: Sequence[GenerationReportData], destination: Path
    ) -> None:
        """Write objective-wise survivor extrema per generation atomically to ``destination``."""
        rows = []
        for generation in generation_data:
            survivors = self._survivor_points(generation)
            rows.append(
                (
                    generation.generation,
                    len(survivors),
                    max(point.keff for point in survivors),
                    min(point.ppf_3d for point in survivors),
                    len(self._pareto_points(generation)),
                    (
                        "feasible_solutions"
                        if self._pareto_points(generation)
                        else "no_feasible_solutions"
                    ),
                )
            )
        self._atomic_write_csv(
            destination,
            (
                "generation",
                "survivor_count",
                "max_keff",
                "min_ppf_3d",
                "feasible_pareto_count",
                "solution_status",
            ),
            rows,
        )

    @staticmethod
    def _pareto_points(generation: GenerationReportData) -> tuple[ObjectivePoint, ...]:
        """Return feasible nondominated solutions in the parent-offspring scope of ``generation``."""
        first_front_ids = set(generation.fronts[0])
        return tuple(
            point
            for point in generation.eligible_points
            if point.feasible and point.individual_id in first_front_ids
        )

    def _write_pareto_csv(self, generation: GenerationReportData, destination: Path) -> None:
        """Write only feasible nondominated solutions for ``generation`` to ``destination``."""
        rows = [
            (
                generation.generation,
                point.individual_id,
                point.keff,
                point.ppf_3d,
                point.feasible,
                point.keff_violation,
                point.individual_id in generation.survivor_ids,
            )
            for point in self._pareto_points(generation)
        ]
        self._atomic_write_csv(
            destination,
            (
                "generation",
                "individual_id",
                "keff",
                "ppf_3d",
                "feasible",
                "keff_violation",
                "is_survivor",
            ),
            rows,
        )

    def _plot_history(
        self, generation_data: Sequence[GenerationReportData], destination: Path
    ) -> None:
        """Render survivor keff and 3D PPF histories from ``generation_data`` to ``destination``."""
        figure = Figure(figsize=(10, 8), constrained_layout=True)
        FigureCanvasAgg(figure)
        keff_axis, ppf_axis = figure.subplots(2, 1, sharex=True)
        generation_numbers = [generation.generation for generation in generation_data]
        minimum_negative_keff = []
        minimum_ppf_3d = []
        for generation in generation_data:
            survivors = self._survivor_points(generation)
            for feasible, marker in ((True, "o"), (False, "x")):
                points = tuple(point for point in survivors if point.feasible is feasible)
                label = (
                    ("Feasible" if feasible else "Infeasible; violation pada CSV")
                    if generation is generation_data[0]
                    else None
                )
                keff_axis.scatter(
                    [generation.generation] * len(points),
                    [point.negative_keff for point in points],
                    color="#4C78A8",
                    alpha=0.35,
                    s=20,
                    marker=marker,
                    label=label,
                )
                ppf_axis.scatter(
                    [generation.generation] * len(points),
                    [point.ppf_3d for point in points],
                    color="#F58518",
                    alpha=0.35,
                    s=20,
                    marker=marker,
                    label=label,
                )
            minimum_negative_keff.append(min(point.negative_keff for point in survivors))
            minimum_ppf_3d.append(min(point.ppf_3d for point in survivors))
        keff_axis.plot(
            generation_numbers,
            minimum_negative_keff,
            color="#1F4E79",
            marker="o",
            linewidth=2,
            label="Minimum -keff survivor",
        )
        ppf_axis.plot(
            generation_numbers,
            minimum_ppf_3d,
            color="#B24A00",
            marker="o",
            linewidth=2,
            label="Minimum PPF 3D survivor",
        )
        keff_axis.set_ylabel("-keff (lebih kecil lebih baik)")
        ppf_axis.set_ylabel("PPF 3D")
        ppf_axis.set_xlabel("Indeks generasi (0-based)")
        keff_axis.set_title("Riwayat Objective Survivor NSGA-II")
        for axis in (keff_axis, ppf_axis):
            axis.grid(True, alpha=0.25)
            axis.legend()
        self._atomic_save_figure(figure, destination)

    def _plot_pareto(self, generation: GenerationReportData, destination: Path) -> None:
        """Render eligible candidates and first Pareto front for ``generation`` to ``destination``."""
        figure = Figure(figsize=(9, 7), constrained_layout=True)
        FigureCanvasAgg(figure)
        axis = figure.subplots(1, 1)
        points_by_id = {point.individual_id: point for point in generation.eligible_points}
        all_points = generation.eligible_points
        first_front = self._pareto_points(generation)
        survivors = tuple(points_by_id[individual_id] for individual_id in generation.survivor_ids)
        axis.scatter(
            [point.ppf_3d for point in all_points],
            [point.negative_keff for point in all_points],
            color="#BAB0AC",
            alpha=0.65,
            s=30,
            label="Kandidat eligible",
        )
        ordered_front = sorted(
            first_front, key=lambda point: (point.ppf_3d, point.negative_keff, point.individual_id)
        )
        axis.plot(
            [point.ppf_3d for point in ordered_front],
            [point.negative_keff for point in ordered_front],
            color="#E45756",
            linewidth=1.5,
            alpha=0.8,
        )
        axis.scatter(
            [point.ppf_3d for point in first_front],
            [point.negative_keff for point in first_front],
            color="#E45756",
            s=55,
            label="Pareto front pertama (feasible)",
            zorder=3,
        )
        axis.scatter(
            [point.ppf_3d for point in survivors],
            [point.negative_keff for point in survivors],
            facecolors="none",
            edgecolors="#1F77B4",
            linewidths=1.2,
            s=85,
            label="Survivor",
            zorder=4,
        )
        if not first_front:
            axis.text(
                0.5,
                0.5,
                "Tidak ada solusi feasible; Pareto solution set kosong.",
                ha="center",
                transform=axis.transAxes,
            )
        axis.set_xlabel("PPF 3D (lebih kecil lebih baik)")
        axis.set_ylabel("-keff (lebih kecil lebih baik)")
        axis.set_title(f"Pareto Front Generation {generation.generation:03d}")
        axis.grid(True, alpha=0.25)
        axis.legend()
        self._atomic_save_figure(figure, destination)

    @staticmethod
    def _atomic_write_csv(
        destination: Path, header: Iterable[str], rows: Iterable[Iterable[Any]]
    ) -> None:
        """Atomically write ``header`` and ``rows`` to CSV ``destination`` and return no value."""
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerows(rows)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, destination)
        except Exception as error:
            temporary_path.unlink(missing_ok=True)
            raise OptimizationReportingError(f"Gagal menulis report CSV {destination}.") from error

    @staticmethod
    def _atomic_save_figure(figure: Figure, destination: Path) -> None:
        """Atomically save ``figure`` as PNG ``destination`` and return no value."""
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        try:
            figure.savefig(
                temporary_path,
                format="png",
                dpi=160,
                metadata={"Software": "ta_nsga2_runtime OptimizationReporter"},
            )
            os.replace(temporary_path, destination)
        except Exception as error:
            temporary_path.unlink(missing_ok=True)
            raise OptimizationReportingError(f"Gagal menulis report PNG {destination}.") from error
        finally:
            figure.clear()
