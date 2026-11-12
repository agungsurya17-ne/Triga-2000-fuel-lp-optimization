"""Test ppf 3d runtime without neutron transport.

Run with ``python -m unittest discover -s tests -v``."""

import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = RUNTIME_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from modules.OpenMCResultEvaluator import OpenMCResultEvaluator
from modules.nsga2 import dominates, NSGA2Candidate
from modules.keff_constraint import KeffConstraint

TEST_CONSTRAINT = KeffConstraint(1.0, 0.0)
from modules.optimization_reporting import OptimizationReporter, OptimizationReportingError


def make_segment_records() -> tuple[dict[str, object], ...]:
    """Return three fuel pins with fifteen ordered axial segment records each."""
    records = []
    cell_id = 1
    for genome_index, slot_name in enumerate(("C1", "C2", "C3")):
        for axial_index in range(1, 16):
            records.append(
                {
                    "cell_id": cell_id,
                    "slot_name": slot_name,
                    "genome_index": genome_index,
                    "gene_id": f"fuel_{genome_index}",
                    "axial_index": axial_index,
                }
            )
            cell_id += 1
    return tuple(records)


def make_reporting_state() -> dict[str, object]:
    """Return a minimal completed generation using the PPF 3D objective contract."""
    individuals = {
        "P0": {
            "status": "completed",
            "results": {
                "keff": {"mean": 1.02},
                "ppf_3d": 1.40,
                "radial_ppf": 1.10,
                "objectives": [-1.02, 1.40],
            },
        },
        "O0": {
            "status": "completed",
            "results": {
                "keff": {"mean": 1.01},
                "ppf_3d": 1.30,
                "radial_ppf": 1.05,
                "objectives": [-1.01, 1.30],
            },
        },
    }
    for record in individuals.values():
        record["results"]["constraint"] = TEST_CONSTRAINT.from_results(record["results"])
    replacement = {
        "constraint_contract": TEST_CONSTRAINT.identity(),
        "objectives": ["-keff", "ppf_3d"],
        "eligible_parent_ids": ["P0"],
        "eligible_offspring_ids": ["O0"],
        "fronts": [["P0", "O0"]],
        "survivor_ids": ["P0"],
    }
    generation = {
        "generation": 0,
        "status": "completed",
        "survivor_ids": ["P0"],
        "summary": {"replacement": replacement},
    }
    return {
        "constraint_contract": TEST_CONSTRAINT.identity(),
        "individuals": individuals,
        "generations": [generation],
    }


class RuntimePPF3DTests(unittest.TestCase):
    """Check segment aggregation, objective wiring, and generated report fields."""

    def test_ppf_3d_uses_segments_without_pin_summation(self) -> None:
        """Distinguish a local segment peak from equal radial pin totals."""
        records = make_segment_records()
        tally_means = np.asarray([15.0] + [0.0] * 14 + [1.0] * 30)
        results = OpenMCResultEvaluator._calculate_power_results(tally_means, records)
        self.assertEqual(results["radial_ppf"], 1.0)
        self.assertEqual(results["ppf_3d"], 15.0)
        self.assertEqual(
            OpenMCResultEvaluator.build_objectives(1.02, results["ppf_3d"]), (-1.02, 15.0)
        )

    def test_reporting_uses_ppf_3d_columns_and_outputs(self) -> None:
        """Generate CSV and plots whose second physical objective is PPF 3D."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            reporter = OptimizationReporter(temporary_directory)
            report = reporter.generate(make_reporting_state())
            with report.history_csv.open(encoding="utf-8", newline="") as stream:
                history_rows = list(csv.reader(stream))
            with report.generation_summary_csv.open(encoding="utf-8", newline="") as stream:
                summary_rows = list(csv.reader(stream))
            self.assertEqual(
                history_rows[0],
                ["generation", "individual_id", "keff", "ppf_3d", "feasible", "keff_violation"],
            )
            self.assertEqual(
                summary_rows[0],
                [
                    "generation",
                    "survivor_count",
                    "max_keff",
                    "min_ppf_3d",
                    "feasible_pareto_count",
                    "solution_status",
                ],
            )
            self.assertTrue(report.history_plot.is_file())
            self.assertTrue(report.pareto_plots[0].is_file())

    def test_reporting_rejects_radial_or_inconsistent_objectives(self) -> None:
        """Fail closed for old radial metadata and mismatched objective values."""
        old_state = make_reporting_state()
        old_state["generations"][0]["summary"]["replacement"]["objectives"] = [
            "-keff",
            "radial_ppf",
        ]
        with tempfile.TemporaryDirectory() as temporary_directory, self.assertRaisesRegex(
            OptimizationReportingError, "ppf_3d"
        ):
            OptimizationReporter(temporary_directory).generate(old_state)

        inconsistent_state = make_reporting_state()
        inconsistent_state["individuals"]["P0"]["results"]["objectives"][1] = 9.99
        with tempfile.TemporaryDirectory() as temporary_directory, self.assertRaisesRegex(
            OptimizationReportingError, "tidak konsisten"
        ):
            OptimizationReporter(temporary_directory).generate(inconsistent_state)

    def test_minimization_prefers_larger_keff_at_equal_ppf_3d(self) -> None:
        """Confirm that minimizing negative keff prefers the larger physical keff."""
        higher_keff = OpenMCResultEvaluator.build_objectives(1.05, 1.30)
        lower_keff = OpenMCResultEvaluator.build_objectives(1.02, 1.30)
        self.assertEqual(higher_keff, (-1.05, 1.30))
        self.assertTrue(
            dominates(
                NSGA2Candidate("A", higher_keff, TEST_CONSTRAINT),
                NSGA2Candidate("B", lower_keff, TEST_CONSTRAINT),
            )
        )
        self.assertFalse(
            dominates(
                NSGA2Candidate("B", lower_keff, TEST_CONSTRAINT),
                NSGA2Candidate("A", higher_keff, TEST_CONSTRAINT),
            )
        )

    def test_objective_plots_display_negative_keff(self) -> None:
        """Verify that objective plots use raw negative-keff values and minimization labels."""
        captured_figures = {}

        def capture_figure(figure: object, destination: Path) -> None:
            """Retain each generated figure by filename for plot-coordinate assertions."""
            captured_figures[destination.name] = figure

        with tempfile.TemporaryDirectory() as temporary_directory, patch.object(
            OptimizationReporter, "_atomic_save_figure", side_effect=capture_figure
        ):
            OptimizationReporter(temporary_directory).generate(make_reporting_state())

        history_axis = captured_figures["optimization_history.png"].axes[0]
        pareto_axis = captured_figures["pareto_generation_000.png"].axes[0]
        np.testing.assert_allclose(history_axis.collections[0].get_offsets()[:, 1], [-1.02])
        np.testing.assert_allclose(pareto_axis.collections[0].get_offsets()[:, 1], [-1.02, -1.01])
        self.assertEqual(history_axis.get_ylabel(), "-keff (lebih kecil lebih baik)")
        self.assertEqual(pareto_axis.get_ylabel(), "-keff (lebih kecil lebih baik)")
        self.assertEqual(pareto_axis.get_xlabel(), "PPF 3D (lebih kecil lebih baik)")


if __name__ == "__main__":
    unittest.main()
