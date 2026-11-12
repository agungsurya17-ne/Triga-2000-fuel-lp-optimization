"""Test constraint reporting without neutron transport.

Run with ``python -m unittest discover -s tests -v``."""

import ast
import csv
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME_ROOT / "src"))

from modules.config_loader import ConfigLoader
from modules.keff_constraint import KeffConstraint
from modules.nsga2 import extract_eligible_candidates, rank_candidates
from modules.optimization_reporting import OptimizationReporter, OptimizationReportingError

CONTRACT = KeffConstraint(1.0, 0.0)


def reporting_state(metrics):
    """Return a completed replacement report fixture for ID-to-(keff, PPF) ``metrics``."""
    individuals = {}
    for identifier, (keff, ppf) in metrics.items():
        results = {
            "keff": {"mean": keff},
            "ppf_3d": ppf,
            "objectives": [-keff, ppf],
            "constraint": CONTRACT.evaluate(keff),
        }
        individuals[identifier] = {"status": "completed", "results": results}
    fronts = rank_candidates(extract_eligible_candidates(individuals)).fronts
    ids = list(individuals)
    replacement = {
        "constraint_contract": CONTRACT.identity(),
        "objectives": ["-keff", "ppf_3d"],
        "fronts": [list(front) for front in fronts],
        "eligible_parent_ids": ids[:1],
        "eligible_offspring_ids": ids[1:],
    }
    return {
        "constraint_contract": CONTRACT.identity(),
        "individuals": individuals,
        "generations": [
            {
                "generation": 0,
                "status": "completed",
                "survivor_ids": ids,
                "summary": {"replacement": replacement},
            }
        ],
    }


class ConstraintReportingTests(unittest.TestCase):
    """Check actual CSV contents, plot membership, empty solutions, and fail-closed reporting."""

    def test_pareto_csv_is_feasible_only_and_plot_shows_full_selection_scope(self):
        """Keep Pareto CSV feasible-only while plotting all eligible points and survivors."""
        state = reporting_state({"F": (1.02, 1.2), "DOMINATED": (1.01, 1.3), "I": (0.99, 1.0)})
        figures = {}

        def capture(figure, destination):
            """Capture generated figure ``destination`` for membership assertions without writing PNG."""
            figures[destination.name] = figure

        with tempfile.TemporaryDirectory() as directory, patch.object(
            OptimizationReporter, "_atomic_save_figure", side_effect=capture
        ):
            report = OptimizationReporter(directory).generate(state)
            with report.pareto_csv_files[0].open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["individual_id"] for row in rows], ["F"])
            self.assertEqual(rows[0]["feasible"], "True")
            with report.history_csv.open() as stream:
                diagnostics = {row["individual_id"]: row for row in csv.DictReader(stream)}
            self.assertEqual(diagnostics["I"]["feasible"], "False")
            self.assertAlmostEqual(float(diagnostics["I"]["keff_violation"]), 0.01)
            axis = figures["pareto_generation_000.png"].axes[0]
            self.assertCountEqual(
                axis.collections[0].get_offsets().tolist(),
                [[1.2, -1.02], [1.3, -1.01], [1.0, -0.99]],
            )
            self.assertEqual(axis.collections[1].get_offsets().tolist(), [[1.2, -1.02]])
            self.assertCountEqual(
                axis.collections[2].get_offsets().tolist(),
                [[1.2, -1.02], [1.3, -1.01], [1.0, -0.99]],
            )

    def test_no_feasible_means_empty_solution_csv_and_explicit_status(self):
        """Keep internal infeasible fronts but emit no final Pareto solution or fallback."""
        state = reporting_state({"A": (0.95, 1.2), "B": (0.99, 1.3)})
        with tempfile.TemporaryDirectory() as directory:
            report = OptimizationReporter(directory).generate(state)
            with report.pareto_csv_files[0].open() as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])
            with report.generation_summary_csv.open() as stream:
                summary = list(csv.DictReader(stream))[0]
            self.assertEqual(summary["feasible_pareto_count"], "0")
            self.assertEqual(summary["solution_status"], "no_feasible_solutions")
            self.assertTrue(report.pareto_plots[0].is_file())

    def test_stale_fronts_or_missing_metadata_rejected_before_output(self):
        """Reject stale ordinary-Pareto membership and absent contract metadata before writing files."""
        base = reporting_state({"F": (1.0, 3.0), "I": (0.9, 1.0)})
        for mode in ("fronts", "candidate", "contract", "threshold"):
            state = deepcopy(base)
            if mode == "fronts":
                state["generations"][0]["summary"]["replacement"]["fronts"] = [["F", "I"]]
            elif mode == "candidate":
                del state["individuals"]["I"]["results"]["constraint"]
            elif mode == "contract":
                del state["constraint_contract"]
            else:
                state["constraint_contract"]["keff_min"] = 1.01
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(OptimizationReportingError):
                    OptimizationReporter(directory).generate(state)
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_active_runtime_contract_matches_main_source(self):
        """Compare shared modules and changed checkpoint/composition methods with main implementation."""
        main_root = RUNTIME_ROOT.parent
        shared_names = (
            "keff_constraint.py",
            "nsga2.py",
            "parent_selection.py",
            "single_candidate_pipeline.py",
        )
        scoped_methods = (
            (
                "modules/checkpoint.py",
                (
                    "create_state",
                    "_require_constraint_contract",
                    "mark_individual_completed",
                    "mark_individual_not_converged",
                    "validate_compatibility",
                    "validate_state",
                ),
            ),
            ("main.py", ("select_parent_ids", "complete_active_generation")),
        )
        source_paths = [main_root / "modules" / name for name in shared_names]
        source_paths.extend(main_root / name for name, _ in scoped_methods)
        if not all(path.is_file() for path in source_paths):
            self.skipTest("Sumber induk tidak tersedia dalam paket runtime mandiri.")

        def normalized_tree(path):
            """Read Python ``path`` and return an AST with docstrings removed for implementation comparison."""
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    if (
                        node.body
                        and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)
                    ):
                        node.body.pop(0)
            return tree

        for name in shared_names:
            self.assertEqual(
                ast.dump(normalized_tree(main_root / "modules" / name)),
                ast.dump(normalized_tree(RUNTIME_ROOT / "src/modules" / name)),
            )
        for name, methods in scoped_methods:
            implementations = []
            for path in (main_root / name, RUNTIME_ROOT / "src" / name):
                tree = normalized_tree(path)
                implementations.append(
                    {
                        node.name: ast.dump(node)
                        for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name in methods
                    }
                )
            self.assertEqual(implementations[0], implementations[1])
        ga = ConfigLoader(RUNTIME_ROOT / "config/config.ini").get_params()["ga"]
        self.assertEqual((ga["keff_min"], ga["feasibility_tolerance"]), (1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
