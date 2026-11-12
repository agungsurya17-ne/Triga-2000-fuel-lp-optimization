"""Test offspring evaluation scheduler without neutron transport.

Run with ``python -m unittest discover -s tests -v``."""

from __future__ import annotations

import sys
from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME_ROOT / "src"))

from modules.keff_constraint import KeffConstraint


from collections import Counter
from pathlib import Path
import tempfile
from copy import deepcopy
import unittest
from itertools import permutations

from modules.checkpoint import CheckpointManager, CheckpointValidationError
from modules.genome_sorter import GenomeSorter
from modules.input_provenance import build_input_provenance
from modules.offspring_evaluation_scheduler import (
    OffspringEvaluationError,
    OffspringEvaluationScheduler,
)
from modules.single_candidate_pipeline import SingleCandidatePipeline


TEST_CONSTRAINT = KeffConstraint(1.0, 0.0)


class FakeBuilderFactory:
    """Build plain Python models and retain each checkpoint-derived genome."""

    def __init__(self) -> None:
        """Create an empty record of fake model construction calls."""
        self.build_calls = []

    def __call__(self, genome, material_file, tally_name, tally_score):
        """Return a fake builder for one complete physical genome."""
        factory = self

        class FakeBuilder:
            """Return one model dictionary and ordered fake tally records."""

            def build_model(self, **settings):
                """Record injected settings and return fake builder output."""
                factory.build_calls.append(
                    {
                        "genome": tuple(genome),
                        "settings": dict(settings),
                        "material_file": Path(material_file),
                    }
                )
                return {"genome": tuple(genome)}, {
                    "tally_name": tally_name,
                    "tally_score": tally_score,
                    "tally_records": ({"cell_index": 0},),
                }

        return FakeBuilder()


class FakeRunner:
    """Create fake statepoints and inject deterministic technical failures."""

    def __init__(self, work_root, failures_before_success=None) -> None:
        """Store output root, per-child failure counts, and call history."""
        self.work_root = Path(work_root)
        self.failures_before_success = dict(failures_before_success or {})
        self.calls = []

    def run(self, model, run_id):
        """Record a fake transport attempt or return one fake statepoint path."""
        individual_id = self._individual_id(run_id)
        self.calls.append({"individual_id": individual_id, "run_id": run_id, "model": model})
        remaining = self.failures_before_success.get(individual_id, 0)
        if remaining > 0:
            self.failures_before_success[individual_id] = remaining - 1
            raise RuntimeError(f"fake transport failure for {individual_id}")
        work_dir = self.work_root / run_id
        work_dir.mkdir(parents=True, exist_ok=True)
        statepoint_path = work_dir / "statepoint.fake.h5"
        statepoint_path.write_bytes(b"fake statepoint")
        return {"statepoint_path": statepoint_path, "work_dir": work_dir}

    @staticmethod
    def _individual_id(run_id):
        """Extract the durable offspring ID from an attempt run ID."""
        run_prefix = run_id.rsplit("-A", 1)[0]
        return run_prefix[run_prefix.rfind("-G") + 1 :]


class FakeEvaluator:
    """Return small JSON-safe objectives with configurable convergence modes."""

    tally_name = "fake_tally"
    tally_score = "fake_score"

    def __init__(self, modes=None) -> None:
        """Store per-child terminal convergence modes and evaluator calls."""
        self.modes = dict(modes or {})
        self.calls = []

    def evaluate(self, statepoint_path, tally_records):
        """Return deterministic fake objectives for the statepoint's child ID."""
        individual_id = FakeRunner._individual_id(Path(statepoint_path).parent.name)
        self.calls.append({"individual_id": individual_id, "tally_records": tuple(tally_records)})
        return {
            "keff": {"mean": -(-1.0)},
            "objectives": [-1.0, 1.1],
            "fake_mode": self.modes.get(individual_id, "completed"),
        }


class FakeConvergenceGate:
    """Translate fake evaluator modes into real lifecycle gate decisions."""

    def evaluate(self, results):
        """Return a failed gate only for the requested non-converged mode."""
        passed = results["fake_mode"] != "not_converged"
        return {
            "enabled": True,
            "passed": passed,
            "failed_criteria": [] if passed else ["fake_non_convergence"],
        }


class OffspringEvaluationSchedulerTests(unittest.TestCase):
    """Verify P2-4 child-only lifecycle, resume, retry, and seed contracts."""

    def setUp(self) -> None:
        """Create a small generation with two completed parents and registered children."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "config.ini"
        self.inventory_path = self.root / "fuel_inventory.json"
        self.materials_path = self.root / "materials.xml"
        self.config_path.write_text("[ga]\nrandom_seed = 1000\n", encoding="utf-8")
        self.inventory_path.write_text("{}", encoding="utf-8")
        self.materials_path.write_text("<materials/>", encoding="utf-8")
        self.genes = (("fuel", "fuel_1"), ("fuel", "fuel_2"), ("empty_hole", "EMPTY_1"))
        self.sorter = GenomeSorter(
            slot_order=("A1", "B1", "B2", "B3"), excluded_genes={"A1": ("fixed", "FIXED")}
        )
        self.contract = CheckpointManager.build_genome_contract(
            "included_position_based_permutation",
            self.sorter.slot_order,
            self.sorter.excluded_genes,
            self.genes,
        )
        provenance = build_input_provenance(
            self.config_path, self.inventory_path, self.materials_path, self.contract
        )
        self.manager = CheckpointManager(
            self.root / "checkpoint.json",
            "P2-4-TEST",
            provenance,
            constraint_contract=TEST_CONSTRAINT,
        )

    def tearDown(self) -> None:
        """Remove temporary fake checkpoints and statepoint files."""
        self.temporary_directory.cleanup()

    def _pipeline(
        self,
        runner=None,
        evaluator=None,
        max_attempts=2,
        on_max_attempts="continue_campaign",
        interrupted_counts=False,
    ):
        """Build the shared lifecycle pipeline using fake scientific components only."""
        builder = FakeBuilderFactory()
        selected_runner = FakeRunner(self.root / "runs") if runner is None else runner
        selected_evaluator = FakeEvaluator() if evaluator is None else evaluator
        pipeline = SingleCandidatePipeline(
            genome_sorter=self.sorter,
            builder_factory=builder,
            runner=selected_runner,
            result_evaluator=selected_evaluator,
            checkpoint_manager=self.manager,
            material_file=self.materials_path,
            work_root=self.root / "runs",
            tally_name=selected_evaluator.tally_name,
            tally_score=selected_evaluator.tally_score,
            particles=10,
            inactive_batches=1,
            active_batches=2,
            keff_trigger_enabled=False,
            keff_trigger_stdev_pcm=50.0,
            trigger_batch_interval=1,
            trigger_max_batches=3,
            tally_trigger_enabled=False,
            tally_trigger_type="rel_err",
            tally_trigger_threshold=0.1,
            tally_trigger_ignore_zeros=False,
            convergence_gate=FakeConvergenceGate(),
            openmc_seed=1000,
            max_attempts=max_attempts,
            interrupted_counts_as_failed_attempt=interrupted_counts,
            on_max_attempts=on_max_attempts,
        )
        return pipeline, builder, selected_runner, selected_evaluator

    def _registered_state(self, offspring_count=3):
        """Return a durable P2-3 registry with two completed parents and child genomes."""
        state = self.manager.create_state(
            self.contract, ga_settings={"population_size": offspring_count, "random_seed": 1000}
        )
        parent_ids = ("G000-I000", "G000-I001")
        for index, individual_id in enumerate(parent_ids):
            genome = self.genes[index:] + self.genes[:index]
            self.manager.register_individual(state, individual_id, genome, 0, "initial")
        self.manager.begin_generation(state, 0, parent_ids)
        for index, individual_id in enumerate(parent_ids):
            self.manager.mark_individual_running(
                state,
                individual_id,
                f"PARENT-{index}",
                self.root / "parents" / str(index),
                900 + index,
            )
            self.manager.mark_individual_completed(
                state, individual_id, {"keff": {"mean": -(-1.0)}, "objectives": [-1.0, 1.1]}
            )
        self.manager.mark_parent_evaluation_complete(state, 0)
        self.manager.begin_offspring_creation(state, 0)
        offspring_records = []
        parent_genomes = {
            tuple(map(tuple, state["individuals"][key]["genome"])) for key in parent_ids
        }
        unique_genomes = [
            genome for genome in permutations(self.genes) if genome not in parent_genomes
        ]
        for index in range(offspring_count):
            genome = unique_genomes[index]
            offspring_records.append(
                {
                    "genome": genome,
                    "parent_ids": parent_ids,
                    "crossover": {"method": "fake", "parent_ids": list(parent_ids)},
                    "mutation": {"method": "fake"},
                }
            )
        self.manager.register_generation_offspring(state, 0, offspring_records)
        self.manager.save(state)
        return state

    def _scheduler(self, state, pipeline, resample_duplicate_offspring=True):
        """Return the P2-4 scheduler for one registered checkpoint state."""
        return OffspringEvaluationScheduler(
            pipeline,
            self.manager,
            state,
            "P2-4-TEST",
            resample_duplicate_offspring=resample_duplicate_offspring,
        )

    def test_legacy_duplicate_is_blocked_before_pipeline_or_state_changes(self):
        """Reject a saved parent clone or sibling clone without builder, runner, or evaluator calls."""
        for duplicate_id in ("G000-I000", "G000-O001"):
            with self.subTest(duplicate_id=duplicate_id):
                state = self._registered_state()
                state["individuals"]["G000-O000"]["genome"] = state["individuals"][duplicate_id][
                    "genome"
                ]
                pipeline, builder, runner, evaluator = self._pipeline()
                before = deepcopy(state)
                with self.assertRaisesRegex(OffspringEvaluationError, "duplicate"):
                    self._scheduler(state, pipeline).run()
                self.assertEqual(state, before)
                self.assertEqual(runner.calls, [])
                self.assertEqual(evaluator.calls, [])

    def test_parent_duplicate_reuses_complete_evaluation_without_transport(self):
        """Reuse a completed parent's exact result only after its saved evaluation
        context matches the current pipeline."""
        state = self._registered_state(offspring_count=1)
        pipeline, builder, runner, evaluator = self._pipeline()
        parent_id = state["generations"][0]["parent_ids"][0]
        child_id = state["generations"][0]["offspring_ids"][0]
        state["individuals"][parent_id]["operator"][
            "evaluation_context"
        ] = pipeline.evaluation_context()
        state["individuals"][child_id]["genome"] = deepcopy(
            state["individuals"][parent_id]["genome"]
        )
        state["individuals"][child_id]["operator"]["duplicate_handling"] = {
            "mode": "reuse_parent_fitness",
            "duplicate_parent_id": parent_id,
            "fitness_source_parent_id": None,
            "fitness_reused": False,
        }

        result = self._scheduler(state, pipeline, resample_duplicate_offspring=False).run()

        child = state["individuals"][child_id]
        metadata = child["operator"]["duplicate_handling"]
        self.assertEqual(result.phase, "offspring_evaluated")
        self.assertEqual(result.evaluated_ids, (child_id,))
        self.assertEqual(child["status"], "completed")
        self.assertEqual(child["results"], state["individuals"][parent_id]["results"])
        self.assertEqual(child["attempts"], [])
        self.assertTrue(metadata["fitness_reused"])
        self.assertEqual(metadata["fitness_source_parent_id"], parent_id)
        self.assertEqual(metadata["reuse_reason"], "identical_genome_and_evaluation_context")
        self.assertEqual(builder.build_calls, [])
        self.assertEqual(runner.calls, [])
        self.assertEqual(evaluator.calls, [])
        self.manager.load(self.contract)

    def test_parent_duplicate_with_context_mismatch_is_evaluated_normally(self):
        """Keep the duplicate offspring but run its normal evaluation when the
        parent's stored context differs from the active pipeline context."""
        state = self._registered_state(offspring_count=1)
        pipeline, builder, runner, evaluator = self._pipeline()
        parent_id = state["generations"][0]["parent_ids"][0]
        child_id = state["generations"][0]["offspring_ids"][0]
        parent_context = pipeline.evaluation_context()
        parent_context["model_version"] = "different-model/v9"
        state["individuals"][parent_id]["operator"]["evaluation_context"] = parent_context
        state["individuals"][child_id]["genome"] = deepcopy(
            state["individuals"][parent_id]["genome"]
        )
        state["individuals"][child_id]["operator"]["duplicate_handling"] = {
            "mode": "reuse_parent_fitness",
            "duplicate_parent_id": parent_id,
            "fitness_source_parent_id": None,
            "fitness_reused": False,
        }

        result = self._scheduler(state, pipeline, resample_duplicate_offspring=False).run()

        metadata = state["individuals"][child_id]["operator"]["duplicate_handling"]
        self.assertEqual(result.phase, "offspring_evaluated")
        self.assertFalse(metadata["fitness_reused"])
        self.assertIsNone(metadata["fitness_source_parent_id"])
        self.assertEqual(metadata["reuse_reason"], "evaluation_context_mismatch_or_missing")
        self.assertEqual(len(builder.build_calls), 1)
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(len(evaluator.calls), 1)

    def test_evaluates_only_checkpoint_offspring_with_success_nonconvergence_retry_and_failure(
        self,
    ):
        """Keep parent records untouched while each fake child reaches its terminal lifecycle state."""
        state = self._registered_state()
        parent_before = {
            individual_id: state["individuals"][individual_id].copy()
            for individual_id in state["generations"][0]["parent_ids"]
        }
        runner = FakeRunner(self.root / "runs", {"G000-O002": 2})
        evaluator = FakeEvaluator({"G000-O001": "not_converged"})
        pipeline, builder, runner, _ = self._pipeline(runner=runner, evaluator=evaluator)

        result = self._scheduler(state, pipeline).run()

        self.assertEqual(result.phase, "offspring_evaluated")
        self.assertEqual(result.status_counts, {"completed": 1, "failed": 1, "not_converged": 1})
        self.assertEqual(result.evaluated_ids, ("G000-O000", "G000-O001", "G000-O002"))
        self.assertEqual(
            [call["individual_id"] for call in runner.calls],
            ["G000-O000", "G000-O001", "G000-O002", "G000-O002"],
        )
        self.assertEqual(
            {individual_id: state["individuals"][individual_id] for individual_id in parent_before},
            parent_before,
        )
        self.assertEqual(
            builder.build_calls[0]["genome"],
            (
                ("fixed", "FIXED"),
                *tuple(tuple(gene) for gene in state["individuals"]["G000-O000"]["genome"]),
            ),
        )
        self.assertTrue(state["individuals"]["G000-O001"]["attempts"][-1]["statepoint_retained"])
        self.assertIsNotNone(state["individuals"]["G000-O001"]["results"])
        self.assertIsNone(state["individuals"]["G000-O002"]["results"])
        self.manager.set_nsga2_metrics(state, "G000-O000", 0, None, True)
        for individual_id in ("G000-O001", "G000-O002"):
            with self.subTest(individual_id=individual_id), self.assertRaisesRegex(
                CheckpointValidationError, "completed"
            ):
                self.manager.set_nsga2_metrics(state, individual_id, 0, None, True)
        seeds = [call["settings"]["openmc_seed"] for call in builder.build_calls]
        self.assertEqual(seeds, [1006, 1010, 1016, 1017])
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(
            [
                state["individuals"][individual_id]["operator"]["openmc_seed_base"]
                for individual_id in state["generations"][0]["offspring_ids"]
            ],
            [1006, 1010, 1016],
        )
        self.assertEqual(Counter(result.status_counts.values()), Counter((1, 1, 1)))
        resumed_result = self._scheduler(state, pipeline).run()
        self.assertEqual(resumed_result.evaluated_ids, ())
        self.assertEqual(len(runner.calls), 4)

    def test_resume_recovers_nonconsuming_interruption_without_changing_seed(self):
        """Recover a running child to pending and reuse its seed when policy does not consume it."""
        state = self._registered_state(offspring_count=2)
        self.manager.begin_offspring_evaluation(state, 0)
        self.manager.mark_individual_running(
            state, "G000-O000", "P2-4-TEST-G000-O000-A01", self.root / "runs" / "interrupted", 1006
        )
        self.manager.save(state)
        loaded = self.manager.load(expected_genome_contract=self.contract)
        pipeline, builder, runner, _ = self._pipeline(interrupted_counts=False)

        result = self._scheduler(loaded, pipeline).run()

        self.assertEqual(result.recovered_ids, ("G000-O000",))
        self.assertEqual(result.phase, "offspring_evaluated")
        self.assertEqual(
            [call["run_id"] for call in runner.calls],
            ["P2-4-TEST-G000-O000-A02", "P2-4-TEST-G000-O001-A01"],
        )
        self.assertEqual(builder.build_calls[0]["settings"]["openmc_seed"], 1006)
        attempts = loaded["individuals"]["G000-O000"]["attempts"]
        self.assertEqual([attempt["status"] for attempt in attempts], ["interrupted", "completed"])
        self.assertEqual([attempt["openmc_seed"] for attempt in attempts], [1006, 1006])

    def test_resume_skips_terminal_offspring_and_wrong_phase_is_rejected(self):
        """Make the terminal boundary idempotent and fail closed before child evaluation in another phase."""
        state = self._registered_state(offspring_count=1)
        pipeline, _, runner, _ = self._pipeline()
        first_result = self._scheduler(state, pipeline).run()
        resumed_result = self._scheduler(state, pipeline).run()

        self.assertEqual(first_result.evaluated_ids, ("G000-O000",))
        self.assertEqual(resumed_result.evaluated_ids, ())
        self.assertEqual(len(runner.calls), 1)

        invalid_state = self._registered_state(offspring_count=1)
        self.manager.set_progress(invalid_state, 0, "parents_evaluated")
        invalid_pipeline, _, invalid_runner, _ = self._pipeline()
        with self.assertRaisesRegex(
            OffspringEvaluationError, "offspring_registered atau evaluating_offspring"
        ):
            self._scheduler(invalid_state, invalid_pipeline).run()
        self.assertEqual(invalid_runner.calls, [])


if __name__ == "__main__":
    unittest.main()
