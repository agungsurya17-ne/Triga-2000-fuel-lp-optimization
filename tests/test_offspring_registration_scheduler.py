"""Test offspring registration scheduler without neutron transport.

Run with ``python -m unittest discover -s tests -v``."""

from __future__ import annotations

import sys
from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME_ROOT / "src"))

from modules.keff_constraint import KeffConstraint


import random
import tempfile
import unittest
from unittest.mock import patch
from itertools import permutations
from types import SimpleNamespace
from copy import deepcopy
from pathlib import Path

from modules.checkpoint import CheckpointManager, CheckpointValidationError
from modules.crossover import CrossoverDispatcher, build_default_crossover_registry
from modules.genome_sorter import GenomeSorter
from modules.input_provenance import build_input_provenance
from modules.mutation import MutationDispatcher, build_default_mutation_registry
from modules.mutable_genome import MutableGenomeContract
from modules.offspring_registration_scheduler import (
    OffspringRegistrationError,
    OffspringRegistrationScheduler,
)


TEST_CONSTRAINT = KeffConstraint(1.0, 0.0)


class CountingSelector:
    """Return fixed parent pairs while recording scheduler calls."""

    def __init__(self, parent_ids: tuple[str, str]) -> None:
        """Store one valid pair returned repeatedly for every requested pair."""
        self.parent_ids = parent_ids
        self.calls = 0

    def __call__(self, parent_count: int) -> tuple[str, ...]:
        """Return ordered pairs totalling ``parent_count`` IDs."""
        self.calls += 1
        return self.parent_ids * (parent_count // 2)


class CountingCrossover:
    """Delegate crossover while exposing whether resume called it."""

    def __init__(self, delegate: CrossoverDispatcher) -> None:
        """Store the real mutable-genome crossover dispatcher."""
        self.delegate = delegate
        self.rng = delegate.rng
        self.calls = 0

    def crossover(self, parent_ids, checkpoint_records):
        """Run the delegate and increment the pure-operator call counter."""
        self.calls += 1
        return self.delegate.crossover(parent_ids, checkpoint_records)


class CountingMutation:
    """Delegate mutation while exposing whether resume called it."""

    def __init__(self, delegate: MutationDispatcher) -> None:
        """Store the real mutable-genome mutation dispatcher."""
        self.delegate = delegate
        self.rng = delegate.rng
        self.calls = 0

    def mutate(self, genome):
        """Run the delegate and increment the pure-operator call counter."""
        self.calls += 1
        return self.delegate.mutate(genome)


class OffspringRegistrationSchedulerTests(unittest.TestCase):
    """Verify durable P2-2 offspring registration without OpenMC evaluation."""

    def setUp(self) -> None:
        """Build a small completed fake generation followed by generation one."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        config_path = self.root / "config.ini"
        inventory_path = self.root / "fuel_inventory.json"
        materials_path = self.root / "materials.xml"
        config_path.write_text("[ga]\nrandom_seed = 12345\n", encoding="utf-8")
        inventory_path.write_text('{"fuel_1": 1.0}', encoding="utf-8")
        materials_path.write_text("<materials/>", encoding="utf-8")
        self.sorter = GenomeSorter(("A1", "B1", "B2", "B3"), {"A1": ("irradiation_tube", "IT")})
        self.genes = (("fuel", "fuel_1"), ("fuel", "fuel_2"), ("empty_hole", "EMPTY_1"))
        self.contract = CheckpointManager.build_genome_contract(
            "included_position_based_permutation",
            self.sorter.slot_order,
            self.sorter.excluded_genes,
            self.genes,
        )
        provenance = build_input_provenance(
            config_path, inventory_path, materials_path, self.contract
        )
        self.manager = CheckpointManager(
            self.root / "checkpoint.json",
            "P2-2-TEST",
            provenance,
            constraint_contract=TEST_CONSTRAINT,
        )
        self.state = self._make_generation_one_state()
        self.rng = random.Random(827)
        mutable_contract = MutableGenomeContract.from_genome_sorter(self.genes, self.sorter)
        crossover = CrossoverDispatcher(
            "pmx", 1.0, build_default_crossover_registry(), mutable_contract, self.sorter, self.rng
        )
        mutation = MutationDispatcher(
            "swap_mutation",
            1.0,
            1,
            build_default_mutation_registry(),
            mutable_contract,
            self.sorter,
            self.rng,
        )
        self.selector = CountingSelector(("G000-I000", "G000-I001"))
        self.crossover = CountingCrossover(crossover)
        self.mutation = CountingMutation(mutation)
        self.scheduler = OffspringRegistrationScheduler(
            self.manager, self.state, 2, self.selector, self.crossover, self.mutation, self.rng
        )

    def tearDown(self) -> None:
        """Remove isolated fake checkpoint artifacts."""
        self.temporary_directory.cleanup()

    def _make_generation_one_state(self) -> dict:
        """Return generation one at parents_evaluated using only fake results."""
        state = self.manager.create_state(
            self.contract, {"population_size": 2, "random_seed": 12345}
        )
        first_genome = self.genes
        second_genome = self.genes[1:] + self.genes[:1]
        self.manager.register_individual(state, "G000-I000", first_genome, 0, "initial")
        self.manager.register_individual(state, "G000-I001", second_genome, 0, "initial")
        self.manager.begin_generation(state, 0, ("G000-I000", "G000-I001"))
        self._complete(state, "G000-I000", 1001)
        self._complete(state, "G000-I001", 1002)
        self.manager.mark_parent_evaluation_complete(state, 0)
        self.manager.begin_offspring_creation(state, 0)
        self.manager.register_individual(
            state,
            "G000-O000",
            first_genome,
            0,
            "offspring",
            ("G000-I000", "G000-I001"),
            {"legacy": "fake"},
        )
        self.manager.register_individual(
            state,
            "G000-O001",
            second_genome,
            0,
            "offspring",
            ("G000-I000", "G000-I001"),
            {"legacy": "fake"},
        )
        self.manager.set_generation_offspring(state, 0, ("G000-O000", "G000-O001"))
        self.manager.set_progress(state, 0, "evaluating_offspring")
        self._complete(state, "G000-O000", 1003)
        self._complete(state, "G000-O001", 1004)
        self.manager.complete_generation(state, 0, ("G000-I000", "G000-I001"))
        self.manager.begin_generation(state, 1, ("G000-I000", "G000-I001"))
        self.manager.mark_parent_evaluation_complete(state, 1)
        return state

    def _complete(self, state: dict, individual_id: str, seed: int) -> None:
        """Mark one fake record completed without constructing an OpenMC model."""
        run_id = f"{individual_id}-A01"
        self.manager.mark_individual_running(
            state, individual_id, run_id, self.root / "runs" / run_id, seed
        )
        self.manager.mark_individual_completed(
            state,
            individual_id,
            {"objectives": [-(1.01), 1.2], "keff": {"mean": 1.01, "std_dev": 0.0004}},
            self.root / "runs" / run_id / "statepoint.250.h5",
        )

    def test_registers_complete_batch_before_evaluation_and_persists_provenance(self) -> None:
        """Create two pending offspring, linked only after both records exist."""
        result = self.scheduler.run()

        self.assertFalse(result.resumed)
        self.assertEqual(result.offspring_ids, ("G001-O000", "G001-O001"))
        self.assertEqual(self.state["progress"]["phase"], "offspring_registered")
        self.assertEqual(self.state["generations"][1]["offspring_ids"], list(result.offspring_ids))
        self.assertEqual(self.selector.calls, self.crossover.calls)
        self.assertEqual(self.crossover.calls, 3)
        self.assertEqual(self.mutation.calls, 5)
        for individual_id in result.offspring_ids:
            individual = self.state["individuals"][individual_id]
            self.assertEqual(individual["generation_created"], 1)
            self.assertEqual(individual["origin"], "offspring")
            self.assertEqual(individual["parent_ids"], ["G000-I000", "G000-I001"])
            self.assertEqual(individual["status"], "pending")
            self.assertIn("crossover", individual["operator"])
            self.assertIn("mutation", individual["operator"])
            mutable = tuple(tuple(gene) for gene in individual["genome"])
            self.assertEqual(set(mutable), set(self.genes))
            self.assertEqual(
                self.sorter.extract_included_genome(self.sorter.construct_complete_genome(mutable)),
                mutable,
            )
        self.assertTrue(self.manager.checkpoint_path.is_file())

    def test_resume_complete_registry_uses_no_additional_rng_or_operators(self) -> None:
        """Reuse durable offspring without selecting parents or advancing shared RNG."""
        first = self.scheduler.run()
        rng_state = self.rng.getstate()
        loaded = self.manager.load(self.contract)
        resumed_rng = random.Random()
        self.manager.restore_rng_state(loaded, python_rng=resumed_rng)
        resumed_selector = CountingSelector(("G000-I000", "G000-I001"))
        mutable_contract = MutableGenomeContract.from_genome_sorter(self.genes, self.sorter)
        resumed_crossover = CountingCrossover(
            CrossoverDispatcher(
                "pmx",
                1.0,
                build_default_crossover_registry(),
                mutable_contract,
                self.sorter,
                resumed_rng,
            )
        )
        resumed_mutation = CountingMutation(
            MutationDispatcher(
                "swap_mutation",
                1.0,
                1,
                build_default_mutation_registry(),
                mutable_contract,
                self.sorter,
                resumed_rng,
            )
        )
        resumed_scheduler = OffspringRegistrationScheduler(
            self.manager,
            loaded,
            2,
            resumed_selector,
            resumed_crossover,
            resumed_mutation,
            resumed_rng,
        )

        resumed = resumed_scheduler.run()

        self.assertTrue(resumed.resumed)
        self.assertEqual(resumed.offspring_ids, first.offspring_ids)
        self.assertEqual(resumed_selector.calls, 0)
        self.assertEqual(resumed_rng.getstate(), rng_state)
        self.assertEqual(self.crossover.calls, 3)
        self.assertEqual(self.mutation.calls, 5)
        self.assertEqual(resumed_crossover.calls, 0)
        self.assertEqual(resumed_mutation.calls, 0)

    def test_partial_registry_fails_closed_without_regeneration(self) -> None:
        """Reject one durable but unlinked child before selection or RNG use."""
        self.manager.begin_offspring_creation(self.state, 1)
        self.manager.register_individual(
            self.state,
            "G001-O000",
            self.genes,
            1,
            "offspring",
            ("G000-I000", "G000-I001"),
            {"legacy": "partial"},
        )
        rng_state = self.rng.getstate()

        with self.assertRaisesRegex(OffspringRegistrationError, "terisi sebagian"):
            self.scheduler.run()

        self.assertEqual(self.selector.calls, 0)
        self.assertEqual(self.crossover.calls, 0)
        self.assertEqual(self.mutation.calls, 0)
        self.assertEqual(self.rng.getstate(), rng_state)

    def test_checkpoint_api_rejects_invalid_parent_and_round_trips(self) -> None:
        """Reject foreign/ineligible parents and reload a valid complete batch."""
        self.manager.begin_offspring_creation(self.state, 1)
        valid_metadata = {"method": "fake", "parent_ids": ["G000-I000", "G000-I001"]}
        with self.assertRaisesRegex(CheckpointValidationError, "bukan parent generation"):
            self.manager.register_generation_offspring(
                self.state,
                1,
                (
                    {
                        "genome": self.genes,
                        "parent_ids": ("G000-I000", "G000-O000"),
                        "crossover": valid_metadata,
                        "mutation": {"method": "fake"},
                    },
                ),
            )
        with self.assertRaisesRegex(CheckpointValidationError, "ID duplikat"):
            self.manager.register_generation_offspring(
                self.state,
                1,
                (
                    {
                        "genome": self.genes,
                        "parent_ids": ("G000-I000", "G000-I000"),
                        "crossover": valid_metadata,
                        "mutation": {"method": "fake"},
                    },
                ),
            )
        self.state["individuals"]["G000-I001"]["status"] = "failed"
        with self.assertRaisesRegex(CheckpointValidationError, "completed"):
            self.manager.register_generation_offspring(
                self.state,
                1,
                (
                    {
                        "genome": self.genes,
                        "parent_ids": ("G000-I000", "G000-I001"),
                        "crossover": valid_metadata,
                        "mutation": {"method": "fake"},
                    },
                ),
            )

    def test_odd_population_creates_exact_count_from_two_deterministic_pairs(self) -> None:
        """Keep only the first child of the final pair for an odd population."""
        scheduler = OffspringRegistrationScheduler(
            self.manager, self.state, 3, self.selector, self.crossover, self.mutation, self.rng
        )

        result = scheduler.run()

        self.assertEqual(result.offspring_ids, ("G001-O000", "G001-O001", "G001-O002"))
        self.assertEqual(self.selector.calls, self.crossover.calls)
        self.assertEqual(self.crossover.calls, 3)
        self.assertEqual(self.mutation.calls, 6)
        self.assertEqual(
            self.state["individuals"]["G001-O002"]["parent_ids"], ["G000-I000", "G000-I001"]
        )

    def test_skipped_crossover_children_are_still_mutated(self) -> None:
        """Run mutation for both copied children when crossover probability is zero."""
        mutable_contract = MutableGenomeContract.from_genome_sorter(self.genes, self.sorter)
        crossover = CrossoverDispatcher(
            "pmx", 0.0, build_default_crossover_registry(), mutable_contract, self.sorter, self.rng
        )
        mutation = MutationDispatcher(
            "swap_mutation",
            1.0,
            1,
            build_default_mutation_registry(),
            mutable_contract,
            self.sorter,
            self.rng,
        )
        scheduler = OffspringRegistrationScheduler(
            self.manager, self.state, 2, self.selector, crossover, mutation, self.rng
        )

        result = scheduler.run()

        for individual_id in result.offspring_ids:
            operator = self.state["individuals"][individual_id]["operator"]
            self.assertFalse(operator["crossover"]["applied"])
            self.assertTrue(operator["mutation"]["applied"])

    def test_hcmx_dispatch_requires_and_uses_compatible_converged_parent_results(self) -> None:
        """Dispatch HCMX only after both fake parent tally records pass its preflight."""
        for parent_id in ("G000-I000", "G000-I001"):
            genome = tuple(tuple(gene) for gene in self.state["individuals"][parent_id]["genome"])
            self.state["individuals"][parent_id]["results"].update(self._hcmx_results(genome))
        mutable_contract = MutableGenomeContract.from_genome_sorter(self.genes, self.sorter)
        options = {
            "mask_size_min": 1,
            "mask_size_max": 2,
            "selection_alpha": 2.0,
            "epsilon": 0.01,
            "nonfuel_policy": "inherit_donor_layout",
            "require_parent_power_tally": True,
            "require_convergence_passed": True,
            "expected_tally_name": "fuel_segment_fission_q_recoverable",
            "expected_tally_score": "fission-q-recoverable",
        }
        crossover = CrossoverDispatcher(
            "triga_hcmx_power_mask",
            1.0,
            build_default_crossover_registry(),
            mutable_contract,
            self.sorter,
            self.rng,
            operator_options=options,
        )
        mutation = MutationDispatcher(
            "swap_mutation",
            0.0,
            1,
            build_default_mutation_registry(),
            mutable_contract,
            self.sorter,
            self.rng,
        )
        scheduler = OffspringRegistrationScheduler(
            self.manager, self.state, 2, self.selector, crossover, mutation, self.rng
        )

        dispatched = crossover.crossover(self.selector.parent_ids, self.state["individuals"])
        self.assertEqual(dispatched.metadata["method"], "triga_hcmx_power_mask")
        self.assertTrue(dispatched.metadata["applied"])
        scheduler.max_reproduction_attempts = 3
        with self.assertRaisesRegex(OffspringRegistrationError, "Batas reproduction"):
            scheduler.run()
        self.assertEqual(self.state["generations"][-1]["offspring_ids"], [])

    def test_creation_keeps_parent_payloads_unchanged_and_child_evaluation_blocked(self) -> None:
        """Preserve parent science payloads and stop before any child transport attempt."""
        parent_before = {
            individual_id: deepcopy(self.state["individuals"][individual_id])
            for individual_id in ("G000-I000", "G000-I001")
        }

        result = self.scheduler.run()

        self.assertEqual(
            {
                individual_id: self.state["individuals"][individual_id]
                for individual_id in parent_before
            },
            parent_before,
        )
        self.assertEqual(self.state["progress"]["phase"], "offspring_registered")
        for individual_id in result.offspring_ids:
            child = self.state["individuals"][individual_id]
            self.assertEqual(child["status"], "pending")
            self.assertEqual(child["attempts"], [])
            self.assertIsNone(child["results"])
        with self.assertRaisesRegex(
            CheckpointValidationError, "bukan anggota populasi yang sedang dievaluasi"
        ):
            self.manager.mark_individual_running(
                self.state,
                result.offspring_ids[0],
                "P2-3-CHILD",
                self.root / "runs" / "P2-3-CHILD",
                5001,
            )

    def test_invalid_lifecycle_phase_is_rejected_without_rng_or_record_changes(self) -> None:
        """Fail closed before selection when parent evaluation has not been closed."""
        self.manager.set_progress(self.state, 1, "evaluating_parents")
        state_before = deepcopy(self.state)
        rng_state = self.rng.getstate()

        with self.assertRaisesRegex(OffspringRegistrationError, "parents_evaluated"):
            self.scheduler.run()

        self.assertEqual(self.state, state_before)
        self.assertEqual(self.rng.getstate(), rng_state)
        self.assertEqual(self.selector.calls, 0)

    def test_noop_variation_is_rejected_and_normal_draws_repeat(self) -> None:
        """Allow real no-op draws, reject clones, then accept only normally varied unique children."""
        self.crossover.delegate.probability = 0.0
        self.mutation.delegate.probability_per_individual = 0.5
        # The first pair skips both mutations; the next draw applies a swap.
        with patch.object(self.rng, "random", side_effect=[0.9, 0.9, 0.9] + [0.0] * 100):
            with self.assertLogs("modules.offspring_registration_scheduler", level="INFO") as logs:
                result = self.scheduler.run()
        audit = self.state["metadata"]["reproduction"]["1"]
        self.assertEqual(audit["duplicate_rejections"][0]["duplicate_of"], "G000-I000")
        self.assertEqual(audit["duplicate_rejections"][1]["duplicate_of"], "G000-I001")
        self.assertFalse(audit["duplicate_rejections"][0]["crossover_applied"])
        self.assertFalse(audit["duplicate_rejections"][0]["mutation_applied"])
        self.assertGreater(self.selector.calls, 1)
        self.assertEqual(self.selector.calls, self.crossover.calls)
        self.assertTrue(any("Rejected duplicate" in message for message in logs.output))
        genomes = [
            tuple(map(tuple, self.state["individuals"][key]["genome"]))
            for key in result.offspring_ids
        ]
        self.assertEqual(len(set(genomes)), 2)
        historical = {
            tuple(map(tuple, record["genome"]))
            for key, record in self.state["individuals"].items()
            if key not in result.offspring_ids
        }
        self.assertTrue(set(genomes).isdisjoint(historical))
        self.assertEqual(self.manager.load(self.contract)["metadata"]["reproduction"]["1"], audit)
        self.assertEqual(self.crossover.delegate.probability, 0.0)
        self.assertEqual(self.mutation.delegate.probability_per_individual, 0.5)

    def test_parent_duplicates_are_registered_without_resampling_when_disabled(self) -> None:
        """Accept both no-op parent clones in one reproduction draw while leaving
        their fitness pending for the evaluation-context guard."""
        self.crossover.delegate.probability = 0.0
        self.mutation.delegate.probability_per_individual = 0.0
        scheduler = OffspringRegistrationScheduler(
            self.manager,
            self.state,
            2,
            self.selector,
            self.crossover,
            self.mutation,
            self.rng,
            resample_duplicate_offspring=False,
        )

        result = scheduler.run()

        self.assertEqual(self.selector.calls, 1)
        self.assertEqual(self.crossover.calls, 1)
        self.assertEqual(self.mutation.calls, 2)
        self.assertEqual(
            self.state["metadata"]["reproduction"]["1"]["policy"],
            "accept_parent_duplicate_conditional_fitness_reuse",
        )
        for child_id, parent_id in zip(
            result.offspring_ids, ("G000-I000", "G000-I001"), strict=True
        ):
            metadata = self.state["individuals"][child_id]["operator"]["duplicate_handling"]
            self.assertEqual(metadata["mode"], "reuse_parent_fitness")
            self.assertEqual(metadata["duplicate_parent_id"], parent_id)
            self.assertIsNone(metadata["fitness_source_parent_id"])
            self.assertFalse(metadata["fitness_reused"])

    def test_rejects_historical_and_same_batch_duplicates(self) -> None:
        """Exclude nonparent history and accepted siblings using ordered genome keys."""
        parents = {tuple(map(tuple, item["genome"])) for item in self.state["individuals"].values()}
        unseen = [genome for genome in permutations(self.genes) if genome not in parents]
        self.state["individuals"]["G000-O000"]["genome"] = list(unseen[0])
        self.mutation.delegate.probability_per_individual = 0.0
        metadata = {"method": "fake", "parent_ids": list(self.selector.parent_ids), "applied": True}
        pairs = [(unseen[0], unseen[1]), (unseen[1], unseen[2])]
        results = [SimpleNamespace(children=pair, metadata=metadata) for pair in pairs]
        with patch.object(self.crossover, "crossover", side_effect=results):
            result = self.scheduler.run()
        audit = self.state["metadata"]["reproduction"]["1"]
        self.assertEqual(
            [entry["duplicate_of"] for entry in audit["duplicate_rejections"]],
            ["G000-O000", "G001-O000"],
        )
        self.assertEqual(self.selector.calls, 2)
        self.assertEqual(
            [
                tuple(map(tuple, self.state["individuals"][key]["genome"]))
                for key in result.offspring_ids
            ],
            unseen[1:3],
        )

    def test_noop_exhaustion_preserves_checkpoint_state_and_rng(self) -> None:
        """Never force variation when Pc/Pm are zero; bounded retries leave no partial batch."""
        self.crossover.delegate.probability = 0.0
        self.mutation.delegate.probability_per_individual = 0.0
        self.scheduler.max_reproduction_attempts = 3
        self.manager.save(self.state, python_rng=self.rng)
        before = deepcopy(self.state)
        saved = self.manager.checkpoint_path.read_bytes()
        rng_before = self.rng.getstate()
        with self.assertRaisesRegex(OffspringRegistrationError, "duplicate=6"):
            self.scheduler.run()
        self.assertEqual(self.selector.calls, 3)
        self.assertEqual(self.crossover.calls, 3)
        self.assertEqual(self.mutation.calls, 6)
        self.assertEqual(self.state, before)
        self.assertEqual(self.rng.getstate(), rng_before)
        self.assertEqual(self.manager.checkpoint_path.read_bytes(), saved)

    def test_failed_save_replays_same_batch_and_rng(self) -> None:
        """Replay an interrupted atomic batch deterministically after restoring its saved boundary."""
        before = deepcopy(self.state)
        rng_before = self.rng.getstate()
        with patch.object(self.manager, "save", side_effect=OSError("injected save failure")):
            with self.assertRaisesRegex(OSError, "injected"):
                self.scheduler.run()
        self.assertEqual(self.state, before)
        self.assertEqual(self.rng.getstate(), rng_before)
        result = self.scheduler.run()
        expected = [
            (self.state["individuals"][key]["genome"], self.state["individuals"][key]["operator"])
            for key in result.offspring_ids
        ]
        rng_after = self.rng.getstate()
        self.state.clear()
        self.state.update(before)
        self.rng.setstate(rng_before)
        repeated = self.scheduler.run()
        self.assertEqual(
            [
                (
                    self.state["individuals"][key]["genome"],
                    self.state["individuals"][key]["operator"],
                )
                for key in repeated.offspring_ids
            ],
            expected,
        )
        self.assertEqual(self.rng.getstate(), rng_after)

    def _hcmx_results(self, genome: tuple[tuple[str, str], ...]) -> dict:
        """Return compatible converged tally data for one tiny fake mutable genome."""
        fuel_pairs = [
            (slot, gene_id)
            for slot, (gene_type, gene_id) in zip(
                self.sorter.included_positions, genome, strict=True
            )
            if gene_type == "fuel"
        ]
        power_by_slot = {slot: 1.0 + index * 0.1 for index, (slot, _) in enumerate(fuel_pairs)}
        return {
            "keff": {"mean": -(-1.01)},
            "objectives": [-1.01, 1.2],
            "convergence": {"passed": True},
            "metadata": {
                "tally_name": "fuel_segment_fission_q_recoverable",
                "tally_score": "fission-q-recoverable",
                "power_normalization": "relative_to_mean_fission_q_recoverable",
                "tally_quantity": "integrated_fission_q_recoverable_per_segment",
                "radial_ppf_aggregation": "sum_segment_scores_per_fuel_slot",
                "included_slot_types": ["fuel"],
                "ffcr_included": False,
                "fuel_slot_count": len(fuel_pairs),
            },
            "power_by_slot": power_by_slot,
            "fuel_id_by_slot": dict(fuel_pairs),
        }
