"""Test generation extension without neutron transport.

Run with ``python -m unittest discover -s tests -v``."""

from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = RUNTIME_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from modules.checkpoint import CheckpointCompatibilityError, CheckpointManager
from modules.input_provenance import build_input_provenance
from modules.keff_constraint import KeffConstraint
from modules.optimization_loop_controller import OptimizationLoopController


TEST_CONSTRAINT = KeffConstraint(1.0, 0.0)


class FailingInitialScheduler:
    """Reject unexpected initial-population work during an extension."""

    def run(self, registration_only: bool = False) -> None:
        """Raise because an extended completed campaign must not initialize again."""
        raise AssertionError(f"Initial scheduler dipanggil; registration_only={registration_only}.")


class CompletedGenerationTransition:
    """Append lightweight completed generations without transport evaluations."""

    def __init__(
        self, manager: CheckpointManager, state: dict, num_generations: int, rng: random.Random
    ) -> None:
        """Store the real checkpoint manager, mutable state, target, and RNG."""
        self.manager = manager
        self.state = state
        self.num_generations = num_generations
        self.rng = rng
        self.calls: list[int] = []

    def run(self, target_generation: int) -> SimpleNamespace:
        """Complete ``target_generation`` from prior survivors and return a stub result."""
        self.calls.append(target_generation)
        parent_ids = tuple(self.state["generations"][-1]["survivor_ids"])
        self.manager.begin_generation(self.state, target_generation, parent_ids)
        self.manager.complete_generation(self.state, target_generation, parent_ids)
        self.manager.save(self.state, python_rng=self.rng)
        return SimpleNamespace(generation=target_generation)


class GenerationExtensionTests(unittest.TestCase):
    """Check safe 30-to-40 migration, continuation, and repeated extension."""

    def setUp(self) -> None:
        """Create isolated inputs and a valid finished schema-v3 checkpoint."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "config.ini"
        self.inventory_path = self.root / "fuel_inventory.json"
        self.materials_path = self.root / "materials.xml"
        self.checkpoint_path = self.root / "checkpoint.json"
        self.inventory_path.write_text('{"fuel_1": 1.0}', encoding="utf-8")
        self.materials_path.write_text("<materials/>", encoding="utf-8")
        self._write_config(30, mutation_probability="0.35")
        self.contract = CheckpointManager.build_genome_contract(
            "included_position_based_permutation",
            ("A1", "B1"),
            {"A1": ("fixed", "FIXED")},
            (("fuel", "fuel_1"),),
        )
        self.ga_settings = {"population_size": 1, "num_generations": 30, "random_seed": 12345}
        legacy_manager = self._manager()
        self.state = legacy_manager.create_state(
            self.contract,
            self.ga_settings,
            metadata={"algorithm": "NSGA-II"},
            python_rng=random.Random(17),
        )
        legacy_manager.register_individual(
            self.state, "G000-I000", (("fuel", "fuel_1"),), 0, "initial"
        )
        legacy_manager.begin_generation(self.state, 0, ("G000-I000",))
        legacy_manager.mark_individual_running(
            self.state, "G000-I000", "G000-I000-A01", self.root / "runs/G000-I000-A01", 101
        )
        legacy_manager.mark_individual_completed(
            self.state,
            "G000-I000",
            {"objectives": [-1.01, 1.2], "keff": {"mean": 1.01, "std_dev": 0.0004}},
            self.root / "statepoint.h5",
        )
        legacy_manager.complete_generation(self.state, 0, ("G000-I000",))
        for generation in range(1, 30):
            legacy_manager.begin_generation(self.state, generation, ("G000-I000",))
            legacy_manager.complete_generation(self.state, generation, ("G000-I000",))
        legacy_manager.mark_finished(self.state)
        legacy_provenance = deepcopy(self.state["input_provenance"])
        legacy_provenance["schema_version"] = 1
        legacy_provenance["inputs"]["config.ini"].pop("compatibility_sha256")
        legacy_provenance["inputs"]["config.ini"].pop("compatibility_format")
        self.state["schema_version"] = 3
        self.state["input_provenance"] = legacy_provenance
        self.checkpoint_path.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")

    def tearDown(self) -> None:
        """Remove all isolated checkpoint and input files."""
        self.temporary_directory.cleanup()

    def _write_config(self, target: int, mutation_probability: str) -> None:
        """Write a compact test INI with ``target`` and one immutable option."""
        self.config_path.write_text(
            "[ga]\n"
            f"num_generations = {target}\n"
            "population_size = 1\n"
            "random_seed = 12345\n"
            "[mutation]\n"
            f"probability = {mutation_probability}\n",
            encoding="utf-8",
        )

    def _manager(self) -> CheckpointManager:
        """Return a checkpoint manager using fingerprints from current test inputs."""
        provenance = build_input_provenance(
            self.config_path, self.inventory_path, self.materials_path, self.contract
        )
        return CheckpointManager(
            self.checkpoint_path,
            "GENERATION-EXTENSION-TEST",
            provenance,
            constraint_contract=TEST_CONSTRAINT,
        )

    def _current_ga(self, target: int) -> dict:
        """Return runtime GA settings matching the legacy fixture except its target."""
        return {**self.ga_settings, "num_generations": target}

    def test_migrates_finished_v3_and_preserves_scientific_state(self) -> None:
        """Back up v3, reopen generation 29, and preserve prior results and RNG."""
        original_bytes = self.checkpoint_path.read_bytes()
        original_individuals = deepcopy(self.state["individuals"])
        original_generations = deepcopy(self.state["generations"])
        original_rng_state = deepcopy(self.state["rng_state"])
        self._write_config(40, mutation_probability="0.35")

        loaded = self._manager().load(self.contract, current_ga_settings=self._current_ga(40))

        self.assertEqual((self.root / "checkpoint.v3.backup.json").read_bytes(), original_bytes)
        self.assertEqual(loaded["schema_version"], 4)
        self.assertEqual(
            loaded["progress"],
            {"generation": 29, "phase": "generation_completed", "active_individual_id": None},
        )
        self.assertEqual(loaded["ga_settings"]["num_generations"], 40)
        self.assertEqual(loaded["individuals"], original_individuals)
        self.assertEqual(loaded["generations"], original_generations)
        self.assertEqual(loaded["rng_state"], original_rng_state)
        self.assertEqual(loaded["metadata"]["generation_limit_extensions"][0]["from"], 30)
        self.assertEqual(loaded["metadata"]["generation_limit_extensions"][0]["to"], 40)

    def test_continues_only_generations_30_through_39_and_is_idempotent(self) -> None:
        """Use previous survivors, finish at index 39, and make equal-target reload a no-op."""
        self._write_config(40, mutation_probability="0.35")
        manager = self._manager()
        loaded = manager.load(self.contract, current_ga_settings=self._current_ga(40))
        rng = random.Random()
        manager.restore_rng_state(loaded, python_rng=rng)
        transition = CompletedGenerationTransition(manager, loaded, 40, rng)
        controller = OptimizationLoopController(
            manager, loaded, 40, FailingInitialScheduler(), transition, rng
        )

        result = controller.run()

        self.assertEqual(transition.calls, list(range(30, 40)))
        self.assertEqual(result.final_generation, 39)
        self.assertEqual(loaded["progress"]["phase"], "finished")
        self.assertEqual(len(loaded["generations"]), 40)
        self.assertEqual(
            loaded["generations"][30]["parent_ids"], loaded["generations"][29]["survivor_ids"]
        )
        checkpoint_after_finish = self.checkpoint_path.read_bytes()
        reloaded = self._manager().load(self.contract, current_ga_settings=self._current_ga(40))
        self.assertEqual(self.checkpoint_path.read_bytes(), checkpoint_after_finish)
        self.assertEqual(reloaded["progress"]["phase"], "finished")

    def test_repeated_extension_and_decrease_rejection(self) -> None:
        """Accept 30-to-40-to-50 increases and reject a later target decrease."""
        self._write_config(40, mutation_probability="0.35")
        manager_40 = self._manager()
        loaded_40 = manager_40.load(self.contract, current_ga_settings=self._current_ga(40))
        manager_40.save(loaded_40)
        backup_bytes = (self.root / "checkpoint.v3.backup.json").read_bytes()
        self._write_config(50, mutation_probability="0.35")

        loaded_50 = self._manager().load(self.contract, current_ga_settings=self._current_ga(50))

        self.assertEqual(loaded_50["ga_settings"]["num_generations"], 50)
        self.assertEqual(
            [
                (item["from"], item["to"])
                for item in loaded_50["metadata"]["generation_limit_extensions"]
            ],
            [(30, 40), (40, 50)],
        )
        self.assertEqual((self.root / "checkpoint.v3.backup.json").read_bytes(), backup_bytes)
        self._write_config(45, mutation_probability="0.35")
        checkpoint_before_rejection = self.checkpoint_path.read_bytes()
        with self.assertRaisesRegex(CheckpointCompatibilityError, "tidak boleh diturunkan"):
            self._manager().load(self.contract, current_ga_settings=self._current_ga(45))
        self.assertEqual(self.checkpoint_path.read_bytes(), checkpoint_before_rejection)

    def test_rejects_other_config_change_and_conflicting_backup(self) -> None:
        """Fail before migration for immutable changes or a conflicting backup file."""
        self._write_config(40, mutation_probability="0.50")
        checkpoint_before_rejection = self.checkpoint_path.read_bytes()
        with self.assertRaisesRegex(CheckpointCompatibilityError, "berubah selain"):
            self._manager().load(self.contract, current_ga_settings=self._current_ga(40))
        self.assertEqual(self.checkpoint_path.read_bytes(), checkpoint_before_rejection)
        self._write_config(40, mutation_probability="0.35")
        (self.root / "checkpoint.v3.backup.json").write_bytes(b"conflict")
        with self.assertRaisesRegex(CheckpointCompatibilityError, "Backup legacy"):
            self._manager().load(self.contract, current_ga_settings=self._current_ga(40))
        self.assertEqual(self.checkpoint_path.read_bytes(), checkpoint_before_rejection)

    def test_failed_migration_save_keeps_original_checkpoint(self) -> None:
        """Keep v3 authoritative when the post-backup atomic save fails."""
        original_bytes = self.checkpoint_path.read_bytes()
        self._write_config(40, mutation_probability="0.35")
        manager = self._manager()

        with patch.object(manager, "save", side_effect=OSError("simulated save failure")):
            with self.assertRaisesRegex(OSError, "simulated save failure"):
                manager.load(self.contract, current_ga_settings=self._current_ga(40))

        self.assertEqual(self.checkpoint_path.read_bytes(), original_bytes)
        self.assertEqual((self.root / "checkpoint.v3.backup.json").read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()
