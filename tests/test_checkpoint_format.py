"""Test checkpoint format without neutron transport.

Run with ``python -m unittest discover -s tests -v``."""

import json
from pathlib import Path
import sys
import tempfile
import unittest


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = RUNTIME_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from modules.checkpoint import CheckpointManager
from modules.keff_constraint import KeffConstraint


from modules.input_provenance import build_input_provenance


TEST_CONSTRAINT = KeffConstraint(1.0, 0.0)


class RuntimeCheckpointFormatTests(unittest.TestCase):
    """Check readable default serialization and semantic JSON round trips."""

    def test_default_checkpoint_is_indented_and_loads_identically(self) -> None:
        """Save multiline JSON by default and recover the same parsed state."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "config.ini"
            inventory_path = root / "fuel_inventory.json"
            materials_path = root / "materials.xml"
            config_path.write_text("[ga]\nrandom_seed = 12345\n", encoding="utf-8")
            inventory_path.write_text('{"fuel_1": 1.0}', encoding="utf-8")
            materials_path.write_text("<materials/>", encoding="utf-8")
            contract = CheckpointManager.build_genome_contract(
                "included_position_based_permutation",
                ("A1", "B1"),
                {"A1": ("irradiation_tube", "IT")},
                (("fuel", "fuel_1"),),
            )
            provenance = build_input_provenance(
                config_path, inventory_path, materials_path, contract
            )
            manager = CheckpointManager(
                root / "checkpoint.json",
                "FORMAT-TEST",
                provenance,
                constraint_contract=TEST_CONSTRAINT,
            )
            state = manager.create_state(
                contract, ga_settings={"population_size": 1, "random_seed": 12345}
            )
            manager.save(state)
            checkpoint_text = manager.checkpoint_path.read_text(encoding="utf-8")
            self.assertTrue(checkpoint_text.startswith('{\n  "schema_version"'))
            self.assertGreater(len(checkpoint_text.splitlines()), 1)
            self.assertEqual(
                json.loads(checkpoint_text), manager.load(expected_genome_contract=contract)
            )


if __name__ == "__main__":
    unittest.main()
