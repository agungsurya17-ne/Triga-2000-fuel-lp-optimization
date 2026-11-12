"""Static geometry regressions with public fuel identities; no neutron transport."""

import configparser
import csv
import itertools
import math
from pathlib import Path
import re
import sys
import unittest
import warnings
import xml.etree.ElementTree as ET

import openmc


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.openmc_model_builder import OpenMCModelBuilder
from final_evaluation.model import FinalModelBuilder


def reference_genome():
    """Return the recorded 127-position loading used by geometry regressions."""
    import json

    return json.loads((ROOT / "tests/fixtures/geometry/reference_genome.json").read_text())


class BaselineGeometryTests(unittest.TestCase):
    """Check corrected material regions, their neighbors, and unchanged fuel."""

    @classmethod
    def setUpClass(cls):
        """Build reference CSG in memory and load audit probes; return None."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", openmc.IDWarning)
            cls.builder = OpenMCModelBuilder(
                reference_genome(),
                ROOT / "config/materials.xml",
                "fuel_segment_q",
                "fission-q-recoverable",
            )
            cls.geometry, cls.slots, cls.lattice, _, _ = cls.builder._build_geometry()
        cls.centers = {}
        for x in range(-6, 7):
            for y in range(-6, 7):
                if max(abs(x), abs(y), abs(x + y)) <= 6:
                    universe = cls.lattice.get_universe((x, y))
                    local = cls.lattice.get_local_coordinates((0.0, 0.0, 0.0), (x, y))
                    cls.centers[universe.id] = (-local[0], -local[1])
        with (ROOT / "tests/fixtures/geometry/geometry_probe_differences.csv").open() as stream:
            cls.old_differences = list(csv.DictReader(stream))

    def assert_material(self, point, expected):
        """Assert the deepest cell at point (cm) has expected material; return None."""
        cells = [item for item in self.geometry.find(point) if isinstance(item, openmc.Cell)]
        self.assertTrue(cells, msg=f"Unreachable point: {point}")
        self.assertIsInstance(cells[-1].fill, openmc.Material)
        self.assertEqual(cells[-1].fill.name, expected, msg=f"Point: {point}")

    def global_point(self, slot, radius, z):
        """Translate local +x radius and z (cm) at slot; return a global point."""
        x, y = self.centers[self.slots[slot]["universe"].id]
        return (x + radius, y, z)

    def test_all_48_audited_differences_and_perturbations_are_aluminum(self):
        """Check all former mismatches and 27 nearby points each; return None."""
        self.assertEqual(len(self.old_differences), 48)
        for row in self.old_differences:
            point = tuple(float(row[key]) for key in ("x_cm", "y_cm", "z_cm"))
            for shift in itertools.product((-1e-4, 0.0, 1e-4), repeat=3):
                shifted = tuple(value + delta for value, delta in zip(point, shift))
                with self.subTest(point=shifted):
                    self.assert_material(shifted, "aluminum")

    def test_peripheral_slabs_partition_outer_universe_without_overlap(self):
        """Check both sides of four plate planes and the water complement; return None."""
        outer = self.lattice.outer
        for boundary, before, after in (
            (0.0, "water", "aluminum"),
            (1.905, "aluminum", "water"),
            (66.9925, "water", "aluminum"),
            (68.58, "aluminum", "water"),
        ):
            for offset, expected in ((-1e-4, before), (1e-4, after)):
                point = (0.0, 0.0, boundary + offset)
                cells = [cell for cell in outer.cells.values() if point in cell.region]
                self.assertEqual(len(cells), 1, msg=f"Outer partition at {point}")
                self.assertEqual(cells[0].fill.name, expected)
        for z in (0.9525, 67.78625):
            self.assertFalse(
                self.lattice.is_valid_index(self.lattice.find_element((0.0, 25.4, z))[0])
            )
            self.assert_material((0.0, 25.4, z), "aluminum")

    def test_grid_material_at_all_corrected_and_control_rod_slots(self):
        """Check aluminum grid regions at all seven corrected slots and two CRs; return None."""
        for slot in ("A1", "G10", "G17", "G32", "D2", "D13", "D16", "D5", "D9"):
            for z in (0.9525, 67.78625):
                with self.subTest(slot=slot, z=z):
                    self.assert_material(self.global_point(slot, 2.02, z), "aluminum")

    def test_rod_extensions_and_neighboring_materials(self):
        """Check all five Al annuli, adjacent water, and intact SS cladding; return None."""
        for slot in ("D2", "D5", "D9", "D13", "D16"):
            for z in (107.02935, 120.0, 548.9601, 698.9599):
                self.assert_material(self.global_point(slot, 1.5, z), "aluminum")
            for radius in (1.32075, 1.72095):
                self.assert_material(self.global_point(slot, radius, 120.0), "water")
            self.assert_material(self.global_point(slot, 1.5, 107.02915), "ss304")
        for slot in ("D2", "D13", "D16"):
            self.assert_material(self.global_point(slot, 0.1, 30.0), "zirconium_rod")
            self.assert_material(self.global_point(slot, 1.7, 30.0), "ss304")

    def test_all_1665_fuel_segments_preserve_material_bounds_volume_and_position(self):
        """Match fuel CSG against public geometric bounds in cm and cm³; return None."""
        with (ROOT / "tests/fixtures/geometry/cell_mapping_verified.csv").open() as stream:
            records = list(csv.DictReader(stream))
        self.assertEqual(len(records), 1665)
        for row in records:
            slot, axial = row["slot"], int(row["axial"])
            cell = self.slots[slot]["fuel_cells"][axial - 1]
            self.assertEqual(cell.fill.name, row["material_name"])
            z_bounds, radii = {}, {}
            for halfspace in cell.region:
                if isinstance(halfspace.surface, openmc.ZPlane):
                    z_bounds[halfspace.side] = halfspace.surface.z0
                elif isinstance(halfspace.surface, openmc.ZCylinder):
                    radii[halfspace.side] = halfspace.surface.r
            for actual, key in (
                (z_bounds["+"], "z_low_cm"),
                (z_bounds["-"], "z_high_cm"),
                (radii["+"], "inner_radius_cm"),
                (radii["-"], "outer_radius_cm"),
            ):
                self.assertAlmostEqual(actual, float(row[key]), places=11)
            volume = math.pi * (radii["-"] ** 2 - radii["+"] ** 2) * (z_bounds["-"] - z_bounds["+"])
            self.assertAlmostEqual(volume, float(row["openmc_volume_cm3"]), places=10)
            center = self.centers[self.slots[slot]["universe"].id]
            self.assertAlmostEqual(center[0], float(row["x_cm"]), places=11)
            self.assertAlmostEqual(center[1], float(row["y_cm"]), places=11)
            point = self.global_point(
                slot, (radii["+"] + radii["-"]) / 2, (z_bounds["+"] + z_bounds["-"]) / 2
            )
            self.assert_material(point, row["material_name"])

    def test_regular_fuel_tally_domain_is_unchanged(self):
        """Check the historical 1620-bin regular-fuel tally contract; return None."""
        tallies, records = self.builder._build_fission_q_tally(self.slots)
        self.assertEqual(len(records), 1620)
        self.assertEqual(len({row["cell_id"] for row in records}), 1620)
        self.assertEqual(tallies[0].scores, ["fission-q-recoverable"])
        self.assertEqual({row["axial_index"] for row in records}, set(range(1, 16)))
        self.assertFalse({row["slot_name"] for row in records} & {"D2", "D13", "D16"})

    def test_geometry_change_has_a_distinct_evaluation_identity(self):
        """Check base and inherited model identities distinguish corrected physics; return None."""
        self.assertEqual(OpenMCModelBuilder.MODEL_VERSION, "triga2000_openmc_model/v2")
        self.assertIn(OpenMCModelBuilder.MODEL_VERSION, FinalModelBuilder.MODEL_VERSION)


if __name__ == "__main__":
    unittest.main()
