"""Independent arithmetic and motion isolation tests for the IV.4.5 path."""

import math
from pathlib import Path
import sys
import unittest
import warnings
import numpy as np
import openmc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from final_evaluation.evaluator import (
    power_metrics,
    sdm_metrics,
    budget_scenarios,
    K_LIMIT,
)
from final_evaluation.model import FinalModelBuilder, RODS


class FinalArithmeticTests(unittest.TestCase):
    """Check density weighting and conditional SDM with independent small examples."""

    def test_unequal_volumes_and_separate_ppfs(self):
        """Use unequal volumes and off-center peaks; return None after oracle checks."""
        records = [
            dict(slot="B1", axial=1, volume_cm3=1.0),
            dict(slot="B1", axial=2, volume_cm3=3.0),
            dict(slot="D2", axial=1, volume_cm3=2.0),
            dict(slot="D2", axial=2, volume_cm3=4.0),
        ]
        result = power_metrics([4, 3, 2, 8], records)
        self.assertAlmostEqual(result["ppf_3d"], 40 / 17)
        self.assertAlmostEqual(result["ppf_radial"], (7 / 4) / (17 / 10))
        self.assertAlmostEqual(result["ppf_axial"], 2 / (17 / 10))
        self.assertNotAlmostEqual(result["ppf_3d"], result["ppf_radial"] * result["ppf_axial"])
        self.assertNotAlmostEqual(result["ppf_3d"], 8 / (17 / 4))

    def test_uniform_density_and_scaling(self):
        """Check unit PPF at uniform density and normalization invariance; return None."""
        records = [
            dict(slot=str(i // 2), axial=i % 2, volume_cm3=v) for i, v in enumerate([1, 2, 3, 4])
        ]
        result = power_metrics([5, 10, 15, 20], records)
        for key in ("ppf_3d", "ppf_radial", "ppf_axial"):
            self.assertAlmostEqual(result[key], 1.0)
        self.assertEqual(result["ppf_uncertainty"], None)

    def test_sdm_boundary_and_monotonic_uncertainty(self):
        """Check exact half-dollar boundary and conservative direction; return None."""
        self.assertAlmostEqual(K_LIMIT, 0.9963662522779424)
        self.assertAlmostEqual(sdm_metrics(K_LIMIT, 0)["sdm_lower_dollar"], 0.5)
        low = sdm_metrics(0.99, 0.0002)
        high = sdm_metrics(0.99, 0.0004)
        self.assertLess(high["sdm_lower_dollar"], low["sdm_lower_dollar"])
        self.assertAlmostEqual(low["sigma_rho_pcm"], 20 / 0.99**2)
        self.assertNotAlmostEqual(low["sigma_rho_pcm"], low["sigma_k_times_1e5"])

    def test_batch_budget_ceil_and_inactive(self):
        """Verify active-first scaling and ceiling at fractional budgets; return None."""
        rows = budget_scenarios(0.000301, 150, 300, 100)
        self.assertEqual(rows[1]["estimated_active_batches"], 152)
        self.assertEqual(rows[1]["estimated_total_batches"], 202)
        for row in rows:
            self.assertEqual(row["estimated_total_batches"], row["estimated_active_batches"] + 50)

    def test_invalid_response(self):
        """Reject negative, empty, or nonfinite responses; return None."""
        for values in ([-1.0], [float("nan")], [0.0]):
            with self.assertRaises(ValueError):
                power_metrics(values, [dict(slot="B1", axial=1, volume_cm3=1)])


class IndependentRodTests(unittest.TestCase):
    """Check per-rod motion without loading confidential candidate inventories."""

    def test_shutdown_state_cannot_use_nominal_ppf(self):
        """Reject nominal tally construction for an IN rod state; return None."""
        builder = object.__new__(FinalModelBuilder)
        builder.rod_states = {rod: "IN" for rod in RODS}
        with self.assertRaisesRegex(ValueError, "Nominal PPF"):
            builder.build_final(shutdown=False)

    def test_one_rod_change_leaves_four_rods_out(self):
        """Validate independent numeric plane states and plane-38 rounding; return None."""
        # Only moving_z is exercised; no materials or geometry construction is required.
        builder = object.__new__(FinalModelBuilder)
        builder.rod_states = {rod: "OUT" for rod in RODS}
        builder.rod_states["D2"] = "IN"
        self.assertAlmostEqual(builder.moving_z("D2", 41), 17.653)
        self.assertAlmostEqual(builder.moving_z("D2", 38), -37.67138)
        for rod in RODS[1:]:
            self.assertAlmostEqual(builder.moving_z(rod, 41), 55.753)


if __name__ == "__main__":
    unittest.main()
