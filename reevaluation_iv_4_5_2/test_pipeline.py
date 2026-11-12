"""Independent scientific arithmetic and checkpoint contract regression tests."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from common import (
    HERE,
    ROOT,
    NOMINAL,
    BASELINE_ID,
    atomic_path,
    write_json,
    read_json,
    power_metrics,
    classify,
    dominates,
    digest,
    object_digest,
)
from inputs import load_cases, reference_segment_geometry
from model import (
    build_model,
    validate_records,
    validate_model,
    extract_statepoint,
    common_model_signature,
)
from reevaluate_final_front import verify_completed, execute_campaign
from analyze_reevaluation import analyze, make_tables
from modules.OpenMCResultEvaluator import OpenMCResultEvaluator


class ArithmeticTests(unittest.TestCase):
    """Validate physics arithmetic against independent exact examples."""

    def test_unequal_volumes(self):
        """Check unequal-volume 2-element/2-layer oracle; return None."""
        records = [
            dict(slot=s, axial=j, volume_cm3=v)
            for s, j, v in [("B1", 1, 1), ("B1", 2, 3), ("D2", 1, 2), ("D2", 2, 4)]
        ]
        result = power_metrics([4, 3, 2, 8], records)
        self.assertAlmostEqual(result["ppf_3d"], 40 / 17)
        self.assertAlmostEqual(result["ppf_radial"], 35 / 34)
        self.assertAlmostEqual(result["ppf_axial"], 20 / 17)
        self.assertNotAlmostEqual(result["ppf_3d"], 8 / (17 / 4))
        self.assertIsNone(result["ppf_uncertainty"])

    def test_uniform_density_and_scale(self):
        """Check PPF=1 for uniform density and scale invariance; return None."""
        records = [
            dict(slot=str(i // 2), axial=i % 2, volume_cm3=v) for i, v in enumerate([1, 2, 3, 4])
        ]
        for factor in [1, 1000]:
            result = power_metrics(np.array([1, 2, 3, 4]) * factor, records)
            for key in ["ppf_3d", "ppf_radial", "ppf_axial"]:
                self.assertEqual(result[key], 1.0)

    def test_invalid_scores_and_volumes(self):
        """Reject nonfinite/negative/empty Q or nonpositive V; return None."""
        for q, records in [
            ([], []),
            ([0], [dict(slot="B1", axial=1, volume_cm3=1)]),
            ([-1], [dict(slot="B1", axial=1, volume_cm3=1)]),
            ([float("nan")], [dict(slot="B1", axial=1, volume_cm3=1)]),
            ([1], [dict(slot="B1", axial=1, volume_cm3=0)]),
        ]:
            with self.assertRaises(ValueError):
                power_metrics(q, records)

    def test_classification_exact_and_weak_ties(self):
        """Verify all five classes plus unrounded and equal-objective boundaries; return None."""
        ref = (1.05, 2.3)
        cases = [
            ((1.06, 2.2), "mendominasi acuan"),
            ((1.04, 2.4), "didominasi acuan"),
            ((1.06, 2.4), "Trade-off A"),
            ((1.04, 2.2), "Trade-off B"),
            (ref, "setara"),
            ((1.05, 2.2), "mendominasi acuan"),
            ((1.04, 2.3), "didominasi acuan"),
            ((1.050000000001, 2.3), "mendominasi acuan"),
        ]
        for point, label in cases:
            self.assertEqual(classify(point, ref), label)
        self.assertFalse(dominates(ref, ref))

    def test_atomic_rollback_and_numpy(self):
        """Check failed write retains old content and NumPy results serialize; return None."""
        with tempfile.TemporaryDirectory(dir=HERE / "validation") as temporary:
            path = Path(temporary) / "result.json"
            write_json(path, dict(value=np.int32(2)))
            before = path.read_bytes()
            with self.assertRaises(RuntimeError):
                with atomic_path(path) as staging:
                    staging.write_text("partial")
                    raise RuntimeError("injected interrupted write")
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(read_json(path), {"value": 2})


class GeometryTests(unittest.TestCase):
    """Check actual loading, geometric fuel domain, and historical definitions without transport."""

    @classmethod
    def setUpClass(cls):
        """Build actual candidate/reference fixtures once; return None."""
        from inputs import CHECKPOINT

        if CHECKPOINT.is_file():
            cls.cases, cls.audit = load_cases()
        else:
            genome = read_json(ROOT / "tests/fixtures/geometry/reference_genome.json")
            cls.cases = [dict(candidate_id="PUBLICATION_GEOMETRY_FIXTURE", genome=genome)]
            cls.audit = dict(purpose="Geometry fixture only; no completed public cohort.")
        cls.model, cls.records, cls.volumes = build_model(cls.cases[0], 123)

    @unittest.skipUnless(
        (ROOT / "results/TRIGA2000_PUBLIC_FRESH_DEMO/checkpoint.json").is_file(),
        "A completed public optimizer checkpoint is not distributed.",
    )
    def test_cohort_and_reference(self):
        """Verify the public final-front membership and reference fixture; return None."""
        self.assertEqual(len(self.cases), self.audit["actual_front_count"] + 1)
        self.assertGreater(self.audit["actual_front_count"], 0)
        self.assertTrue(self.audit["independent_front_verified"])
        self.assertEqual(len(reference_segment_geometry()), 1665)
        self.assertEqual(self.cases[-1]["candidate_id"], BASELINE_ID)

    def test_domain_identity_and_volume(self):
        """Verify actual regular/follower counts and analytic volumes; return None."""
        self.assertEqual(self.volumes["regular_segments"], 1620)
        self.assertEqual(self.volumes["ffcr_segments"], 45)
        self.assertAlmostEqual(self.volumes["total_volume_cm3"], 42597.02648433863)
        self.assertNotEqual(
            self.volumes["regular_segment_volume_cm3"], self.volumes["ffcr_segment_volume_cm3"]
        )
        self.assertEqual(len({r["cell_id"] for r in self.records}), 1665)

    def test_corrupt_identity_rejected(self):
        """Reject swapped axial/material identities and duplicate cells; return None."""
        for field, value in [
            ("cell_id", self.records[1]["cell_id"]),
            ("axial", 2),
            ("fuel_id", "wrong"),
            ("volume_cm3", -1),
        ]:
            records = deepcopy(self.records)
            records[0][field] = value
            with self.assertRaises(ValueError):
                validate_records(records, self.cases[0]["genome"])

    def test_historical_regular_ppfs(self):
        """Compare all three regular PPFs to the actual historical implementation on identical Q; return None."""
        records = [r for r in self.records if r["category"] == "REGULAR"]
        q = np.random.default_rng(217).lognormal(size=len(records))
        converted = [
            dict(slot_name=r["slot"], axial_index=r["axial"], gene_id=r["fuel_id"]) for r in records
        ]
        old = OpenMCResultEvaluator._calculate_power_results(q, converted)
        new = power_metrics(q, records)
        for historical, current in [
            ("ppf_3d", "ppf_3d"),
            ("radial_ppf", "ppf_radial"),
            ("axial_ppf", "ppf_axial"),
        ]:
            self.assertAlmostEqual(old[historical], new[current], places=12)

    @unittest.skipUnless(
        (ROOT / "results/TRIGA2000_PUBLIC_FRESH_DEMO/checkpoint.json").is_file(),
        "A completed public optimizer checkpoint is not distributed.",
    )
    def test_actual_historical_response_values(self):
        """Reproduce stored candidate PPFs on its identical normalized segment Q; return None."""
        from inputs import CHECKPOINT

        item = read_json(CHECKPOINT)["individuals"][self.cases[0]["candidate_id"]]["results"]
        records = [r for r in self.records if r["category"] == "REGULAR"]
        q = [item["power_by_segment"][r["slot"]][f"{r['axial']:02d}"] for r in records]
        result = power_metrics(q, records)
        for old, new in [
            ("ppf_3d", "ppf_3d"),
            ("radial_ppf", "ppf_radial"),
            ("axial_ppf", "ppf_axial"),
        ]:
            self.assertAlmostEqual(item[old], result[new], places=12)

    def test_nominal_budget_and_disabled_diagnostics(self):
        """Reject changed budget, trigger and entropy settings; return None."""
        validate_model(self.model, 123, NOMINAL)
        self.model.settings.batches = 499
        with self.assertRaises(ValueError):
            validate_model(self.model, 123, NOMINAL)
        self.model.settings.batches = 500
        self.assertIsNone(self.model.settings.entropy_mesh)
        self.assertFalse(self.model.settings.keff_trigger)

    @unittest.skipUnless(
        (ROOT / "results/TRIGA2000_PUBLIC_FRESH_DEMO/checkpoint.json").is_file(),
        "A completed public optimizer checkpoint is not distributed.",
    )
    def test_common_model_loading_only(self):
        """Verify model equality beyond seed/loading against reference; return None."""
        reference, records, audit = build_model(self.cases[-1], 124)
        with tempfile.TemporaryDirectory(dir=HERE / "validation") as temporary:
            a, b = Path(temporary) / "a.xml", Path(temporary) / "b.xml"
            self.model.export_to_model_xml(a)
            reference.export_to_model_xml(b)
            self.assertEqual(
                common_model_signature(a, self.records), common_model_signature(b, records)
            )
            self.assertTrue(audit["reference_geometry_fixture_verified"])


class SmokeArtifactTests(unittest.TestCase):
    """Exercise real smoke statepoint verification and restart logic without more transport."""

    @classmethod
    def setUpClass(cls):
        """Find the latest completed smoke fixture or skip; return None."""
        choices = [
            p
            for p in (HERE / "validation").glob("smoke*/manifest.json")
            if read_json(p).get("finished")
        ]
        if not choices:
            raise unittest.SkipTest("Run --smoke-test before artifact tests.")
        cls.output = sorted(choices, key=lambda p: p.stat().st_mtime)[-1].parent
        cls.manifest = read_json(cls.output / "manifest.json")
        cls.entry = next(iter(cls.manifest["cases"].values()))

    def test_real_statepoint_verifies(self):
        """Verify real final smoke statepoint and every stored numerical result; return None."""
        result = verify_completed(self.output, self.entry)
        self.assertEqual(len(result["segments"]), 1665)
        self.assertEqual(result["current_batch"], 6)

    def test_partial_or_seed_mismatch_rejected(self):
        """Reject smoke statepoint as nominal production or wrong-seed result; return None."""
        path = self.output / self.entry["statepoint_path"]
        records = read_json(self.output / self.entry["attempt_path"] / "segments.json")
        with self.assertRaises(ValueError):
            extract_statepoint(path, records, self.entry["seed"], NOMINAL)
        with self.assertRaises(ValueError):
            extract_statepoint(path, records, self.entry["seed"] + 1, self.manifest["parameters"])

    def test_tally_permutation_invariance(self):
        """Verify record order cannot silently permute OpenMC tally identities; return None."""
        path = self.output / self.entry["statepoint_path"]
        records = read_json(self.output / self.entry["attempt_path"] / "segments.json")
        result = extract_statepoint(
            path, records[::-1], self.entry["seed"], self.manifest["parameters"]
        )
        self.assertEqual(result["keff"], verify_completed(self.output, self.entry)["keff"])
        self.assertEqual(result["segments"][0]["cell_id"], records[0]["cell_id"])

    def test_altered_artifact_rejected(self):
        """Reject a changed statepoint hash without reading it as completed; return None."""
        entry = deepcopy(self.entry)
        entry["artifacts"][entry["statepoint_path"]] = "0" * 64
        with self.assertRaises(ValueError):
            verify_completed(self.output, entry)

    def test_checkpoint_failure_and_completed_skip(self):
        """Inject build failure and prove running/failed restart while completed case survives; return None."""
        manifest = deepcopy(self.manifest)
        first, second = list(manifest["cases"].values())
        first["status"] = "running"
        cases = manifest["identity"]["cases"]
        with tempfile.TemporaryDirectory(dir=HERE / "validation") as temporary:
            out = Path(temporary)
            with patch(
                "reevaluate_final_front.prepare_case",
                side_effect=RuntimeError("injected build failure"),
            ) as prepare, patch(
                "reevaluate_final_front.verify_completed", return_value={}
            ) as verify, patch(
                "analyze_reevaluation.analyze", return_value={"verification_errors": {}}
            ):
                code = execute_campaign(out, cases, manifest, Path("/unused/openmc"))
            self.assertEqual(code, 1)
            self.assertEqual(prepare.call_count, 1)
            self.assertEqual(verify.call_count, 1)
            saved = read_json(out / "manifest.json")
            failed = saved["cases"][first["candidate_id"]]
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["attempt_number"], first["attempt_number"])
            self.assertEqual(failed["previous_attempts"][-1]["status"], "running")
            self.assertEqual(saved["cases"][second["candidate_id"]]["status"], "completed")
            self.assertFalse(saved["finished"])

    def test_failed_and_pending_rows_retained(self):
        """Verify reports retain all selected rows without asserting incomplete results; return None."""
        manifest = deepcopy(self.manifest)
        entries = list(manifest["cases"].values())
        entries[0]["status"] = "failed"
        entries[1]["status"] = "pending"
        results, segments, profiles, hotspots, relations, errors = make_tables(
            manifest, self.output
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(results.keff.isna().all())
        self.assertEqual(len(segments), 0)
        self.assertEqual(len(relations), 0)

    def test_every_output_generated_without_transport(self):
        """Regenerate nine required artifacts without a transport function call; return None."""
        with patch(
            "reevaluate_final_front.run_transport",
            side_effect=AssertionError("transport forbidden"),
        ):
            report = analyze(self.output)
        self.assertTrue(report["complete"])
        for name in [
            "reevaluation_results.csv",
            "comparison.xlsx",
            "segment_responses.csv",
            "radial_axial_profiles.csv",
            "pareto_reevaluation.png",
            "comparison_nominal.png",
            "hotspot_summary.csv",
            "summary.md",
            "manifest.json",
        ]:
            self.assertTrue((self.output / name).is_file(), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
