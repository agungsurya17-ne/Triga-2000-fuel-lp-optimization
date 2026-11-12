"""Non-transport SDM tests: public cohort contracts and clearly synthetic attempts."""

import copy
import json
import os
import shutil
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from final_evaluation.sdm_contract import (
    ROOT,
    atomic_json,
    cohort,
    digest,
    load_config,
    lock,
    planned_jobs,
    sha256,
)
from final_evaluation.sdm_campaign import (
    attempt_binding,
    execute_job,
    execution_plan,
    run_campaign,
    verify_completed,
    verify_input,
)
from final_evaluation.sdm_results import (
    inspect_attempt,
    metrics,
    read_statepoint,
    report,
    summarize,
)
from final_evaluation.model import FinalModelBuilder, RODS, FFCR, rod_cases
from final_evaluation.evaluator import K_LIMIT, BETA_EFF


def synthetic_statepoint(path, seed, *, inactive=45, mean=0.99, sigma=0.0002):
    """Write minimal synthetic OpenMC-format fixture at path; no neutron transport occurs."""
    with h5py.File(path, "w") as out:
        out.attrs.update(
            filetype=np.bytes_("statepoint"),
            version=np.array([18, 1]),
            openmc_version=np.array([0, 15, 3]),
            source_present=0,
            tallies_present=0,
        )
        for key, value in dict(
            n_particles=100000,
            n_batches=200,
            current_batch=200,
            n_inactive=inactive,
            generations_per_batch=1,
            seed=seed,
            n_realizations=200 - inactive,
        ).items():
            out[key] = value
        out["run_mode"] = np.bytes_("eigenvalue")
        out["k_combined"] = [mean, sigma]
        out["k_generation"] = np.full(200, mean)
        out["entropy"] = np.linspace(9, 9.1, 200)
        out.create_group("runtime")["total"] = 0.01


def synthetic_transport(
    directory, job, config, *, returncode=0, inactive=45, sigma=0.0002, snapshots=True
):
    """Write labeled fake output at directory for job and return subprocess-like result."""
    from types import SimpleNamespace

    if returncode == 0:
        synthetic_statepoint(
            directory / "statepoint.200.h5", job["seed"], inactive=inactive, sigma=sigma
        )
        if snapshots:
            dtype = np.dtype([("r", [("x", "f8"), ("y", "f8"), ("z", "f8")])])
            bank = np.zeros(100000, dtype=dtype)
            for batch in config["source_snapshots"]:
                bank["r"]["z"] = batch / 10
                with h5py.File(directory / f"source.{batch:03d}.h5", "w") as source:
                    source.attrs["filetype"] = np.bytes_("source")
                    source.create_dataset("source_bank", data=bank, compression="gzip")
    return SimpleNamespace(returncode=returncode)


class ArithmeticAndCohortTests(unittest.TestCase):
    """Check independent arithmetic, fixed survivor selection and rod-state contracts."""

    def test_exact_formulas_signs_and_conversion(self):
        """Verify rho, reciprocal-k SDM, conservative sigma and negative supercritical values."""
        result = metrics(0.99, 0.0003)
        self.assertAlmostEqual(result["rho"], -0.01 / 0.99)
        self.assertAlmostEqual(result["k_U"], 0.9904935)
        self.assertAlmostEqual(result["SDM_nominal_dollar"], (1 / 0.99 - 1) / BETA_EFF)
        self.assertAlmostEqual(0.5 * BETA_EFF * 1e5, 364.7)
        self.assertNotAlmostEqual(result["SDM_nominal_pcm"], (1 - 0.99) * 1e5)
        self.assertTrue(metrics(K_LIMIT, 0)["conditional_sdm_pass"])
        self.assertFalse(metrics(K_LIMIT + 1e-8, 0)["conditional_sdm_pass"])
        self.assertLess(metrics(1.01, 0)["SDM_L_dollar"], 0)
        self.assertLess(
            metrics(0.99, 0.0003)["SDM_L_dollar"], metrics(0.99, 0.0001)["SDM_L_dollar"]
        )
        for mean, sigma in ((0, 0), (float("nan"), 0), (1, -1)):
            with self.assertRaises(ValueError):
                metrics(mean, sigma)

    @unittest.skipUnless(
        (ROOT / "results/TRIGA2000_PUBLIC_FRESH_DEMO/checkpoint.json").is_file(),
        "A completed public optimization checkpoint is not distributed.",
    )
    def test_cohort_and_seed_contract(self):
        """Resolve a completed public front and verify five unique seeds per candidate."""
        config = load_config()
        ids, genomes, _ = cohort(config)
        jobs = planned_jobs(ids, genomes, config)
        self.assertGreater(len(ids), 0)
        self.assertEqual(len(jobs), len(ids) * 5)
        self.assertEqual(len({j["seed"] for j in jobs}), len(ids) * 5)
        self.assertEqual(jobs, planned_jobs(ids, genomes, config))
        self.assertTrue(all(len(g) == 127 for g in genomes.values()))
        self.assertEqual(config["transport"]["inactive"], 45)
        self.assertEqual(config["transport"]["active"], 155)
        self.assertEqual(config["transport"]["batches"], 200)

    def test_all_stuck_states_and_motion(self):
        """Check five one-OUT matrices, independent offsets and three FFCR designations."""
        self.assertEqual(set(FFCR), {"D2", "D13", "D16"})
        for case, states in rod_cases().items():
            self.assertEqual(sum(s == "OUT" for s in states.values()), 1)
            self.assertEqual(states[case.split("-")[1]], "OUT")
            builder = object.__new__(FinalModelBuilder)
            builder.rod_states = states
            for rod in RODS:
                self.assertAlmostEqual(
                    builder.moving_z(rod, 41), 55.753 if states[rod] == "OUT" else 17.653
                )

    @unittest.skipUnless(
        (ROOT / "results/TRIGA2000_PUBLIC_FRESH_DEMO/checkpoint.json").is_file(),
        "A completed public optimization checkpoint is not distributed.",
    )
    def test_actual_model_settings_roundtrip_and_rejection(self):
        """Build real candidate XML without transport; reject inactive/overwrite/trigger changes."""
        import warnings
        import openmc
        from lxml import etree as ET

        config = load_config()
        ids, genomes, _ = cohort(config)
        job = planned_jobs(ids, genomes, config)[0]
        with tempfile.TemporaryDirectory() as temporary, warnings.catch_warnings():
            warnings.simplefilter("ignore", openmc.IDWarning)
            builder = FinalModelBuilder(
                genomes[job["candidate_id"]], ROOT / config["materials"], job["rod_states"]
            )
            model, _ = builder.build_final(
                shutdown=True, particles=100000, total_batches=200, inactive=45, seed=job["seed"]
            )
            model.settings.sourcepoint = dict(
                batches=config["source_snapshots"], separate=True, write=True
            )
            path = Path(temporary) / "model.xml"
            model.export_to_model_xml(path)
            actual = verify_input(path, job, config)
            self.assertEqual(actual["inactive"], 45)
            self.assertEqual(actual["active"], 155)
            original = path.read_bytes()
            for tag, value in (("inactive", "50"), ("trigger/active", "true")):
                tree = ET.fromstring(original)
                tree.find("settings/" + tag).text = value
                path.write_bytes(ET.tostring(tree))
                with self.assertRaises(ValueError):
                    verify_input(path, job, config)
            tree = ET.fromstring(original)
            ET.SubElement(tree.find("settings/source_point"), "overwrite_latest").text = "false"
            path.write_bytes(ET.tostring(tree))
            with self.assertRaises(ValueError):
                verify_input(path, job, config)


class SyntheticExecutionTests(unittest.TestCase):
    """Exercise execution/resume/report with mocked process launch and synthetic HDF5 only."""

    def setUp(self):
        """Create five private synthetic job directories; return None."""
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = load_config()
        self.jobs = planned_jobs(["TEST"], {"TEST": ["synthetic"] * 127}, self.config)[:5]
        for job in self.jobs:
            directory = self.root / job["directory"]
            (directory / "input").mkdir(parents=True)
            (directory / "attempts").mkdir()
            (directory / "input/model.xml").write_text("<model>SYNTHETIC TEST ONLY</model>")
            job["input_sha256"] = sha256(directory / "input/model.xml")
            job["job_sha256"] = digest(job)
            atomic_json(directory / "status.json", dict(state="prepared", attempt=None))
        self.manifest = dict(
            jobs=self.jobs,
            config=self.config,
            cohort=["TEST"],
            context_sha256="synthetic-context",
            manifest_sha256="synthetic-manifest",
            context={},
        )

    def tearDown(self):
        """Remove synthetic temporary artifacts; return None."""
        self.temporary.cleanup()

    def launcher(self, command, *, cwd, env, stdout, stderr):
        """Mock subprocess.run, writing synthetic result/log for current job; return fake process."""
        self.assertEqual(command, [self.config["executable"], "-s", "32"])
        self.assertEqual(env["OMP_NUM_THREADS"], "32")
        job = next(j for j in self.jobs if j["case"] == Path(cwd).parent.parent.name)
        stdout.write(
            "SYNTHETIC TEST ONLY\nVersion | 0.15.3\nMPI Processes | 1\nOpenMP Threads | 32\n"
            "WARNING: Negative probability table synthetic warning\n"
        )
        stdout.flush()
        return synthetic_transport(Path(cwd), job, self.config)

    def execute(self, job=None, launcher=None):
        """Execute one job through a mocked transport function; return None."""
        with patch(
            "final_evaluation.sdm_campaign.subprocess.run", side_effect=launcher or self.launcher
        ):
            execute_job(self.root, job or self.jobs[0], self.manifest)

    def test_serial_launch_resume_skips_and_preserves_all_files(self):
        """Mock five sequential launches then resume; verify every completed file is preserved."""
        with patch("final_evaluation.sdm_campaign.preflight", return_value=self.manifest), patch(
            "final_evaluation.sdm_campaign.running_openmc", return_value=[]
        ), patch("final_evaluation.sdm_campaign.unchanged_context"), patch(
            "final_evaluation.sdm_campaign.subprocess.run", side_effect=self.launcher
        ) as launched:
            run_campaign(self.root)
            self.assertEqual(launched.call_count, 5)
            hashes = {str(p): sha256(p) for p in self.root.rglob("*") if p.is_file()}
            run_campaign(self.root, resume=True)
            self.assertEqual(launched.call_count, 5)
            self.assertEqual(
                hashes, {str(p): sha256(p) for p in self.root.rglob("*") if p.is_file()}
            )
        rows = execution_plan(self.root, self.manifest, resume=True)
        self.assertTrue(all(row["action"] == "skip_verified" for row in rows))
        output = report(self.root, self.manifest)
        self.assertTrue((output / "diagnostics/TEST/STUCK-D2/diagnostics.png").exists())
        summary = json.loads((output / "summary.json").read_text())["candidates"][0]
        self.assertTrue(summary["five_cases_complete"])
        self.assertTrue(summary["conditional_sdm_pass"])
        self.assertIn("source_review_required", summary["evaluation_status"])
        artifact_root = os.environ.get("SDM_SYNTHETIC_ARTIFACTS")
        if artifact_root:
            target = Path(artifact_root)
            shutil.copytree(self.root, target)
            (target / "README.md").write_text(
                "SYNTHETIC TEST ONLY. Five fake HDF5 cases from mocked subprocess. "
                "No neutron transport; no real candidate; never reuse as final results.\n"
            )

    def test_dry_run_cannot_launch(self):
        """Assert run/resume dry-run plans do not invoke the transport executor."""
        with patch("final_evaluation.sdm_campaign.preflight", return_value=self.manifest), patch(
            "final_evaluation.sdm_campaign.execute_job"
        ) as executor:
            run_campaign(self.root, dry_run=True)
            run_campaign(self.root, resume=True, dry_run=True)
            executor.assert_not_called()

    def test_failed_attempt_explicit_retry_preserves_log(self):
        """Retain failed attempt 001, block implicit retry, then create new attempt 002."""

        def fail(command, *, cwd, env, stdout, stderr):
            """Return synthetic launch failure after writing retained failure log."""
            from types import SimpleNamespace

            stdout.write("SYNTHETIC FAILURE\n")
            return SimpleNamespace(returncode=17)

        with self.assertRaises(RuntimeError):
            self.execute(launcher=fail)
        directory = self.root / self.jobs[0]["directory"]
        old = sha256(directory / "attempts/001/openmc.log")
        actions = execution_plan(self.root, self.manifest, resume=True)
        self.assertEqual(actions[0]["action"], "blocked_explicit_retry_required")
        self.assertEqual(
            execution_plan(self.root, self.manifest, resume=True, retry_failed=True)[0]["action"],
            "retry",
        )
        self.execute()
        self.assertEqual(sha256(directory / "attempts/001/openmc.log"), old)
        self.assertEqual(read_json_file(directory / "status.json")["attempt"], "002")

    def test_reject_reuse_candidate_seed_config_and_input_mismatch(self):
        """Reject reuse of completed outputs after candidate, seed, transport or input changes."""
        self.execute()
        job = self.jobs[0]
        directory = self.root / job["directory"]
        state = read_json_file(directory / "status.json")
        for key, value in (("candidate_id", "OTHER"), ("seed", 9), ("genome_sha256", "wrong")):
            changed = copy.deepcopy(job)
            changed[key] = value
            with self.assertRaises(ValueError):
                verify_completed(self.root, changed, self.manifest, state)
        attempt = directory / "attempts/001"
        for manifest_hash, context_hash in (
            ("wrong-manifest", "synthetic-context"),
            ("synthetic-manifest", "wrong-context"),
        ):
            with self.assertRaises(ValueError):
                inspect_attempt(attempt, job, self.config, manifest_hash, context_hash)
        with h5py.File(attempt / "statepoint.200.h5", "r+") as statepoint:
            statepoint.attrs["openmc_version"] = [0, 16, 0]
        with self.assertRaises(ValueError):
            verify_completed(self.root, job, self.manifest, state)
        with h5py.File(attempt / "statepoint.200.h5", "r+") as statepoint:
            statepoint.attrs["openmc_version"] = [0, 15, 3]
            statepoint["n_inactive"][()] = 50
        with self.assertRaises(ValueError):
            verify_completed(self.root, job, self.manifest, state)
        (attempt / "model.xml").write_text("changed")
        with self.assertRaises(ValueError):
            verify_completed(self.root, job, self.manifest, state)

    def test_incomplete_precision_and_manual_review_are_separate(self):
        """Keep conditional numeric pass when sigma is large, and never pass partial cohorts."""
        self.execute()
        directory = self.root / self.jobs[0]["directory"] / "attempts/001"
        with h5py.File(directory / "statepoint.200.h5", "r+") as out:
            out["k_combined"][...] = [0.99, 0.0005]
        row, _, _ = inspect_attempt(
            directory, self.jobs[0], self.config, "synthetic-manifest", "synthetic-context"
        )
        self.assertTrue(row["final_valid"])
        self.assertFalse(row["precision_pass"])
        self.assertTrue(row["conditional_sdm_pass"])
        self.assertEqual(row["source_review_status"], "pending_manual_review")
        rows = [dict(row, candidate_id="TEST", case=j["case"]) for j in self.jobs]
        result = summarize("TEST", rows)
        self.assertTrue(result["conditional_sdm_pass"])
        self.assertIn("precision_target_not_met", result["evaluation_status"])
        self.assertIsNone(summarize("TEST", rows[:4])["conditional_sdm_pass"])
        rows[0]["SDM_L_dollar"] = -1
        rows[0]["conditional_sdm_pass"] = False
        result = summarize("TEST", rows)
        self.assertEqual(result["limiting_case"], "STUCK-D2")
        self.assertFalse(result["conditional_sdm_pass"])
        (directory / "source.045.h5").unlink()
        row, _, _ = inspect_attempt(
            directory, self.jobs[0], self.config, "synthetic-manifest", "synthetic-context"
        )
        self.assertTrue(row["final_valid"])
        self.assertFalse(row["diagnostics_complete"])
        self.assertEqual(row["source_review_status"], "diagnostics_incomplete")
        (directory / "statepoint.200.h5").unlink()
        row, _, _ = inspect_attempt(
            directory, self.jobs[0], self.config, "synthetic-manifest", "synthetic-context"
        )
        self.assertFalse(row["statepoint_complete"])
        self.assertIsNone(row["SDM_L_dollar"])

    def test_missing_entropy_preserves_valid_conditional_sdm(self):
        """Keep combined-keff SDM when entropy is absent, flagging diagnostics incomplete."""
        self.execute()
        directory = self.root / self.jobs[0]["directory"] / "attempts/001"
        with h5py.File(directory / "statepoint.200.h5", "r+") as out:
            del out["entropy"]
        row, generation, entropy = inspect_attempt(
            directory, self.jobs[0], self.config, "synthetic-manifest", "synthetic-context"
        )
        self.assertTrue(row["final_valid"])
        self.assertTrue(row["conditional_sdm_pass"])
        self.assertFalse(row["diagnostics_complete"])
        self.assertEqual(len(generation), 200)
        self.assertEqual(len(entropy), 0)
        self.assertEqual(row["source_review_status"], "diagnostics_incomplete")

    def test_changed_input_is_rejected_before_launch(self):
        """Refuse altered prepared XML before creating an attempt or calling subprocess."""
        directory = self.root / self.jobs[0]["directory"]
        (directory / "input/model.xml").write_text("changed input")
        with patch("final_evaluation.sdm_campaign.subprocess.run") as launched:
            with self.assertRaises(ValueError):
                execute_job(self.root, self.jobs[0], self.manifest)
            launched.assert_not_called()
        self.assertEqual(list((directory / "attempts").iterdir()), [])

    def test_exclusive_lock_and_atomic_json(self):
        """Reject a concurrent holder and preserve last complete JSON on serialization failure."""
        path = self.root / "state.json"
        atomic_json(path, {"state": "valid"})
        with self.assertRaises(ValueError):
            atomic_json(path, {"bad": float("nan")})
        self.assertEqual(read_json_file(path), {"state": "valid"})
        with lock(self.root / ".lock"):
            with self.assertRaises(RuntimeError):
                with lock(self.root / ".lock"):
                    pass


def read_json_file(path):
    """Read test JSON path and return decoded value without importing implementation helpers."""
    return json.loads(path.read_text())


if __name__ == "__main__":
    unittest.main()
