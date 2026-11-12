"""Test evaluation cache without neutron transport.

Run with ``python -m unittest discover -s tests -v``."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME_ROOT / "src"))

from modules.checkpoint import CheckpointManager
from modules.evaluation_cache import EvaluationCache, EvaluationCacheError
from modules.input_provenance import build_input_provenance
from modules.keff_constraint import KeffConstraint


class EvaluationCacheTests(unittest.TestCase):
    """Verify source priority, cache-hit lifecycle, and catalog snapshot stability."""

    def setUp(self) -> None:
        """Create a small target contract and temporary provenance files; return None."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "config.ini"
        self.inventory_path = self.root / "fuel_inventory.json"
        self.materials_path = self.root / "materials.xml"
        self.config_path.write_text("[ga]\nnum_generations = 1\n", encoding="utf-8")
        self.inventory_path.write_text("{}", encoding="utf-8")
        self.materials_path.write_text("<materials/>", encoding="utf-8")
        self.genes = (("fuel", "fuel_1"), ("fuel", "fuel_2"))
        self.contract = CheckpointManager.build_genome_contract(
            "included_position_based_permutation", ("A1", "B1"), {}, self.genes
        )
        self.provenance = build_input_provenance(
            self.config_path, self.inventory_path, self.materials_path, self.contract
        )
        self.constraint = KeffConstraint(1.0, 0.0)
        self.manager = CheckpointManager(
            self.root / "target" / "checkpoint.json",
            "TARGET",
            self.provenance,
            constraint_contract=self.constraint,
        )
        inputs = self.provenance["inputs"]
        self.context = {
            "schema_version": 2,
            "model_version": "test-model/v1",
            "evaluator_version": "test-evaluator/v1",
            "openmc_version": "0.15.3",
            "scientific_inputs": {
                name: inputs[name]["sha256"]
                for name in (
                    "fuel_inventory.json",
                    "materials.xml",
                    "canonical_genome_contract",
                )
            },
            "evaluation_config": {
                "tally_name": "test_tally",
                "tally_score": "fission-q-recoverable",
                "constraint": self.constraint.identity(),
                "convergence": {
                    "enabled": True,
                    "max_keff_std_dev_pcm": 50.0,
                    "max_tally_mean_relative_std_dev": 0.05,
                    "max_tally_p95_relative_std_dev": 0.1,
                    "max_tally_max_relative_std_dev": 0.2,
                    "allow_zero_mean_bins": False,
                },
            },
            "openmc_parameters": {
                "particles": 1000,
                "inactive_batches": 1,
                "active_batches": 2,
            },
        }
        self.state = self.manager.create_state(
            self.contract, {"population_size": 1, "num_generations": 1, "random_seed": 1}
        )
        self.manager.save(self.state)

    def tearDown(self) -> None:
        """Remove temporary cache and checkpoint test files; return None."""
        self.temporary_directory.cleanup()

    def _results(self, ppf: float) -> dict:
        """Return minimal valid completed-result data for the requested PPF and OpenMC version."""
        keff = 1.02
        return {
            "keff": {"mean": keff, "std_dev": 0.0002},
            "ppf_3d": ppf,
            "objectives": [-keff, ppf],
            "constraint": self.constraint.from_results({"keff": {"mean": keff}}),
            "convergence": {"enabled": True, "passed": True},
        }

    def _write_source(self, name: str, ppf: float, *, phase: str = "finished") -> Path:
        """Write a synthetic strict-context checkpoint with the supplied name and results; return its path."""
        path = self.root / f"{name}.json"
        source = {
            "schema_version": 4,
            "campaign": {"campaign_id": name},
            "progress": {"phase": phase},
            "constraint_contract": self.constraint.identity(),
            "input_provenance": deepcopy(self.provenance),
            "genome_contract": deepcopy(self.contract),
            "individuals": {
                "G000-I000": {
                    "status": "completed",
                    "genome": [list(gene) for gene in self.genes],
                    "operator": {"evaluation_context": deepcopy(self.context)},
                    "results": self._results(ppf),
                }
            },
        }
        path.write_text(json.dumps(source), encoding="utf-8")
        return path

    def _write_legacy_source(self, name: str, *, openmc_version: str = "0.15.3") -> Path:
        """Write a synthetic named checkpoint with statistical evidence and no evaluation context; return its path."""
        path = self._write_source(name, 1.20)
        source = json.loads(path.read_text(encoding="utf-8"))
        individual = source["individuals"]["G000-I000"]
        individual["operator"] = {}
        individual["results"].update(
            {
                "metadata": {
                    "tally_name": "test_tally",
                    "tally_score": "fission-q-recoverable",
                    "objective_names": ["negative_keff", "ppf_3d"],
                    "objective_orientation": "minimize",
                    "ppf_3d_aggregation": "max_segment_score_over_mean_segment_score",
                },
                "statistics": {
                    "statepoint": {
                        "openmc_version": openmc_version,
                        "particles_per_generation": 1000,
                        "inactive_batches": 1,
                        "active_batches_completed": 2,
                    }
                },
                "convergence": {
                    "enabled": True,
                    "passed": True,
                    "criteria": {
                        "keff_std_dev_pcm": {"observed": 20.0},
                        "tally_mean_relative_std_dev": {"observed": 0.02},
                        "tally_p95_relative_std_dev": {"observed": 0.04},
                        "tally_max_relative_std_dev": {"observed": 0.08},
                        "zero_mean_bins": {"observed": 0},
                    },
                },
            }
        )
        path.write_text(json.dumps(source), encoding="utf-8")
        return path

    def _build_cache(
        self,
        sources: tuple[Path, ...],
        *,
        allow_legacy: bool = False,
        compatible_versions: tuple[tuple[str, str], ...] = (),
    ) -> EvaluationCache:
        """Return a target EvaluationCache for the ordered source paths and selected legacy/version policy."""
        cache = EvaluationCache(
            enabled=True,
            source_checkpoints=sources,
            allow_legacy_context=allow_legacy,
            compatible_openmc_version_pairs=compatible_versions,
            project_root=self.root,
            campaign_root=self.root / "target",
            target_checkpoint=self.manager.checkpoint_path,
            evaluation_context=self.context,
            genome_contract=self.contract,
            state=self.state,
            checkpoint_manager=self.manager,
        )
        if cache._connection is not None:
            self.addCleanup(cache._connection.close)
        return cache

    def test_first_source_wins_and_hit_completes_without_attempt(self) -> None:
        """Verify first-source priority and completion on a cache hit without a transport attempt; return None."""
        first = self._write_source("SOURCE-A", 1.20)
        second = self._write_source("SOURCE-B", 1.40)
        cache = self._build_cache((first, second))
        self.manager.register_individual(self.state, "G000-I000", self.genes, 0, "initial")
        self.manager.begin_generation(self.state, 0, ("G000-I000",))

        self.assertTrue(cache.try_reuse(self.state, "G000-I000"))
        individual = self.state["individuals"]["G000-I000"]
        self.assertEqual(individual["status"], "completed")
        self.assertEqual(individual["attempts"], [])
        self.assertEqual(individual["results"]["ppf_3d"], 1.20)
        self.assertEqual(
            individual["operator"]["evaluation_cache"]["source_campaign_id"], "SOURCE-A"
        )
        self.assertEqual(cache.summary()["priority_conflicts"], 1)
        self.manager.save(self.state)

    def test_catalog_resume_survives_archived_source(self) -> None:
        """Verify a validated catalog remains usable after its source is archived; return None."""
        source = self._write_source("SOURCE-A", 1.20)
        cache = self._build_cache((source,))
        original_summary = cache.summary()
        source.unlink()

        resumed = self._build_cache((source,))

        self.assertEqual(resumed.summary(), original_summary)

    def test_active_source_reuses_only_completed_results_without_mutating_source(self) -> None:
        """Reuse completed active-source results and skip other statuses without changing the source; return None."""
        source = self._write_source("SOURCE-A", 1.20, phase="evaluating_offspring")
        payload = json.loads(source.read_bytes())
        for status in ("pending", "running", "failed", "not_converged", "interrupted"):
            record = deepcopy(payload["individuals"]["G000-I000"])
            record["status"] = status
            record["results"] = None
            payload["individuals"][status] = record
        source.write_text(json.dumps(payload), encoding="utf-8")
        original_bytes = source.read_bytes()
        cache = self._build_cache((source,))
        self.manager.register_individual(self.state, "G000-I000", self.genes, 0, "initial")
        self.manager.begin_generation(self.state, 0, ("G000-I000",))

        self.assertTrue(cache.try_reuse(self.state, "G000-I000"))
        individual = self.state["individuals"]["G000-I000"]
        self.assertEqual(individual["attempts"], [])
        self.assertEqual(individual["results"]["ppf_3d"], 1.20)
        self.assertEqual(cache.summary()["entry_count"], 1)
        provenance = individual["operator"]["evaluation_cache"]
        self.assertEqual(
            provenance["source_checkpoint_sha256"], hashlib.sha256(original_bytes).hexdigest()
        )
        self.assertEqual(source.read_bytes(), original_bytes)
        snapshot = self.state["metadata"]["evaluation_cache"]["sources"][0]
        self.assertEqual(snapshot["snapshot_phase"], "evaluating_offspring")
        self.manager.save(self.state)

    def test_active_source_update_does_not_change_existing_catalog(self) -> None:
        """Verify an atomic source update does not change the frozen catalog on resume; return None."""
        source = self._write_source("SOURCE-A", 1.20, phase="evaluating_parents")
        cache = self._build_cache((source,))
        original_hash = self.state["metadata"]["evaluation_cache"]["sources"][0][
            "checkpoint_sha256"
        ]
        payload = json.loads(source.read_bytes())
        payload["individuals"]["G000-I000"]["results"] = self._results(1.40)
        replacement = source.with_suffix(".tmp")
        replacement.write_text(json.dumps(payload), encoding="utf-8")
        replacement.replace(source)

        resumed = self._build_cache((source,))
        self.manager.register_individual(self.state, "G000-I000", self.genes, 0, "initial")
        self.manager.begin_generation(self.state, 0, ("G000-I000",))
        self.assertTrue(resumed.try_reuse(self.state, "G000-I000"))
        self.assertEqual(self.state["individuals"]["G000-I000"]["results"]["ppf_3d"], 1.20)
        self.assertEqual(
            self.state["metadata"]["evaluation_cache"]["sources"][0]["checkpoint_sha256"],
            original_hash,
        )

    def test_unknown_source_phase_is_rejected(self) -> None:
        """Reject source phases outside the runtime lifecycle; return None."""
        source = self._write_source("SOURCE-A", 1.20, phase="unknown")
        with self.assertRaisesRegex(EvaluationCacheError, "Phase checkpoint sumber tidak valid"):
            self._build_cache((source,))

    def test_transport_mismatch_cannot_fall_back_to_legacy(self) -> None:
        """Reject particle and batch mismatches even with legacy fallback enabled; return None."""
        for parameter in ("particles", "inactive_batches", "active_batches"):
            with self.subTest(parameter=parameter):
                source = self._write_source("SOURCE-A", 1.20, phase="evaluating_offspring")
                payload = json.loads(source.read_bytes())
                transport = payload["individuals"]["G000-I000"]["operator"]["evaluation_context"][
                    "openmc_parameters"
                ]
                transport[parameter] += 1
                source.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(EvaluationCacheError, f"Parameter {parameter} berbeda"):
                    self._build_cache((source,), allow_legacy=True)

    def test_legacy_active_batch_count_must_match_exactly(self) -> None:
        """Reject legacy active-batch counts below or above the exact target; return None."""
        for active_batches in (1, 3):
            with self.subTest(active_batches=active_batches):
                source = self._write_legacy_source("LEGACY-A")
                payload = json.loads(source.read_bytes())
                payload["progress"]["phase"] = "evaluating_parents"
                payload["individuals"]["G000-I000"]["results"]["statistics"]["statepoint"][
                    "active_batches_completed"
                ] = active_batches
                source.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(EvaluationCacheError, "Active batch legacy berbeda"):
                    self._build_cache((source,), allow_legacy=True)

    def test_changed_genome_is_a_cache_miss(self) -> None:
        """Require reevaluation when allele order changes despite an unchanged inventory; return None."""
        source = self._write_source("SOURCE-A", 1.20, phase="evaluating_parents")
        cache = self._build_cache((source,))
        self.manager.register_individual(
            self.state, "G000-I000", tuple(reversed(self.genes)), 0, "initial"
        )
        self.manager.begin_generation(self.state, 0, ("G000-I000",))

        self.assertFalse(cache.try_reuse(self.state, "G000-I000"))
        self.assertEqual(self.state["individuals"]["G000-I000"]["status"], "pending")
        self.assertEqual(cache.summary()["misses"], 1)

    def test_active_source_completed_result_requires_convergence(self) -> None:
        """Reject completed active-source results without passing convergence evidence; return None."""
        source = self._write_source("SOURCE-A", 1.20, phase="evaluating_parents")
        payload = json.loads(source.read_bytes())
        payload["individuals"]["G000-I000"]["results"]["convergence"]["passed"] = False
        source.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(EvaluationCacheError, "tidak lulus convergence"):
            self._build_cache((source,))

    def test_legacy_context_is_validated_and_openmc_mismatch_rejected(self) -> None:
        """Accept matching legacy context and reject a different OpenMC version; return None."""
        compatible = self._write_legacy_source("LEGACY-OK")
        cache = self._build_cache((compatible,), allow_legacy=True)
        self.assertEqual(cache.summary()["entry_count"], 1)
        source_mode = self.state["metadata"]["evaluation_cache"]["sources"][0]["validation_modes"]
        self.assertEqual(source_mode, ["legacy"])

        # Use a separate target so the incompatible source is validated again.
        mismatch = self._write_legacy_source("LEGACY-BAD", openmc_version="9.9")
        other_manager = CheckpointManager(
            self.root / "other" / "checkpoint.json",
            "OTHER",
            self.provenance,
            constraint_contract=self.constraint,
        )
        other_state = other_manager.create_state(
            self.contract, {"population_size": 1, "num_generations": 1, "random_seed": 1}
        )
        other_manager.save(other_state)
        with self.assertRaisesRegex(EvaluationCacheError, "Versi OpenMC"):
            EvaluationCache(
                enabled=True,
                source_checkpoints=(mismatch,),
                allow_legacy_context=True,
                project_root=self.root,
                campaign_root=self.root / "other",
                target_checkpoint=other_manager.checkpoint_path,
                evaluation_context=self.context,
                genome_contract=self.contract,
                state=other_state,
                checkpoint_manager=other_manager,
            )

    def test_explicit_openmc_version_pair_allows_audited_legacy_reuse(self) -> None:
        """Allow only the explicitly configured directed OpenMC version transition; return None."""
        source = self._write_legacy_source("LEGACY-COMPATIBLE", openmc_version="0.15.3")
        self.context["openmc_version"] = "0.16.0"
        cache = self._build_cache(
            (source,),
            allow_legacy=True,
            compatible_versions=(("0.15.3", "0.16.0"),),
        )
        self.manager.register_individual(self.state, "G000-I000", self.genes, 0, "initial")
        self.manager.begin_generation(self.state, 0, ("G000-I000",))

        self.assertTrue(cache.try_reuse(self.state, "G000-I000"))
        provenance = self.state["individuals"]["G000-I000"]["operator"]["evaluation_cache"]
        self.assertEqual(provenance["validation_mode"], "legacy-compatible-openmc-version")
        self.assertEqual(provenance["source_openmc_version"], "0.15.3")
        self.assertEqual(provenance["target_openmc_version"], "0.16.0")
        self.assertEqual(
            self.state["metadata"]["evaluation_cache"]["compatible_openmc_version_pairs"],
            [["0.15.3", "0.16.0"]],
        )
        self.manager.save(self.state)


if __name__ == "__main__":
    unittest.main()
