"""Test transport configuration without neutron transport.

Run with ``python -m unittest discover -s tests -v``."""

import configparser
from pathlib import Path
import sys
import tempfile
import unittest


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = RUNTIME_ROOT / "src"
for source_path in (RUNTIME_ROOT, SOURCE_ROOT):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from main import MainOptimizer
from modules.config_loader import ConfigLoader
from modules.input_provenance import canonical_config_contract_sha256
from run_optimization import _build_transport_summary


class RuntimeTransportConfigurationTests(unittest.TestCase):
    """Check configured threads, geometry debug, and pre-transport reporting."""

    def test_configured_threads_and_geometry_debug_reach_runner(self) -> None:
        """Compose a temporary campaign and verify explicit runner settings."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            optimizer = MainOptimizer(
                config_path=RUNTIME_ROOT / "config" / "config.ini",
                campaign_id_override="TRANSPORT-CONFIG-TEST",
                run_root_override=temporary_directory,
                evaluation_cache_enabled_override=False,
            )
            summary = _build_transport_summary(optimizer)
            configured_openmc = ConfigLoader(RUNTIME_ROOT / "config" / "config.ini").get_params()[
                "openmc"
            ]
            self.assertEqual(
                optimizer.offspring_creation_scheduler.max_reproduction_attempts, 10000
            )
            self.assertEqual(optimizer.openmc_runner.threads, configured_openmc["threads"])
            self.assertFalse(optimizer.openmc_runner.geometry_debug)
            self.assertEqual(summary["threads"], configured_openmc["threads"])
            self.assertFalse(summary["geometry_debug"])
            self.assertEqual(summary["particles_per_batch"], configured_openmc["particles"])
            self.assertEqual(summary["inactive_batches"], configured_openmc["inactive"])
            self.assertEqual(
                summary["active_batches"],
                configured_openmc["batches"] - configured_openmc["inactive"],
            )
            self.assertIn(
                f"trigger cap {configured_openmc['trigger_max_batches']}", summary["stopping_rule"]
            )

    def test_reproduction_limit_validation_and_injection(self) -> None:
        """Validate optional pair-attempt limits and inject a custom value without transport; return None."""
        parser = configparser.ConfigParser()
        parser.optionxform = str
        parser.read(RUNTIME_ROOT / "config/config.ini")
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.ini"
            for value in ("0", "-1", "1.5", "true", "17", None):
                if value is None:
                    parser.remove_option("ga", "max_reproduction_attempts")
                else:
                    parser.set("ga", "max_reproduction_attempts", value)
                with config_path.open("w") as stream:
                    parser.write(stream)
                with self.subTest(value=value):
                    if value in ("0", "-1", "1.5", "true"):
                        with self.assertRaises(ValueError):
                            ConfigLoader(config_path)
                    else:
                        optimizer = MainOptimizer(
                            config_path=config_path,
                            campaign_id_override=f"REPRODUCTION-{value}",
                            run_root_override=directory,
                            evaluation_cache_enabled_override=False,
                        )
                        self.assertEqual(
                            optimizer.offspring_creation_scheduler.max_reproduction_attempts,
                            10000 if value is None else 17,
                        )

    def test_duplicate_offspring_boolean_validation_and_injection(self) -> None:
        """Validate true/false values, reject an invalid token, and verify that
        registration and evaluation receive the same configured mode."""
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.ini"
            for raw_value, expected in (("true", True), ("false", False), ("sometimes", None)):
                parser = configparser.ConfigParser()
                parser.optionxform = str
                parser.read(RUNTIME_ROOT / "config/config.ini")
                parser.set("ga", "resample_duplicate_offspring", raw_value)
                with config_path.open("w") as stream:
                    parser.write(stream)
                with self.subTest(raw_value=raw_value):
                    if expected is None:
                        with self.assertRaises(ValueError):
                            ConfigLoader(config_path)
                        continue
                    loaded = ConfigLoader(config_path).get_params()["ga"][
                        "resample_duplicate_offspring"
                    ]
                    self.assertIs(loaded, expected)
                    optimizer = MainOptimizer(
                        config_path=config_path,
                        campaign_id_override=f"DUPLICATE-MODE-{raw_value.upper()}",
                        run_root_override=directory,
                        evaluation_cache_enabled_override=False,
                    )
                    self.assertIs(
                        optimizer.offspring_creation_scheduler.resample_duplicate_offspring,
                        expected,
                    )
                    self.assertIs(
                        optimizer.offspring_evaluation_scheduler.resample_duplicate_offspring,
                        expected,
                    )

    def test_evaluation_cache_multiline_sources_and_duplicate_rejection(self) -> None:
        """Parse explicit source priority without scanning the results directory."""
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.ini"
            parser = configparser.ConfigParser()
            parser.optionxform = str
            parser.read(RUNTIME_ROOT / "config/config.ini")
            parser.set(
                "evaluation_cache",
                "source_checkpoints",
                "results/A/checkpoint.json\nresults/B/checkpoint.json",
            )
            parser.set("evaluation_cache", "compatible_openmc_version_pairs", "0.15.3 -> 0.16.0")
            with config_path.open("w") as stream:
                parser.write(stream)
            loaded = ConfigLoader(config_path).get_params()["evaluation_cache"]
            self.assertEqual(
                loaded["source_checkpoints"],
                ("results/A/checkpoint.json", "results/B/checkpoint.json"),
            )
            self.assertEqual(
                loaded["compatible_openmc_version_pairs"],
                (("0.15.3", "0.16.0"),),
            )

            parser.set(
                "evaluation_cache",
                "source_checkpoints",
                "results/A/checkpoint.json\nresults/A/checkpoint.json",
            )
            with config_path.open("w") as stream:
                parser.write(stream)
            with self.assertRaisesRegex(ValueError, "duplikat"):
                ConfigLoader(config_path)

    def test_openmc_cache_version_policy_is_not_a_physics_input(self) -> None:
        """Verify cache version allowlist changes preserve the scientific input fingerprint; return None."""
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.ini"
            parser = configparser.ConfigParser()
            parser.optionxform = str
            parser.read(RUNTIME_ROOT / "config/config.ini")
            parser.set("evaluation_cache", "compatible_openmc_version_pairs", "")
            with config_path.open("w") as stream:
                parser.write(stream)
            without_pair = canonical_config_contract_sha256(config_path)

            parser.set(
                "evaluation_cache",
                "compatible_openmc_version_pairs",
                "0.15.3 -> 0.16.0",
            )
            with config_path.open("w") as stream:
                parser.write(stream)

            self.assertEqual(canonical_config_contract_sha256(config_path), without_pair)


if __name__ == "__main__":
    unittest.main()
