"""Check fresh compositions, mass inventory and rejection of incompatible public cohorts."""

from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from modules.keff_constraint import KeffConstraint
from modules.public_inputs import file_digest, public_contract, reference_loading, select_front


def synthetic_checkpoint(responses):
    """Return a labeled in-memory checkpoint from ID-to-(keff, PPF) ``responses``.

    It contains no neutron-transport results and is used only for input validation
    tests. Every individual uses a permutation of the public reference inventory.
    """
    sorter, alleles = public_contract()
    genome = [list(g) for g in sorter.extract_included_genome(reference_loading())]
    constraint = KeffConstraint(1.0, 0.0)
    individuals = {}
    for index, (key, (keff, ppf)) in enumerate(responses.items()):
        individuals[key] = dict(
            status="completed",
            genome=genome[index:] + genome[:index],
            results=dict(
                keff=dict(mean=keff, std_dev=0.0001),
                ppf_3d=ppf,
                radial_ppf=ppf,
                axial_ppf=1.0,
                objectives=[-keff, ppf],
                constraint=constraint.evaluate(keff),
            ),
        )
    return dict(
        purpose="SYNTHETIC TEST ONLY; no transport",
        progress=dict(phase="finished"),
        constraint_contract=constraint.identity(),
        input_provenance=dict(
            inputs={
                name: dict(sha256=file_digest(ROOT / "config" / name))
                for name in ("materials.xml", "fuel_inventory.json")
            }
        ),
        genome_contract=dict(
            slot_order=list(sorter.slot_order),
            excluded_genes={s: list(g) for s, g in sorter.excluded_genes.items()},
            mutable_alleles=[list(g) for g in alleles],
        ),
        individuals=individuals,
        generations=[
            dict(
                generation=0,
                status="completed",
                survivor_ids=list(individuals),
                summary=dict(replacement=dict(fronts=[list(individuals)])),
            )
        ],
    )


class FreshMaterialTests(unittest.TestCase):
    """Check independently supplied card ratios, densities and geometric mass units."""

    def test_fresh_fuel_weights_and_no_burnup_isotopes(self):
        """Match all axial compositions to the author's three mass-weight cards; return None."""
        materials = ET.parse(ROOT / "config/materials.xml").getroot().findall("material")
        assignments = json.loads((ROOT / "config/public_fuel_assignments.json").read_text())
        expected = {
            "M8": (5.79875, dict(U235=38, U238=152, Zr=2009.761910, H1=35.5322)),
            "M9": (5.94499, dict(U235=55, U238=220, Zr=1981.6318, H1=35.03487)),
            "M10": (
                6.40732,
                dict(U235=99, U238=396, Zr=1934.17172, H1=34.19578, Er166=3.85564, Er167=2.66335),
            ),
        }
        self.assertEqual(len(materials), 1675)
        self.assertEqual(len(assignments), 111)
        regular_counts = Counter(
            a["fuel_type"] for a in assignments.values() if a["role"] == "regular"
        )
        self.assertEqual(regular_counts, dict(M8=36, M9=36, M10=36))
        for material in materials:
            name = material.get("name")
            if not name.startswith("fuel_"):
                continue
            density, weights = expected[assignments[name.rsplit("_", 1)[0]]["fuel_type"]]
            self.assertEqual(material.find("density").get("units"), "g/cm3")
            self.assertAlmostEqual(float(material.find("density").get("value")), density)
            self.assertEqual(float(material.get("temperature")), 300.0)
            actual = Counter()
            for nuclide in material.findall("nuclide"):
                n = nuclide.get("name")
                self.assertIn(
                    n,
                    {
                        "U235",
                        "U238",
                        "H1",
                        "Zr90",
                        "Zr91",
                        "Zr92",
                        "Zr94",
                        "Zr96",
                        "Er166",
                        "Er167",
                    },
                )
                actual["Zr" if n.startswith("Zr") else n] += float(nuclide.get("wo"))
            self.assertEqual(set(actual), set(weights))
            for n, weight in weights.items():
                self.assertAlmostEqual(actual[n], weight / sum(weights.values()), places=13)
            self.assertEqual([s.get("name") for s in material.findall("sab")], ["c_H_in_ZrH"])

    def test_u235_mass_matches_density_volume_and_fraction(self):
        """Check inventory U-235 in g against independently fixed regular-pin geometry; return None."""
        inventory = json.loads((ROOT / "config/fuel_inventory.json").read_text())
        definitions = json.loads((ROOT / "config/public_fuel_definition.json").read_text())[
            "fuel_types"
        ]
        self.assertEqual(inventory["units"], "g")
        self.assertEqual(len(inventory["fuels"]), 108)
        volume = math.pi * (1.82245**2 - 0.3175**2) * 38.1
        for value in inventory["fuels"].values():
            definition = definitions[value["public_fuel_type"]]
            weights = definition["mass_weights"]
            expected = (
                volume * definition["density_g_cm3"] * weights["U235"] / sum(weights.values())
            )
            self.assertAlmostEqual(value["u_235_fresh"], expected, places=10)
            self.assertEqual(value["u_235_current"], value["u_235_fresh"])
        self.assertEqual(len(reference_loading()), 127)


class PublicCohortTests(unittest.TestCase):
    """Use synthetic records to test selection without producing published physics results."""

    def select(self, checkpoint):
        """Write synthetic ``checkpoint`` temporarily and return the verified selection."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.json"
            path.write_text(json.dumps(checkpoint))
            return select_front(path)

    def test_front_size_is_dynamic_and_constraint_is_respected(self):
        """Accept a two-member trade-off; exclude an attractive but infeasible point; return None."""
        checkpoint = synthetic_checkpoint(dict(A=(1.04, 1.4), B=(1.03, 1.2), I=(0.99, 1.0)))
        checkpoint["generations"][0]["summary"]["replacement"]["fronts"] = [["A", "B"], ["I"]]
        ids, genomes, _, audit = self.select(checkpoint)
        self.assertEqual(ids, ["A", "B"])
        self.assertEqual(audit["population_size"], 3)
        self.assertEqual(len(genomes["B"]), 127)

    def test_equal_infeasible_violations_do_not_use_objective_ties(self):
        """Retain equal-violation individuals despite different PPF objectives; return None."""
        checkpoint = synthetic_checkpoint(dict(A=(0.99, 1.4), B=(0.99, 1.2)))
        self.assertEqual(self.select(checkpoint)[0], ["A", "B"])

    def test_reject_changed_inputs_fixed_ids_and_stale_front(self):
        """Reject material fingerprints, fixed genes, permutations and stale selection; return None."""
        original = synthetic_checkpoint(dict(A=(1.04, 1.4), B=(1.03, 1.2)))
        changed = deepcopy(original)
        changed["input_provenance"]["inputs"]["materials.xml"]["sha256"] = "incompatible"
        with self.assertRaisesRegex(ValueError, "Checkpoint input"):
            self.select(changed)
        changed = deepcopy(original)
        changed["genome_contract"]["excluded_genes"]["D2"] = ["ffcr", "fuel_1"]
        with self.assertRaisesRegex(ValueError, "genome contract"):
            self.select(changed)
        changed = deepcopy(original)
        changed["individuals"]["B"]["genome"][0] = changed["individuals"]["B"]["genome"][1]
        with self.assertRaisesRegex(ValueError, "Invalid completed"):
            self.select(changed)
        changed = deepcopy(original)
        changed["generations"][0]["summary"]["replacement"]["fronts"] = [["A"]]
        with self.assertRaisesRegex(ValueError, "reconstructed"):
            self.select(changed)
        changed = deepcopy(original)
        changed["individuals"]["B"]["results"]["ppf_3d"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite positive"):
            self.select(changed)


class PublicWorkflowTests(unittest.TestCase):
    """Build genuine public CSG from synthetic selection records without transport."""

    def test_public_sdm_preparation_and_preflight(self):
        """Prepare all five rod cases and verify their XML/provenance with a fake environment hash."""
        import openmc
        from final_evaluation.sdm_campaign import prepare, preflight
        from final_evaluation.sdm_contract import load_config

        with tempfile.TemporaryDirectory() as temporary, warnings.catch_warnings():
            warnings.simplefilter("ignore", openmc.IDWarning)
            root = Path(temporary)
            checkpoint = root / "checkpoint.json"
            checkpoint.write_text(json.dumps(synthetic_checkpoint(dict(TEST=(1.01, 1.4)))))
            config = load_config()
            config["checkpoint"] = str(checkpoint)
            config_path = root / "sdm.json"
            config_path.write_text(json.dumps(config))
            with patch(
                "final_evaluation.sdm_campaign.environment",
                return_value={"purpose": "SYNTHETIC TEST ONLY"},
            ), patch("final_evaluation.sdm_campaign.subprocess.run") as transport:
                manifest = prepare(root / "sdm", config_path)
                verified = preflight(root / "sdm", verify_environment=False)
                transport.assert_not_called()
            self.assertEqual(len(manifest["jobs"]), 5)
            self.assertEqual(verified["manifest_sha256"], manifest["manifest_sha256"])
            self.assertFalse(manifest["validation_scope"]["independent_mcnp_benchmark"])
            self.assertTrue(
                all(
                    (root / "sdm" / j["directory"] / "geometry.json").is_file()
                    for j in manifest["jobs"]
                )
            )

    def test_public_nominal_reference_fixture(self):
        """Build candidate and public reference models from a one-member synthetic cohort."""
        sys.path.insert(0, str(ROOT / "reevaluation_iv_4_5_2"))
        from inputs import load_cases
        from model import build_model

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.json"
            checkpoint.write_text(json.dumps(synthetic_checkpoint(dict(TEST=(1.01, 1.4)))))
            cases, audit = load_cases(checkpoint)
            self.assertEqual(audit["actual_front_count"], 1)
            self.assertEqual(len(cases), 2)
            for index, case in enumerate(cases):
                _, records, geometry = build_model(case, 701 + index)
                self.assertEqual(len(records), 1665)
                self.assertEqual(geometry["regular_segments"], 1620)
            self.assertTrue(geometry["reference_geometry_fixture_verified"])


if __name__ == "__main__":
    unittest.main()
