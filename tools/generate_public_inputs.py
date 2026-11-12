"""Rebuild the public fresh-fuel example without reading private research inputs."""

import argparse
import json
import math
from pathlib import Path
import random
import re
import sys

import openmc

sys.dont_write_bytecode = True


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    """Write JSON-compatible ``value`` at ``path``; return None."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def add_component(material, name, weight):
    """Add isotope or natural element ``name`` with relative mass ``weight``; return None."""
    if any(char.isdigit() for char in name):
        material.add_nuclide(name, weight, percent_type="wo")
    else:
        # Expand natural elements independently of the installed nuclear library.
        abundances = {
            isotope: abundance
            for isotope, abundance in openmc.data.NATURAL_ABUNDANCE.items()
            if re.fullmatch(name + r"\d+", isotope)
        }
        total = sum(a * openmc.data.atomic_mass(n) for n, a in abundances.items())
        if total <= 0:
            raise ValueError(f"No natural isotopic composition for {name}.")
        for isotope, abundance in abundances.items():
            material.add_nuclide(
                isotope, weight * abundance * openmc.data.atomic_mass(isotope) / total, "wo"
            )


def material_recipe(name, definition, material_id):
    """Build a 300 K material from normalized mass weights and g/cm³ density; return it."""
    material = openmc.Material(material_id=material_id, name=name, temperature=300.0)
    density = definition["density_g_cm3"]
    weights = definition["mass_weights"]
    if (
        density <= 0
        or not math.isfinite(density)
        or any(w <= 0 or not math.isfinite(w) for w in weights.values())
    ):
        raise ValueError("Require positive finite density and component mass weights.")
    material.set_density("g/cm3", density)
    total = sum(weights.values())
    for component, weight in weights.items():
        add_component(material, component, weight / total)
    for name in definition.get("thermal_scattering", []):
        material.add_s_alpha_beta(name)
    return material


def generate(root=ROOT):
    """Rebuild fresh materials, inventory, INI IDs and shuffled reference under ``root``.

    Read only public recipes and settings. Refuse to modify a copy with existing
    optimizer or reevaluation checkpoints. Return material and element counts.
    """
    root = Path(root)
    if any((root / "results").rglob("checkpoint.json")) or any(
        (root / "reevaluation_iv_4_5_2").rglob("manifest.json")
    ):
        raise RuntimeError("Generate inputs in a fresh copy without existing campaign outputs.")
    definition = json.loads((root / "config/public_fuel_definition.json").read_text())
    openmc.reset_auto_ids()
    materials = openmc.Materials()
    for index, (name, recipe) in enumerate(definition["structural_materials"].items(), 1):
        materials.append(material_recipe(name, recipe, index))
    inventory, assignments, ordinary_ids = {}, {}, []
    regular_volume = math.pi * (1.82245**2 - 0.3175**2) * 38.1
    follower_volume = math.pi * (1.67005**2 - 0.3175**2) * 38.1
    for offset in range(111):
        serial = str(900001 + offset)
        name = "fuel_" + serial
        # Fictional equal groups: 36 regular pins per type and one follower per type.
        kind = ("M8", "M9", "M10")[offset // 36 if offset < 108 else offset - 108]
        recipe = definition["fuel_types"][kind]
        volume = regular_volume if offset < 108 else follower_volume
        mass_u235 = (
            recipe["density_g_cm3"]
            * volume
            * recipe["mass_weights"]["U235"]
            / sum(recipe["mass_weights"].values())
        )
        assignments[name] = dict(fuel_type=kind, role="regular" if offset < 108 else "ffcr")
        for axial in range(1, 16):
            material = material_recipe(f"{name}_{axial:02d}", recipe, 1000 + offset * 15 + axial)
            material.volume = volume / 15
            materials.append(material)
        if offset < 108:
            ordinary_ids.append(name)
            inventory[serial] = dict(
                fuel_type="fuel",
                public_fuel_type=kind,
                u_235_fresh=mass_u235,
                u_235_current=mass_u235,
            )
    materials.cross_sections = "nuclear_data/cross_sections.xml"
    material_path = root / "config/materials.xml"
    materials.export_to_xml(material_path)
    # OpenMC resolves this setter to an absolute path. Keep the distributed
    # placeholder relative; local setup replaces it before any transport run.
    text, count = re.subn(
        r"(<cross_sections>).*?(</cross_sections>)",
        r"\g<1>../nuclear_data/cross_sections.xml\g<2>",
        material_path.read_text(),
    )
    if count != 1:
        raise ValueError("Expected exactly one cross-section index in exported materials.")
    material_path.write_text(text)
    write_json(root / "config/fuel_inventory.json", dict(units="g", fuels=inventory))
    write_json(root / "config/public_fuel_assignments.json", assignments)
    for filename, campaign in (
        ("config.ini", "TRIGA2000_PUBLIC_FRESH_DEMO"),
        ("smoke_generation0.ini", "TRIGA2000_PUBLIC_FRESH_SMOKE"),
    ):
        path = root / "config" / filename
        text = re.sub(
            r"^campaign_id = .*$", "campaign_id = " + campaign, path.read_text(), flags=re.M
        )
        text = re.sub(
            r"^fuel_pin_ids = .*$", "fuel_pin_ids = " + ", ".join(ordinary_ids), text, flags=re.M
        )
        for slot, offset in zip(("D2", "D13", "D16"), (109, 110, 111)):
            text = re.sub(
                rf"^{slot} = ffcr, .*$", f"{slot} = ffcr, fuel_{900000 + offset}", text, flags=re.M
            )
        # Cached research responses must never seed this different material context.
        text = re.sub(
            r"compatible_openmc_version_pairs =\n(?:    .*\n)*",
            "compatible_openmc_version_pairs =\n",
            text,
        )
        path.write_text(text)
    sys.path.insert(0, str(root / "src"))
    from modules.config_loader import ConfigLoader
    from modules.genome_sorter import GenomeSorter
    from modules.openmc_model_builder import OpenMCModelBuilder

    config = ConfigLoader(root / "config/config.ini").get_params()
    # Public seed creates a new ordering; there is no mapping from confidential IDs.
    random.Random(definition["reference_seed"]).shuffle(ordinary_ids)
    sorter = GenomeSorter(OpenMCModelBuilder.SLOT_ORDER, config["chromosome"]["excluded_genes"])
    genome = list(sorter.construct_complete_genome([("fuel", name) for name in ordinary_ids]))
    write_json(root / "tests/fixtures/geometry/reference_genome.json", genome)
    return dict(materials=len(materials), regular_pins=108, followers=3)


def main():
    """Parse the optional root, regenerate public inputs, and print counts; return None."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(generate(parser.parse_args().root), indent=2))


if __name__ == "__main__":
    main()
