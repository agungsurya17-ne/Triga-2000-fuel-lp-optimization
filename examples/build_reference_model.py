"""Export the reference-loading geometry-v2 model without running neutron transport."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from modules.openmc_model_builder import OpenMCModelBuilder


def build_reference(output):
    """Write a fresh model and segment map to ``output``; return model identity and counts."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    genome = json.loads((ROOT / "tests/fixtures/geometry/reference_genome.json").read_text())
    builder = OpenMCModelBuilder(
        genome,
        ROOT / "config/materials.xml",
        "fuel_segment_fission_q_recoverable",
        "fission-q-recoverable",
    )
    model, metadata = builder.build_model(
        particles=100000,
        inactive_batches=45,
        active_batches=105,
        keff_trigger_enabled=False,
        tally_trigger_enabled=False,
        openmc_seed=12345,
    )
    model.export_to_model_xml(output / "model.xml")
    (output / "tally_records.json").write_text(
        json.dumps(metadata["tally_records"], indent=2) + "\n"
    )
    result = dict(
        model_version=builder.MODEL_VERSION,
        physical_positions=len(genome),
        regular_fuel_segments=len(metadata["tally_records"]),
        transport_launched=False,
        purpose="Public fresh-fuel geometry export; no transport results.",
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    """Parse an optional fresh output directory and print the export summary; return None."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/reference_model")
    arguments = parser.parse_args()
    print(json.dumps(build_reference(arguments.output), indent=2))


if __name__ == "__main__":
    main()
