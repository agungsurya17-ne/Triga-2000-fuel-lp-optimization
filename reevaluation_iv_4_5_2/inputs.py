"""Select a public optimizer front and its independently shuffled reference loading."""

import csv
import sys

from common import ROOT, BASELINE_ID

sys.path.insert(0, str(ROOT / "src"))
from modules.public_inputs import DEFAULT_CHECKPOINT, reference_loading, select_front

CHECKPOINT = DEFAULT_CHECKPOINT


def load_cases(checkpoint_path=CHECKPOINT):
    """Return nominal cases and selection audit for the finished public ``checkpoint_path``."""
    ids, genomes, checkpoint, audit = select_front(checkpoint_path)
    cases = []
    for candidate_id in ids:
        response = checkpoint["individuals"][candidate_id]["results"]
        cases.append(
            dict(
                candidate_id=candidate_id,
                genome=genomes[candidate_id],
                historical_keff=response["keff"]["mean"],
                historical_keff_std_dev=response["keff"]["std_dev"],
                historical_ppf_regular=response["ppf_3d"],
                historical_radial_regular=response["radial_ppf"],
                historical_axial_regular=response["axial_ppf"],
            )
        )
    cases.append(
        dict(
            candidate_id=BASELINE_ID,
            genome=[list(g) for g in reference_loading()],
            historical_keff=None,
            historical_keff_std_dev=None,
            historical_ppf_regular=None,
            historical_radial_regular=None,
            historical_axial_regular=None,
        )
    )
    audit["reference"] = dict(
        source="tests/fixtures/geometry/reference_genome.json",
        purpose="Public seeded loading; unrelated to the confidential reactor loading.",
        assignment_count=111,
        segment_count=1665,
    )
    return cases, audit


def reference_segment_geometry():
    """Return public fixture fuel bounds/volumes in cm and cm³, keyed by slot/layer."""
    result = {}
    with (ROOT / "tests/fixtures/geometry/cell_mapping_verified.csv").open() as stream:
        for row in csv.DictReader(stream):
            result[(row["slot"], int(row["axial"]))] = dict(
                fuel_id=row["material_name"].rsplit("_", 1)[0],
                z_low_cm=float(row["z_low_cm"]),
                z_high_cm=float(row["z_high_cm"]),
                volume_cm3=float(row["openmc_volume_cm3"]),
            )
    if len(result) != 1665:
        raise ValueError("Public geometry fixture must contain 1665 unique segments.")
    return result
