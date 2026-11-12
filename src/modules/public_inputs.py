"""Load public reference inputs and select a completed demonstration Pareto front."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from modules.config_loader import ConfigLoader
from modules.genome_sorter import GenomeSorter
from modules.keff_constraint import KeffConstraint
from modules.openmc_model_builder import OpenMCModelBuilder


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "results/TRIGA2000_PUBLIC_FRESH_DEMO/checkpoint.json"


def read_json(path):
    """Decode JSON from ``path`` and return its value."""
    return json.loads(Path(path).read_text())


def file_digest(path):
    """Return the SHA-256 digest of the existing file at ``path``."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_contract(root=ROOT):
    """Return the configured sorter and mutable allele multiset under ``root``."""
    config = ConfigLoader(Path(root) / "config/config.ini").get_params()["chromosome"]
    sorter = GenomeSorter(OpenMCModelBuilder.SLOT_ORDER, config["excluded_genes"])
    expected = Counter(("fuel", name) for name in config["fuel_pin_ids"])
    return sorter, expected


def reference_loading(root=ROOT):
    """Return the validated, fictional 127-position reference loading under ``root``."""
    root = Path(root)
    genome = [tuple(g) for g in read_json(root / "tests/fixtures/geometry/reference_genome.json")]
    sorter, expected = public_contract(root)
    mutable = sorter.extract_included_genome(genome)
    if Counter(map(tuple, mutable)) != expected:
        raise ValueError("Public reference does not match the configured fresh inventory.")
    return genome


def select_front(checkpoint_path=DEFAULT_CHECKPOINT, root=ROOT):
    """Return final-front IDs, complete genomes, checkpoint and audit for public inputs.

    ``checkpoint_path`` must describe a finished optimizer run using the current
    material and inventory bytes. Validate permutations, fixed genes and the
    constrained Pareto front; never reuse confidential or incompatible results.
    """
    root, checkpoint_path = Path(root), Path(checkpoint_path)
    checkpoint = read_json(checkpoint_path)
    for name in ("materials.xml", "fuel_inventory.json"):
        recorded = checkpoint["input_provenance"]["inputs"][name]["sha256"]
        if recorded != file_digest(root / "config" / name):
            raise ValueError(f"Checkpoint input differs from public {name}.")
    if checkpoint["progress"]["phase"] != "finished" or not checkpoint["generations"]:
        raise ValueError("Select a finished optimization campaign.")
    generation = checkpoint["generations"][-1]
    survivors = generation["survivor_ids"]
    ids = generation["summary"]["replacement"]["fronts"][0]
    if (
        generation["status"] != "completed"
        or not survivors
        or not ids
        or len(set(survivors)) != len(survivors)
        or len(set(ids)) != len(ids)
        or not set(ids).issubset(survivors)
    ):
        raise ValueError("Require a completed final population and a unique survivor front.")
    sorter, expected = public_contract(root)
    contract = checkpoint["genome_contract"]
    if (
        contract["slot_order"] != list(sorter.slot_order)
        or {s: tuple(g) for s, g in contract["excluded_genes"].items()} != sorter.excluded_genes
        or Counter(map(tuple, contract["mutable_alleles"])) != expected
    ):
        raise ValueError("Checkpoint genome contract differs from the public configuration.")
    constraint = KeffConstraint.from_identity(checkpoint["constraint_contract"])
    genomes, points = {}, {}
    for candidate_id in survivors:
        if not isinstance(candidate_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", candidate_id
        ):
            raise ValueError("Candidate ID must be safe for a job directory name.")
        item = checkpoint["individuals"][candidate_id]
        if item["status"] != "completed" or Counter(map(tuple, item["genome"])) != expected:
            raise ValueError(f"Invalid completed public individual: {candidate_id}.")
        result = item["results"]
        constraint.validate_results(result)
        mean, sigma, ppf = result["keff"]["mean"], result["keff"]["std_dev"], result["ppf_3d"]
        if (
            not all(math.isfinite(v) for v in (mean, sigma, ppf))
            or mean <= 0
            or sigma < 0
            or ppf <= 0
        ):
            raise ValueError("Require finite positive keff/PPF and nonnegative keff uncertainty.")
        objectives = tuple(result["objectives"])
        if objectives != (-result["keff"]["mean"], result["ppf_3d"]):
            raise ValueError("Stored objectives differ from raw response values.")
        points[candidate_id] = (constraint.evaluate(-objectives[0]), objectives)
        genomes[candidate_id] = [list(g) for g in sorter.construct_complete_genome(item["genome"])]

    def dominates(first, second):
        """Return whether constraint/objective pair ``first`` dominates ``second``."""
        a, x = first
        b, y = second
        if a["feasible"] != b["feasible"]:
            return a["feasible"]
        if not a["feasible"]:
            return a["keff_violation"] < b["keff_violation"]
        return all(i <= j for i, j in zip(x, y)) and any(i < j for i, j in zip(x, y))

    first_front = {
        key
        for key, point in points.items()
        if not any(dominates(other, point) for name, other in points.items() if name != key)
    }
    if first_front != set(ids):
        raise ValueError("Saved front differs from the reconstructed constrained first front.")
    audit = dict(
        path=str(checkpoint_path.resolve()),
        sha256=file_digest(checkpoint_path),
        generation=generation["generation"],
        population_size=len(survivors),
        evolution_cycles=len(checkpoint["generations"]),
        actual_front_count=len(ids),
        cohort_ids=ids,
        independent_front_verified=True,
        dataset="public-fresh-fuel/v1",
    )
    return ids, {key: genomes[key] for key in ids}, checkpoint, audit
