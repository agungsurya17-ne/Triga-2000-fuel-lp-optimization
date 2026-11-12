"""Canonical geometry v2 with an explicit 111-element tracklength tally."""

import math
import sys
import warnings
import numpy as np
from common import ROOT, NOMINAL, TALLY_NAME, BASELINE_ID

sys.path.insert(0, str(ROOT / "src"))
import openmc
from modules.openmc_model_builder import OpenMCModelBuilder


def validate_records(records, genome):
    """Check all 1665 unique geometry-derived records against complete genome; return totals."""
    if (
        len(records) != 1665
        or len({r["cell_id"] for r in records}) != 1665
        or len({(r["slot"], r["axial"]) for r in records}) != 1665
    ):
        raise ValueError("Require 1665 unique cells and slot/layer pairs.")
    for index, (slot, gene) in enumerate(zip(OpenMCModelBuilder.SLOT_ORDER, genome, strict=True)):
        selected = [r for r in records if r["slot"] == slot]
        if gene[0] not in ("fuel", "ffcr"):
            if selected:
                raise ValueError("Nonfuel cell in tally domain.")
            continue
        if {r["axial"] for r in selected} != set(range(1, 16)):
            raise ValueError(f"Missing layers at {slot}")
        for row in selected:
            if (
                row["fuel_id"] != gene[1]
                or row["genome_index"] != index
                or row["category"] != ("FFCR" if gene[0] == "ffcr" else "REGULAR")
            ):
                raise ValueError("Tally identity does not match genome.")
            if (
                not math.isfinite(row["volume_cm3"])
                or row["volume_cm3"] <= 0
                or not math.isclose(row["z_high_cm"] - row["z_low_cm"], 2.54, abs_tol=1e-9)
            ):
                raise ValueError("Invalid fuel-only volume or axial layer height.")
            if not math.isclose(row["z_low_cm"], 17.653 + (row["axial"] - 1) * 2.54, abs_tol=1e-9):
                raise ValueError("Axial layer does not coincide between regular fuel and follower.")
    regular = [r for r in records if r["category"] == "REGULAR"]
    followers = [r for r in records if r["category"] == "FFCR"]
    if (
        len(regular) != 1620
        or len(followers) != 45
        or {r["slot"] for r in followers} != {"D2", "D13", "D16"}
    ):
        raise ValueError("Invalid regular/follower domain.")
    if not np.allclose(
        [r["volume_cm3"] for r in regular], regular[0]["volume_cm3"], rtol=0, atol=1e-10
    ):
        raise ValueError("Historical regular PPF assumes equal segment volumes.")
    return dict(
        regular_segments=len(regular),
        ffcr_segments=len(followers),
        regular_segment_volume_cm3=regular[0]["volume_cm3"],
        ffcr_segment_volume_cm3=followers[0]["volume_cm3"],
        regular_total_volume_cm3=math.fsum(r["volume_cm3"] for r in regular),
        ffcr_total_volume_cm3=math.fsum(r["volume_cm3"] for r in followers),
        total_volume_cm3=math.fsum(r["volume_cm3"] for r in records),
    )


def validate_model(model, seed, parameters):
    """Compare ``model`` against the expected ``seed`` and ``parameters``.

    Check run budget, source, disabled triggers, and the 1,665-bin tally contract.
    Return None on success; raise if any required model setting differs."""
    settings = model.settings
    for key, value in parameters.items():
        if getattr(settings, key) != value:
            raise ValueError(f"Unexpected {key}: {getattr(settings, key)}")
    if (
        settings.run_mode != "eigenvalue"
        or settings.energy_mode != "continuous-energy"
        or settings.seed != seed
    ):
        raise ValueError("Require continuous-energy eigenvalue mode with recorded seed.")
    if settings.trigger_active or settings.keff_trigger or settings.entropy_mesh is not None:
        raise ValueError("Triggers and entropy diagnostic must be absent.")
    if len(model.tallies) != 1:
        raise ValueError("Require one nominal tally.")
    tally = model.tallies[0]
    if (
        tally.triggers
        or tally.estimator != "tracklength"
        or tally.scores != ["fission-q-recoverable"]
        or len(tally.filters) != 1
        or not isinstance(tally.filters[0], openmc.CellFilter)
    ):
        raise ValueError("Unexpected tally estimator, score, filter, or triggers.")
    if tally.filters[0].num_bins != 1665:
        raise ValueError("Incorrect tally bin count.")
    source_xml = settings.source[0].to_xml_element()
    box = [float(x) for x in source_xml.find("space/parameters").text.split()]
    if (
        box != [-26.0, -26.0, 17.653, 26.0, 26.0, 55.753]
        or source_xml.find("constraints/fissionable").text != "true"
    ):
        raise ValueError("Source differs from original research model.")


def build_model(case, seed, parameters=None):
    """Return a nominal model, segment records, and audit data for ``case``.

    ``case`` supplies the complete physical genome and candidate ID. ``seed`` sets
    the random stream, and optional ``parameters`` overrides the fixed batch budget.
    Geometry comes from the canonical v2 builder; the tally includes FFCR followers."""
    parameters = dict(NOMINAL if parameters is None else parameters)
    openmc.reset_auto_ids()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", openmc.IDWarning)
        builder = OpenMCModelBuilder(
            case["genome"], ROOT / "config/materials.xml", TALLY_NAME, "fission-q-recoverable"
        )
        geometry, slots, lattice, core, outer = builder._build_geometry()
        settings = builder.build_settings(
            particles=parameters["particles"],
            inactive_batches=parameters["inactive"],
            active_batches=parameters["batches"] - parameters["inactive"],
            keff_trigger_enabled=False,
            tally_trigger_enabled=False,
            openmc_seed=seed,
        )
    settings.generations_per_batch = 1
    settings.energy_mode = "continuous-energy"
    settings.trigger_active = False
    settings.statepoint = {"batches": [parameters["batches"]]}
    settings.sourcepoint = {"write": False}
    settings.output = {"summary": True, "tallies": False}
    centers = {}
    for x in range(-6, 7):
        for a in range(-6, 7):
            if max(abs(x), abs(a), abs(x + a)) <= 6:
                universe = lattice.get_universe((x, a))
                local = lattice.get_local_coordinates((0.0, 0.0, 0.0), (x, a))
                centers[universe.id] = (-float(local[0]), -float(local[1]))
    records, cells = [], []
    for genome_index, slot in enumerate(builder.SLOT_ORDER):
        component = slots[slot]
        if component["slot_type"] not in ("fuel", "ffcr"):
            continue
        center_x, center_y = centers[component["universe"].id]
        for axial, cell in enumerate(component["fuel_cells"], 1):
            # Fuel meat is the intersection of two z planes and two concentric cylinders.
            if not isinstance(cell.region, openmc.Intersection) or len(cell.region) != 4:
                raise ValueError("Unexpected fuel CSG region; analytic volume is not justified.")
            z_bounds, radii = {}, {}
            for half in cell.region:
                if not isinstance(half, openmc.Halfspace):
                    raise ValueError("Unexpected compound fuel halfspace.")
                if isinstance(half.surface, openmc.ZPlane):
                    z_bounds[half.side] = half.surface.z0
                elif isinstance(half.surface, openmc.ZCylinder):
                    if half.surface.x0 != 0 or half.surface.y0 != 0:
                        raise ValueError("Expected local concentric cylinders.")
                    radii[half.side] = half.surface.r
                else:
                    raise ValueError("Unsupported fuel-volume surface.")
            low, high = z_bounds["+"], z_bounds["-"]
            inner, radius = radii["+"], radii["-"]
            volume = math.pi * (radius**2 - inner**2) * (high - low)
            if (
                radius <= inner
                or high <= low
                or cell.fill.name != f"{component['gene_id']}_{axial:02d}"
            ):
                raise ValueError("Fuel region/material layer mismatch.")
            cell.volume = volume
            probe = (center_x + (inner + radius) / 2, center_y, (low + high) / 2)
            found = [item.id for item in geometry.find(probe) if isinstance(item, openmc.Cell)]
            if cell.id not in found:
                raise ValueError(
                    f"Fuel metadata is not reachable through geometry at {slot}/{axial}"
                )
            records.append(
                dict(
                    cell_id=cell.id,
                    slot=slot,
                    axial=axial,
                    fuel_id=component["gene_id"],
                    genome_index=genome_index,
                    category="FFCR" if component["slot_type"] == "ffcr" else "REGULAR",
                    x_cm=center_x,
                    y_cm=center_y,
                    z_low_cm=low,
                    z_high_cm=high,
                    inner_radius_cm=inner,
                    outer_radius_cm=radius,
                    volume_cm3=volume,
                    material_id=cell.fill.id,
                    material_name=cell.fill.name,
                )
            )
            cells.append(cell)
    tally = openmc.Tally(name=TALLY_NAME)
    tally.filters = [openmc.CellFilter(cells)]
    tally.scores = ["fission-q-recoverable"]
    tally.estimator = "tracklength"
    model = openmc.Model(geometry, builder.materials, settings, openmc.Tallies([tally]))
    audit = validate_records(records, case["genome"])
    if case["candidate_id"] == BASELINE_ID:
        from inputs import reference_segment_geometry

        reference = reference_segment_geometry()
        for record in records:
            original = reference[(record["slot"], record["axial"])]
            if original["fuel_id"] != record["fuel_id"] or any(
                not math.isclose(record[k], original[k], rel_tol=0, abs_tol=1e-8)
                for k in ["z_low_cm", "z_high_cm", "volume_cm3"]
            ):
                raise ValueError(
                    "Reference fuel geometry differs from the public regression fixture."
                )
        audit["reference_geometry_fixture_verified"] = True
    validate_model(model, seed, parameters)
    return model, records, audit


def extract_statepoint(path, records, seed, parameters):
    """Return combined keff statistics and segment Q means/errors from ``path``.

    Require a complete run with the expected ``seed`` and transport ``parameters``.
    Match tally identities to ``records`` before attaching responses to fuel segments."""
    from common import power_metrics
    import h5py

    with h5py.File(path, "r") as handle:
        if int(handle["seed"][()]) != seed:
            raise ValueError("Statepoint seed mismatch.")
        if handle["energy_mode"][()].decode() != "continuous-energy":
            raise ValueError("Statepoint energy mode mismatch.")
        if (
            int(handle["n_batches"][()]) != parameters["batches"]
            or int(handle["generations_per_batch"][()]) != 1
        ):
            raise ValueError("Statepoint budget mismatch.")
    with openmc.StatePoint(path, autolink=False) as sp:
        if (
            sp.current_batch != parameters["batches"]
            or sp.n_inactive != parameters["inactive"]
            or sp.n_particles != parameters["particles"]
            or sp.run_mode != "eigenvalue"
        ):
            raise ValueError(
                "Partial or incompatible statepoint cannot complete nominal evaluation."
            )
        if sp.n_realizations != parameters["batches"] - parameters["inactive"]:
            raise ValueError("Unexpected number of active realizations.")
        tally = sp.get_tally(name=TALLY_NAME)
        if (
            tally.estimator != "tracklength"
            or tally.scores != ["fission-q-recoverable"]
            or len(tally.filters) != 1
            or tally.nuclides != ["total"]
        ):
            raise ValueError("Statepoint tally contract mismatch.")
        if tally.num_realizations != parameters["batches"] - parameters["inactive"]:
            raise ValueError("Incomplete tally realizations.")
        ids = tally.find_filter(openmc.CellFilter).bins.reshape(-1).tolist()
        by_id = {r["cell_id"]: r for r in records}
        if len(ids) != 1665 or len(set(ids)) != 1665 or set(ids) != set(by_id):
            raise ValueError("Statepoint cells do not match saved tally identities.")
        ordered = [by_id[cell_id] for cell_id in ids]
        scores, sigma = tally.mean.reshape(-1), tally.std_dev.reshape(-1)
        if len(scores) != 1665 or not np.isfinite(sigma).all() or (sigma < 0).any():
            raise ValueError("Invalid tally shape/uncertainty.")
        result = dict(
            keff=float(sp.keff.n),
            keff_std_dev=float(sp.keff.s),
            current_batch=int(sp.current_batch),
            active_realizations=int(sp.n_realizations),
            openmc_version=[int(value) for value in sp.version],
            all_fuel=power_metrics(scores, ordered),
        )
        if (
            not math.isfinite(result["keff"])
            or result["keff"] <= 0
            or not math.isfinite(result["keff_std_dev"])
            or result["keff_std_dev"] < 0
        ):
            raise ValueError("Invalid combined estimator.")
        regular = np.array([r["category"] == "REGULAR" for r in ordered])
        result["regular"] = power_metrics(
            scores[regular], [r for r in ordered if r["category"] == "REGULAR"]
        )
        segments = []
        for row, q, sd in zip(ordered, scores, sigma, strict=True):
            segments.append(
                dict(
                    row,
                    q_eV_per_source=float(q),
                    q_std_dev_eV_per_source=float(sd),
                    density_eV_per_source_cm3=float(q / row["volume_cm3"]),
                    density_std_dev_eV_per_source_cm3=float(sd / row["volume_cm3"]),
                    relative_density=float(
                        q / row["volume_cm3"] / result["all_fuel"]["mean_density_eV_per_source_cm3"]
                    ),
                    tally_relative_std_dev=float(sd / q) if q > 0 else None,
                )
            )
        result["segments"] = segments
        result["zero_mean_bins"] = int(np.sum(scores == 0))
        return result


def common_model_signature(xml_path, records):
    """Hash exported XML excluding seed and fuel assignments; return common nominal physics hash."""
    import hashlib
    import re
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path).getroot()
    fuel_cells = {str(r["cell_id"]): f"FUEL_AT_{r['slot']}_{r['axial']:02d}" for r in records}
    geometry = tree.find("geometry")
    for cell in geometry.findall("cell"):
        if cell.get("id") in fuel_cells:
            cell.set("material", fuel_cells[cell.get("id")])
        if "name" in cell.attrib:
            cell.set("name", re.sub(r"fuel_\d+", "fuel_LOADING", cell.get("name")))
    for universe in geometry.findall("universe"):
        if "name" in universe.attrib:
            universe.set("name", re.sub(r"fuel_\d+", "fuel_LOADING", universe.get("name")))
    seed = tree.find("settings/seed")
    tree.find("settings").remove(seed)
    return hashlib.sha256(ET.tostring(tree)).hexdigest()
