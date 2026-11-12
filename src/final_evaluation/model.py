"""Independent rod motion with audited candidate materials (coordinates in cm)."""

import math
import numpy as np
import openmc
from modules.openmc_model_builder import OpenMCModelBuilder

RODS = ("D2", "D5", "D9", "D13", "D16")
FFCR = ("D2", "D13", "D16")
MOVING_OUT = {
    37: -0.127,
    38: 0.42863,
    39: 1.143,
    40: 17.653,
    41: 55.753,
    42: 93.853,
    43: 104.80675,
    44: 107.02925,
    45: 110.09757,
    **{46 + i: 20.193 + 2.54 * i for i in range(14)},
}
PREFIX = {
    "D2": "surf_ffcr_d2",
    "D5": "surf_control_rod_d5",
    "D9": "surf_control_rod_d9",
    "D13": "surf_ffcr_d13",
    "D16": "surf_ffcr_d16",
}


def rod_cases():
    """Return the five ordered one-OUT/four-IN state dictionaries."""
    return {
        f"STUCK-{stuck}": {rod: "OUT" if rod == stuck else "IN" for rod in RODS} for stuck in RODS
    }


def replace_surfaces(region, replacements):
    """Return region with selected surface IDs replaced; retain halfspace signs."""
    if isinstance(region, openmc.Halfspace):
        return openmc.Halfspace(replacements.get(region.surface.id, region.surface), region.side)
    if isinstance(region, openmc.Complement):
        return ~replace_surfaces(region.node, replacements)
    return type(region)([replace_surfaces(node, replacements) for node in region])


class FinalModelBuilder(OpenMCModelBuilder):
    """Extend canonical geometry v2 with a 111-element tally and independent rod states.

    The SDM extension revision is independent of the inherited nominal geometry
    revision; MODEL_VERSION includes both identities for evaluation provenance."""

    MODEL_VERSION = f"triga2000_final_sdm/v1+{OpenMCModelBuilder.MODEL_VERSION}"

    def __init__(self, genome, material_file, rod_states=None):
        """Store complete genome/material XML and optional states; return None."""
        super().__init__(genome, material_file, "final_fuel_segment_q", "fission-q-recoverable")
        self.rod_states = {rod: "OUT" for rod in RODS}
        if rod_states is not None:
            if set(rod_states) != set(RODS) or set(rod_states.values()) - {"IN", "OUT"}:
                raise ValueError("Specify exactly five rods, each IN or OUT.")
            self.rod_states.update(rod_states)
        self.rod_components = {}

    def moving_z(self, rod, surface_id):
        """Return MCNP moving-plane z in cm for rod and surface_id (37..59)."""
        z = MOVING_OUT[surface_id] - (38.1 if self.rod_states[rod] == "IN" else 0.0)
        # Reproduce the actual IN input, including the 1e-5 cm rounding residue.
        if surface_id == 38 and self.rod_states[rod] == "IN":
            z -= 0.00001
        return z

    def _moving_universe(self, universe, rod, components):
        """Move rod planes in universe and rebuild water complement; return universe."""
        old_cells = list(universe.cells.values())
        moved = {}
        for cell in old_cells:
            for surface in cell.region.get_surfaces().values():
                if isinstance(surface, openmc.ZPlane):
                    matches = [
                        key
                        for key, z in MOVING_OUT.items()
                        if math.isclose(surface.z0, z, abs_tol=1e-8)
                    ]
                    if matches and surface.id not in moved:
                        moved[surface.id] = openmc.ZPlane(z0=self.moving_z(rod, matches[0]))
        planes = {key: openmc.ZPlane(z0=self.moving_z(rod, key)) for key in (37, 38, 44, 45)}
        fixed_top = openmc.ZPlane(z0=698.96)
        # Preserve the canonical builder cell ordering and audited material fills.
        offset = 16 if rod in FFCR else 0
        indices = {
            "bottom_fit": 1,
            "bottom_void": 3,
            "b4c": 4 + offset,
            "top_void": 5 + offset,
            "top_fit": 6 + offset,
            "clad": 7 + offset,
            "extension": 8 + offset,
            "top_grid": 10 + offset,
            "bottom_grid": 11 + offset,
        }
        kept = [cell for index, cell in enumerate(old_cells) if index not in (0, 2, 9 + offset)]
        for cell in kept:
            cell.region = replace_surfaces(cell.region, moved)
        old_cells[indices["bottom_fit"]].region &= +planes[37]
        old_cells[indices["extension"]].region &= -fixed_top
        occupied = openmc.Union([cell.region for cell in kept])
        water = openmc.Cell(
            name=f"{rod} complementary water",
            fill=self._get_material("water"),
            region=~occupied,
        )
        universe.clear_cells()
        universe.add_cells([*kept, water])
        components.update(
            {
                "universe": universe,
                "moving_planes": planes,
                "b4c_cell": old_cells[indices["b4c"]],
                "component_cells": {key: old_cells[index] for key, index in indices.items()},
            }
        )
        self.rod_components[rod] = components
        return universe

    def _build_ffcr(self, fuel_name, slot_name):
        """Build FFCR at slot_name with its original 15 materials; return pair."""
        universe, components = super()._build_ffcr(fuel_name, slot_name)
        self._moving_universe(universe, slot_name, components)
        return universe, components

    def _build_control_rod(self, slot_name):
        """Build nonfollower rod at slot_name in requested state; return universe."""
        universe = super()._build_control_rod(slot_name)
        return self._moving_universe(universe, slot_name, {})

    def _build_outer_core(self):
        """Move per-rod lower water boundaries; return legacy outer cells and metadata."""
        cells, components = super()._build_outer_core()
        surfaces = components["surfaces"]
        shared = surfaces["surf_ffcr_lower_end_fitting"]
        for rod in RODS:
            radial = surfaces[PREFIX[rod] + "_clad"]
            plane = openmc.ZPlane(z0=self.moving_z(rod, 37))
            surfaces[f"{rod}_lower_end"] = plane
            for cell in cells:
                used = cell.region.get_surfaces()
                # The central lower-water cell uses the end-fitting cylinder only.
                end = surfaces[PREFIX[rod] + "_end_fitting"]
                if shared.id in used and (radial.id in used or end.id in used):
                    cell.region = replace_surfaces(cell.region, {shared.id: plane})
        return cells, components

    def _build_core_lattice_region(self, outer_core_components):
        """Return parent region with separate lower extension bounds for each rod."""
        surfaces = outer_core_components["surfaces"]
        shim = openmc.Union([cell.region for cell in outer_core_components["cell_groups"]["shim"]])
        lower = (
            +surfaces["surf_core_bottom_grid"]
            & -surfaces["surf_core_wall_top"]
            & -surfaces["surf_inner_core_wall"]
            & ~shim
        )
        upper = (
            +surfaces["surf_core_wall_top"]
            & -surfaces["surf_extended_pool_water_surface"]
            & -surfaces["surf_upper_part_rotary_rack_1"]
        )
        extensions = [
            +surfaces[f"{rod}_lower_end"]
            & -surfaces["surf_core_bottom_grid"]
            & -surfaces[PREFIX[rod] + "_clad"]
            for rod in RODS
        ]
        return openmc.Union([lower, upper, *extensions])

    def build_final(
        self,
        *,
        particles=100000,
        total_batches=500,
        inactive=50,
        seed=20261020,
        shutdown=False,
    ):
        """Return the fixed-budget OpenMC model and ordered fuel-segment records.

        ``particles`` is histories per generation; ``total_batches`` includes
        ``inactive`` discarded batches. ``seed`` initializes transport. With
        ``shutdown=True`` the model omits the nominal PPF tally; otherwise every rod
        must be OUT and the tally includes regular fuel and FFCR followers."""
        if not shutdown and any(state != "OUT" for state in self.rod_states.values()):
            raise ValueError("Nominal PPF requires every rod OUT; shutdown models omit PPF.")
        geometry, slots, lattice, core, outer = self._build_geometry()
        from .geometry_checks import centers_from_lattice

        centers = centers_from_lattice(dict(lattice=lattice, slots=slots))
        center_corrections = {}
        for rod, (x, y) in centers.items():
            for key, surface in outer["surfaces"].items():
                if key.startswith(PREFIX[rod] + "_") and hasattr(surface, "x0"):
                    center_corrections[key] = (
                        float(x - surface.x0),
                        float(y - surface.y0),
                    )
                    surface.x0, surface.y0 = x, y
        records, cells = [], []
        for slot, component in slots.items():
            if component["slot_type"] not in ("fuel", "ffcr"):
                continue
            for axial, cell in enumerate(component["fuel_cells"], 1):
                bounds = [
                    (half.surface.z0, half.side)
                    for half in cell.region
                    if isinstance(half, openmc.Halfspace)
                    and isinstance(half.surface, openmc.ZPlane)
                ]
                low = max(z for z, side in bounds if side == "+")
                high = min(z for z, side in bounds if side == "-")
                radii = [
                    (half.surface.r, half.side)
                    for half in cell.region
                    if isinstance(half, openmc.Halfspace)
                    and isinstance(half.surface, openmc.ZCylinder)
                ]
                inner = max(r for r, side in radii if side == "+")
                radius = min(r for r, side in radii if side == "-")
                volume = math.pi * (radius**2 - inner**2) * (high - low)
                cell.volume = volume
                records.append(
                    dict(
                        cell_id=cell.id,
                        slot=slot,
                        kind=component["slot_type"],
                        axial=axial,
                        volume_cm3=volume,
                        z_low=low,
                        z_high=high,
                    )
                )
                cells.append(cell)
        if len(records) != 1665:
            raise ValueError("Final tally requires exactly 111 x 15 fuel segments.")
        settings = self.build_settings(
            particles,
            inactive,
            total_batches - inactive,
            keff_trigger_enabled=False,
            tally_trigger_enabled=False,
            openmc_seed=seed,
            trigger_max_batches=total_batches,
        )
        settings.generations_per_batch = 1
        settings.trigger_active = False
        # Cover ordinary fuel and every translated follower; reject nonfissionable samples.
        lower_z = min(record["z_low"] for record in records)
        upper_z = max(record["z_high"] for record in records)
        settings.source = openmc.IndependentSource(
            space=openmc.stats.Box((-28.0, -28.0, lower_z), (28.0, 28.0, upper_z)),
            constraints={"fissionable": True},
        )
        entropy = openmc.RegularMesh()
        entropy.lower_left = (-28.0, -28.0, lower_z - 1e-4)
        entropy.upper_right = (28.0, 28.0, upper_z + 1e-4)
        entropy.dimension = (12, 12, 20)
        settings.entropy_mesh = entropy
        settings.statepoint = {"batches": [total_batches]}
        settings.sourcepoint = {"batches": [total_batches], "write": True}
        tallies = openmc.Tallies()
        if not shutdown:
            tally = openmc.Tally(name=self.tally_name)
            tally.filters = [openmc.CellFilter(cells)]
            tally.scores = [self.tally_score]
            tallies.append(tally)
        model = openmc.Model(geometry, self.materials, settings, tallies)
        return model, dict(
            records=records,
            slots=slots,
            lattice=lattice,
            core=core,
            outer=outer,
            center_corrections_cm=center_corrections,
        )
