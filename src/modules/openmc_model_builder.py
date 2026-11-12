"""Build the canonical TRIGA 2000 OpenMC model using geometry revision 2.

OpenMCModelBuilder is the single nominal geometry implementation used by the
active runtime. It maps a complete 127-position genome onto the core lattice,
loads materials from XML, and builds transport settings and fuel-segment tallies.
All geometry coordinates and dimensions are in cm. MODEL_VERSION remains part
of the evaluation identity so results from different geometries cannot be
silently interchanged. Archived transport outputs retain their original origin.

MCNP cell and surface labels in comments identify the reference deck entries.
They are traceability labels, not OpenMC-assigned object IDs."""

# This geometry is specific to the TRIGA 2000 research reactor.

from pathlib import Path
import math
from numbers import Integral, Real
import warnings

import numpy as np
import openmc


class OpenMCModelBuilder:
    """Assemble the canonical revision-2 TRIGA 2000 model from 127 core positions."""

    MODEL_VERSION = "triga2000_openmc_model/v2"
    OPENMC_VERSION = str(getattr(openmc, "__version__", "unknown"))

    RING_SIZE = {
        "A": 1,
        "B": 6,
        "C": 12,
        "D": 18,
        "E": 24,
        "F": 30,
        "G": 36,
    }
    RING_ORDER = ("A", "B", "C", "D", "E", "F", "G")
    SLOT_ORDER = tuple(
        f"{ring}{number}"
        for ring, ring_size in (
            ("A", 1),
            ("B", 6),
            ("C", 12),
            ("D", 18),
            ("E", 24),
            ("F", 30),
            ("G", 36),
        )
        for number in range(1, ring_size + 1)
    )
    WORLD_BOUNDARY_CENTER_Z_CM = 215.0
    WORLD_BOUNDARY_RADIUS_CM = 700.0

    material_file = Path("materials.xml")
    keff_stdev_tolerance = 50  # pcm

    def __init__(self, genome, material_file, tally_name, tally_score):
        """Initialize one builder for a complete physical-core genome.

        Parameters
        ----------
        genome : list or tuple
            Complete 127-position loading ordered from A1 through G36.
        material_file : str or pathlib.Path
            Path to ``materials.xml``.
        tally_name, tally_score : str
            Injected identity of the fuel-segment relative-power tally.

        Returns
        -------
        None
            The builder stores the genome, materials, and tally identity."""
        self.genome = genome
        self.material_file = Path(material_file)
        self.tally_name = self._require_non_empty_string("tally_name", tally_name)
        self.tally_score = self._require_non_empty_string("tally_score", tally_score)
        self.materials = self._load_materials(self.material_file)
        self.MAT = {material.name: material for material in self.materials}
        self.MAT_ID = {material.id: material for material in self.materials}
        if len(self.MAT) != len(self.materials):
            raise ValueError("Terdapat nama material duplikat dalam materials.xml.")
        if len(self.MAT_ID) != len(self.materials):
            raise ValueError("Terdapat ID material duplikat dalam materials.xml.")

    @staticmethod
    def _require_non_empty_string(name, value):
        """Validate one string parameter and return its stripped value."""
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"{name} harus berupa string non-kosong.")
        return value.strip()

    def _load_materials(self, material_file):
        """Load the OpenMC materials collection from an XML file.

        Parameters
        ----------
        material_file : str or pathlib.Path
            Location of the materials XML file.

        Returns
        -------
        openmc.Materials
            Materials parsed from the requested file."""
        material_file = Path(material_file)
        if not material_file.is_file():
            raise FileNotFoundError(f"materials.xml tidak ditemukan: {material_file.resolve()}.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", openmc.IDWarning)
            materials = openmc.Materials.from_xml(material_file)
        return materials

    def _get_material(self, name: str) -> openmc.Material:
        """Return the material named ``name`` from the loaded materials XML."""
        try:
            return self.MAT[name]
        except KeyError as exc:
            raise KeyError(f"Material {name!r} tidak ditemukan.") from exc

    def _get_fuel(self, element_number: int, index: int) -> openmc.Material:
        """Return the fuel material for ``element_number`` and its axial segment ``index``."""

        return self._get_material(f"fuel_{element_number}_{index:02d}")

    def _build_ffcr(self, fuel_name, slot_name):
        """Build a fuel-followed control rod for ``fuel_name`` at ``slot_name``.

        Return the slot universe and component metadata, including the 15 axial fuel
        segments below the absorber. Geometry dimensions are in cm."""
        if not isinstance(fuel_name, str) or not fuel_name.strip():
            raise ValueError("fuel_name harus berupa string dan tidak boleh kosong.")

        if not isinstance(slot_name, str) or not slot_name.strip():
            raise ValueError("slot_name harus berupa string dan tidak boleh kosong.")

        fuel_name = fuel_name.strip()
        slot_name = slot_name.strip().upper()
        uzrh_bottom = openmc.ZPlane(z0=17.65300)
        uzrh_top = openmc.ZPlane(z0=55.75300)
        b4c_top = openmc.ZPlane(z0=93.85300)
        ffcr_end_fit_2 = openmc.ZPlane(z0=0.42863)
        ffcr_end_fit_3 = openmc.ZPlane(z0=1.14300)
        ffcr_top_fit_1 = openmc.ZPlane(z0=104.80675)
        ffcr_top_fit_2 = openmc.ZPlane(z0=107.02925)
        ffcr_top_fit_3 = openmc.ZPlane(z0=110.09757)
        r_end_fit = openmc.ZCylinder(r=1.11125)
        r_top_fit = openmc.ZCylinder(r=1.19063)
        r_uzrh = openmc.ZCylinder(r=1.67005)
        r_zr_rod = openmc.ZCylinder(r=0.31750)
        r_clad = openmc.ZCylinder(r=1.72085)
        r_extension_rod = openmc.ZCylinder(r=1.32085)
        tally_plane_ffcr_1 = openmc.ZPlane(z0=20.19300)
        tally_plane_ffcr_2 = openmc.ZPlane(z0=22.73300)
        tally_plane_ffcr_3 = openmc.ZPlane(z0=25.27300)
        tally_plane_ffcr_4 = openmc.ZPlane(z0=27.81300)
        tally_plane_ffcr_5 = openmc.ZPlane(z0=30.35300)
        tally_plane_ffcr_6 = openmc.ZPlane(z0=32.89300)
        tally_plane_ffcr_7 = openmc.ZPlane(z0=35.43300)
        tally_plane_ffcr_8 = openmc.ZPlane(z0=37.97300)
        tally_plane_ffcr_9 = openmc.ZPlane(z0=40.51300)
        tally_plane_ffcr_10 = openmc.ZPlane(z0=43.05300)
        tally_plane_ffcr_11 = openmc.ZPlane(z0=45.59300)
        tally_plane_ffcr_12 = openmc.ZPlane(z0=48.13300)
        tally_plane_ffcr_13 = openmc.ZPlane(z0=50.67300)
        tally_plane_ffcr_14 = openmc.ZPlane(z0=53.21300)
        extended_pool_water_surface = openmc.ZPlane(z0=698.96000)
        pool_water_surface = openmc.ZPlane(z0=548.96000)

        bottom_grid_plate_1 = openmc.ZPlane(z0=0.0)
        bottom_grid_plate_2 = openmc.ZPlane(z0=1.90500)
        top_grid_plate_1 = openmc.ZPlane(z0=66.99250)
        top_grid_plate_2 = openmc.ZPlane(z0=68.58000)
        surf_845 = openmc.ZCylinder(r=1.91135)
        cell_bottom_end_fit = openmc.Cell(
            fill=self._get_material("ss304"),
            region=((-ffcr_end_fit_2 & -r_end_fit) | (+ffcr_end_fit_2 & -ffcr_end_fit_3 & -r_uzrh)),
        )
        cell_bottom_annulus = openmc.Cell(
            fill=self._get_material("water"),
            region=(
                (-bottom_grid_plate_1 & +r_end_fit)
                | (+bottom_grid_plate_1 & -ffcr_end_fit_2 & +r_end_fit & -r_clad)
            ),
        )
        cell_bottom_void = openmc.Cell(fill=None, region=(+ffcr_end_fit_3 & -uzrh_bottom & -r_uzrh))
        cell_zr_rod = openmc.Cell(
            fill=self._get_material("zirconium_rod"), region=(+uzrh_bottom & -uzrh_top & -r_zr_rod)
        )
        cell_uzrh_ffcr_1 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_01"),
            region=(+uzrh_bottom & -tally_plane_ffcr_1 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_ffcr_2 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_02"),
            region=(+tally_plane_ffcr_1 & -tally_plane_ffcr_2 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_ffcr_3 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_03"),
            region=(+tally_plane_ffcr_2 & -tally_plane_ffcr_3 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_ffcr_4 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_04"),
            region=(+tally_plane_ffcr_3 & -tally_plane_ffcr_4 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_ffcr_5 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_05"),
            region=(+tally_plane_ffcr_4 & -tally_plane_ffcr_5 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_ffcr_6 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_06"),
            region=(+tally_plane_ffcr_5 & -tally_plane_ffcr_6 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_ffcr_7 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_07"),
            region=(+tally_plane_ffcr_6 & -tally_plane_ffcr_7 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_ffcr_8 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_08"),
            region=(+tally_plane_ffcr_7 & -tally_plane_ffcr_8 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_ffcr_9 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_09"),
            region=(+tally_plane_ffcr_8 & -tally_plane_ffcr_9 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_ffcr_10 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_10"),
            region=(+tally_plane_ffcr_9 & -tally_plane_ffcr_10 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_ffcr_11 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_11"),
            region=(+tally_plane_ffcr_10 & -tally_plane_ffcr_11 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_ffcr_12 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_12"),
            region=(+tally_plane_ffcr_11 & -tally_plane_ffcr_12 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_ffcr_13 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_13"),
            region=(+tally_plane_ffcr_12 & -tally_plane_ffcr_13 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_ffcr_14 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_14"),
            region=(+tally_plane_ffcr_13 & -tally_plane_ffcr_14 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_ffcr_15 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_15"),
            region=(+tally_plane_ffcr_14 & -uzrh_top & -r_uzrh & +r_zr_rod),
        )
        cell_b4c = openmc.Cell(
            fill=self._get_material("b4c"), region=(+uzrh_top & -b4c_top & -r_uzrh)
        )
        cell_top_void = openmc.Cell(fill=None, region=(+b4c_top & -ffcr_top_fit_1 & -r_uzrh))
        cell_top_end_fit = openmc.Cell(
            fill=self._get_material("ss304"),
            region=(
                (+ffcr_top_fit_1 & -ffcr_top_fit_2 & -r_uzrh)
                | (+ffcr_top_fit_2 & -ffcr_top_fit_3 & -r_top_fit)
            ),
        )
        cell_cladding = openmc.Cell(
            fill=self._get_material("ss304"),
            region=(+ffcr_end_fit_2 & -ffcr_top_fit_2 & -r_clad & +r_uzrh),
        )
        cell_extension_rod = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(+ffcr_top_fit_2 & +r_extension_rod & -r_clad),
        )
        water_region = (
            (+pool_water_surface & -r_extension_rod)
            | (+ffcr_top_fit_2 & -pool_water_surface & +r_top_fit & -r_extension_rod)
            | (+ffcr_top_fit_3 & -pool_water_surface & -r_top_fit)
            | (+bottom_grid_plate_2 & -top_grid_plate_1 & +r_clad)
            | (+top_grid_plate_1 & -top_grid_plate_2 & +r_clad & -surf_845)
            | (+bottom_grid_plate_1 & -bottom_grid_plate_2 & +r_clad & -surf_845)
        )
        cell_water = openmc.Cell(fill=self._get_material("water"), region=(water_region))
        cell_top_grid = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(+top_grid_plate_1 & -top_grid_plate_2 & +surf_845),
        )
        cell_bottom_grid = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(+bottom_grid_plate_1 & -bottom_grid_plate_2 & +surf_845),
        )
        water_full_cell = openmc.Cell(
            fill=self._get_material("water"), region=(+top_grid_plate_2 & +r_clad)
        )
        fuel_cells = [
            cell_uzrh_ffcr_1,
            cell_uzrh_ffcr_2,
            cell_uzrh_ffcr_3,
            cell_uzrh_ffcr_4,
            cell_uzrh_ffcr_5,
            cell_uzrh_ffcr_6,
            cell_uzrh_ffcr_7,
            cell_uzrh_ffcr_8,
            cell_uzrh_ffcr_9,
            cell_uzrh_ffcr_10,
            cell_uzrh_ffcr_11,
            cell_uzrh_ffcr_12,
            cell_uzrh_ffcr_13,
            cell_uzrh_ffcr_14,
            cell_uzrh_ffcr_15,
        ]
        ffcr_cells = [
            water_full_cell,
            cell_bottom_end_fit,
            cell_bottom_annulus,
            cell_bottom_void,
            cell_zr_rod,
            *fuel_cells,
            cell_b4c,
            cell_top_void,
            cell_top_end_fit,
            cell_cladding,
            cell_extension_rod,
            cell_water,
            cell_top_grid,
            cell_bottom_grid,
        ]
        ffcr_universe = openmc.Universe(name=f"FFCR_{slot_name}_{fuel_name}", cells=ffcr_cells)
        # Keep cell references for tally mapping and postprocessing.
        components = {
            "slot_name": slot_name,
            "fuel_name": fuel_name,
            "fuel_cells": fuel_cells,
            "b4c_cell": cell_b4c,
            "zr_rod_cell": cell_zr_rod,
            "cladding_cell": cell_cladding,
        }
        return ffcr_universe, components

    def _build_irradiation_tube(self, slot_name):
        """Build the aluminum irradiation tube and its water-filled interior.

        Return the universe at ``slot_name``; all dimensions are in cm."""
        bottom_grid_plate_1 = openmc.ZPlane(z0=0.0)
        bottom_grid_plate_2 = openmc.ZPlane(z0=1.90500)
        top_grid_plate_1 = openmc.ZPlane(z0=66.99250)
        top_grid_plate_2 = openmc.ZPlane(z0=68.58000)
        irradiation_tube_bottom_wall = openmc.ZPlane(z0=2.00500)
        irradiation_tube_tally_plane_1 = openmc.ZPlane(z0=15.00250)
        irradiation_tube_tally_plane_2 = openmc.ZPlane(z0=28.00000)
        irradiation_tube_tally_plane_3 = openmc.ZPlane(z0=40.99750)
        irradiation_tube_tally_plane_4 = openmc.ZPlane(z0=53.99500)
        r_irradiation_tube_1 = openmc.ZCylinder(r=1.48750)
        r_irradiation_tube_2 = openmc.ZCylinder(r=1.58750)
        extended_pool_water_surface = openmc.ZPlane(z0=698.96000)
        surf_751 = openmc.ZCylinder(r=1.91135)
        surf_752 = openmc.ZCylinder(r=1.58750)
        surf_753 = openmc.ZCylinder(r=0.35687)
        cell_ir_1 = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(+bottom_grid_plate_2 & -irradiation_tube_bottom_wall & -r_irradiation_tube_1),
        )
        cell_ir_2 = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +bottom_grid_plate_2
                & -extended_pool_water_surface
                & -r_irradiation_tube_2
                & +r_irradiation_tube_1
            ),
        )
        cell_air_incore_1 = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                +irradiation_tube_bottom_wall
                & -irradiation_tube_tally_plane_1
                & -r_irradiation_tube_1
            ),
        )
        cell_air_incore_2 = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                +irradiation_tube_tally_plane_1
                & -irradiation_tube_tally_plane_2
                & -r_irradiation_tube_1
            ),
        )
        cell_air_incore_3 = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                +irradiation_tube_tally_plane_2
                & -irradiation_tube_tally_plane_3
                & -r_irradiation_tube_1
            ),
        )
        cell_air_incore_4 = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                +irradiation_tube_tally_plane_3
                & -irradiation_tube_tally_plane_4
                & -r_irradiation_tube_1
            ),
        )
        cell_air_incore_5 = openmc.Cell(
            fill=self._get_material("air"),
            region=(+irradiation_tube_tally_plane_4 & -top_grid_plate_1 & -r_irradiation_tube_1),
        )
        cell_air_excore = openmc.Cell(
            fill=self._get_material("air"),
            region=(+top_grid_plate_1 & -extended_pool_water_surface & -r_irradiation_tube_1),
        )
        cell_water = openmc.Cell(
            fill=self._get_material("water"),
            region=(
                (-top_grid_plate_1 & +bottom_grid_plate_2 & +r_irradiation_tube_2)
                | (+top_grid_plate_1 & -top_grid_plate_2 & +r_irradiation_tube_2 & -surf_751)
                | (+bottom_grid_plate_1 & -bottom_grid_plate_2 & +surf_753 & -surf_752)
            ),
        )
        cell_top_grid = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(+top_grid_plate_1 & -top_grid_plate_2 & +surf_751),
        )
        cell_bottom_grid = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                (+bottom_grid_plate_1 & -bottom_grid_plate_2 & -surf_753)
                | (+bottom_grid_plate_1 & -bottom_grid_plate_2 & +surf_752)
            ),
        )
        water_full_cell = openmc.Cell(
            fill=self._get_material("water"),
            region=(+top_grid_plate_2 & -extended_pool_water_surface & +r_irradiation_tube_2),
        )
        irradiation_tube_universe = openmc.Universe(
            name=f"irradiation_tube_{slot_name}",
            cells=[
                water_full_cell,
                cell_ir_1,
                cell_ir_2,
                cell_air_incore_1,
                cell_air_incore_2,
                cell_air_incore_3,
                cell_air_incore_4,
                cell_air_incore_5,
                cell_air_excore,
                cell_water,
                cell_top_grid,
                cell_bottom_grid,
            ],
        )
        return irradiation_tube_universe

    def _build_fuel_pin(self, fuel_name, slot_name):
        """Build a unique segmented fuel-pin universe at one physical core slot.

        Parameters
        ----------
        fuel_name : str
            Fuel material base name, for example ``fuel_900001``.
        slot_name : str
            Physical core slot, for example ``B1`` or ``C6``.

        Returns
        -------
        tuple
            The slot universe and cell metadata for tally mapping and postprocessing.
            Axial segment coordinates and dimensions are in cm."""
        if not isinstance(fuel_name, str) or not fuel_name.strip():
            raise ValueError("fuel_name harus berupa string dan tidak boleh kosong.")

        if not isinstance(slot_name, str) or not slot_name.strip():
            raise ValueError("slot_name harus berupa string dan tidak boleh kosong.")

        fuel_name = fuel_name.strip()
        slot_name = slot_name.strip().upper()
        extended_pool_water_surface = openmc.ZPlane(z0=698.96000)
        bottom_grid_plate_1 = openmc.ZPlane(z0=0.0)
        bottom_grid_plate_2 = openmc.ZPlane(z0=1.90500)
        top_grid_plate_1 = openmc.ZPlane(z0=66.99250)
        top_grid_plate_2 = openmc.ZPlane(z0=68.58000)
        fuel_end_fit = openmc.ZPlane(z0=6.98500)
        bottom_fuel_graphite_plug = openmc.ZPlane(z0=8.25500)
        uzrh_bottom = openmc.ZPlane(z0=17.65300)
        uzrh_top = openmc.ZPlane(z0=55.75300)
        top_fuel_graphite_plug = openmc.ZPlane(z0=62.35700)
        fuel_top_fit = openmc.ZPlane(z0=63.62700)
        tally_plane_fuel_1 = openmc.ZPlane(z0=20.19300)
        tally_plane_fuel_2 = openmc.ZPlane(z0=22.73300)
        tally_plane_fuel_3 = openmc.ZPlane(z0=25.27300)
        tally_plane_fuel_4 = openmc.ZPlane(z0=27.81300)
        tally_plane_fuel_5 = openmc.ZPlane(z0=30.35300)
        tally_plane_fuel_6 = openmc.ZPlane(z0=32.89300)
        tally_plane_fuel_7 = openmc.ZPlane(z0=35.43300)
        tally_plane_fuel_8 = openmc.ZPlane(z0=37.97300)
        tally_plane_fuel_9 = openmc.ZPlane(z0=40.51300)
        tally_plane_fuel_10 = openmc.ZPlane(z0=43.05300)
        tally_plane_fuel_11 = openmc.ZPlane(z0=45.59300)
        tally_plane_fuel_12 = openmc.ZPlane(z0=48.13300)
        tally_plane_fuel_13 = openmc.ZPlane(z0=50.67300)
        tally_plane_fuel_14 = openmc.ZPlane(z0=53.21300)
        r_end_fit = openmc.ZCylinder(r=0.93663)
        r_zr_rod = openmc.ZCylinder(r=0.31750)
        r_uzrh = openmc.ZCylinder(r=1.82245)
        r_clad = openmc.ZCylinder(r=1.87325)
        surf_757 = openmc.ZCylinder(r=1.91135)  # Same boundary as MCNP surface 754.
        surf_758 = openmc.ZCylinder(r=1.58750)  # Same boundary as MCNP surface 755.
        surf_759 = openmc.ZCylinder(r=0.35687)  # Same boundary as MCNP surface 756.
        cell_bottom_end_fit = openmc.Cell(
            fill=self._get_material("ss304"),
            region=(
                (+bottom_grid_plate_2 & -fuel_end_fit & -r_end_fit)
                | (+fuel_end_fit & -bottom_fuel_graphite_plug & -r_uzrh)
            ),
        )
        cell_graphite_plug_bottom = openmc.Cell(
            fill=self._get_material("graphite"),
            region=(+bottom_fuel_graphite_plug & -uzrh_bottom & -r_uzrh),
        )
        cell_zr_rod = openmc.Cell(
            fill=self._get_material("zirconium_rod"), region=(+uzrh_bottom & -uzrh_top & -r_zr_rod)
        )
        cell_uzrh_fuel_1 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_01"),
            region=(+uzrh_bottom & -tally_plane_fuel_1 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_fuel_2 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_02"),
            region=(+tally_plane_fuel_1 & -tally_plane_fuel_2 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_fuel_3 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_03"),
            region=(+tally_plane_fuel_2 & -tally_plane_fuel_3 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_fuel_4 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_04"),
            region=(+tally_plane_fuel_3 & -tally_plane_fuel_4 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_fuel_5 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_05"),
            region=(+tally_plane_fuel_4 & -tally_plane_fuel_5 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_fuel_6 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_06"),
            region=(+tally_plane_fuel_5 & -tally_plane_fuel_6 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_fuel_7 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_07"),
            region=(+tally_plane_fuel_6 & -tally_plane_fuel_7 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_fuel_8 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_08"),
            region=(+tally_plane_fuel_7 & -tally_plane_fuel_8 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_fuel_9 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_09"),
            region=(+tally_plane_fuel_8 & -tally_plane_fuel_9 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_fuel_10 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_10"),
            region=(+tally_plane_fuel_9 & -tally_plane_fuel_10 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_fuel_11 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_11"),
            region=(+tally_plane_fuel_10 & -tally_plane_fuel_11 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_fuel_12 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_12"),
            region=(+tally_plane_fuel_11 & -tally_plane_fuel_12 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_fuel_13 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_13"),
            region=(+tally_plane_fuel_12 & -tally_plane_fuel_13 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_fuel_14 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_14"),
            region=(+tally_plane_fuel_13 & -tally_plane_fuel_14 & -r_uzrh & +r_zr_rod),
        )
        cell_uzrh_fuel_15 = openmc.Cell(
            fill=self._get_material(f"{fuel_name}_15"),
            region=(+tally_plane_fuel_14 & -uzrh_top & -r_uzrh & +r_zr_rod),
        )
        cell_graphite_plug_top = openmc.Cell(
            fill=self._get_material("graphite"),
            region=(+uzrh_top & -top_fuel_graphite_plug & -r_uzrh),
        )
        cell_top_end_fit = openmc.Cell(
            fill=self._get_material("ss304"),
            region=(
                (+top_fuel_graphite_plug & -fuel_top_fit & -r_uzrh)
                | (+fuel_top_fit & -top_grid_plate_1 & -r_end_fit)
            ),
        )
        cell_cladding = openmc.Cell(
            fill=self._get_material("ss304"),
            region=(+fuel_end_fit & -fuel_top_fit & -r_clad & +r_uzrh),
        )
        cell_water = openmc.Cell(
            fill=self._get_material("water"),
            region=(
                (+bottom_grid_plate_2 & -fuel_end_fit & +r_end_fit)
                | (+fuel_end_fit & -fuel_top_fit & +r_clad)
                | (+fuel_top_fit & -top_grid_plate_1 & +r_end_fit)
                | (+top_grid_plate_1 & -top_grid_plate_2 & +r_end_fit & -surf_757)
                | (+bottom_grid_plate_1 & -bottom_grid_plate_2 & +surf_759 & -surf_758)
            ),
        )
        cell_top_grid_ss304 = openmc.Cell(
            fill=self._get_material("ss304"),
            region=(+top_grid_plate_1 & -top_grid_plate_2 & -r_end_fit),
        )
        cell_top_grid_aluminum = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(+top_grid_plate_1 & -top_grid_plate_2 & +surf_757),
        )
        cell_bottom_grid_ss304 = openmc.Cell(
            fill=self._get_material("ss304"),
            region=(+bottom_grid_plate_1 & -bottom_grid_plate_2 & -surf_759),
        )
        cell_bottom_grid_aluminum = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(+bottom_grid_plate_1 & -bottom_grid_plate_2 & +surf_758),
        )
        water_full_cell = openmc.Cell(
            fill=self._get_material("water"),
            region=(+top_grid_plate_2 & -extended_pool_water_surface),
        )

        cell_uzrh_fuel_1.name = f"{slot_name}_fuel_axial_01"
        cell_uzrh_fuel_2.name = f"{slot_name}_fuel_axial_02"
        cell_uzrh_fuel_3.name = f"{slot_name}_fuel_axial_03"
        cell_uzrh_fuel_4.name = f"{slot_name}_fuel_axial_04"
        cell_uzrh_fuel_5.name = f"{slot_name}_fuel_axial_05"
        cell_uzrh_fuel_6.name = f"{slot_name}_fuel_axial_06"
        cell_uzrh_fuel_7.name = f"{slot_name}_fuel_axial_07"
        cell_uzrh_fuel_8.name = f"{slot_name}_fuel_axial_08"
        cell_uzrh_fuel_9.name = f"{slot_name}_fuel_axial_09"
        cell_uzrh_fuel_10.name = f"{slot_name}_fuel_axial_10"
        cell_uzrh_fuel_11.name = f"{slot_name}_fuel_axial_11"
        cell_uzrh_fuel_12.name = f"{slot_name}_fuel_axial_12"
        cell_uzrh_fuel_13.name = f"{slot_name}_fuel_axial_13"
        cell_uzrh_fuel_14.name = f"{slot_name}_fuel_axial_14"
        cell_uzrh_fuel_15.name = f"{slot_name}_fuel_axial_15"
        cell_bottom_end_fit.name = f"{slot_name}_bottom_end_fit"
        cell_graphite_plug_bottom.name = f"{slot_name}_graphite_bottom"
        cell_zr_rod.name = f"{slot_name}_zr_rod"
        cell_graphite_plug_top.name = f"{slot_name}_graphite_top"
        cell_top_end_fit.name = f"{slot_name}_top_end_fit"
        cell_cladding.name = f"{slot_name}_cladding"
        cell_water.name = f"{slot_name}_water"

        cell_top_grid_ss304.name = f"{slot_name}_top_grid_ss304"
        cell_top_grid_aluminum.name = f"{slot_name}_top_grid_aluminum"
        cell_bottom_grid_ss304.name = f"{slot_name}_bottom_grid_ss304"
        cell_bottom_grid_aluminum.name = f"{slot_name}_bottom_grid_aluminum"
        fuel_cells = [
            cell_uzrh_fuel_1,
            cell_uzrh_fuel_2,
            cell_uzrh_fuel_3,
            cell_uzrh_fuel_4,
            cell_uzrh_fuel_5,
            cell_uzrh_fuel_6,
            cell_uzrh_fuel_7,
            cell_uzrh_fuel_8,
            cell_uzrh_fuel_9,
            cell_uzrh_fuel_10,
            cell_uzrh_fuel_11,
            cell_uzrh_fuel_12,
            cell_uzrh_fuel_13,
            cell_uzrh_fuel_14,
            cell_uzrh_fuel_15,
        ]
        # Cells forming the complete fuel pin.
        fuel_pin_cells = [
            water_full_cell,
            cell_bottom_end_fit,
            cell_graphite_plug_bottom,
            cell_zr_rod,
            *fuel_cells,
            cell_graphite_plug_top,
            cell_top_end_fit,
            cell_cladding,
            cell_water,
            cell_top_grid_ss304,
            cell_top_grid_aluminum,
            cell_bottom_grid_ss304,
            cell_bottom_grid_aluminum,
        ]
        fuel_pin_universe = openmc.Universe(
            name=f"fuel_pin_{slot_name}_{fuel_name}", cells=fuel_pin_cells
        )
        components = {
            "slot_name": slot_name,
            "slot_type": "fuel_pin",
            "fuel_name": fuel_name,
            "universe": fuel_pin_universe,
            "universe_id": fuel_pin_universe.id,
            # Fuel cells ordered from bottom to top.
            "fuel_cells": fuel_cells,
            "fuel_cell_ids": [cell.id for cell in fuel_cells],
            # Fuel-pin components.
            "zr_rod_cell": cell_zr_rod,
            "cladding_cell": cell_cladding,
            "water_cell": cell_water,
            "bottom_graphite_cell": cell_graphite_plug_bottom,
            "top_graphite_cell": cell_graphite_plug_top,
            "bottom_end_fit_cell": cell_bottom_end_fit,
            "top_end_fit_cell": cell_top_end_fit,
            # Core grid plates.
            "top_grid_ss304_cell": cell_top_grid_ss304,
            "top_grid_aluminum_cell": cell_top_grid_aluminum,
            "bottom_grid_ss304_cell": cell_bottom_grid_ss304,
            "bottom_grid_aluminum_cell": cell_bottom_grid_aluminum,
            # All cells in the slot universe.
            "all_cells": fuel_pin_cells,
        }
        return fuel_pin_universe, components

    def _build_control_rod(self, slot_name):
        """Return the control-rod universe for physical core ``slot_name``."""
        bottom_grid_plate_1 = openmc.ZPlane(z0=0.0)
        bottom_grid_plate_2 = openmc.ZPlane(z0=1.90500)
        top_grid_plate_1 = openmc.ZPlane(z0=66.99250)
        top_grid_plate_2 = openmc.ZPlane(z0=68.58000)
        ffcr_end_fit_2 = openmc.ZPlane(z0=0.42863)
        ffcr_end_fit_3 = openmc.ZPlane(z0=1.14300)
        uzrh_top = openmc.ZPlane(z0=55.75300)
        b4c_top = openmc.ZPlane(z0=93.85300)
        ffcr_top_fit_1 = openmc.ZPlane(z0=104.80675)
        ffcr_top_fit_2 = openmc.ZPlane(z0=107.02925)
        ffcr_top_fit_3 = openmc.ZPlane(z0=110.09757)
        r_end_fit = openmc.ZCylinder(r=1.11125)
        r_top_fit = openmc.ZCylinder(r=1.19063)
        r_b4c = openmc.ZCylinder(r=1.67005)
        r_clad = openmc.ZCylinder(r=1.72085)
        r_extension_rod = openmc.ZCylinder(r=1.32085)
        extended_pool_water_surface = openmc.ZPlane(z0=698.96000)
        pool_water_surface = openmc.ZPlane(z0=548.96000)
        surf_828 = openmc.ZCylinder(r=1.91135)
        cell_bottom_end_fit = openmc.Cell(
            fill=self._get_material("ss304"),
            region=((-ffcr_end_fit_2 & -r_end_fit) | (+ffcr_end_fit_2 & -ffcr_end_fit_3 & -r_b4c)),
        )
        cell_bottom_annulus = openmc.Cell(
            fill=self._get_material("water"),
            region=(
                (-bottom_grid_plate_1 & +r_end_fit)
                | (+bottom_grid_plate_1 & -ffcr_end_fit_2 & +r_end_fit & -r_clad)
            ),
        )
        cell_bottom_void = openmc.Cell(fill=None, region=(+ffcr_end_fit_3 & -uzrh_top & -r_b4c))
        cell_b4c = openmc.Cell(
            fill=self._get_material("b4c"), region=(+uzrh_top & -b4c_top & -r_b4c)
        )
        cell_top_void = openmc.Cell(fill=None, region=(+b4c_top & -ffcr_top_fit_1 & -r_b4c))
        cell_top_end_fit = openmc.Cell(
            fill=self._get_material("ss304"),
            region=(
                (+ffcr_top_fit_1 & -ffcr_top_fit_2 & -r_b4c)
                | (+ffcr_top_fit_2 & -ffcr_top_fit_3 & -r_top_fit)
            ),
        )
        cell_cladding = openmc.Cell(
            fill=self._get_material("ss304"),
            region=(+ffcr_end_fit_2 & -ffcr_top_fit_2 & -r_clad & +r_b4c),
        )
        cell_extension_rod = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(+ffcr_top_fit_2 & +r_extension_rod & -r_clad),
        )
        cell_water = openmc.Cell(
            fill=self._get_material("water"),
            region=(
                (+pool_water_surface & -r_extension_rod)
                | (+ffcr_top_fit_2 & -pool_water_surface & +r_top_fit & -r_extension_rod)
                | (+ffcr_top_fit_3 & -pool_water_surface & -r_top_fit)
                | (+bottom_grid_plate_2 & -top_grid_plate_1 & +r_clad)
                | (+top_grid_plate_1 & -top_grid_plate_2 & +r_clad & -surf_828)
                | (+bottom_grid_plate_1 & -bottom_grid_plate_2 & +r_clad & -surf_828)
            ),
        )
        cell_top_grid = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(+top_grid_plate_1 & -top_grid_plate_2 & +surf_828),
        )
        cell_bottom_grid = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(+bottom_grid_plate_1 & -bottom_grid_plate_2 & +surf_828),
        )
        water_full_cell = openmc.Cell(
            fill=self._get_material("water"), region=(+top_grid_plate_2 & +r_clad)
        )
        control_rod_cells = [
            water_full_cell,
            cell_bottom_end_fit,
            cell_bottom_annulus,
            cell_bottom_void,
            cell_b4c,
            cell_top_void,
            cell_top_end_fit,
            cell_cladding,
            cell_extension_rod,
            cell_water,
            cell_top_grid,
            cell_bottom_grid,
        ]
        control_rod_universe = openmc.Universe(
            name=f"control_rod_{slot_name}", cells=control_rod_cells
        )
        return control_rod_universe

    def _build_support_structure(self, slot_name):
        """Return the support-structure universe for a valid Ring-G ``slot_name``."""
        if not isinstance(slot_name, str):
            raise TypeError("slot_name harus berupa string.")
        slot_name = slot_name.strip().upper()
        valid_slots = ("G1", "G7", "G13", "G19", "G25", "G31")
        if slot_name not in valid_slots:
            raise ValueError(
                f"slot_name '{slot_name}' tidak valid. "
                f"Slot yang diperbolehkan: {', '.join(valid_slots)}."
            )

        extended_pool_water_surface = openmc.ZPlane(z0=698.96000)
        bottom_grid_plate_1 = openmc.ZPlane(z0=0.0)  # MCNP surface 8
        bottom_grid_plate_2 = openmc.ZPlane(z0=1.90500)  # MCNP surface 9
        top_grid_plate_1 = openmc.ZPlane(z0=66.99250)  # MCNP surface 15
        top_grid_plate_2 = openmc.ZPlane(z0=68.58000)  # MCNP surface 16
        r_dummy_cylinder = openmc.ZCylinder(r=0.77625)  # MCNP surface 453
        r_cylinder_support_rod = openmc.ZCylinder(r=0.47625)  # MCNP surface 1014
        cell_water = openmc.Cell(
            fill=self._get_material("water"),
            region=(
                (+bottom_grid_plate_2 & -top_grid_plate_1 & +r_dummy_cylinder)
                | (
                    +bottom_grid_plate_2
                    & -top_grid_plate_1
                    & -r_dummy_cylinder
                    & +r_cylinder_support_rod
                )
            ),
        )
        cell_ss_support_rod = openmc.Cell(
            fill=self._get_material("ss304"),
            region=(+bottom_grid_plate_1 & -top_grid_plate_2 & -r_cylinder_support_rod),
        )
        cell_top_grid = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(+top_grid_plate_1 & -top_grid_plate_2 & +r_cylinder_support_rod),
        )
        cell_bottom_grid = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(+bottom_grid_plate_1 & -bottom_grid_plate_2 & +r_cylinder_support_rod),
        )
        water_full_cell = openmc.Cell(
            fill=self._get_material("water"),
            region=(+top_grid_plate_2 & -extended_pool_water_surface),
        )
        support_structure_cells = [
            water_full_cell,
            cell_bottom_grid,
            cell_ss_support_rod,
            cell_top_grid,
            cell_water,
        ]
        support_structure_universe = openmc.Universe(
            name=f"support_rod{slot_name}", cells=support_structure_cells
        )
        return support_structure_universe

    def _build_empty_hole(self, slot_name):
        """Build the aluminum tube and water inside an unoccupied core slot.

        Return the empty-hole universe at ``slot_name`` for placement in the core lattice."""
        bottom_grid_plate_1 = openmc.ZPlane(z0=0.0)  # MCNP surface 8
        bottom_grid_plate_2 = openmc.ZPlane(z0=1.90500)  # MCNP surface 9
        top_grid_plate_1 = openmc.ZPlane(z0=66.99250)  # MCNP surface 15
        top_grid_plate_2 = openmc.ZPlane(z0=68.58000)  # MCNP surface 16
        r_dummy_cylinder = openmc.ZCylinder(r=1.48750)  # MCNP surface 454
        surf_1015 = openmc.ZCylinder(r=1.91135)
        surf_1016 = openmc.ZCylinder(r=1.58750)
        extended_pool_water_surface = openmc.ZPlane(z0=698.96000)

        cell_water = openmc.Cell(
            fill=self._get_material("water"),
            region=(
                (+bottom_grid_plate_2 & -top_grid_plate_1 & +r_dummy_cylinder)
                | (
                    +bottom_grid_plate_2 & -top_grid_plate_1 & -r_dummy_cylinder  # water in G-2)
                    | (+top_grid_plate_1 & -top_grid_plate_2 & -surf_1015)  # top grid
                    | (+bottom_grid_plate_1 & -bottom_grid_plate_2 & -surf_1016)
                )
            ),
        )  # bottom grid
        cell_top_grid = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(+top_grid_plate_1 & -top_grid_plate_2 & +surf_1015),
        )
        cell_bottom_grid = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(+bottom_grid_plate_1 & -bottom_grid_plate_2 & +surf_1016),
        )
        water_full_cell = openmc.Cell(
            fill=self._get_material("water"),
            region=(+top_grid_plate_2 & -extended_pool_water_surface),
        )
        empty_hole_cells = [water_full_cell, cell_water, cell_top_grid, cell_bottom_grid]
        empty_hole_universe = openmc.Universe(name=f"empty_hole{slot_name}", cells=empty_hole_cells)
        return empty_hole_universe

    def _build_outer_core(self):
        """Build fixed geometry outside the 127-slot TRIGA core lattice.

        All dimensions are in cm; comments label the reference MCNP deck entries.
        Returns the 146 fixed root cells plus grouped cells and named surfaces
        needed when assembling the lattice-filled core and root universe."""
        # Core wall and safety plate.
        surf_core_wall_bottom = openmc.ZPlane(z0=-42.54500)  # MCNP surface 1
        surf_safety_plate = openmc.ZPlane(z0=-41.27500)  # MCNP surface 2
        surf_core_wall_top = openmc.ZPlane(z0=68.5800)  # MCNP surface 16
        surf_inner_core_wall = openmc.ZCylinder(r=26.51125)  # MCNP surface 74
        surf_outer_core_wall = openmc.ZCylinder(r=27.14625)  # MCNP surface 75
        cell_core_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=+surf_core_wall_bottom
            & -surf_core_wall_top
            & -surf_outer_core_wall
            & +surf_inner_core_wall,
        )
        cell_safety_plate = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(+surf_core_wall_bottom & -surf_safety_plate & -surf_inner_core_wall),
        )

        # Rotary specimen rack (Lazy Susan).
        surf_reflector_top = openmc.ZPlane(z0=64.92875)  # MCNP surface 14
        surf_upper_part_rotary_rack_1 = openmc.ZCylinder(r=26.22550)  # MCNP surface 630
        surf_upper_part_rotary_rack_2 = openmc.ZCylinder(r=26.86050)  # MCNP surface 631
        surf_lower_part_rotary_rack_1 = openmc.ZCylinder(r=30.51810)  # MCNP surface 632
        surf_lower_part_rotary_rack_2 = openmc.ZCylinder(r=31.15310)  # MCNP surface 633
        surf_outer_part_rotary_rack_1 = openmc.ZCylinder(r=35.68700)  # MCNP surface 634
        surf_outer_part_rotary_rack_2 = openmc.ZCylinder(r=36.32200)  # MCNP surface 635
        surf_top_wall_rotary_rack_1 = openmc.ZPlane(z0=78.57490)  # MCNP surface 636
        surf_top_wall_rotary_rack_2 = openmc.ZPlane(z0=77.48270)  # MCNP surface 637
        surf_middle_wall_rotary_rack = openmc.ZPlane(z0=69.85000)  # MCNP surface 638
        surf_bottom_wall_rotary_rack = openmc.ZPlane(z0=44.76750)  # MCNP surface 639
        surf_shaft_assembly_1 = openmc.ZCylinder(
            x0=-30.759400, y0=0.0, r=2.54000
        )  # MCNP surface 640
        surf_shaft_assembly_2 = openmc.ZCylinder(
            x0=-30.75940, y0=0.0, r=2.32918
        )  # MCNP surface 641
        surf_rotating_shaft_rod = openmc.ZCylinder(
            x0=-32.02940, y0=0, r=0.47625
        )  # MCNP surface 642
        surf_locking_shaft_rod = openmc.ZCylinder(x0=-29.48940, y0=0, r=0.34925)  # MCNP surface 643
        surf_specimen_removal_assembly_1 = openmc.ZCylinder(
            x0=33.19780, y0=0, r=1.90500
        )  # MCNP surface 644
        surf_specimen_removal_assembly_2 = openmc.ZCylinder(
            x0=33.19780, y0=0, r=1.69418
        )  # MCNP surface 645
        surf_ls_well_bottom_wall = openmc.ZPlane(z0=44.13250)  # MCNP surface 704
        surf_ls_well_wall = openmc.ZCylinder(r=37.14750)  # MCNP surface 705
        surf_top_of_extended_tank = openmc.ZPlane(z0=718.96000)  # MCNP surface 750
        cell_top_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                -surf_top_wall_rotary_rack_1
                & +surf_top_wall_rotary_rack_2
                & -surf_outer_part_rotary_rack_2
                & +surf_upper_part_rotary_rack_1
                & +surf_shaft_assembly_1
                & +surf_specimen_removal_assembly_1
            ),
        )
        cell_middle_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                -surf_middle_wall_rotary_rack
                & +surf_core_wall_top
                & -surf_lower_part_rotary_rack_2
                & +surf_upper_part_rotary_rack_1
            ),
        )
        cell_bottom_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                -surf_bottom_wall_rotary_rack
                & +surf_ls_well_bottom_wall
                & -surf_outer_part_rotary_rack_2
                & +surf_lower_part_rotary_rack_1
            ),
        )
        cell_upper_part_inner_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                -surf_top_wall_rotary_rack_2
                & +surf_middle_wall_rotary_rack
                & -surf_upper_part_rotary_rack_2
                & +surf_upper_part_rotary_rack_1
            ),
        )
        cell_lower_part_inner_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                -surf_core_wall_top
                & +surf_bottom_wall_rotary_rack
                & -surf_lower_part_rotary_rack_2
                & +surf_lower_part_rotary_rack_1
            ),
        )
        cell_outer_wall_rotary_rack = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                -surf_top_wall_rotary_rack_2
                & +surf_bottom_wall_rotary_rack
                & -surf_outer_part_rotary_rack_2
                & +surf_outer_part_rotary_rack_1
            ),
        )
        cell_shaft_assembly_tube = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_top_wall_rotary_rack_2
                & -surf_top_of_extended_tank
                & -surf_shaft_assembly_1
                & +surf_shaft_assembly_2
            ),
        )
        cell_air_shaft_assembly = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                +surf_top_wall_rotary_rack_2
                & -surf_top_of_extended_tank
                & -surf_shaft_assembly_2
                & +surf_rotating_shaft_rod
                & +surf_locking_shaft_rod
            ),
        )
        cell_rotating_shaft = openmc.Cell(
            fill=self._get_material("ss304"),
            region=(
                +surf_top_wall_rotary_rack_2 & -surf_top_of_extended_tank & -surf_rotating_shaft_rod
            ),
        )
        cell_locking_shaft = openmc.Cell(
            fill=self._get_material("ss304"),
            region=(
                +surf_top_wall_rotary_rack_2 & -surf_top_of_extended_tank & -surf_locking_shaft_rod
            ),
        )
        cell_specimen_removal_tube = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_top_wall_rotary_rack_2
                & -surf_top_of_extended_tank
                & -surf_specimen_removal_assembly_1
                & +surf_specimen_removal_assembly_2
            ),
        )
        cell_air_specimen_removal = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                +surf_top_wall_rotary_rack_2
                & -surf_top_of_extended_tank
                & -surf_specimen_removal_assembly_2
            ),
        )
        cell_air_upper_rotary_rack = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                -surf_top_wall_rotary_rack_2
                & +surf_middle_wall_rotary_rack
                & -surf_outer_part_rotary_rack_1
                & +surf_upper_part_rotary_rack_2
            ),
        )
        cell_air_lower_rotary_rack = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                -surf_middle_wall_rotary_rack
                & +surf_bottom_wall_rotary_rack
                & -surf_outer_part_rotary_rack_1
                & +surf_lower_part_rotary_rack_2
            ),
        )
        cell_water_ls_well = openmc.Cell(
            fill=self._get_material("water"),
            region=(
                -surf_reflector_top
                & +surf_ls_well_bottom_wall
                & -surf_ls_well_wall
                & +surf_outer_part_rotary_rack_2
            ),
        )
        core_and_rotary_rack_cells = [
            cell_core_wall,
            cell_safety_plate,
            cell_top_wall,
            cell_middle_wall,
            cell_bottom_wall,
            cell_upper_part_inner_wall,
            cell_lower_part_inner_wall,
            cell_outer_wall_rotary_rack,
            cell_shaft_assembly_tube,
            cell_air_shaft_assembly,
            cell_rotating_shaft,
            cell_locking_shaft,
            cell_specimen_removal_tube,
            cell_air_specimen_removal,
            cell_air_upper_rotary_rack,
            cell_air_lower_rotary_rack,
            cell_water_ls_well,
        ]
        # Thermal column.
        surf_thermal_column_ambiguity_plane = openmc.YPlane(y0=0.0)  # MCNP surface 707
        surf_thermal_column_outer_front_wall = openmc.ZCylinder(r=58.57875)  # MCNP surface 708
        surf_thermal_column_inner_front_wall = openmc.ZCylinder(r=56.03875)  # MCNP surface 709
        surf_thermal_column_upper_top_wall = openmc.ZPlane(z0=97.15500)  # MCNP surface 710
        surf_thermal_column_lower_top_wall = openmc.ZPlane(z0=96.52000)  # MCNP surface 711
        surf_thermal_column_upper_bottom_wall = openmc.ZPlane(z0=-25.40000)  # MCNP surface 712
        surf_thermal_column_lower_bottom_wall = openmc.ZPlane(z0=-26.03500)  # MCNP surface 713
        surf_thermal_column_outer_right_wall = openmc.XPlane(x0=60.96000)  # MCNP surface 714
        surf_thermal_column_inner_right_wall = openmc.XPlane(x0=60.32500)  # MCNP surface 715
        surf_thermal_column_inner_left_wall = openmc.XPlane(x0=-60.32500)  # MCNP surface 716
        surf_thermal_column_outer_left_wall = openmc.XPlane(x0=-60.96000)  # MCNP surface 717
        surf_thermal_column_outer_left_diagonal = openmc.Plane(
            a=-0.83910, b=1.0, c=0.0, d=-11.60507
        )  # MCNP surface 718
        surf_thermal_column_inner_left_diagonal = openmc.Plane(
            a=-0.83910, b=1.0, c=0.0, d=-12.43400
        )  # MCNP surface 719
        surf_thermal_column_inner_right_diagonal = openmc.Plane(
            a=0.83910, b=1.0, c=0.0, d=-12.43400
        )  # MCNP surface 720
        surf_thermal_column_outer_right_diagonal = openmc.Plane(
            a=0.83910, b=1.0, c=0.0, d=-11.60507
        )  # MCNP surface 721
        surf_thermal_column_graphite_back_wall = openmc.YPlane(y0=-219.23375)  # MCNP surface 722

        # Thermalizing column.
        surf_thermalizing_inner_front_wall = openmc.ZCylinder(r=56.03875)  # MCNP surface 723
        surf_thermalizing_outer_front_wall = openmc.ZCylinder(r=58.57875)  # MCNP surface 724
        surf_thermalizing_upper_top_wall = openmc.ZPlane(z0=67.31000)  # MCNP surface 725
        surf_thermalizing_lower_top_wall = openmc.ZPlane(z0=66.67500)  # MCNP surface 726
        surf_thermalizing_upper_bottom_wall = openmc.ZPlane(z0=5.71500)  # MCNP surface 727
        surf_thermalizing_lower_bottom_wall = openmc.ZPlane(z0=5.08000)  # MCNP surface 728
        surf_thermalizing_ambiguity_plane = openmc.YPlane(y0=0.0)  # MCNP surface 729
        surf_thermalizing_outer_right_wall = openmc.XPlane(x0=31.11500)  # MCNP surface 730
        surf_thermalizing_inner_right_wall = openmc.XPlane(x0=30.48000)  # MCNP surface 731
        surf_thermalizing_inner_left_wall = openmc.XPlane(x0=-30.48000)  # MCNP surface 732
        surf_thermalizing_outer_left_wall = openmc.XPlane(x0=-31.11500)  # MCNP surface 733
        surf_thermalizing_graphite_wall = openmc.YPlane(y0=58.57975)  # MCNP surface 734
        surf_thermalizing_lead_wall_1 = openmc.YPlane(y0=78.89875)  # MCNP surface 735
        surf_thermalizing_lead_wall_2 = openmc.YPlane(y0=83.97875)  # MCNP surface 736
        surf_thermalizing_outer_graphite_wall = openmc.YPlane(y0=129.54000)  # MCNP surface 737

        # Geometry boundaries shared by both columns.
        surf_outer_tank_wall = openmc.ZCylinder(r=99.69500)  # MCNP surface 739
        surf_south_top_wall = openmc.YPlane(y0=190.50000)  # MCNP surface 1134

        # Thermal column.
        cell_thermal_column_front_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_thermal_column_inner_front_wall
                & -surf_thermal_column_outer_front_wall
                & +surf_thermal_column_lower_bottom_wall
                & -surf_thermal_column_upper_top_wall
                & -surf_thermal_column_outer_left_diagonal
                & -surf_thermal_column_outer_right_diagonal
                & -surf_thermal_column_ambiguity_plane
            ),
        )
        cell_thermal_column_left_outer_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_thermal_column_outer_front_wall
                & +surf_thermal_column_outer_left_wall
                & +surf_thermal_column_lower_bottom_wall
                & -surf_thermal_column_upper_top_wall
                & +surf_thermal_column_inner_left_diagonal
                & -surf_thermal_column_outer_left_diagonal
                & -surf_thermal_column_ambiguity_plane
            ),
        )

        cell_thermal_column_right_outer_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_thermal_column_outer_front_wall
                & -surf_thermal_column_outer_right_wall
                & +surf_thermal_column_lower_bottom_wall
                & -surf_thermal_column_upper_top_wall
                & +surf_thermal_column_inner_right_diagonal
                & -surf_thermal_column_outer_right_diagonal
                & -surf_thermal_column_ambiguity_plane
            ),
        )

        cell_thermal_column_left_side_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                -surf_thermal_column_inner_left_diagonal
                & -surf_thermal_column_inner_right_diagonal
                & -surf_outer_tank_wall
                & +surf_thermal_column_lower_bottom_wall
                & -surf_thermal_column_upper_top_wall
                & +surf_thermal_column_outer_left_wall
                & -surf_thermal_column_inner_left_wall
                & -surf_thermal_column_ambiguity_plane
            ),
        )

        cell_thermal_column_right_side_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                -surf_thermal_column_inner_left_diagonal
                & -surf_thermal_column_inner_right_diagonal
                & -surf_outer_tank_wall
                & +surf_thermal_column_lower_bottom_wall
                & -surf_thermal_column_upper_top_wall
                & +surf_thermal_column_inner_right_wall
                & -surf_thermal_column_outer_right_wall
                & -surf_thermal_column_ambiguity_plane
            ),
        )

        cell_thermal_column_bottom_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                -surf_thermal_column_inner_left_diagonal
                & -surf_thermal_column_inner_right_diagonal
                & -surf_outer_tank_wall
                & +surf_thermal_column_outer_front_wall
                & +surf_thermal_column_lower_bottom_wall
                & -surf_thermal_column_upper_bottom_wall
                & +surf_thermal_column_inner_left_wall
                & -surf_thermal_column_inner_right_wall
                & -surf_thermal_column_ambiguity_plane
            ),
        )

        cell_thermal_column_top_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                -surf_thermal_column_inner_left_diagonal
                & -surf_thermal_column_inner_right_diagonal
                & -surf_outer_tank_wall
                & +surf_thermal_column_outer_front_wall
                & +surf_thermal_column_lower_top_wall
                & -surf_thermal_column_upper_top_wall
                & +surf_thermal_column_inner_left_wall
                & -surf_thermal_column_inner_right_wall
                & -surf_thermal_column_ambiguity_plane
            ),
        )

        cell_thermal_column_graphite = openmc.Cell(
            fill=self._get_material("graphite"),
            region=(
                -surf_thermal_column_inner_left_diagonal
                & -surf_thermal_column_inner_right_diagonal
                & +surf_thermal_column_graphite_back_wall
                & +surf_thermal_column_outer_front_wall
                & +surf_thermal_column_upper_bottom_wall
                & -surf_thermal_column_lower_top_wall
                & +surf_thermal_column_inner_left_wall
                & -surf_thermal_column_inner_right_wall
                & -surf_thermal_column_ambiguity_plane
            ),
        )

        thermal_column_cells = [
            cell_thermal_column_front_wall,
            cell_thermal_column_left_outer_wall,
            cell_thermal_column_right_outer_wall,
            cell_thermal_column_left_side_wall,
            cell_thermal_column_right_side_wall,
            cell_thermal_column_bottom_wall,
            cell_thermal_column_top_wall,
            cell_thermal_column_graphite,
        ]

        # Thermalizing column.
        cell_thermalizing_front_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_thermalizing_inner_front_wall
                & -surf_thermalizing_outer_front_wall
                & +surf_thermalizing_lower_bottom_wall
                & -surf_thermalizing_upper_top_wall
                & +surf_thermalizing_outer_left_wall
                & -surf_thermalizing_outer_right_wall
                & +surf_thermalizing_ambiguity_plane
            ),
        )

        cell_thermalizing_left_outer_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_thermalizing_outer_front_wall
                & -surf_outer_tank_wall
                & +surf_thermalizing_lower_bottom_wall
                & -surf_thermalizing_upper_top_wall
                & +surf_thermalizing_outer_left_wall
                & -surf_thermalizing_inner_left_wall
                & +surf_thermalizing_ambiguity_plane
            ),
        )

        cell_thermalizing_right_outer_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_thermalizing_outer_front_wall
                & -surf_outer_tank_wall
                & +surf_thermalizing_lower_bottom_wall
                & -surf_thermalizing_upper_top_wall
                & +surf_thermalizing_inner_right_wall
                & -surf_thermalizing_outer_right_wall
                & +surf_thermalizing_ambiguity_plane
            ),
        )

        cell_thermalizing_bottom_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_thermalizing_outer_front_wall
                & -surf_outer_tank_wall
                & +surf_thermalizing_lower_bottom_wall
                & -surf_thermalizing_upper_bottom_wall
                & +surf_thermalizing_inner_left_wall
                & -surf_thermalizing_inner_right_wall
                & +surf_thermalizing_ambiguity_plane
            ),
        )

        cell_thermalizing_top_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_thermalizing_outer_front_wall
                & -surf_outer_tank_wall
                & +surf_thermalizing_lower_top_wall
                & -surf_thermalizing_upper_top_wall
                & +surf_thermalizing_inner_left_wall
                & -surf_thermalizing_inner_right_wall
                & +surf_thermalizing_ambiguity_plane
            ),
        )

        cell_thermalizing_air_gap = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                +surf_thermalizing_outer_front_wall
                & -surf_thermalizing_graphite_wall
                & +surf_thermalizing_upper_bottom_wall
                & -surf_thermalizing_lower_top_wall
                & +surf_thermalizing_inner_left_wall
                & -surf_thermalizing_inner_right_wall
                & +surf_thermalizing_ambiguity_plane
            ),
        )

        cell_thermalizing_inner_graphite = openmc.Cell(
            fill=self._get_material("graphite"),
            region=(
                -surf_thermalizing_lower_top_wall
                & +surf_thermalizing_upper_bottom_wall
                & -surf_thermalizing_inner_right_wall
                & +surf_thermalizing_inner_left_wall
                & +surf_thermalizing_graphite_wall
                & -surf_thermalizing_lead_wall_1
            ),
        )

        cell_thermalizing_lead = openmc.Cell(
            fill=self._get_material("lead"),
            region=(
                -surf_thermalizing_lower_top_wall
                & +surf_thermalizing_upper_bottom_wall
                & -surf_thermalizing_inner_right_wall
                & +surf_thermalizing_inner_left_wall
                & +surf_thermalizing_lead_wall_1
                & -surf_thermalizing_lead_wall_2
            ),
        )

        cell_thermalizing_outer_air_gap = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                -surf_thermalizing_lower_top_wall
                & +surf_thermalizing_upper_bottom_wall
                & -surf_thermalizing_inner_right_wall
                & +surf_thermalizing_inner_left_wall
                & +surf_thermalizing_lead_wall_2
                & -surf_thermalizing_outer_graphite_wall
            ),
        )

        cell_thermalizing_outer_graphite = openmc.Cell(
            fill=self._get_material("graphite"),
            region=(
                -surf_thermalizing_lower_top_wall
                & +surf_thermalizing_upper_bottom_wall
                & -surf_thermalizing_inner_right_wall
                & +surf_thermalizing_inner_left_wall
                & +surf_thermalizing_outer_graphite_wall
                & -surf_south_top_wall
            ),
        )
        thermalizing_column_cells = [
            cell_thermalizing_front_wall,
            cell_thermalizing_left_outer_wall,
            cell_thermalizing_right_outer_wall,
            cell_thermalizing_bottom_wall,
            cell_thermalizing_top_wall,
            cell_thermalizing_air_gap,
            cell_thermalizing_inner_graphite,
            cell_thermalizing_lead,
            cell_thermalizing_outer_air_gap,
            cell_thermalizing_outer_graphite,
        ]
        # Reflector walls.
        # Surfaces 14, 16, 75, 632, 704, and 705 are already defined.

        surf_reflector_bottom = openmc.ZPlane(z0=7.62000)  # MCNP surface 10
        surf_ls_well_bottom = openmc.ZPlane(z0=43.49750)  # MCNP surface 13

        surf_reflector_wall_lower_bottom = openmc.ZPlane(z0=8.57250)  # MCNP surface 701
        surf_reflector_wall_lower_top = openmc.ZPlane(z0=63.97625)  # MCNP surface 702
        surf_reflector_wall_inner = openmc.ZCylinder(r=53.34000)  # MCNP surface 703
        surf_ls_well_inner_wall = openmc.ZCylinder(r=29.88310)  # MCNP surface 706

        surf_ls_well_outer_wall = openmc.ZCylinder(r=37.78250)  # MCNP surface 81
        surf_outer_reflector = openmc.ZCylinder(r=54.29250)  # MCNP surface 82

        # Surfaces 649 and 650 use *TR1:
        # translate z by 29.21 cm and rotate +62.7 degrees about the z-axis.
        surf_piercing_beamport = openmc.Cylinder(
            x0=0.0,
            y0=0.0,
            z0=29.21000,
            r=8.89000,
            dx=-0.8886172326549489,
            dy=0.4586495544843149,
            dz=0.0,
        )  # 649

        surf_piercing_beamport_ambiguity_plane = openmc.Plane(
            a=-0.8886172326549489,
            b=0.4586495544843149,
            c=0.0,
            d=0.0,
        )  # MCNP surface 650

        # Reflector walls.
        cell_reflector_wall_upper_ring = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                -surf_reflector_top
                & +surf_reflector_wall_lower_top
                & -surf_outer_reflector
                & +surf_ls_well_wall
            ),
        )  # MCNP cell 46

        cell_reflector_wall_lower_ring = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_reflector_bottom
                & -surf_reflector_wall_lower_bottom
                & -surf_outer_reflector
                & +surf_outer_core_wall
            ),
        )  # MCNP cell 47

        cell_reflector_wall_main = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_reflector_wall_lower_bottom
                & -surf_reflector_wall_lower_top
                & -surf_outer_reflector
                & +surf_reflector_wall_inner
                & ~((-surf_piercing_beamport) & (+surf_piercing_beamport_ambiguity_plane))
            ),
        )  # MCNP cell 48; #(-649 650)

        cell_ls_well_upper_annulus = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_ls_well_bottom
                & -surf_reflector_wall_lower_top
                & -surf_ls_well_outer_wall
                & +surf_ls_well_wall
            ),
        )  # MCNP cell 49

        cell_ls_well_bottom_annulus = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_ls_well_bottom
                & -surf_ls_well_bottom_wall
                & +surf_ls_well_inner_wall
                & -surf_ls_well_wall
            ),
        )  # MCNP cell 50

        cell_rotary_rack_lower_annulus = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_ls_well_bottom_wall
                & -surf_reflector_wall_lower_top
                & -surf_lower_part_rotary_rack_1
                & +surf_ls_well_inner_wall
            ),
        )  # MCNP cell 51

        cell_rotary_rack_upper_annulus = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                -surf_core_wall_top
                & +surf_reflector_wall_lower_top
                & -surf_lower_part_rotary_rack_1
                & +surf_outer_core_wall
            ),
        )  # MCNP cell 52

        cell_ls_well_inner_bottom_annulus = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_ls_well_bottom
                & -surf_ls_well_bottom_wall
                & -surf_ls_well_inner_wall
                & +surf_outer_core_wall
            ),
        )  # MCNP cell 53

        cell_void_under_rotary_specimen_rack = openmc.Cell(
            fill=None,
            region=(
                -surf_reflector_wall_lower_top
                & +surf_ls_well_bottom_wall
                & -surf_ls_well_inner_wall
                & +surf_outer_core_wall
            ),
        )  # MCNP cell 54
        reflector_cells = [
            cell_reflector_wall_upper_ring,
            cell_reflector_wall_lower_ring,
            cell_reflector_wall_main,
            cell_ls_well_upper_annulus,
            cell_ls_well_bottom_annulus,
            cell_rotary_rack_lower_annulus,
            cell_rotary_rack_upper_annulus,
            cell_ls_well_inner_bottom_annulus,
            cell_void_under_rotary_specimen_rack,
        ]
        import numpy as np

        def apply_mcnp_surface_transform(
            surface: openmc.Surface,
            translation: np.ndarray,
            rotation: np.ndarray,
        ) -> openmc.Surface:
            """Apply an MCNP TR transform to an OpenMC surface.

            Parameters are the translation in cm and the 3x3 rotation matrix.
            Returns the equivalent surface in global OpenMC coordinates."""
            # OpenMC rotation may create temporary replacement surfaces with reused IDs.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", openmc.IDWarning)
                transformed = surface.translate(translation, inplace=True)
                return transformed.rotate(rotation, pivot=translation, inplace=True)

        TR1_TRANSLATION = np.array([0.0, 0.0, 29.21000])
        TR2_TRANSLATION = np.array([0.0, 0.0, 29.21000])
        TR3_TRANSLATION = np.array([0.0, 0.0, 29.21000])
        TR4_TRANSLATION = np.array([0.0, 0.0, 29.21000])

        TR1_ROTATION = np.array(
            [
                [0.4586495544843149, -0.8886172326549487, 0.0],
                [0.8886172326549489, 0.4586495544843149, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        TR2_ROTATION = np.array(
            [
                [0.8886172326549489, -0.4586495544843146, 0.0],
                [0.4586495544843149, 0.8886172326549489, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        TR3_ROTATION = np.array(
            [
                [0.7660444431189780, 0.6427876096865394, 0.0],
                [-0.6427876096865394, 0.7660444431189780, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        TR4_ROTATION = np.array(
            [
                [0.6427876096865394, -0.7660444431189779, 0.0],
                [0.7660444431189780, 0.6427876096865394, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        TR11_TRANSLATION = np.array([145.0, -350.0609665, 0.0])
        TR12_TRANSLATION = np.array([172.5, 423.5, 0.0])
        TR13_TRANSLATION = np.array([-172.5, 423.5, 0.0])
        TR14_TRANSLATION = np.array([-145.0, -350.0609665, 0.0])

        TR11_ROTATION = np.array(
            [
                [0.7071067811865476, 0.7071067811865476, 0.0],
                [-0.7071067811865475, 0.7071067811865476, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        TR12_ROTATION = np.array(
            [
                [0.4869345989179495, 0.8734384330773496, 0.0],
                [-0.8734384330773495, 0.4869345989179495, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        TR13_ROTATION = np.array(
            [
                [-0.8734384330773495, -0.4869345989179494, 0.0],
                [0.4869345989179495, -0.8734384330773495, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        TR14_ROTATION = np.array(
            [
                [-0.7071067811865475, -0.7071067811865475, 0.0],
                [0.7071067811865476, -0.7071067811865475, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        # Through beam port: surfaces 76-79 and 646-651.
        surf_piercing_core_end = apply_mcnp_surface_transform(
            openmc.YPlane(y0=27.14625), TR1_TRANSLATION, TR1_ROTATION
        )  # 76
        surf_piercing_reflector_end = apply_mcnp_surface_transform(
            openmc.YPlane(y0=28.09875), TR1_TRANSLATION, TR1_ROTATION
        )  # 77
        surf_piercing_reflector_flange = apply_mcnp_surface_transform(
            openmc.YPlane(y0=69.53250), TR1_TRANSLATION, TR1_ROTATION
        )  # 78
        surf_piercing_tank_flange = apply_mcnp_surface_transform(
            openmc.YPlane(y0=91.44000), TR1_TRANSLATION, TR1_ROTATION
        )  # 79

        surf_other_beam_tube_end = openmc.ZCylinder(r=55.24500)  # MCNP surface 80

        surf_piercing_beam_hole_outer = apply_mcnp_surface_transform(
            openmc.YCylinder(r=8.57250), TR1_TRANSLATION, TR1_ROTATION
        )  # 646
        surf_piercing_beam_hole_inner = apply_mcnp_surface_transform(
            openmc.YCylinder(r=7.62000), TR1_TRANSLATION, TR1_ROTATION
        )  # 647
        surf_bellows_outer_wall = apply_mcnp_surface_transform(
            openmc.YCylinder(r=10.47750), TR1_TRANSLATION, TR1_ROTATION
        )  # 648
        surf_piercing_beam_tube = apply_mcnp_surface_transform(
            openmc.YCylinder(r=7.93750), TR1_TRANSLATION, TR1_ROTATION
        )  # 651

        # Time-of-flight beam port: surfaces 652-654.
        surf_tof_beam_hole = apply_mcnp_surface_transform(
            openmc.XCylinder(r=7.62000), TR2_TRANSLATION, TR2_ROTATION
        )  # 652
        surf_tof_ambiguity_plane = apply_mcnp_surface_transform(
            openmc.XPlane(x0=0.0), TR2_TRANSLATION, TR2_ROTATION
        )  # 653
        surf_tof_beam_tube = apply_mcnp_surface_transform(
            openmc.XCylinder(r=8.57250), TR2_TRANSLATION, TR2_ROTATION
        )  # 654

        # Neutron radiography port: surfaces 655-657.
        surf_nr_beam_hole = apply_mcnp_surface_transform(
            openmc.XCylinder(r=7.62000), TR3_TRANSLATION, TR3_ROTATION
        )  # 655
        surf_nr_ambiguity_plane = apply_mcnp_surface_transform(
            openmc.XPlane(x0=0.0), TR3_TRANSLATION, TR3_ROTATION
        )  # 656
        surf_nr_beam_tube = apply_mcnp_surface_transform(
            openmc.XCylinder(r=8.57250), TR3_TRANSLATION, TR3_ROTATION
        )  # 657

        # Tangential beam port: surfaces 658-660.
        surf_tangential_beam_hole = apply_mcnp_surface_transform(
            openmc.XCylinder(y0=30.95625, z0=0.0, r=7.62000),
            TR4_TRANSLATION,
            TR4_ROTATION,
        )  # 658

        # Surface 659 uses TR1 in the MCNP deck, rather than TR4.
        surf_tangential_ambiguity_plane = apply_mcnp_surface_transform(
            openmc.XPlane(x0=0.0), TR1_TRANSLATION, TR1_ROTATION
        )  # 659

        surf_tangential_beam_tube = apply_mcnp_surface_transform(
            openmc.XCylinder(y0=30.95625, z0=0.0, r=8.57250),
            TR4_TRANSLATION,
            TR4_ROTATION,
        )  # 660

        # Steel tubes: surfaces 661-668.
        surf_piercing_steel_outer = apply_mcnp_surface_transform(
            openmc.YCylinder(r=11.11250), TR1_TRANSLATION, TR1_ROTATION
        )  # 661
        surf_piercing_steel_inner = apply_mcnp_surface_transform(
            openmc.YCylinder(r=10.16000), TR1_TRANSLATION, TR1_ROTATION
        )  # 662

        surf_tof_steel_outer = apply_mcnp_surface_transform(
            openmc.XCylinder(r=11.11250), TR2_TRANSLATION, TR2_ROTATION
        )  # 663
        surf_tof_steel_inner = apply_mcnp_surface_transform(
            openmc.XCylinder(r=10.16000), TR2_TRANSLATION, TR2_ROTATION
        )  # 664

        surf_nr_steel_outer = apply_mcnp_surface_transform(
            openmc.XCylinder(r=11.11250), TR3_TRANSLATION, TR3_ROTATION
        )  # 665
        surf_nr_steel_inner = apply_mcnp_surface_transform(
            openmc.XCylinder(r=10.16000), TR3_TRANSLATION, TR3_ROTATION
        )  # 666

        surf_tangential_steel_outer = apply_mcnp_surface_transform(
            openmc.XCylinder(y0=30.95625, z0=0.0, r=11.11250),
            TR4_TRANSLATION,
            TR4_ROTATION,
        )  # 667
        surf_tangential_steel_inner = apply_mcnp_surface_transform(
            openmc.XCylinder(y0=30.95625, z0=0.0, r=10.16000),
            TR4_TRANSLATION,
            TR4_ROTATION,
        )  # 668

        # Shutter boxes: surfaces 669-682.
        surf_piercing_shutter_inner = apply_mcnp_surface_transform(
            openmc.XPlane(x0=-18.41500), TR13_TRANSLATION, TR13_ROTATION
        )  # 669
        surf_piercing_shutter_right = apply_mcnp_surface_transform(
            openmc.XPlane(x0=-31.43250), TR1_TRANSLATION, TR1_ROTATION
        )  # 670
        surf_piercing_shutter_left = apply_mcnp_surface_transform(
            openmc.XPlane(x0=22.54250), TR1_TRANSLATION, TR1_ROTATION
        )  # 671

        surf_tof_shutter_inner = apply_mcnp_surface_transform(
            openmc.YPlane(y0=-18.41500), TR12_TRANSLATION, TR12_ROTATION
        )  # 672
        surf_tof_shutter_left = apply_mcnp_surface_transform(
            openmc.YPlane(y0=-31.43250), TR2_TRANSLATION, TR2_ROTATION
        )  # 673
        surf_tof_shutter_right = apply_mcnp_surface_transform(
            openmc.YPlane(y0=22.54250), TR2_TRANSLATION, TR2_ROTATION
        )  # 674

        surf_nr_shutter_inner = apply_mcnp_surface_transform(
            openmc.XPlane(x0=-18.41500), TR11_TRANSLATION, TR11_ROTATION
        )  # 675
        surf_nr_shutter_left = apply_mcnp_surface_transform(
            openmc.YPlane(y0=-22.54250), TR3_TRANSLATION, TR3_ROTATION
        )  # 676
        surf_nr_shutter_right = apply_mcnp_surface_transform(
            openmc.YPlane(y0=31.43250), TR3_TRANSLATION, TR3_ROTATION
        )  # 677

        surf_tangential_shutter_inner = apply_mcnp_surface_transform(
            openmc.YPlane(y0=-18.41500), TR14_TRANSLATION, TR14_ROTATION
        )  # 678
        surf_tangential_shutter_right = apply_mcnp_surface_transform(
            openmc.YPlane(y0=8.41375), TR4_TRANSLATION, TR4_ROTATION
        )  # 679
        surf_tangential_shutter_left = apply_mcnp_surface_transform(
            openmc.YPlane(y0=62.38875), TR4_TRANSLATION, TR4_ROTATION
        )  # 680

        surf_shutter_top = openmc.ZPlane(z0=54.76875)  # MCNP surface 681
        surf_shutter_bottom = openmc.ZPlane(z0=3.65125)  # MCNP surface 682

        # Beam-port shielding: surfaces 683-700.
        surf_beam_shield_top = openmc.ZPlane(z0=80.01000)  # MCNP surface 683
        surf_beam_shield_bottom = openmc.ZPlane(z0=-21.59000)  # MCNP surface 684

        surf_piercing_shield_outer = apply_mcnp_surface_transform(
            openmc.YPlane(y0=175.89500), TR1_TRANSLATION, TR1_ROTATION
        )  # 685
        surf_piercing_shield_inner = apply_mcnp_surface_transform(
            openmc.YPlane(y0=166.37000), TR1_TRANSLATION, TR1_ROTATION
        )  # 686
        surf_piercing_shield_right = apply_mcnp_surface_transform(
            openmc.XPlane(x0=50.80000), TR1_TRANSLATION, TR1_ROTATION
        )  # 687
        surf_piercing_shield_left = apply_mcnp_surface_transform(
            openmc.XPlane(x0=-50.80000), TR1_TRANSLATION, TR1_ROTATION
        )  # 688

        surf_tof_shield_outer = apply_mcnp_surface_transform(
            openmc.XPlane(x0=175.89500), TR2_TRANSLATION, TR2_ROTATION
        )  # 689
        surf_tof_shield_inner = apply_mcnp_surface_transform(
            openmc.XPlane(x0=166.37000), TR2_TRANSLATION, TR2_ROTATION
        )  # 690
        surf_tof_shield_right = apply_mcnp_surface_transform(
            openmc.YPlane(y0=50.80000), TR2_TRANSLATION, TR2_ROTATION
        )  # 691
        surf_tof_shield_left = apply_mcnp_surface_transform(
            openmc.YPlane(y0=-50.80000), TR2_TRANSLATION, TR2_ROTATION
        )  # 692

        surf_nr_shield_outer = apply_mcnp_surface_transform(
            openmc.XPlane(x0=175.89500), TR3_TRANSLATION, TR3_ROTATION
        )  # 693
        surf_nr_shield_inner = apply_mcnp_surface_transform(
            openmc.XPlane(x0=166.37000), TR3_TRANSLATION, TR3_ROTATION
        )  # 694
        surf_nr_shield_right = apply_mcnp_surface_transform(
            openmc.YPlane(y0=50.80000), TR3_TRANSLATION, TR3_ROTATION
        )  # 695
        surf_nr_shield_left = apply_mcnp_surface_transform(
            openmc.YPlane(y0=-50.80000), TR3_TRANSLATION, TR3_ROTATION
        )  # 696

        surf_tangential_shield_outer = apply_mcnp_surface_transform(
            openmc.XPlane(x0=-174.68614), TR4_TRANSLATION, TR4_ROTATION
        )  # 697
        surf_tangential_shield_inner = apply_mcnp_surface_transform(
            openmc.XPlane(x0=-165.16114), TR4_TRANSLATION, TR4_ROTATION
        )  # 698
        surf_tangential_shield_right = apply_mcnp_surface_transform(
            openmc.YPlane(y0=81.75625), TR4_TRANSLATION, TR4_ROTATION
        )  # 699
        surf_tangential_shield_left = apply_mcnp_surface_transform(
            openmc.YPlane(y0=-19.84375), TR4_TRANSLATION, TR4_ROTATION
        )  # 700

        # Biological-shield base boundaries required by the shutter boxes.
        surf_bio_northwest_base_wall = apply_mcnp_surface_transform(
            openmc.XPlane(x0=0.0), TR11_TRANSLATION, TR11_ROTATION
        )  # 1123
        surf_bio_southwest_base_wall = apply_mcnp_surface_transform(
            openmc.YPlane(y0=0.0), TR12_TRANSLATION, TR12_ROTATION
        )  # 1125
        surf_bio_southeast_base_wall = apply_mcnp_surface_transform(
            openmc.XPlane(x0=0.0), TR13_TRANSLATION, TR13_ROTATION
        )  # 1127
        surf_bio_northeast_base_wall = apply_mcnp_surface_transform(
            openmc.YPlane(y0=0.0), TR14_TRANSLATION, TR14_ROTATION
        )  # 1129

        # Beam ports within and immediately outside the reflector.
        cell_piercing_graphite_hole = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                -surf_piercing_beam_tube
                & -surf_outer_tank_wall
                & +surf_piercing_reflector_end
                & +surf_piercing_beamport_ambiguity_plane
            ),
        )  # MCNP cell 5

        cell_piercing_flange = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                -surf_piercing_beamport
                & +surf_piercing_beam_tube
                & -surf_outer_tank_wall
                & +surf_piercing_tank_flange
                & +surf_piercing_beamport_ambiguity_plane
            ),
        )  # MCNP cell 6

        cell_piercing_reflector_flange = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                -surf_piercing_beamport
                & +surf_piercing_beam_tube
                & -surf_piercing_reflector_flange
                & +surf_piercing_reflector_end
                & +surf_piercing_beamport_ambiguity_plane
            ),
        )  # MCNP cell 7

        cell_piercing_beamport_end_plate = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                -surf_piercing_beamport & -surf_piercing_reflector_end & +surf_piercing_core_end
            ),
        )  # MCNP cell 8

        cell_piercing_void_behind_end_plate = openmc.Cell(
            fill=None,
            region=(
                -surf_piercing_beamport
                & -surf_piercing_core_end
                & +surf_outer_core_wall
                & +surf_piercing_beamport_ambiguity_plane
            ),
        )  # MCNP cell 9

        cell_tof_beamport_air = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                -surf_tof_beam_hole
                & -surf_outer_tank_wall
                & +surf_other_beam_tube_end
                & +surf_tof_ambiguity_plane
            ),
        )  # MCNP cell 10

        cell_tof_graphite_hole = openmc.Cell(
            fill=None,
            region=(
                -surf_tof_beam_hole
                & -surf_reflector_wall_inner
                & +surf_outer_core_wall
                & +surf_tof_ambiguity_plane
            ),
        )  # MCNP cell 11

        cell_tof_beam_tube = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                -surf_tof_beam_tube
                & +surf_tof_beam_hole
                & -surf_outer_tank_wall
                & +surf_other_beam_tube_end
                & +surf_tof_ambiguity_plane
            ),
        )  # MCNP cell 12

        cell_tof_beamport_end_plate = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                -surf_tof_beam_tube
                & -surf_other_beam_tube_end
                & +surf_outer_reflector
                & +surf_tof_ambiguity_plane
            ),
        )  # MCNP cell 13

        cell_nr_beamport_air = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                -surf_nr_beam_hole
                & -surf_outer_tank_wall
                & +surf_other_beam_tube_end
                & +surf_nr_ambiguity_plane
            ),
        )  # MCNP cell 14

        cell_nr_beam_tube = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                -surf_nr_beam_tube
                & +surf_nr_beam_hole
                & -surf_outer_tank_wall
                & +surf_other_beam_tube_end
                & +surf_nr_ambiguity_plane
            ),
        )  # MCNP cell 15

        cell_nr_beamport_end_plate = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                -surf_nr_beam_tube
                & -surf_other_beam_tube_end
                & +surf_outer_reflector
                & +surf_nr_ambiguity_plane
            ),
        )  # MCNP cell 16

        cell_tangential_beamport_air = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                -surf_tangential_beam_hole
                & -surf_outer_tank_wall
                & +surf_other_beam_tube_end
                & -surf_tangential_ambiguity_plane
            ),
        )  # MCNP cell 17

        cell_tangential_graphite_hole = openmc.Cell(
            fill=None,
            region=(
                -surf_tangential_beam_hole
                & -surf_reflector_wall_inner
                & +surf_outer_core_wall
                & -surf_tangential_ambiguity_plane
                & +surf_piercing_beamport
            ),
        )  # MCNP cell 18

        cell_tangential_beam_tube = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                -surf_tangential_beam_tube
                & +surf_tangential_beam_hole
                & -surf_outer_tank_wall
                & +surf_other_beam_tube_end
                & -surf_tangential_ambiguity_plane
            ),
        )  # MCNP cell 19

        cell_tangential_beamport_end_plate = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                -surf_tangential_beam_tube
                & -surf_other_beam_tube_end
                & +surf_outer_reflector
                & -surf_tangential_ambiguity_plane
            ),
        )  # MCNP cell 20

        # Outer beam holes, steel tubes, and shutter boxes.
        cell_piercing_beam_hole_air = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                -surf_piercing_beam_hole_inner
                & +surf_outer_tank_wall
                & +surf_piercing_beamport_ambiguity_plane
                & -surf_piercing_shield_outer
            ),
        )  # MCNP cell 21

        cell_piercing_beam_hole_tube = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                -surf_piercing_beam_hole_outer
                & +surf_piercing_beam_hole_inner
                & +surf_outer_tank_wall
                & +surf_piercing_beamport_ambiguity_plane
                & -surf_piercing_shield_outer
            ),
        )  # MCNP cell 22

        cell_piercing_steel_tube_air = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                -surf_piercing_steel_inner
                & -surf_piercing_shutter_inner
                & +surf_piercing_beamport_ambiguity_plane
                & +surf_piercing_shield_outer
            ),
        )  # MCNP cell 23

        cell_piercing_steel_tube = openmc.Cell(
            fill=self._get_material("steel"),
            region=(
                -surf_piercing_steel_outer
                & +surf_piercing_steel_inner
                & -surf_piercing_shutter_inner
                & +surf_piercing_beamport_ambiguity_plane
                & +surf_piercing_shield_outer
            ),
        )  # MCNP cell 24

        cell_piercing_shutter_box = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                +surf_piercing_shutter_inner
                & -surf_bio_southeast_base_wall
                & +surf_piercing_shutter_right
                & -surf_piercing_shutter_left
                & -surf_shutter_top
                & +surf_shutter_bottom
            ),
        )  # MCNP cell 25

        cell_tof_beam_hole_air = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                -surf_tof_beam_hole
                & +surf_outer_tank_wall
                & +surf_tof_ambiguity_plane
                & -surf_tof_shield_outer
            ),
        )  # MCNP cell 26

        cell_tof_beam_hole_tube = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                -surf_tof_beam_tube
                & +surf_tof_beam_hole
                & +surf_outer_tank_wall
                & +surf_tof_ambiguity_plane
                & -surf_tof_shield_outer
            ),
        )  # MCNP cell 27

        cell_tof_steel_tube_air = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                -surf_tof_steel_inner
                & -surf_tof_shutter_inner
                & +surf_tof_ambiguity_plane
                & +surf_tof_shield_outer
            ),
        )  # MCNP cell 28

        cell_tof_steel_tube = openmc.Cell(
            fill=self._get_material("steel"),
            region=(
                -surf_tof_steel_outer
                & +surf_tof_steel_inner
                & -surf_tof_shutter_inner
                & +surf_tof_ambiguity_plane
                & +surf_tof_shield_outer
            ),
        )  # MCNP cell 29

        cell_tof_shutter_box = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                +surf_tof_shutter_inner
                & -surf_bio_southwest_base_wall
                & +surf_tof_shutter_left
                & -surf_tof_shutter_right
                & -surf_shutter_top
                & +surf_shutter_bottom
            ),
        )  # MCNP cell 30

        cell_nr_beam_hole_air = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                -surf_nr_beam_hole
                & +surf_outer_tank_wall
                & +surf_nr_ambiguity_plane
                & -surf_nr_shield_outer
            ),
        )  # MCNP cell 31

        cell_nr_beam_hole_tube = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                -surf_nr_beam_tube
                & +surf_nr_beam_hole
                & +surf_outer_tank_wall
                & +surf_nr_ambiguity_plane
                & -surf_nr_shield_outer
            ),
        )  # MCNP cell 32

        cell_nr_steel_tube_air = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                -surf_nr_steel_inner
                & -surf_nr_shutter_inner
                & +surf_nr_ambiguity_plane
                & +surf_nr_shield_outer
            ),
        )  # MCNP cell 33

        cell_nr_steel_tube = openmc.Cell(
            fill=self._get_material("steel"),
            region=(
                -surf_nr_steel_outer
                & +surf_nr_steel_inner
                & -surf_nr_shutter_inner
                & +surf_nr_ambiguity_plane
                & +surf_nr_shield_outer
            ),
        )  # MCNP cell 34

        cell_nr_shutter_box = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                +surf_nr_shutter_inner
                & -surf_bio_northwest_base_wall
                & +surf_nr_shutter_left
                & -surf_nr_shutter_right
                & -surf_shutter_top
                & +surf_shutter_bottom
            ),
        )  # MCNP cell 35

        cell_tangential_beam_hole_air = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                -surf_tangential_beam_hole
                & +surf_outer_tank_wall
                & -surf_tangential_ambiguity_plane
                & +surf_tangential_shield_outer
            ),
        )  # MCNP cell 36

        cell_tangential_beam_hole_tube = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                -surf_tangential_beam_tube
                & +surf_tangential_beam_hole
                & +surf_outer_tank_wall
                & -surf_tangential_ambiguity_plane
                & +surf_tangential_shield_outer
            ),
        )  # MCNP cell 37

        cell_tangential_steel_tube_air = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                -surf_tangential_steel_inner
                & -surf_tangential_shutter_inner
                & -surf_tangential_ambiguity_plane
                & -surf_tangential_shield_outer
            ),
        )  # MCNP cell 38

        cell_tangential_steel_tube = openmc.Cell(
            fill=self._get_material("steel"),
            region=(
                -surf_tangential_steel_outer
                & +surf_tangential_steel_inner
                & -surf_tangential_shutter_inner
                & -surf_tangential_ambiguity_plane
                & -surf_tangential_shield_outer
            ),
        )  # MCNP cell 39

        cell_tangential_shutter_box = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                +surf_tangential_shutter_inner
                & -surf_bio_northeast_base_wall
                & +surf_tangential_shutter_right
                & -surf_tangential_shutter_left
                & -surf_shutter_top
                & +surf_shutter_bottom
            ),
        )  # MCNP cell 40

        # Beam shielding and flexible connections.
        cell_piercing_beam_shield = openmc.Cell(
            fill=self._get_material("steel"),
            region=(
                -surf_piercing_shield_outer
                & +surf_piercing_shield_inner
                & -surf_piercing_shield_right
                & +surf_piercing_shield_left
                & -surf_beam_shield_top
                & +surf_beam_shield_bottom
                & +surf_piercing_beam_hole_outer
            ),
        )  # MCNP cell 41

        cell_tof_beam_shield = openmc.Cell(
            fill=self._get_material("steel"),
            region=(
                -surf_tof_shield_outer
                & +surf_tof_shield_inner
                & -surf_tof_shield_right
                & +surf_tof_shield_left
                & -surf_beam_shield_top
                & +surf_beam_shield_bottom
                & +surf_tof_beam_tube
            ),
        )  # MCNP cell 42

        cell_nr_beam_shield = openmc.Cell(
            fill=self._get_material("steel"),
            region=(
                -surf_nr_shield_outer
                & +surf_nr_shield_inner
                & -surf_nr_shield_right
                & +surf_nr_shield_left
                & -surf_beam_shield_top
                & +surf_beam_shield_bottom
                & +surf_nr_beam_tube
            ),
        )  # MCNP cell 43

        cell_tangential_beam_shield = openmc.Cell(
            fill=self._get_material("steel"),
            region=(
                +surf_tangential_shield_outer
                & -surf_tangential_shield_inner
                & -surf_tangential_shield_right
                & +surf_tangential_shield_left
                & -surf_beam_shield_top
                & +surf_beam_shield_bottom
                & +surf_tangential_beam_tube
            ),
        )  # MCNP cell 44

        cell_piercing_bellows = openmc.Cell(
            fill=self._get_material("ss304"),
            region=(
                -surf_bellows_outer_wall
                & +surf_piercing_beam_tube
                & -surf_piercing_tank_flange
                & +surf_piercing_reflector_flange
                & +surf_piercing_beamport_ambiguity_plane
            ),
        )  # MCNP cell 45
        beamport_cells = [
            cell_piercing_graphite_hole,
            cell_piercing_flange,
            cell_piercing_reflector_flange,
            cell_piercing_beamport_end_plate,
            cell_piercing_void_behind_end_plate,
            cell_tof_beamport_air,
            cell_tof_graphite_hole,
            cell_tof_beam_tube,
            cell_tof_beamport_end_plate,
            cell_nr_beamport_air,
            cell_nr_beam_tube,
            cell_nr_beamport_end_plate,
            cell_tangential_beamport_air,
            cell_tangential_graphite_hole,
            cell_tangential_beam_tube,
            cell_tangential_beamport_end_plate,
            cell_piercing_beam_hole_air,
            cell_piercing_beam_hole_tube,
            cell_piercing_steel_tube_air,
            cell_piercing_steel_tube,
            cell_piercing_shutter_box,
            cell_tof_beam_hole_air,
            cell_tof_beam_hole_tube,
            cell_tof_steel_tube_air,
            cell_tof_steel_tube,
            cell_tof_shutter_box,
            cell_nr_beam_hole_air,
            cell_nr_beam_hole_tube,
            cell_nr_steel_tube_air,
            cell_nr_steel_tube,
            cell_nr_shutter_box,
            cell_tangential_beam_hole_air,
            cell_tangential_beam_hole_tube,
            cell_tangential_steel_tube_air,
            cell_tangential_steel_tube,
            cell_tangential_shutter_box,
            cell_piercing_beam_shield,
            cell_tof_beam_shield,
            cell_nr_beam_shield,
            cell_tangential_beam_shield,
            cell_piercing_bellows,
        ]
        # Shared surfaces for the pool tank, bulk-shield tank, and biological shield.
        # Reuse apply_mcnp_surface_transform() from the beam-port block.

        surf_inner_tank_wall = openmc.ZCylinder(r=99.06000)  # MCNP surface 738
        surf_lead_stack_wall = openmc.ZCylinder(r=104.77500)  # MCNP surface 740
        surf_tori_boundary = openmc.ZPlane(z0=-45.72000)  # MCNP surface 741
        surf_pool_tank_upper_bottom_wall = openmc.ZPlane(z0=-57.15000)  # MCNP surface 742
        surf_pool_tank_lower_bottom_wall = openmc.ZPlane(z0=-57.78500)  # MCNP surface 743
        surf_tori_cylinder = openmc.ZCylinder(r=87.63000)  # MCNP surface 744
        surf_inner_torus = openmc.ZTorus(
            x0=0.0, y0=0.0, z0=-45.72000, a=87.63000, b=11.42900, c=11.42900
        )  # MCNP surface 745
        surf_outer_torus = openmc.ZTorus(
            x0=0.0, y0=0.0, z0=-45.72000, a=87.63000, b=12.06500, c=12.06500
        )  # MCNP surface 746
        surf_pool_water_surface = openmc.ZPlane(z0=548.96000)  # MCNP surface 747
        surf_pool_tank_top = openmc.ZPlane(z0=568.96000)  # MCNP surface 748
        surf_extended_pool_water_surface = openmc.ZPlane(z0=698.96000)  # MCNP surface 749

        surf_lower_beamport_plane = openmc.ZPlane(z0=20.32000)  # MCNP surface 11
        surf_upper_beamport_plane = openmc.ZPlane(z0=38.10000)  # MCNP surface 12
        surf_core_bottom_grid = openmc.ZPlane(z0=0.0)  # MCNP surface 8

        # MCNP graphite-reflector cells 55-58. Cell 56 excludes three
        # beam holes intersecting this axial reflector layer.
        graphite_beamport_cutout = (
            (-surf_piercing_beamport & +surf_piercing_beamport_ambiguity_plane)
            | (-surf_tof_beam_hole & +surf_tof_ambiguity_plane)
            | (-surf_tangential_beam_hole & -surf_tangential_ambiguity_plane)
        )
        cell_graphite_reflector_55 = openmc.Cell(
            fill=self._get_material("graphite"),
            region=(
                +surf_reflector_wall_lower_bottom
                & -surf_lower_beamport_plane
                & -surf_reflector_wall_inner
                & +surf_outer_core_wall
            ),
            name="MCNP 55 graphite reflector",
        )
        cell_graphite_reflector_56 = openmc.Cell(
            fill=self._get_material("graphite"),
            region=(
                +surf_lower_beamport_plane
                & -surf_upper_beamport_plane
                & -surf_reflector_wall_inner
                & +surf_outer_core_wall
                & ~graphite_beamport_cutout
            ),
            name="MCNP 56 graphite reflector",
        )
        cell_graphite_reflector_57 = openmc.Cell(
            fill=self._get_material("graphite"),
            region=(
                +surf_upper_beamport_plane
                & -surf_ls_well_bottom
                & -surf_reflector_wall_inner
                & +surf_outer_core_wall
            ),
            name="MCNP 57 graphite reflector",
        )
        cell_graphite_reflector_58 = openmc.Cell(
            fill=self._get_material("graphite"),
            region=(
                +surf_ls_well_bottom
                & -surf_reflector_wall_lower_top
                & -surf_reflector_wall_inner
                & +surf_ls_well_outer_wall
            ),
            name="MCNP 58 graphite reflector",
        )
        reflector_cells.extend(
            [
                cell_graphite_reflector_55,
                cell_graphite_reflector_56,
                cell_graphite_reflector_57,
                cell_graphite_reflector_58,
            ]
        )

        surf_ffcr_lower_end_fitting = openmc.ZPlane(z0=-0.12700)  # MCNP surface 37
        surf_ffcr_upper_end_fitting = openmc.ZPlane(z0=0.42863)  # MCNP surface 38
        surf_ffcr_top_fitting_lower = openmc.ZPlane(z0=107.02925)  # MCNP surface 44
        surf_ffcr_top_fitting_upper = openmc.ZPlane(z0=110.09757)  # MCNP surface 45

        # Bulk-shield tank: surfaces 1106-1113.
        surf_bulk_base_floor = openmc.ZPlane(z0=-68.40000)  # MCNP surface 1106
        surf_bulk_west_wall = openmc.XPlane(x0=124.00000)  # MCNP surface 1107
        surf_bulk_east_wall = openmc.XPlane(x0=-124.00000)  # MCNP surface 1108
        surf_bulk_south_wall = openmc.YPlane(y0=468.50000)  # MCNP surface 1109
        surf_bulk_outer_west_wall = openmc.XPlane(x0=172.50000)  # MCNP surface 1110
        surf_bulk_outer_east_wall = openmc.XPlane(x0=-172.50000)  # MCNP surface 1111
        surf_bulk_outer_south_wall = openmc.YPlane(y0=518.50000)  # MCNP surface 1112
        surf_bulk_water_surface = openmc.ZPlane(z0=284.60000)  # MCNP surface 1113

        # Biological shield: surfaces are defined directly.
        surf_bio_south_deck_wall = openmc.YPlane(y0=350.06097)  # MCNP surface 1114
        surf_reactor_deck = openmc.ZPlane(z0=583.96000)  # MCNP surface 1115
        surf_reactor_deck_hole_west = openmc.XPlane(x0=-127.50000)  # MCNP surface 1116
        surf_reactor_deck_hole_south = openmc.YPlane(y0=127.50000)  # MCNP surface 1117
        surf_reactor_deck_hole_east = openmc.XPlane(x0=127.50000)  # MCNP surface 1118
        surf_reactor_deck_hole_north = openmc.YPlane(y0=-127.50000)  # MCNP surface 1119

        surf_bio_lower_base = openmc.ZPlane(z0=-157.78500)  # MCNP surface 1120
        surf_bio_lower_step = openmc.ZPlane(z0=304.60000)  # MCNP surface 1121
        surf_bio_north_base_wall = openmc.YPlane(y0=-350.06097)  # MCNP surface 1122
        surf_bio_west_base_wall = openmc.XPlane(x0=350.06097)  # MCNP surface 1124
        surf_bio_south_base_wall = openmc.YPlane(y0=423.50000)  # MCNP surface 1126
        surf_bio_east_base_wall = openmc.XPlane(x0=-350.06097)  # MCNP surface 1128

        # Surfaces 1123, 1125, 1127, and 1129 are defined in the beam-port block:
        # surf_bio_northwest_base_wall
        # surf_bio_southwest_base_wall
        # surf_bio_southeast_base_wall
        # surf_bio_northeast_base_wall

        TR21_TRANSLATION = np.array([78.90768363, -190.50000, 0.0])
        TR22_TRANSLATION = np.array([78.90768363, 190.50000, 0.0])
        TR23_TRANSLATION = np.array([-78.90768363, 190.50000, 0.0])
        TR24_TRANSLATION = np.array([-78.90768363, -190.50000, 0.0])

        TR21_ROTATION = np.array(
            [
                [0.7071067811865476, 0.7071067811865476, 0.0],
                [-0.7071067811865475, 0.7071067811865476, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        TR23_ROTATION = np.array(
            [
                [-0.7071067811865475, -0.7071067811865475, 0.0],
                [0.7071067811865476, -0.7071067811865475, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        surf_bio_north_top_wall = openmc.YPlane(y0=-190.50000)  # MCNP surface 1130
        surf_bio_northwest_top_wall = apply_mcnp_surface_transform(
            openmc.XPlane(x0=0.0), TR21_TRANSLATION, TR21_ROTATION
        )  # 1131
        surf_bio_west_top_wall = openmc.XPlane(x0=190.50000)  # MCNP surface 1132
        surf_bio_southwest_top_wall = apply_mcnp_surface_transform(
            openmc.YPlane(y0=0.0), TR22_TRANSLATION, TR21_ROTATION
        )  # 1133

        # Reuse surf_south_top_wall for surface 1134.
        surf_bio_southeast_top_wall = apply_mcnp_surface_transform(
            openmc.XPlane(x0=0.0), TR23_TRANSLATION, TR23_ROTATION
        )  # 1135
        surf_bio_east_top_wall = openmc.XPlane(x0=-190.50000)  # MCNP surface 1136
        surf_bio_northeast_top_wall = apply_mcnp_surface_transform(
            openmc.YPlane(y0=0.0), TR24_TRANSLATION, TR23_ROTATION
        )  # 1137

        surf_bio_horizontal_plane = openmc.YPlane(y0=0.0)  # MCNP surface 1138
        surf_bio_vertical_plane = openmc.XPlane(x0=0.0)  # MCNP surface 1139

        # MCNP cells 2-4 fill the air surrounding the reactor structure.
        # The parent world sphere supplies the shared -99990 boundary.
        base_shield_outside = (
            -surf_bio_north_base_wall
            | +surf_bio_northwest_base_wall
            | +surf_bio_west_base_wall
            | +surf_bio_southwest_base_wall
            | +surf_bio_south_base_wall
            | +surf_bio_southeast_base_wall
            | -surf_bio_east_base_wall
            | +surf_bio_northeast_base_wall
        )
        bulk_tank_outside = (
            -surf_bio_south_base_wall
            | -surf_bulk_outer_east_wall
            | +surf_bulk_outer_south_wall
            | +surf_bulk_outer_west_wall
        )
        top_shield_outside = (
            -surf_bio_north_top_wall
            | +surf_bio_northwest_top_wall
            | +surf_bio_west_top_wall
            | +surf_bio_southwest_top_wall
            | +surf_south_top_wall
            | +surf_bio_southeast_top_wall
            | -surf_bio_east_top_wall
            | +surf_bio_northeast_top_wall
        )
        deck_shield_outside = (
            -surf_bio_north_base_wall
            | +surf_bio_northwest_base_wall
            | +surf_bio_west_base_wall
            | +surf_bio_southwest_base_wall
            | +surf_bio_south_deck_wall
            | +surf_bio_southeast_base_wall
            | -surf_bio_east_base_wall
            | +surf_bio_northeast_base_wall
        )
        cell_surrounding_air_lower = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                -surf_bio_lower_base
                | (
                    +surf_bio_lower_base
                    & -surf_bio_lower_step
                    & base_shield_outside
                    & bulk_tank_outside
                )
            ),
            name="MCNP 2 surrounding air lower",
        )
        cell_surrounding_air_upper_shield = openmc.Cell(
            fill=self._get_material("air"),
            region=(+surf_bio_lower_step & -surf_pool_tank_top & top_shield_outside),
            name="MCNP 3 surrounding air upper shield",
        )
        cell_surrounding_air_top = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                +surf_top_of_extended_tank
                | (
                    -surf_top_of_extended_tank
                    & +surf_extended_pool_water_surface
                    & +surf_outer_tank_wall
                )
                | (-surf_extended_pool_water_surface & +surf_reactor_deck & +surf_lead_stack_wall)
                | (-surf_reactor_deck & +surf_pool_tank_top & deck_shield_outside)
            ),
            name="MCNP 4 surrounding air top",
        )

        # Shared TR500 surfaces bound water around FFCRs, control rods, and irradiation tubes.
        TR500_TRANSLATION = np.zeros(3)
        TR500_ROTATION = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        def make_tr500_cylinder(x0: float, y0: float, radius: float) -> openmc.Surface:
            """Create a Z-cylinder transformed by MCNP TR500.

            Parameters are local MCNP center coordinates and cylinder radius in cm.
            Returns the corresponding global OpenMC surface."""
            return apply_mcnp_surface_transform(
                openmc.ZCylinder(x0=x0, y0=y0, r=radius),
                TR500_TRANSLATION,
                TR500_ROTATION,
            )

        # FFCR D-2: 161, 162, 165, 166
        surf_ffcr_d2_end_fitting = make_tr500_cylinder(-3.69792, -10.67498, 1.11125)  # 161
        surf_ffcr_d2_top_fitting = make_tr500_cylinder(-3.69792, -10.67498, 1.19063)  # 162
        surf_ffcr_d2_clad = make_tr500_cylinder(-3.69792, -10.67498, 1.72085)  # 165
        surf_ffcr_d2_extension_rod = make_tr500_cylinder(-3.69792, -10.67498, 1.32085)  # 166

        # Control rod D5: surfaces 175, 176, 178, and 179.
        surf_control_rod_d5_end_fitting = make_tr500_cylinder(-11.09377, -2.13500, 1.11125)  # 175
        surf_control_rod_d5_top_fitting = make_tr500_cylinder(-11.09377, -2.13500, 1.19063)  # 176
        surf_control_rod_d5_clad = make_tr500_cylinder(-11.09377, -2.13500, 1.72085)  # 178
        surf_control_rod_d5_extension_rod = make_tr500_cylinder(-11.09377, -2.13500, 1.32085)  # 179

        # Control rod D9: surfaces 192, 193, 195, and 196.
        surf_control_rod_d9_end_fitting = make_tr500_cylinder(-3.69792, 10.67498, 1.11125)  # 192
        surf_control_rod_d9_top_fitting = make_tr500_cylinder(-3.69792, 10.67498, 1.19063)  # 193
        surf_control_rod_d9_clad = make_tr500_cylinder(-3.69792, 10.67498, 1.72085)  # 195
        surf_control_rod_d9_extension_rod = make_tr500_cylinder(-3.69792, 10.67498, 1.32085)  # 196

        # FFCR D-13: 209, 210, 213, 214
        surf_ffcr_d13_end_fitting = make_tr500_cylinder(11.09377, 6.40499, 1.11125)  # 209
        surf_ffcr_d13_top_fitting = make_tr500_cylinder(11.09377, 6.40499, 1.19063)  # 210
        surf_ffcr_d13_clad = make_tr500_cylinder(11.09377, 6.40499, 1.72085)  # 213
        surf_ffcr_d13_extension_rod = make_tr500_cylinder(11.09377, 6.40499, 1.32085)  # 214

        # FFCR D-16: 223, 224, 227, 228
        surf_ffcr_d16_end_fitting = make_tr500_cylinder(11.09377, -6.40499, 1.11125)  # 223
        surf_ffcr_d16_top_fitting = make_tr500_cylinder(11.09377, -6.40499, 1.19063)  # 224
        surf_ffcr_d16_clad = make_tr500_cylinder(11.09377, -6.40499, 1.72085)  # 227
        surf_ffcr_d16_extension_rod = make_tr500_cylinder(11.09377, -6.40499, 1.32085)  # 228

        # Irradiation-tube outer walls: surfaces 84, 478, 501, and 544.
        surf_irradiation_tube_a1_outer = openmc.ZCylinder(r=1.58750)  # MCNP surface 84
        surf_irradiation_tube_g10_outer = make_tr500_cylinder(-22.18754, 0.00000, 1.58750)  # 478
        surf_irradiation_tube_g17_outer = make_tr500_cylinder(-7.39585, 21.34997, 1.58750)  # 501
        surf_irradiation_tube_g32_outer = make_tr500_cylinder(18.48962, -14.94498, 1.58750)  # 544
        # Pool tank.
        cell_pool_tank_bottom_plate = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_pool_tank_lower_bottom_wall
                & -surf_pool_tank_upper_bottom_wall
                & -surf_tori_cylinder
            ),
        )  # MCNP cell 92

        cell_pool_tank_torus_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_tori_cylinder & -surf_tori_boundary & +surf_inner_torus & -surf_outer_torus
            ),
        )  # MCNP cell 93

        cell_extended_pool_lead_shield = openmc.Cell(
            fill=self._get_material("lead"),
            region=(
                +surf_pool_tank_top
                & -surf_extended_pool_water_surface
                & -surf_lead_stack_wall
                & +surf_outer_tank_wall
            ),
        )  # MCNP cell 94

        cell_extended_tank_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_tori_boundary
                & -surf_top_of_extended_tank
                & +surf_inner_tank_wall
                & -surf_outer_tank_wall
                & ~(
                    ((-surf_piercing_beamport) & (+surf_piercing_beamport_ambiguity_plane))
                    | ((-surf_tof_beam_tube) & (+surf_tof_ambiguity_plane))
                    | ((-surf_nr_beam_tube) & (+surf_nr_ambiguity_plane))
                    | ((-surf_tangential_beam_tube) & (-surf_tangential_ambiguity_plane))
                )
                & ~(
                    (+surf_thermalizing_outer_left_wall)
                    & (-surf_thermalizing_outer_right_wall)
                    & (+surf_thermalizing_lower_bottom_wall)
                    & (-surf_thermalizing_upper_top_wall)
                    & (+surf_thermalizing_ambiguity_plane)
                )
                & ~(
                    (+surf_thermal_column_outer_left_wall)
                    & (-surf_thermal_column_outer_right_wall)
                    & (+surf_thermal_column_lower_bottom_wall)
                    & (-surf_thermal_column_upper_top_wall)
                    & (-surf_thermal_column_ambiguity_plane)
                )
            ),
        )  # MCNP cell 95

        # Bulk-shield tank.
        cell_bulk_shielding_water = openmc.Cell(
            fill=self._get_material("water"),
            region=(
                +surf_bulk_base_floor
                & +surf_south_top_wall
                & -surf_bulk_west_wall
                & -surf_bulk_south_wall
                & +surf_bulk_east_wall
                & -surf_bulk_water_surface
            ),
        )  # MCNP cell 96

        cell_bulk_shielding_air = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                +surf_south_top_wall
                & -surf_bulk_west_wall
                & -surf_bulk_south_wall
                & +surf_bulk_east_wall
                & +surf_bulk_water_surface
                & -surf_bio_lower_step
            ),
        )  # MCNP cell 97

        # Biological shield.
        cell_bio_northwest_concrete = openmc.Cell(
            fill=self._get_material("concrete"),
            region=(
                +surf_bio_lower_base
                & -surf_bio_lower_step
                & -surf_bio_horizontal_plane
                & +surf_bio_north_base_wall
                & -surf_bio_northwest_base_wall
                & -surf_bio_west_base_wall
                & +surf_bio_east_base_wall
                & -surf_bio_northeast_base_wall
                & +surf_outer_tank_wall
                & ~((-surf_nr_beam_tube) & (+surf_nr_ambiguity_plane) & (-surf_nr_shield_outer))
                & ~(
                    (-surf_tangential_beam_tube)
                    & (-surf_tangential_ambiguity_plane)
                    & (+surf_tangential_shield_outer)
                )
                & ~((-surf_nr_steel_outer) & (-surf_nr_shutter_inner) & (+surf_nr_shield_outer))
                & ~(
                    (-surf_tangential_steel_outer)
                    & (-surf_tangential_shutter_inner)
                    & (-surf_tangential_shield_outer)
                )
                & ~(
                    (+surf_thermal_column_graphite_back_wall)
                    & (+surf_thermal_column_upper_bottom_wall)
                    & (-surf_thermal_column_lower_top_wall)
                    & (+surf_thermal_column_inner_left_wall)
                    & (-surf_thermal_column_inner_right_wall)
                )
                & ~(
                    (-surf_nr_shield_outer)
                    & (+surf_nr_shield_inner)
                    & (-surf_nr_shield_right)
                    & (+surf_nr_shield_left)
                    & (-surf_beam_shield_top)
                    & (+surf_beam_shield_bottom)
                )
                & ~(
                    (+surf_tangential_shield_outer)
                    & (-surf_tangential_shield_inner)
                    & (-surf_tangential_shield_right)
                    & (+surf_tangential_shield_left)
                    & (-surf_beam_shield_top)
                    & (+surf_beam_shield_bottom)
                )
                & ~(
                    (+surf_nr_shutter_inner)
                    & (+surf_nr_shutter_left)
                    & (-surf_nr_shutter_right)
                    & (-surf_shutter_top)
                    & (+surf_shutter_bottom)
                )
                & ~(
                    (+surf_tangential_shutter_inner)
                    & (+surf_tangential_shutter_right)
                    & (-surf_tangential_shutter_left)
                    & (-surf_shutter_top)
                    & (+surf_shutter_bottom)
                )
            ),
        )  # MCNP cell 98

        cell_bio_southwest_concrete = openmc.Cell(
            fill=self._get_material("concrete"),
            region=(
                +surf_bio_lower_base
                & -surf_bio_lower_step
                & +surf_bio_horizontal_plane
                & -surf_bio_vertical_plane
                & (
                    (
                        (-surf_bio_south_base_wall)
                        & (-surf_bio_southeast_base_wall)
                        & (+surf_bio_east_base_wall)
                        & (+surf_outer_tank_wall)
                    )
                    | (
                        (+surf_bio_south_base_wall)
                        & (-surf_bulk_outer_south_wall)
                        & (+surf_bulk_outer_east_wall)
                    )
                )
                & ~(
                    (-surf_piercing_beam_hole_outer)
                    & (+surf_piercing_beamport_ambiguity_plane)
                    & (-surf_piercing_shield_outer)
                )
                & ~(
                    (-surf_piercing_steel_outer)
                    & (-surf_piercing_shutter_inner)
                    & (+surf_piercing_shield_outer)
                )
                & ~(
                    (-surf_thermalizing_lower_top_wall)
                    & (+surf_thermalizing_upper_bottom_wall)
                    & (+surf_thermalizing_inner_left_wall)
                    & (-surf_south_top_wall)
                )
                & ~(
                    (+surf_south_top_wall)
                    & (-surf_bulk_south_wall)
                    & (+surf_bulk_east_wall)
                    & (+surf_bulk_base_floor)
                )
                & ~(
                    (-surf_piercing_shield_outer)
                    & (+surf_piercing_shield_inner)
                    & (-surf_piercing_shield_right)
                    & (+surf_piercing_shield_left)
                    & (-surf_beam_shield_top)
                    & (+surf_beam_shield_bottom)
                )
                & ~(
                    (+surf_piercing_shutter_inner)
                    & (+surf_piercing_shutter_right)
                    & (-surf_piercing_shutter_left)
                    & (-surf_shutter_top)
                    & (+surf_shutter_bottom)
                )
            ),
        )  # MCNP cell 99

        cell_bio_southeast_concrete = openmc.Cell(
            fill=self._get_material("concrete"),
            region=(
                +surf_bio_lower_base
                & -surf_bio_lower_step
                & +surf_bio_horizontal_plane
                & +surf_bio_vertical_plane
                & (
                    (
                        (-surf_bio_west_base_wall)
                        & (-surf_bio_southwest_base_wall)
                        & (-surf_bio_south_base_wall)
                        & (+surf_outer_tank_wall)
                    )
                    | (
                        (+surf_bio_south_base_wall)
                        & (-surf_bulk_outer_west_wall)
                        & (-surf_bulk_outer_south_wall)
                    )
                )
                & ~((-surf_tof_beam_tube) & (+surf_tof_ambiguity_plane) & (-surf_tof_shield_outer))
                & ~((-surf_tof_steel_outer) & (-surf_tof_shutter_inner) & (+surf_tof_shield_outer))
                & ~(
                    (-surf_thermalizing_lower_top_wall)
                    & (+surf_thermalizing_upper_bottom_wall)
                    & (-surf_thermalizing_inner_right_wall)
                    & (-surf_south_top_wall)
                )
                & ~(
                    (+surf_south_top_wall)
                    & (-surf_bulk_west_wall)
                    & (-surf_bulk_south_wall)
                    & (+surf_bulk_base_floor)
                )
                & ~(
                    (-surf_tof_shield_outer)
                    & (+surf_tof_shield_inner)
                    & (-surf_tof_shield_right)
                    & (+surf_tof_shield_left)
                    & (-surf_beam_shield_top)
                    & (+surf_beam_shield_bottom)
                )
                & ~(
                    (+surf_tof_shutter_inner)
                    & (+surf_tof_shutter_left)
                    & (-surf_tof_shutter_right)
                    & (-surf_shutter_top)
                    & (+surf_shutter_bottom)
                )
            ),
        )  # MCNP cell 100

        cell_bio_torus_concrete = openmc.Cell(
            fill=self._get_material("concrete"),
            region=(
                -surf_tori_boundary
                & +surf_pool_tank_lower_bottom_wall
                & -surf_outer_tank_wall
                & +surf_outer_torus
                & +surf_tori_cylinder
            ),
        )  # MCNP cell 101

        cell_bio_lower_concrete = openmc.Cell(
            fill=self._get_material("concrete"),
            region=(
                +surf_bio_lower_base & -surf_pool_tank_lower_bottom_wall & -surf_outer_tank_wall
            ),
        )  # MCNP cell 102

        cell_bio_upper_concrete = openmc.Cell(
            fill=self._get_material("concrete"),
            region=(
                +surf_bio_lower_step
                & -surf_pool_tank_top
                & +surf_bio_north_top_wall
                & -surf_bio_northwest_top_wall
                & -surf_bio_west_top_wall
                & -surf_bio_southwest_top_wall
                & -surf_south_top_wall
                & -surf_bio_southeast_top_wall
                & +surf_bio_east_top_wall
                & -surf_bio_northeast_top_wall
                & +surf_outer_tank_wall
            ),
        )  # MCNP cell 103

        cell_reactor_deck_concrete = openmc.Cell(
            fill=self._get_material("concrete"),
            region=(
                +surf_pool_tank_top
                & -surf_reactor_deck
                & +surf_bio_north_base_wall
                & -surf_bio_northwest_base_wall
                & -surf_bio_west_base_wall
                & -surf_bio_southwest_base_wall
                & -surf_bio_south_deck_wall
                & -surf_bio_southeast_base_wall
                & +surf_bio_east_base_wall
                & -surf_bio_northeast_base_wall
                & (
                    (-surf_reactor_deck_hole_north)
                    | (+surf_reactor_deck_hole_east)
                    | (+surf_reactor_deck_hole_south)
                    | (-surf_reactor_deck_hole_west)
                )
            ),
        )  # MCNP cell 104

        cell_reactor_deck_air_hole = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                +surf_pool_tank_top
                & -surf_reactor_deck
                & +surf_lead_stack_wall
                & +surf_reactor_deck_hole_north
                & -surf_reactor_deck_hole_east
                & -surf_reactor_deck_hole_south
                & +surf_reactor_deck_hole_west
            ),
        )  # MCNP cell 105

        cell_extended_tank_air = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                -surf_inner_tank_wall
                & +surf_extended_pool_water_surface
                & -surf_top_of_extended_tank
                & +surf_shaft_assembly_1
                & +surf_specimen_removal_assembly_1
            ),
        )  # MCNP cell 106

        cell_extended_pool_water = openmc.Cell(
            fill=self._get_material("water"),
            region=(
                -surf_inner_tank_wall
                & +surf_pool_water_surface
                & -surf_extended_pool_water_surface
                & +surf_upper_part_rotary_rack_1  # MCNP 630: r > 26.22550 cm
                & +surf_shaft_assembly_1
                & +surf_specimen_removal_assembly_1
                & +surf_ffcr_d2_clad
                & +surf_control_rod_d5_clad
                & +surf_control_rod_d9_clad
                & +surf_ffcr_d13_clad
                & +surf_ffcr_d16_clad
                & +surf_irradiation_tube_a1_outer
                & +surf_irradiation_tube_g10_outer
                & +surf_irradiation_tube_g17_outer
                & +surf_irradiation_tube_g32_outer
            ),
        )  # MCNP cell 107
        cell_extended_pool_water.region = (
            cell_extended_pool_water.region
            & +surf_upper_part_rotary_rack_1  # Only r > 26.22550 cm.
        )

        # Water outside the core.
        cell_pool_water_below_tank = openmc.Cell(
            fill=self._get_material("water"),
            region=(+surf_pool_tank_upper_bottom_wall & -surf_tori_boundary & -surf_tori_cylinder),
        )  # MCNP cell 108

        cell_pool_water_torus = openmc.Cell(
            fill=self._get_material("water"),
            region=(+surf_tori_cylinder & -surf_tori_boundary & -surf_inner_torus),
        )  # MCNP cell 109

        cell_outer_water_lower = openmc.Cell(
            fill=self._get_material("water"),
            region=(
                +surf_tori_boundary
                & -surf_lower_beamport_plane
                & -surf_inner_tank_wall
                & +surf_outer_reflector
                & ~(
                    (-surf_bellows_outer_wall)
                    & (-surf_piercing_tank_flange)
                    & (+surf_piercing_reflector_flange)
                    & (+surf_piercing_beamport_ambiguity_plane)
                )
                & ~(
                    (+surf_thermalizing_inner_front_wall)
                    & (+surf_thermalizing_outer_left_wall)
                    & (-surf_thermalizing_outer_right_wall)
                    & (+surf_thermalizing_lower_bottom_wall)
                    & (+surf_thermalizing_ambiguity_plane)
                )
                & ~(
                    (+surf_thermal_column_inner_front_wall)
                    & (+surf_thermal_column_lower_bottom_wall)
                    & (-surf_thermal_column_outer_left_diagonal)
                    & (-surf_thermal_column_outer_right_diagonal)
                    & (+surf_thermal_column_outer_left_wall)
                    & (-surf_thermal_column_outer_right_wall)
                    & (-surf_thermal_column_ambiguity_plane)
                )
            ),
        )  # MCNP cell 110

        cell_outer_water_beamport_level = openmc.Cell(
            fill=self._get_material("water"),
            region=(
                +surf_lower_beamport_plane
                & -surf_upper_beamport_plane
                & -surf_inner_tank_wall
                & +surf_outer_reflector
                & ~(
                    ((-surf_piercing_beamport) & (+surf_piercing_beamport_ambiguity_plane))
                    | ((-surf_tof_beam_tube) & (+surf_tof_ambiguity_plane))
                    | ((-surf_nr_beam_tube) & (+surf_nr_ambiguity_plane))
                    | ((-surf_tangential_beam_tube) & (-surf_tangential_ambiguity_plane))
                )
                & ~(
                    (-surf_bellows_outer_wall)
                    & (-surf_piercing_tank_flange)
                    & (+surf_piercing_reflector_flange)
                    & (+surf_piercing_beamport_ambiguity_plane)
                )
                & ~(
                    (+surf_thermalizing_inner_front_wall)
                    & (+surf_thermalizing_outer_left_wall)
                    & (-surf_thermalizing_outer_right_wall)
                    & (+surf_thermalizing_ambiguity_plane)
                )
                & ~(
                    (+surf_thermal_column_inner_front_wall)
                    & (-surf_thermal_column_outer_left_diagonal)
                    & (-surf_thermal_column_outer_right_diagonal)
                    & (+surf_thermal_column_outer_left_wall)
                    & (-surf_thermal_column_outer_right_wall)
                    & (-surf_thermal_column_ambiguity_plane)
                )
            ),
        )  # MCNP cell 111

        cell_outer_water_upper = openmc.Cell(
            fill=self._get_material("water"),
            region=(
                +surf_upper_beamport_plane
                & -surf_pool_water_surface
                & -surf_inner_tank_wall
                & +surf_outer_reflector
                & ~(
                    (-surf_bellows_outer_wall)
                    & (-surf_piercing_tank_flange)
                    & (+surf_piercing_reflector_flange)
                    & (+surf_piercing_beamport_ambiguity_plane)
                )
                & ~(
                    (+surf_thermalizing_inner_front_wall)
                    & (+surf_thermalizing_outer_left_wall)
                    & (-surf_thermalizing_outer_right_wall)
                    & (-surf_thermalizing_upper_top_wall)
                    & (+surf_thermalizing_ambiguity_plane)
                )
                & ~(
                    (+surf_thermal_column_inner_front_wall)
                    & (-surf_thermal_column_upper_top_wall)
                    & (-surf_thermal_column_outer_left_diagonal)
                    & (-surf_thermal_column_outer_right_diagonal)
                    & (+surf_thermal_column_outer_left_wall)
                    & (-surf_thermal_column_outer_right_wall)
                    & (-surf_thermal_column_ambiguity_plane)
                )
            ),
        )  # MCNP cell 112

        cell_water_below_reflector = openmc.Cell(
            fill=self._get_material("water"),
            region=(
                +surf_tori_boundary
                & -surf_reflector_bottom
                & -surf_outer_reflector
                & +surf_outer_core_wall
            ),
        )  # MCNP cell 113

        cell_water_below_core_wall = openmc.Cell(
            fill=self._get_material("water"),
            region=(+surf_tori_boundary & -surf_core_wall_bottom & -surf_outer_core_wall),
        )  # MCNP cell 114

        cell_core_bottom_water = openmc.Cell(
            fill=self._get_material("water"),
            region=(
                +surf_safety_plate
                & -surf_inner_core_wall
                & -surf_core_bottom_grid
                & +surf_ffcr_d2_clad
                & +surf_control_rod_d5_clad
                & +surf_control_rod_d9_clad
                & +surf_ffcr_d13_clad
                & +surf_ffcr_d16_clad
            ),
        )  # MCNP cell 115

        cell_ffcr_d2_bottom_annulus = openmc.Cell(
            fill=self._get_material("water"),
            region=(
                +surf_safety_plate
                & -surf_ffcr_lower_end_fitting
                & -surf_ffcr_d2_clad
                & +surf_ffcr_d2_end_fitting
            ),
        )  # MCNP cell 116

        cell_control_rod_d5_bottom_annulus = openmc.Cell(
            fill=self._get_material("water"),
            region=(
                +surf_safety_plate
                & -surf_ffcr_lower_end_fitting
                & -surf_control_rod_d5_clad
                & +surf_control_rod_d5_end_fitting
            ),
        )  # MCNP cell 117

        cell_control_rod_d9_bottom_annulus = openmc.Cell(
            fill=self._get_material("water"),
            region=(
                +surf_safety_plate
                & -surf_ffcr_lower_end_fitting
                & -surf_control_rod_d9_clad
                & +surf_control_rod_d9_end_fitting
            ),
        )  # MCNP cell 118

        cell_ffcr_d13_bottom_annulus = openmc.Cell(
            fill=self._get_material("water"),
            region=(
                +surf_safety_plate
                & -surf_ffcr_lower_end_fitting
                & -surf_ffcr_d13_clad
                & +surf_ffcr_d13_end_fitting
            ),
        )  # MCNP cell 119

        cell_ffcr_d16_bottom_annulus = openmc.Cell(
            fill=self._get_material("water"),
            region=(
                +surf_safety_plate
                & -surf_ffcr_lower_end_fitting
                & -surf_ffcr_d16_clad
                & +surf_ffcr_d16_end_fitting
            ),
        )  # MCNP cell 120

        cell_ffcr_d2_lower_water = openmc.Cell(
            fill=self._get_material("water"),
            region=(+surf_safety_plate & -surf_ffcr_lower_end_fitting & -surf_ffcr_d2_end_fitting),
        )  # MCNP cell 121

        cell_control_rod_d5_lower_water = openmc.Cell(
            fill=self._get_material("water"),
            region=(
                +surf_safety_plate & -surf_ffcr_lower_end_fitting & -surf_control_rod_d5_end_fitting
            ),
        )  # MCNP cell 122

        cell_control_rod_d9_lower_water = openmc.Cell(
            fill=self._get_material("water"),
            region=(
                +surf_safety_plate & -surf_ffcr_lower_end_fitting & -surf_control_rod_d9_end_fitting
            ),
        )  # MCNP cell 123

        cell_ffcr_d13_lower_water = openmc.Cell(
            fill=self._get_material("water"),
            region=(+surf_safety_plate & -surf_ffcr_lower_end_fitting & -surf_ffcr_d13_end_fitting),
        )  # MCNP cell 124

        cell_ffcr_d16_lower_water = openmc.Cell(
            fill=self._get_material("water"),
            region=(+surf_safety_plate & -surf_ffcr_lower_end_fitting & -surf_ffcr_d16_end_fitting),
        )  # MCNP cell 125

        cell_water_above_reflector = openmc.Cell(
            fill=self._get_material("water"),
            region=(
                +surf_reflector_top
                & -surf_pool_water_surface
                & -surf_outer_reflector
                & +surf_outer_part_rotary_rack_2
            ),
        )  # MCNP cell 126

        cell_water_above_rotary_rack = openmc.Cell(
            fill=self._get_material("water"),
            region=(
                +surf_top_wall_rotary_rack_1
                & -surf_pool_water_surface
                & -surf_outer_part_rotary_rack_2
                & +surf_upper_part_rotary_rack_1
                & +surf_shaft_assembly_1
                & +surf_specimen_removal_assembly_1
            ),
        )  # MCNP cell 127
        # Cells 128-138 are omitted, matching their removal from the reference notebook.
        pool_shield_and_water_cells = [
            cell_surrounding_air_lower,
            cell_surrounding_air_upper_shield,
            cell_surrounding_air_top,
            cell_pool_tank_bottom_plate,
            cell_pool_tank_torus_wall,
            cell_extended_pool_lead_shield,
            cell_extended_tank_wall,
            cell_bulk_shielding_water,
            cell_bulk_shielding_air,
            cell_bio_northwest_concrete,
            cell_bio_southwest_concrete,
            cell_bio_southeast_concrete,
            cell_bio_torus_concrete,
            cell_bio_lower_concrete,
            cell_bio_upper_concrete,
            cell_reactor_deck_concrete,
            cell_reactor_deck_air_hole,
            cell_extended_tank_air,
            cell_extended_pool_water,
            cell_pool_water_below_tank,
            cell_pool_water_torus,
            cell_outer_water_lower,
            cell_outer_water_beamport_level,
            cell_outer_water_upper,
            cell_water_below_reflector,
            cell_water_below_core_wall,
            cell_core_bottom_water,
            cell_ffcr_d2_bottom_annulus,
            cell_control_rod_d5_bottom_annulus,
            cell_control_rod_d9_bottom_annulus,
            cell_ffcr_d13_bottom_annulus,
            cell_ffcr_d16_bottom_annulus,
            cell_ffcr_d2_lower_water,
            cell_control_rod_d5_lower_water,
            cell_control_rod_d9_lower_water,
            cell_ffcr_d13_lower_water,
            cell_ffcr_d16_lower_water,
            cell_water_above_reflector,
            cell_water_above_rotary_rack,
        ]
        surf_bottom_grid_plate_top = openmc.ZPlane(z0=1.90500)  # MCNP 9
        surf_top_grid_plate_bottom = openmc.ZPlane(z0=66.99250)  # MCNP 15

        # Surfaces 60 and 61: cylindrical shim walls.
        surf_outer_outer_shim_wall = openmc.ZCylinder(r=26.36520)  # MCNP 60
        surf_inner_outer_shim_wall = openmc.ZCylinder(r=25.88260)  # MCNP 61

        # cos(30 degrees).
        _COS30 = 0.8660254037844386

        # Surfaces 62-67: inner faces of the inner shim.
        # Plane coefficients already include transforms TR401-TR406.

        surf_shim_inner_inner_se = openmc.Plane(
            a=+_COS30, b=-0.5, c=0.0, d=24.83104
        )  # MCNP 62 / TR401

        surf_shim_inner_inner_s = openmc.Plane(a=0.0, b=-1.0, c=0.0, d=24.83104)  # MCNP 63 / TR402

        surf_shim_inner_inner_sw = openmc.Plane(
            a=-_COS30, b=-0.5, c=0.0, d=24.83104
        )  # MCNP 64 / TR403

        surf_shim_inner_inner_nw = openmc.Plane(
            a=-_COS30, b=+0.5, c=0.0, d=24.83104
        )  # MCNP 65 / TR404

        surf_shim_inner_inner_n = openmc.Plane(a=0.0, b=+1.0, c=0.0, d=24.83104)  # MCNP 66 / TR405

        surf_shim_inner_inner_ne = openmc.Plane(
            a=+_COS30, b=+0.5, c=0.0, d=24.83104
        )  # MCNP 67 / TR406

        # Surfaces 68-73: outer faces of the inner shim.

        surf_shim_outer_inner_se = openmc.Plane(
            a=+_COS30, b=-0.5, c=0.0, d=25.31364
        )  # MCNP 68 / TR401

        surf_shim_outer_inner_s = openmc.Plane(a=0.0, b=-1.0, c=0.0, d=25.31364)  # MCNP 69 / TR402

        surf_shim_outer_inner_sw = openmc.Plane(
            a=-_COS30, b=-0.5, c=0.0, d=25.31364
        )  # MCNP 70 / TR403

        surf_shim_outer_inner_nw = openmc.Plane(
            a=-_COS30, b=+0.5, c=0.0, d=25.31364
        )  # MCNP 71 / TR404

        surf_shim_outer_inner_n = openmc.Plane(a=0.0, b=+1.0, c=0.0, d=25.31364)  # MCNP 72 / TR405

        surf_shim_outer_inner_ne = openmc.Plane(
            a=+_COS30, b=+0.5, c=0.0, d=25.31364
        )  # MCNP 73 / TR406
        cell_shim_se_outer_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_bottom_grid_plate_top
                & -surf_top_grid_plate_bottom
                & -surf_outer_outer_shim_wall
                & +surf_inner_outer_shim_wall
                & +surf_shim_inner_inner_se
            ),
        )  # MCNP cell 141

        cell_shim_se_inner_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_bottom_grid_plate_top
                & -surf_top_grid_plate_bottom
                & -surf_inner_outer_shim_wall
                & +surf_shim_inner_inner_se
                & -surf_shim_outer_inner_se
            ),
        )  # MCNP cell 142

        cell_shim_se_air = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                +surf_bottom_grid_plate_top
                & -surf_top_grid_plate_bottom
                & -surf_inner_outer_shim_wall
                & +surf_shim_outer_inner_se
            ),
        )  # MCNP cell 143

        # South shim: TR402, surfaces 63 and 69.
        cell_shim_s_outer_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_bottom_grid_plate_top
                & -surf_top_grid_plate_bottom
                & -surf_outer_outer_shim_wall
                & +surf_inner_outer_shim_wall
                & +surf_shim_inner_inner_s
            ),
        )  # MCNP cell 144

        cell_shim_s_inner_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_bottom_grid_plate_top
                & -surf_top_grid_plate_bottom
                & -surf_inner_outer_shim_wall
                & +surf_shim_inner_inner_s
                & -surf_shim_outer_inner_s
            ),
        )  # MCNP cell 145

        cell_shim_s_air = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                +surf_bottom_grid_plate_top
                & -surf_top_grid_plate_bottom
                & -surf_inner_outer_shim_wall
                & +surf_shim_outer_inner_s
            ),
        )  # MCNP cell 146

        # Southwest shim: TR403, surfaces 64 and 70.
        cell_shim_sw_outer_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_bottom_grid_plate_top
                & -surf_top_grid_plate_bottom
                & -surf_outer_outer_shim_wall
                & +surf_inner_outer_shim_wall
                & +surf_shim_inner_inner_sw
            ),
        )  # MCNP cell 147

        cell_shim_sw_inner_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_bottom_grid_plate_top
                & -surf_top_grid_plate_bottom
                & -surf_inner_outer_shim_wall
                & +surf_shim_inner_inner_sw
                & -surf_shim_outer_inner_sw
            ),
        )  # MCNP cell 148

        cell_shim_sw_air = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                +surf_bottom_grid_plate_top
                & -surf_top_grid_plate_bottom
                & -surf_inner_outer_shim_wall
                & +surf_shim_outer_inner_sw
            ),
        )  # MCNP cell 149

        # Northwest shim: TR404, surfaces 65 and 71.
        cell_shim_nw_outer_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_bottom_grid_plate_top
                & -surf_top_grid_plate_bottom
                & -surf_outer_outer_shim_wall
                & +surf_inner_outer_shim_wall
                & +surf_shim_inner_inner_nw
            ),
        )  # MCNP cell 150

        cell_shim_nw_inner_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_bottom_grid_plate_top
                & -surf_top_grid_plate_bottom
                & -surf_inner_outer_shim_wall
                & +surf_shim_inner_inner_nw
                & -surf_shim_outer_inner_nw
            ),
        )  # MCNP cell 151

        cell_shim_nw_air = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                +surf_bottom_grid_plate_top
                & -surf_top_grid_plate_bottom
                & -surf_inner_outer_shim_wall
                & +surf_shim_outer_inner_nw
            ),
        )  # MCNP cell 152

        # North shim: TR405, surfaces 66 and 72.
        cell_shim_n_outer_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_bottom_grid_plate_top
                & -surf_top_grid_plate_bottom
                & -surf_outer_outer_shim_wall
                & +surf_inner_outer_shim_wall
                & +surf_shim_inner_inner_n
            ),
        )  # MCNP cell 153

        cell_shim_n_inner_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_bottom_grid_plate_top
                & -surf_top_grid_plate_bottom
                & -surf_inner_outer_shim_wall
                & +surf_shim_inner_inner_n
                & -surf_shim_outer_inner_n
            ),
        )  # MCNP cell 154
        cell_shim_n_air = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                +surf_bottom_grid_plate_top
                & -surf_top_grid_plate_bottom
                & -surf_inner_outer_shim_wall
                & +surf_shim_outer_inner_n
            ),
        )  # MCNP cell 155

        # Northeast shim: TR406, surfaces 67 and 73.
        cell_shim_ne_outer_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_bottom_grid_plate_top
                & -surf_top_grid_plate_bottom
                & -surf_outer_outer_shim_wall
                & +surf_inner_outer_shim_wall
                & +surf_shim_inner_inner_ne
            ),
        )  # MCNP cell 156

        cell_shim_ne_inner_wall = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=(
                +surf_bottom_grid_plate_top
                & -surf_top_grid_plate_bottom
                & -surf_inner_outer_shim_wall
                & +surf_shim_inner_inner_ne
                & -surf_shim_outer_inner_ne
            ),
        )  # MCNP cell 157

        cell_shim_ne_air = openmc.Cell(
            fill=self._get_material("air"),
            region=(
                +surf_bottom_grid_plate_top
                & -surf_top_grid_plate_bottom
                & -surf_inner_outer_shim_wall
                & +surf_shim_outer_inner_ne
            ),
        )  # MCNP cell 158

        shim_cells = [
            cell_shim_se_outer_wall,
            cell_shim_se_inner_wall,
            cell_shim_se_air,
            cell_shim_s_outer_wall,
            cell_shim_s_inner_wall,
            cell_shim_s_air,
            cell_shim_sw_outer_wall,
            cell_shim_sw_inner_wall,
            cell_shim_sw_air,
            cell_shim_nw_outer_wall,
            cell_shim_nw_inner_wall,
            cell_shim_nw_air,
            cell_shim_n_outer_wall,
            cell_shim_n_inner_wall,
            cell_shim_n_air,
            cell_shim_ne_outer_wall,
            cell_shim_ne_inner_wall,
            cell_shim_ne_air,
        ]

        cell_groups = {
            "core_and_rotary_rack": core_and_rotary_rack_cells,
            "thermal_column": thermal_column_cells,
            "thermalizing_column": thermalizing_column_cells,
            "reflector_and_lazy_susan": reflector_cells,
            "beamport_shutter_shield_bellows": beamport_cells,
            "pool_shield_and_water": pool_shield_and_water_cells,
            "shim": shim_cells,
        }
        outer_core_cells = [cell for group_cells in cell_groups.values() for cell in group_cells]
        expected_cell_count = 146
        if len(outer_core_cells) != expected_cell_count:
            raise RuntimeError(
                "Jumlah fixed outer-core cells tidak sesuai Model_rapi.ipynb: "
                f"{len(outer_core_cells)} != {expected_cell_count}."
            )

        components = {
            "cell_groups": cell_groups,
            "surfaces": {
                name: surface for name, surface in locals().items() if name.startswith("surf_")
            },
        }
        return outer_core_cells, components

    def _build_gene_universe(self, gene, slot_name):
        """Convert one complete-core gene into a slot universe.

        Parameters
        ----------
        gene : tuple[str, str] or list[str]
            Pair ``(gene_type, gene_id)``, such as ``("fuel", "fuel_900001")``.
        slot_name : str
            Physical core position from A1 through G36.

        Returns
        -------
        tuple
            The slot universe and its component metadata."""
        if not isinstance(gene, (tuple, list)) or len(gene) != 2:
            raise ValueError(f"Gene pada {slot_name} harus berbentuk (jenis, nilai).")

        gene_type, gene_id = gene
        if gene_type == "fuel":
            universe, components = self._build_fuel_pin(gene_id, slot_name)
        elif gene_type == "ffcr":
            universe, components = self._build_ffcr(gene_id, slot_name)
        elif gene_type == "control_rod":
            universe = self._build_control_rod(slot_name)
            components = {}
        elif gene_type == "irradiation_tube":
            universe = self._build_irradiation_tube(slot_name)
            components = {}
        elif gene_type == "empty_hole":
            universe = self._build_empty_hole(slot_name)
            components = {}
        elif gene_type == "support_structure":
            universe = self._build_support_structure(slot_name)
            components = {}
        else:
            raise ValueError(f"Jenis gene {gene_type!r} pada slot {slot_name} tidak dikenal.")

        components.update(
            {
                "slot_name": slot_name,
                "slot_type": gene_type,
                "gene_id": gene_id,
                "universe": universe,
                "universe_id": universe.id,
            }
        )
        return universe, components

    def _assign_genome_to_lattice(self, genome=None):
        """Map a complete 127-gene genome to OpenMC rings G through A.

        Parameters
        ----------
        genome : list or tuple, optional
            Complete A1-to-G36 physical loading. When omitted, use
            ``self.genome``.

        Returns
        -------
        tuple
            OpenMC lattice rings ordered G-to-A and metadata keyed by slot."""
        genome_to_build = self.genome if genome is None else genome
        if not isinstance(genome_to_build, (list, tuple)):
            raise TypeError("genome harus berupa list atau tuple.")
        if len(genome_to_build) != len(self.SLOT_ORDER):
            raise ValueError(
                f"Genome harus berisi {len(self.SLOT_ORDER)} gene, "
                f"tetapi menerima {len(genome_to_build)}."
            )

        rings = {ring_name: [] for ring_name in self.RING_ORDER}
        components_by_slot = {}
        for genome_index, (slot_name, gene) in enumerate(zip(self.SLOT_ORDER, genome_to_build)):
            universe, components = self._build_gene_universe(gene, slot_name)
            components["genome_index"] = genome_index
            rings[slot_name[0]].append(universe)
            components_by_slot[slot_name] = components

        lattice_universes = [
            rings["G"],
            rings["F"],
            rings["E"],
            rings["D"],
            rings["C"],
            rings["B"],
            rings["A"],
        ]
        return lattice_universes, components_by_slot

    def _build_lattice_outer_universe(self):
        """Build baseline aluminum grid plates and water outside valid
        seven-ring indices. All axial coordinates are in cm; the 2-D
        lattice preserves z when translating to an outer-universe tile.

        Returns
        -------
        openmc.Universe
            Axially partitioned universe clipped by the parent core cell."""
        bottom_grid_lower = openmc.ZPlane(z0=0.0)
        bottom_grid_upper = openmc.ZPlane(z0=1.90500)
        top_grid_lower = openmc.ZPlane(z0=66.99250)
        top_grid_upper = openmc.ZPlane(z0=68.58000)
        bottom_grid_region = +bottom_grid_lower & -bottom_grid_upper
        top_grid_region = +top_grid_lower & -top_grid_upper
        # The parent cell supplies the circular plate boundary. Water must
        # exclude both slabs to avoid overlapping the peripheral aluminum.
        grid_region = bottom_grid_region | top_grid_region
        bottom_grid_cell = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=bottom_grid_region,
            name="Core lattice peripheral bottom grid plate",
        )
        top_grid_cell = openmc.Cell(
            fill=self._get_material("aluminum"),
            region=top_grid_region,
            name="Core lattice peripheral top grid plate",
        )
        water_cell = openmc.Cell(
            fill=self._get_material("water"),
            region=~grid_region,
            name="Core lattice peripheral water",
        )
        return openmc.Universe(
            name="Core lattice peripheral grid plates and water",
            cells=[water_cell, bottom_grid_cell, top_grid_cell],
        )

    def _build_triga_core(self, genome=None):
        """Build the seven-ring TRIGA HexLattice.

        Parameters
        ----------
        genome : list or tuple, optional
            Complete 127-position physical loading. Defaults to
            ``self.genome``.

        Returns
        -------
        tuple
            The HexLattice and component metadata keyed by slot name."""
        lattice_universes, components_by_slot = self._assign_genome_to_lattice(genome)
        triga_core = openmc.HexLattice(name="Core TRIGA 2000 Bandung")
        triga_core.center = (0.0, 0.0)
        triga_core.pitch = (4.27,)
        triga_core.orientation = "x"
        triga_core.universes = lattice_universes
        triga_core.outer = self._build_lattice_outer_universe()
        return triga_core, components_by_slot

    def _build_core_lattice_region(self, outer_core_components):
        """Build the parent region occupied by the TRIGA lattice.

        Parameters
        ----------
        outer_core_components : dict
            Surface and fixed-cell metadata returned by
            ``_build_outer_core()``.

        Returns
        -------
        openmc.Region
            Union of the lower core, upper core, and rod lower extensions."""
        surfaces = outer_core_components["surfaces"]
        shim_cells = outer_core_components["cell_groups"]["shim"]

        shim_region = shim_cells[0].region
        for shim_cell in shim_cells[1:]:
            shim_region |= shim_cell.region

        lower_core_region = (
            +surfaces["surf_core_bottom_grid"]
            & -surfaces["surf_core_wall_top"]
            & -surfaces["surf_inner_core_wall"]
            & ~shim_region
        )
        upper_core_region = (
            +surfaces["surf_core_wall_top"]
            & -surfaces["surf_extended_pool_water_surface"]
            & -surfaces["surf_upper_part_rotary_rack_1"]
        )
        fixed_rod_lower_extension = (
            +surfaces["surf_ffcr_lower_end_fitting"]
            & -surfaces["surf_core_bottom_grid"]
            & (
                -surfaces["surf_ffcr_d2_clad"]
                | -surfaces["surf_control_rod_d5_clad"]
                | -surfaces["surf_control_rod_d9_clad"]
                | -surfaces["surf_ffcr_d13_clad"]
                | -surfaces["surf_ffcr_d16_clad"]
            )
        )
        return lower_core_region | upper_core_region | fixed_rod_lower_extension

    def _build_core_cell(self, outer_core_components, genome=None):
        """Build the lattice-filled core cell and slot metadata.

        Parameters
        ----------
        outer_core_components : dict
            Fixed-geometry metadata returned by ``_build_outer_core()``.
        genome : list or tuple, optional
            Complete 127-position physical loading. Defaults to
            ``self.genome``.

        Returns
        -------
        tuple
            Core cell, HexLattice, and component metadata keyed by slot."""
        triga_core, components_by_slot = self._build_triga_core(genome)
        core_cell = openmc.Cell(
            fill=triga_core,
            region=self._build_core_lattice_region(outer_core_components),
            name="Core TRIGA 2000 Bandung",
        )
        return core_cell, triga_core, components_by_slot

    def _build_geometry(self, genome=None):
        """Build the vacuum-bounded root geometry for one complete genome.

        Parameters
        ----------
        genome : list or tuple, optional
            Complete 127-position physical loading. Defaults to
            ``self.genome``.

        Returns
        -------
        tuple
            Geometry, slot metadata, lattice, core cell, and outer metadata."""
        outer_core_cells, outer_core_components = self._build_outer_core()
        core_cell, triga_core, components_by_slot = self._build_core_cell(
            outer_core_components,
            genome,
        )
        physical_universe = openmc.Universe(
            name="TRIGA 2000 Bandung physical geometry",
            cells=[*outer_core_cells, core_cell],
        )

        world_boundary = openmc.Sphere(
            z0=self.WORLD_BOUNDARY_CENTER_Z_CM,
            r=self.WORLD_BOUNDARY_RADIUS_CM,
            boundary_type="vacuum",
            name="TRIGA world vacuum boundary",
        )
        world_cell = openmc.Cell(
            fill=physical_universe,
            region=-world_boundary,
            name="TRIGA bounded world",
        )
        root_universe = openmc.Universe(
            name="TRIGA 2000 Bandung root",
            cells=[world_cell],
        )
        geometry = openmc.Geometry(root_universe)
        return (
            geometry,
            components_by_slot,
            triga_core,
            core_cell,
            outer_core_components,
        )

    def _build_fission_q_tally(
        self,
        components_by_slot,
        tally_trigger_enabled=False,
        tally_trigger_type="rel_err",
        tally_trigger_threshold=0.10,
        tally_trigger_ignore_zeros=False,
    ):
        """Build fuel-segment fission-q-recoverable tallies.

        Parameters
        ----------
        components_by_slot : dict
            Slot metadata produced while assigning the 127-position genome.
        tally_trigger_enabled : bool, optional
            Whether the target tally has an uncertainty trigger.
        tally_trigger_type, tally_trigger_threshold : optional
            OpenMC trigger type and its positive threshold.
        tally_trigger_ignore_zeros : bool, optional
            Whether zero tally bins may be ignored by OpenMC.

        Returns
        -------
        tuple
            OpenMC tallies and ordered records mapping bins to fuel segments."""
        tally_records = []
        for slot_name in self.SLOT_ORDER:
            components = components_by_slot[slot_name]
            if components["slot_type"] != "fuel":
                continue

            for axial_index, fuel_cell in enumerate(
                components["fuel_cells"],
                start=1,
            ):
                tally_records.append(
                    {
                        "cell_id": fuel_cell.id,
                        "slot_name": slot_name,
                        "genome_index": components["genome_index"],
                        "gene_id": components["gene_id"],
                        "axial_index": axial_index,
                    }
                )

        if not tally_records:
            raise ValueError("Tidak ditemukan fuel cell untuk tally.")

        if not isinstance(tally_trigger_enabled, bool):
            raise TypeError("tally_trigger_enabled harus berupa bool.")
        if tally_trigger_type not in {"rel_err", "std_dev", "variance"}:
            raise ValueError("tally_trigger_type harus rel_err, std_dev, atau variance.")
        if isinstance(tally_trigger_threshold, bool) or not isinstance(
            tally_trigger_threshold, Real
        ):
            raise TypeError("tally_trigger_threshold harus numerik dan bukan bool.")
        if not math.isfinite(float(tally_trigger_threshold)) or tally_trigger_threshold <= 0.0:
            raise ValueError("tally_trigger_threshold harus finite dan positif.")
        if not isinstance(tally_trigger_ignore_zeros, bool):
            raise TypeError("tally_trigger_ignore_zeros harus berupa bool.")

        fuel_cell_ids = [record["cell_id"] for record in tally_records]
        fission_q_tally = openmc.Tally(name=self.tally_name)
        fission_q_tally.filters = [openmc.CellFilter(fuel_cell_ids)]
        fission_q_tally.scores = [self.tally_score]
        if tally_trigger_enabled:
            tally_trigger = openmc.Trigger(
                tally_trigger_type, float(tally_trigger_threshold), tally_trigger_ignore_zeros
            )
            tally_trigger.scores = [self.tally_score]
            fission_q_tally.triggers = [tally_trigger]
        return openmc.Tallies([fission_q_tally]), tally_records

    def build_settings(
        self,
        particles,
        inactive_batches,
        active_batches,
        keff_trigger_enabled=True,
        keff_trigger_stdev_pcm=None,
        openmc_seed=1,
        trigger_batch_interval=5,
        trigger_max_batches=300,
        tally_trigger_enabled=False,
    ):
        """Build eigenvalue settings from explicit transport parameters.

        Parameters
        ----------
        particles : int
            Source histories per generation.
        inactive_batches : int
            Batches discarded before statistics are accumulated.
        active_batches : int
            Batches used to accumulate eigenvalue statistics.
        keff_trigger_enabled : bool, optional
            Whether the keff standard-deviation trigger is active.
        keff_trigger_stdev_pcm : float, optional
            Positive trigger threshold in pcm.
        openmc_seed : int, optional
            Positive random-number seed for this transport attempt.
        trigger_batch_interval, trigger_max_batches : int, optional
            Trigger evaluation interval and maximum extended batch count.
        tally_trigger_enabled : bool, optional
            Whether the model's target tally has an active trigger.

        Returns
        -------
        openmc.Settings
            Eigenvalue settings where total batches equal inactive plus active."""
        keff_trigger_stdev_pcm = (
            self.keff_stdev_tolerance if keff_trigger_stdev_pcm is None else keff_trigger_stdev_pcm
        )

        if isinstance(particles, bool) or not isinstance(particles, Integral):
            raise TypeError("particles harus berupa integer dan bukan bool.")
        if particles <= 0:
            raise ValueError("particles harus berupa integer positif.")
        if isinstance(inactive_batches, bool) or not isinstance(inactive_batches, Integral):
            raise TypeError("inactive_batches harus berupa integer dan bukan bool.")
        if inactive_batches < 0:
            raise ValueError("inactive_batches harus integer non-negatif.")
        if isinstance(active_batches, bool) or not isinstance(active_batches, Integral):
            raise TypeError("active_batches harus berupa integer dan bukan bool.")
        if active_batches <= 0:
            raise ValueError("active_batches harus berupa integer positif.")
        if not isinstance(keff_trigger_enabled, bool):
            raise TypeError("keff_trigger_enabled harus berupa bool.")
        if not isinstance(tally_trigger_enabled, bool):
            raise TypeError("tally_trigger_enabled harus berupa bool.")
        if isinstance(keff_trigger_stdev_pcm, bool) or not isinstance(keff_trigger_stdev_pcm, Real):
            raise TypeError("keff_trigger_stdev_pcm harus numerik dan bukan bool.")
        if not math.isfinite(float(keff_trigger_stdev_pcm)) or keff_trigger_stdev_pcm <= 0:
            raise ValueError("keff_trigger_stdev_pcm harus finite dan positif.")
        if isinstance(openmc_seed, bool) or not isinstance(openmc_seed, Integral):
            raise TypeError("openmc_seed harus berupa integer dan bukan bool.")
        if openmc_seed <= 0:
            raise ValueError("openmc_seed harus berupa integer positif.")
        if isinstance(trigger_batch_interval, bool) or not isinstance(
            trigger_batch_interval, Integral
        ):
            raise TypeError("trigger_batch_interval harus berupa integer dan bukan bool.")
        if trigger_batch_interval <= 0:
            raise ValueError("trigger_batch_interval harus berupa integer positif.")
        if isinstance(trigger_max_batches, bool) or not isinstance(trigger_max_batches, Integral):
            raise TypeError("trigger_max_batches harus berupa integer dan bukan bool.")
        if trigger_max_batches <= 0:
            raise ValueError("trigger_max_batches harus berupa integer positif.")

        settings = openmc.Settings()
        settings.run_mode = "eigenvalue"
        settings.particles = particles
        settings.batches = inactive_batches + active_batches
        settings.inactive = inactive_batches
        trigger_active = keff_trigger_enabled or tally_trigger_enabled
        if trigger_active and trigger_max_batches < settings.batches:
            raise ValueError(
                "trigger_max_batches tidak boleh lebih kecil dari total batches ketika trigger aktif."
            )
        if keff_trigger_enabled:
            # Convert the configured pcm threshold to dimensionless OpenMC keff units.
            settings.keff_trigger = {
                "type": "std_dev",
                "threshold": float(keff_trigger_stdev_pcm) * 1.0e-5,
            }
        if trigger_active:
            settings.trigger_active = True
            settings.trigger_batch_interval = int(trigger_batch_interval)
            settings.trigger_max_batches = int(trigger_max_batches)
        settings.seed = int(openmc_seed)
        source = openmc.IndependentSource()
        source.space = openmc.stats.Box(
            (-26.0, -26.0, 17.653),
            (26.0, 26.0, 55.753),
        )
        source.constraints = {"fissionable": True}
        settings.source = source
        settings.max_lost_particles = 1
        settings.max_write_lost_particles = 1
        return settings

    def build_model(
        self,
        particles,
        inactive_batches,
        active_batches,
        keff_trigger_enabled=True,
        keff_trigger_stdev_pcm=None,
        openmc_seed=1,
        trigger_batch_interval=5,
        trigger_max_batches=300,
        tally_trigger_enabled=False,
        tally_trigger_type="rel_err",
        tally_trigger_threshold=0.10,
        tally_trigger_ignore_zeros=False,
    ):
        """Assemble a complete OpenMC model from ``self.genome``.

        Parameters
        ----------
        particles : int
            Source histories per generation.
        inactive_batches : int
            Batches discarded before statistics are accumulated.
        active_batches : int
            Batches used for eigenvalue statistics.
        keff_trigger_enabled : bool, optional
            Whether the keff standard-deviation trigger is active.
        keff_trigger_stdev_pcm : float, optional
            Positive keff trigger threshold in pcm.
        openmc_seed : int, optional
            Positive random-number seed for this transport attempt.
        trigger_batch_interval, trigger_max_batches : int, optional
            Trigger evaluation interval and maximum extended batch count.
        tally_trigger_enabled : bool, optional
            Whether the fuel-segment tally trigger is active.
        tally_trigger_type, tally_trigger_threshold : optional
            OpenMC tally trigger type and positive threshold.
        tally_trigger_ignore_zeros : bool, optional
            Whether OpenMC may ignore zero-mean tally bins.

        Returns
        -------
        tuple
            OpenMC model and build metadata for tally/post-processing."""
        (
            geometry,
            components_by_slot,
            triga_core,
            core_cell,
            outer_core_components,
        ) = self._build_geometry()
        tallies, tally_records = self._build_fission_q_tally(
            components_by_slot,
            tally_trigger_enabled=tally_trigger_enabled,
            tally_trigger_type=tally_trigger_type,
            tally_trigger_threshold=tally_trigger_threshold,
            tally_trigger_ignore_zeros=tally_trigger_ignore_zeros,
        )
        settings = self.build_settings(
            particles=particles,
            inactive_batches=inactive_batches,
            active_batches=active_batches,
            keff_trigger_enabled=keff_trigger_enabled,
            keff_trigger_stdev_pcm=keff_trigger_stdev_pcm,
            openmc_seed=openmc_seed,
            trigger_batch_interval=trigger_batch_interval,
            trigger_max_batches=trigger_max_batches,
            tally_trigger_enabled=tally_trigger_enabled,
        )
        model = openmc.Model(
            geometry=geometry,
            materials=self.materials,
            settings=settings,
            tallies=tallies,
        )
        build_data = {
            "components_by_slot": components_by_slot,
            "tally_name": self.tally_name,
            "tally_score": self.tally_score,
            "tally_records": tally_records,
            "triga_core": triga_core,
            "core_cell": core_cell,
            "outer_core_components": outer_core_components,
        }
        return model, build_data
