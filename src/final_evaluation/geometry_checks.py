"""Deterministic region partitions and independent motion checks, without transport."""

import math
import numpy as np
import openmc
from .model import RODS, FFCR, MOVING_OUT, PREFIX


def vector_contains(region, points, cache=None):
    """Return a Boolean membership mask for ``points`` inside CSG ``region``.

    ``points`` has columns x, y, z in cm. An optional ``cache`` reuses surface
    evaluations while Boolean region operations combine the masks."""
    if cache is None:
        cache = {}
    if isinstance(region, openmc.Halfspace):
        surface = region.surface
        if surface.id in cache:
            values = cache[surface.id]
            return values >= 0 if region.side == "+" else values < 0
        x, y, z = points.T
        if isinstance(surface, openmc.ZPlane):
            values = z - surface.z0
        elif isinstance(surface, openmc.ZCylinder):
            values = (x - surface.x0) ** 2 + (y - surface.y0) ** 2 - surface.r**2
        elif isinstance(surface, openmc.Plane):
            a, b, c, d = surface._get_base_coeffs()
            values = a * x + b * y + c * z - d
        elif hasattr(surface, "get_Abc"):
            a, b, c = surface.get_Abc()
            values = np.einsum("ni,ij,nj->n", points, a, points) + points @ b + c
        elif isinstance(surface, openmc.ZTorus):
            values = (
                (z - surface.z0) ** 2 / surface.b**2
                + (np.hypot(x - surface.x0, y - surface.y0) - surface.a) ** 2 / surface.c**2
                - 1
            )
        else:
            values = np.array([surface.evaluate(point) for point in points])
        cache[surface.id] = values
        return values >= 0 if region.side == "+" else values < 0
    if isinstance(region, openmc.Complement):
        return ~vector_contains(region.node, points, cache)
    values = [vector_contains(node, points, cache) for node in region]
    return (
        np.logical_and.reduce(values)
        if isinstance(region, openmc.Intersection)
        else np.logical_or.reduce(values)
    )


def partition(universe, points):
    """Count cells containing each local point; return counts and last cell IDs."""
    counts = np.zeros(len(points), dtype=int)
    ids = np.full(len(points), -1)
    cache = {}
    for cell in universe.cells.values():
        mask = (
            vector_contains(cell.region, points, cache)
            if cell.region is not None
            else np.ones(len(points), dtype=bool)
        )
        counts[mask] += 1
        ids[mask] = cell.id
    return counts, ids


def rod_points(builder, rod):
    """Return interval midpoints and ±epsilon probes about all rod/fixed boundaries."""
    boundaries = sorted(
        set(
            [builder.moving_z(rod, key) for key in MOVING_OUT]
            + [-41.275, 0.0, 1.905, 66.9925, 68.58, 548.96, 698.96]
        )
    )
    zs = [(a + b) / 2 for a, b in zip(boundaries[:-1], boundaries[1:])]
    zs += [value + sign * 1e-6 for value in boundaries for sign in (-1, 1)]
    radii = [0.0, 0.3175, 1.11125, 1.19063, 1.32085, 1.67005, 1.72085, 1.91135, 2.0]
    rs = [(a + b) / 2 for a, b in zip(radii[:-1], radii[1:])]
    rs += [r + sign * 1e-6 for r in radii[1:-1] for sign in (-1, 1)]
    return np.array(
        [
            (r * math.cos(phi), r * math.sin(phi), z)
            for z in zs
            for r in rs
            for phi in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)
        ]
    )


def check_geometry(builder, model, data):
    """Check model partitions/bounds/inventory for each rod; return aggregate evidence."""
    result = dict(
        rods={},
        passed=True,
        scope="Analytic local axial/radial partitions; targeted full parent hierarchy.",
    )
    core = data["core"]
    physical = next(iter(model.geometry.root_universe.cells.values())).fill
    for rod in RODS:
        points = rod_points(builder, rod)
        component = data["slots"][rod]
        universe = component["universe"]
        counts, ids = partition(universe, points)
        # Parent limits are necessary: the local universe deliberately extends to infinity.
        radial = data["outer"]["surfaces"][PREFIX[rod] + "_clad"]
        center = np.array([radial.x0, radial.y0, 0.0])
        global_points = points + center
        parent_mask = vector_contains(core.region, global_points)
        local_counts = counts[parent_mask]
        parent_counts, parent_ids = partition(physical, global_points)
        in_scope = (global_points[:, 2] > -41.275) & (global_points[:, 2] < 698.96)
        root_bad = (parent_counts != 1) & in_scope
        local_bad = (counts != 1) & parent_mask
        bad_points = global_points[root_bad | local_bad][:8].tolist()
        absorber = builder.rod_components[rod]["component_cells"]["b4c"]
        zplanes = [
            (half.surface.z0, half.side)
            for half in absorber.region
            if isinstance(half, openmc.Halfspace) and isinstance(half.surface, openmc.ZPlane)
        ]
        absorber_bounds = sorted(z for z, _ in zplanes)
        expected = [builder.moving_z(rod, 41), builder.moving_z(rod, 42)]
        bounds_ok = np.allclose(absorber_bounds, expected, rtol=0, atol=1e-10)
        volume_checks = []
        if rod in FFCR:
            for record, cell in zip(
                [r for r in data["records"] if r["slot"] == rod],
                component["fuel_cells"],
                strict=True,
            ):
                # Material objects stay attached to their segment during translation.
                index = record["axial"]
                expected_material = builder._get_material(f"{component['gene_id']}_{index:02d}")
                volume_checks.append(
                    cell.fill is expected_material
                    and math.isclose(
                        record["volume_cm3"],
                        math.pi * (1.67005**2 - 0.3175**2) * 2.54,
                        rel_tol=1e-12,
                    )
                )
        passed = bool(
            not root_bad.any() and not local_bad.any() and bounds_ok and all(volume_checks)
        )
        result["rods"][rod] = dict(
            passed=passed,
            probe_count=len(points),
            local_overlap_count=int(np.sum((counts > 1) & parent_mask)),
            local_gap_count=int(np.sum((counts == 0) & parent_mask)),
            root_overlap_count=int(np.sum((parent_counts > 1) & in_scope)),
            root_gap_count=int(np.sum((parent_counts == 0) & in_scope)),
            absorber_bounds_cm=absorber_bounds,
            follower_inventory_volume_ok=all(volume_checks),
            diagnosis_points_cm=bad_points,
        )
        result["passed"] &= passed
    return result


def centers_from_lattice(data):
    """Return exact lattice centers for rod slots using universe indices and local transform."""
    lattice = data["lattice"]
    centers = {}
    for index in (
        (x, a) for x in range(-6, 7) for a in range(-6, 7) if max(abs(x), abs(a), abs(x + a)) <= 6
    ):
        universe = lattice.get_universe(index)
        for rod in RODS:
            if universe is data["slots"][rod]["universe"]:
                # get_local_coordinates subtracts the exact center even for point (0,0,0).
                local = lattice.get_local_coordinates((0.0, 0.0, 0.0), index)
                centers[rod] = (-local[0], -local[1])
    return centers
