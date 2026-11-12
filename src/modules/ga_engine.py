"""Generate reproducible initial populations with an exact fuel inventory.

PopulationInitializer allocates strategy quotas and produces unique mutable
genomes using low-leakage, out-in, center-relief, ring/sector balancing,
three-color scatter, stratified-random, and uniform-random strategies.
U-235 mass in grams is a heuristic proxy. The GAEngine compatibility facade
is inactive; the campaign lifecycle belongs to MainOptimizer."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import hashlib
from itertools import permutations
import math
import random
from types import MappingProxyType
from typing import TypeAlias

from modules.genome_sorter import GenomeSorter
from modules.mutable_genome import MutableGenomeContract


Gene: TypeAlias = tuple[str, str]


@dataclass(frozen=True)
class InitialCandidate:
    """Store an immutable mutable-position genome and the strategy that generated it."""

    genome: tuple[Gene, ...]
    strategy: str


@dataclass(frozen=True)
class _PositionMetadata:
    """Store internal ring, sector, axial-coordinate, and lattice-color metadata for a slot."""

    ring: str
    radius: int
    within_index: int
    sector: int
    q: int
    r: int
    color: int


class PopulationInitializer:
    """Allocate strategy quotas and generate unique genomes with the exact mutable inventory."""

    STRATEGY_METHODS = MappingProxyType(
        {
            "low_leakage": "_generate_low_leakage",
            "out_in": "_generate_out_in",
            "center_relief": "_generate_center_relief",
            "ring_sector_balanced": "_generate_ring_sector_balanced",
            "three_color_scatter": "_generate_three_color_scatter",
            "stratified_random": "_generate_stratified_random",
            "uniform_random": "_generate_uniform_random",
        }
    )
    TERTILE_NAMES = ("LOW", "MEDIUM", "HIGH")
    MAX_UNIQUENESS_ATTEMPTS = 200

    def __init__(
        self,
        population_size: int,
        mutable_genes: Iterable[Gene],
        genome_sorter: GenomeSorter[Gene],
        reactivity_proxy: Mapping[str, float],
        strategy_percentages: Mapping[str, float],
        random_seed: int,
    ) -> None:
        """Initialize and validate the population-generation inputs; return None.

        ``population_size`` is the number of initial candidates. ``mutable_genes`` and
        ``genome_sorter`` define the allowed inventory and physical slot order.
        ``reactivity_proxy`` maps fuel IDs to current U-235 mass in grams.
        ``strategy_percentages`` defines the seven strategy quotas, and ``random_seed``
        controls reproducible local random-number streams."""
        if isinstance(population_size, bool) or not isinstance(population_size, int):
            raise TypeError("population_size harus berupa integer.")
        if population_size <= 0:
            raise ValueError("population_size harus lebih besar dari nol.")
        if not isinstance(genome_sorter, GenomeSorter):
            raise TypeError("genome_sorter harus berupa GenomeSorter.")
        if isinstance(random_seed, bool) or not isinstance(random_seed, int):
            raise TypeError("random_seed harus berupa integer.")

        self.population_size = population_size
        self.genome_sorter = genome_sorter
        self.random_seed = random_seed
        self.mutable_genome_contract = MutableGenomeContract.from_genome_sorter(
            mutable_genes, genome_sorter
        )
        self.mutable_genes = self.mutable_genome_contract.expected_alleles
        # Separate fuel and nonfuel genes because their placement rules differ.
        self.fuel_genes = tuple(gene for gene in self.mutable_genes if gene[0] == "fuel")
        self.nonfuel_genes = tuple(gene for gene in self.mutable_genes if gene[0] != "fuel")
        self.fuel_count = len(self.fuel_genes)
        self.empty_hole_count = sum(
            gene_type == "empty_hole" for gene_type, _ in self.nonfuel_genes
        )
        self.irradiation_tube_count = sum(
            gene_type == "irradiation_tube" for gene_type, _ in self.nonfuel_genes
        )
        self._validate_gene_inventory()

        self.reactivity_proxy = MappingProxyType(self._validate_reactivity_proxy(reactivity_proxy))
        self.strategy_percentages = MappingProxyType(
            self._validate_strategy_percentages(strategy_percentages)
        )
        metadata, adjacency = self._build_hex_position_metadata(self.genome_sorter.slot_order)
        self.position_metadata = MappingProxyType(metadata)
        self.adjacency = MappingProxyType(
            {position: frozenset(neighbors) for position, neighbors in adjacency.items()}
        )
        self._outer_ring = max(self.position_metadata.values(), key=lambda item: item.radius).ring
        self._fixed_outer_nonfuel = frozenset(
            position
            for position, gene in self.genome_sorter.excluded_genes.items()
            if position.startswith(self._outer_ring) and gene[0] != "fuel"
        )
        # Radial strategies use continuous fuel ranking; scattering strategies use tertiles.
        self.fuel_ranking = self._build_continuous_fuel_ranking()
        self.reactivity_tertiles = MappingProxyType(self._build_reactivity_tertiles())
        self._nonfuel_layouts = self._build_nonfuel_layout_pool()
        self._color_layout_options = MappingProxyType(
            {
                layout: self._calculate_color_layout_options(layout)
                for layout in self._nonfuel_layouts
            }
        )

    def initialize_population(self) -> tuple[InitialCandidate, ...]:
        """Return the complete tuple of unique InitialCandidate genomes and their strategy names."""
        strategy_counts = self._allocate_strategy_counts()
        generators = self._get_strategy_generators()
        candidates: list[InitialCandidate] = []
        seen_genomes: set[tuple[Gene, ...]] = set()

        for strategy, count in strategy_counts.items():
            generator = generators[strategy]
            for candidate_index in range(count):
                for attempt in range(self.MAX_UNIQUENESS_ATTEMPTS):
                    # Per-candidate seeds prevent retries from changing other candidates.
                    rng = self._derive_strategy_rng(strategy, candidate_index, attempt)
                    genome = generator(rng)
                    self._validate_candidate(genome)
                    if genome not in seen_genomes:
                        seen_genomes.add(genome)
                        candidates.append(InitialCandidate(genome, strategy))
                        break
                else:
                    raise RuntimeError(
                        f"Gagal membuat kandidat unik ke-{candidate_index} untuk strategi {strategy!r}."
                    )

        order_rng = self._derive_strategy_rng("population_order", 0, 0)
        order_rng.shuffle(candidates)
        return tuple(candidates)

    def _generate_low_leakage(self, rng: random.Random) -> tuple[Gene, ...]:
        """Use ``rng`` to break priority ties; return a genome placing higher U-235 masses inward."""
        nonfuel = self._build_nonfuel_assignment(rng)
        positions = self._fuel_positions_for_layout(nonfuel)
        ordered = self._order_positions_by_priority(
            positions, lambda position: -self.position_metadata[position].radius, rng
        )
        fuels = reversed(self.fuel_ranking)
        return self._assemble_genome(dict(zip(ordered, fuels)), nonfuel)

    def _generate_out_in(self, rng: random.Random) -> tuple[Gene, ...]:
        """Use ``rng`` to break priority ties; return a genome placing higher U-235 masses outward."""
        nonfuel = self._build_nonfuel_assignment(rng)
        positions = self._fuel_positions_for_layout(nonfuel)
        ordered = self._order_positions_by_priority(
            positions, lambda position: self.position_metadata[position].radius, rng
        )
        fuels = reversed(self.fuel_ranking)
        return self._assemble_genome(dict(zip(ordered, fuels)), nonfuel)

    def _generate_center_relief(self, rng: random.Random) -> tuple[Gene, ...]:
        """Use ``rng`` within radial bands; return a genome prioritizing rings D-E, then C-F, then B-G."""
        nonfuel = self._build_nonfuel_assignment(rng)
        positions = self._fuel_positions_for_layout(nonfuel)

        def annular_priority(position: str) -> int:
            """Return the radial priority of core ``position``, highest in D-E and lower toward the center and periphery."""
            radius = self.position_metadata[position].radius
            if radius < 3:
                return radius - 3
            if radius > 4:
                return 4 - radius
            return 0

        ordered = self._order_positions_by_priority(positions, annular_priority, rng)
        fuels = reversed(self.fuel_ranking)
        return self._assemble_genome(dict(zip(ordered, fuels)), nonfuel)

    def _generate_ring_sector_balanced(self, rng: random.Random) -> tuple[Gene, ...]:
        """Use ``rng`` to return a genome with tertile counts differing by at most one per ring and sector."""
        nonfuel = self._build_nonfuel_assignment(rng)
        positions = self._fuel_positions_for_layout(nonfuel)
        classes = self._build_ring_sector_class_assignment(positions, rng)
        fuels = self._assign_fuels_within_classes(classes, rng)
        return self._assemble_genome(fuels, nonfuel)

    def _generate_three_color_scatter(self, rng: random.Random) -> tuple[Gene, ...]:
        """Use ``rng`` for color mapping and ties; return a genome scattering fuel tertiles over three lattice colors."""
        layout = self._generate_dispersed_nonfuel_layout(rng, prefer_color_balance=True)
        nonfuel = self._assign_nonfuel_tokens(layout, rng)
        fuel_positions = self._fuel_positions_for_layout(nonfuel)
        label_order = rng.choice(self._optimal_color_label_orders(layout))
        color_to_label = dict(zip(range(3), label_order))
        class_assignment: dict[str, str] = {}
        surplus_positions: list[str] = []
        deficits: list[str] = []

        for color, label in color_to_label.items():
            positions = [
                position
                for position in fuel_positions
                if self.position_metadata[position].color == color
            ]
            rng.shuffle(positions)
            target = len(self.reactivity_tertiles[label])
            class_assignment.update({position: label for position in positions[:target]})
            surplus_positions.extend(positions[target:])
            deficits.extend([label] * max(0, target - len(positions)))

        rng.shuffle(deficits)
        for label in deficits:
            # Place spillover where the number of same-class neighbors is smallest.
            scores = {
                position: sum(
                    class_assignment.get(neighbor) == label for neighbor in self.adjacency[position]
                )
                for position in surplus_positions
            }
            if not scores:
                raise RuntimeError("Posisi spillover three-color tidak mencukupi.")
            best_score = min(scores.values())
            chosen = rng.choice(
                [position for position, score in scores.items() if score == best_score]
            )
            class_assignment[chosen] = label
            surplus_positions.remove(chosen)

        if surplus_positions:
            raise RuntimeError("Masih ada posisi three-color tanpa kelas.")
        fuels = self._assign_fuels_within_classes(class_assignment, rng)
        return self._assemble_genome(fuels, nonfuel)

    def _generate_stratified_random(self, rng: random.Random) -> tuple[Gene, ...]:
        """Use ``rng`` to return a random genome with balanced LOW/MEDIUM/HIGH counts in each ring-sector stratum."""
        nonfuel = self._build_nonfuel_assignment(rng)
        fuel_positions = self._fuel_positions_for_layout(nonfuel)
        strata: dict[tuple[str, int], list[str]] = defaultdict(list)
        for position in fuel_positions:
            item = self.position_metadata[position]
            strata[(item.ring, item.sector)].append(position)

        quotas = self._build_balanced_group_quotas(strata, rng)
        class_assignment: dict[str, str] = {}
        for stratum, positions in strata.items():
            positions = list(positions)
            rng.shuffle(positions)
            labels = [label for label in self.TERTILE_NAMES for _ in range(quotas[stratum][label])]
            rng.shuffle(labels)
            class_assignment.update(zip(positions, labels))
        fuels = self._assign_fuels_within_classes(class_assignment, rng)
        return self._assemble_genome(fuels, nonfuel)

    def _generate_uniform_random(self, rng: random.Random) -> tuple[Gene, ...]:
        """Shuffle mutable fuel identities once with ``rng`` and return an unbiased permutation genome."""
        if self.nonfuel_genes:
            raise ValueError(
                "uniform_random hanya mendukung kontrak mutable fuel-only; tetapkan bobotnya nol untuk konfigurasi dengan mutable non-fuel."
            )
        shuffled_fuels = list(self.fuel_genes)
        rng.shuffle(shuffled_fuels)
        return tuple(shuffled_fuels)

    def _validate_gene_inventory(self) -> None:
        """Check mutable inventory length, uniqueness, and gene types; return None or raise on mismatch."""
        if len(self.mutable_genes) != self.genome_sorter.included_length:
            raise ValueError(
                f"Jumlah mutable_genes tidak sama dengan included positions: {len(self.mutable_genes)} != {self.genome_sorter.included_length}."
            )
        if len(set(self.mutable_genes)) != len(self.mutable_genes):
            raise ValueError("mutable_genes tidak boleh mengandung gene duplikat.")
        if self.fuel_count < len(self.TERTILE_NAMES):
            raise ValueError(
                "Jumlah movable fuel minimal tiga agar tertile reaktivitas LOW/MEDIUM/HIGH tidak kosong."
            )
        type_counts = Counter(gene_type for gene_type, _ in self.nonfuel_genes)
        unknown_types = set(type_counts).difference({"empty_hole", "irradiation_tube"})
        if unknown_types:
            raise ValueError(
                "Inventori mutable memuat jenis non-fuel yang tidak dikenal: "
                + ", ".join(sorted(unknown_types))
                + "."
            )

    def _validate_reactivity_proxy(self, reactivity_proxy: Mapping[str, float]) -> dict[str, float]:
        """Validate fuel-to-U-235-mass ``reactivity_proxy`` in grams and return an independent float mapping."""
        if not isinstance(reactivity_proxy, Mapping):
            raise TypeError("reactivity_proxy harus berupa mapping fuel ID ke massa.")
        expected_ids = {gene_id for _, gene_id in self.fuel_genes}
        actual_ids = set(reactivity_proxy)
        if actual_ids != expected_ids:
            raise ValueError(
                f"reactivity_proxy harus tepat mencakup semua movable fuel; missing={sorted(expected_ids - actual_ids)}, extra={sorted(actual_ids - expected_ids)}."
            )
        normalized: dict[str, float] = {}
        for fuel_id, value in reactivity_proxy.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Reactivity proxy {fuel_id} harus numerik.")
            mass = float(value)
            if not math.isfinite(mass) or mass <= 0.0:
                raise ValueError(f"Reactivity proxy {fuel_id} harus finite dan positif.")
            normalized[fuel_id] = mass
        return normalized

    def _validate_strategy_percentages(
        self, strategy_percentages: Mapping[str, float]
    ) -> dict[str, float]:
        """Validate names and quotas in ``strategy_percentages`` and return them in registry order."""
        if not isinstance(strategy_percentages, Mapping):
            raise TypeError("strategy_percentages harus berupa mapping.")
        expected = set(self.STRATEGY_METHODS)
        actual = set(strategy_percentages)
        if actual != expected:
            raise ValueError(
                f"Nama strategi pada konfigurasi harus sama dengan registry; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}."
            )
        percentages: dict[str, float] = {}
        for strategy in self.STRATEGY_METHODS:
            value = strategy_percentages[strategy]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Persentase {strategy} harus numerik.")
            percentage = float(value)
            if not math.isfinite(percentage) or percentage < 0.0:
                raise ValueError(f"Persentase {strategy} harus finite dan tidak negatif.")
            percentages[strategy] = percentage
        if not math.isclose(sum(percentages.values()), 100.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("Total strategy_percentages harus tepat 100.")
        return percentages

    @staticmethod
    def _ring_coordinates(radius: int) -> tuple[tuple[int, int], ...]:
        """Return ordered axial (q, r) coordinates for hexagonal ``radius`` (center 0, ring B 1)."""
        if radius == 0:
            return ((0, 0),)
        q, r = radius, 0
        coordinates: list[tuple[int, int]] = []
        directions = (
            (0, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 0),
            (1, -1),
        )
        for delta_q, delta_r in directions:
            for _ in range(radius):
                coordinates.append((q, r))
                q += delta_q
                r += delta_r
        return tuple(coordinates)

    @classmethod
    def _build_hex_position_metadata(
        cls, slot_order: Iterable[str]
    ) -> tuple[dict[str, _PositionMetadata], dict[str, set[str]]]:
        """Return per-slot metadata and nearest-neighbor maps for the complete physical ``slot_order``."""
        positions = tuple(slot_order)
        ring_sizes = dict(zip("ABCDEFG", (1, 6, 12, 18, 24, 30, 36)))
        expected = tuple(
            f"{ring}{number}" for ring, size in ring_sizes.items() for number in range(1, size + 1)
        )
        if positions != expected:
            raise ValueError("slot_order harus mengikuti urutan fisik A1 hingga G36.")

        metadata: dict[str, _PositionMetadata] = {}
        coordinate_to_position: dict[tuple[int, int], str] = {}
        for radius, (ring, size) in enumerate(ring_sizes.items()):
            coordinates = cls._ring_coordinates(radius)
            if len(coordinates) != size:
                raise RuntimeError(f"Jumlah koordinat Ring {ring} tidak konsisten.")
            for within_index, (q, r) in enumerate(coordinates):
                position = f"{ring}{within_index + 1}"
                # Divide each noncentral ring into six equal angular sectors.
                sector = 0 if radius == 0 else within_index // radius
                metadata[position] = _PositionMetadata(
                    ring=ring,
                    radius=radius,
                    within_index=within_index,
                    sector=sector,
                    q=q,
                    r=r,
                    color=(q - r) % 3,
                )
                coordinate_to_position[(q, r)] = position

        neighbor_deltas = (
            (1, 0),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (0, -1),
            (1, -1),
        )
        adjacency: dict[str, set[str]] = {}
        for position, item in metadata.items():
            adjacency[position] = {
                coordinate_to_position[(item.q + delta_q, item.r + delta_r)]
                for delta_q, delta_r in neighbor_deltas
                if (item.q + delta_q, item.r + delta_r) in coordinate_to_position
            }
        return metadata, adjacency

    def _build_continuous_fuel_ranking(self) -> tuple[Gene, ...]:
        """Return fuel genes ordered by increasing current U-235 mass in grams."""
        return tuple(
            sorted(self.fuel_genes, key=lambda gene: (self.reactivity_proxy[gene[1]], gene[1]))
        )

    def _build_reactivity_tertiles(self) -> dict[str, tuple[Gene, ...]]:
        """Return LOW, MEDIUM, and HIGH fuel groups whose sizes differ by at most one."""
        base_size, remainder = divmod(len(self.fuel_ranking), len(self.TERTILE_NAMES))
        group_sizes = [base_size + (index < remainder) for index in range(len(self.TERTILE_NAMES))]
        tertiles: dict[str, tuple[Gene, ...]] = {}
        start = 0
        for name, size in zip(self.TERTILE_NAMES, group_sizes):
            stop = start + size
            tertiles[name] = self.fuel_ranking[start:stop]
            start = stop
        return tertiles

    def _build_nonfuel_layout_pool(self) -> tuple[tuple[str, ...], ...]:
        """Return feasible Ring-G nonfuel layouts with nonadjacent positions and balanced sector counts."""
        required = len(self.nonfuel_genes)
        if required == 0:
            return ((),)
        candidates = [
            position
            for position in self.genome_sorter.included_positions
            if self.position_metadata[position].ring == self._outer_ring
            and not self.adjacency[position].intersection(self._fixed_outer_nonfuel)
        ]
        layouts: list[tuple[str, ...]] = []

        def search(start: int, selected: list[str], sector_counts: list[int]) -> None:
            """Backtrack Ring-G combinations from ``start``, extending ``selected`` and ``sector_counts``; append valid layouts and return None."""
            if len(selected) == required:
                if max(sector_counts) - min(sector_counts) <= 1:
                    layouts.append(tuple(selected))
                return
            remaining_needed = required - len(selected)
            if len(candidates) - start < remaining_needed:
                return
            maximum_sector_count = math.ceil(required / 6)
            for index in range(start, len(candidates)):
                position = candidates[index]
                sector = self.position_metadata[position].sector
                if sector_counts[sector] >= maximum_sector_count:
                    continue
                # Adjacent nonfuel positions can form a local cluster.
                if any(chosen in self.adjacency[position] for chosen in selected):
                    continue
                selected.append(position)
                sector_counts[sector] += 1
                search(index + 1, selected, sector_counts)
                sector_counts[sector] -= 1
                selected.pop()

        search(0, [], [0] * 6)
        if not layouts:
            raise ValueError("Tidak ditemukan layout Ring G untuk seluruh movable non-fuel.")
        return tuple(layouts)

    def _outer_ring_gap_score(self, layout: tuple[str, ...]) -> int:
        """Return the sum of squared circular gaps in nonfuel ``layout``; smaller values favor even spacing."""
        outer_radius = max(item.radius for item in self.position_metadata.values())
        ring_size = 6 * outer_radius
        occupied = sorted(
            self.position_metadata[position].within_index
            for position in set(layout).union(self._fixed_outer_nonfuel)
        )
        if not occupied:
            return 0
        gaps = [
            (occupied[(index + 1) % len(occupied)] - occupied[index]) % ring_size
            for index in range(len(occupied))
        ]
        return sum(gap * gap for gap in gaps)

    def _generate_dispersed_nonfuel_layout(
        self,
        rng: random.Random,
        prefer_color_balance: bool = False,
    ) -> tuple[str, ...]:
        """Select a Ring-G nonfuel layout with ``rng`` and return its positions.

        When ``prefer_color_balance`` is true, prioritize three-color capacity before
        equalizing circular gaps."""
        layouts = list(self._nonfuel_layouts)
        if prefer_color_balance:
            # Match three-color capacities to fuel-tertile sizes for three_color_scatter.
            color_scores = {layout: self._color_layout_options[layout][0] for layout in layouts}
            best_color_score = min(color_scores.values())
            layouts = [layout for layout in layouts if color_scores[layout] == best_color_score]

        spacing_scores = {layout: self._outer_ring_gap_score(layout) for layout in layouts}
        best_spacing_score = min(spacing_scores.values())
        best_layouts = [
            layout for layout in layouts if spacing_scores[layout] == best_spacing_score
        ]
        return rng.choice(best_layouts)

    def _color_capacities(self, layout: Iterable[str]) -> dict[int, int]:
        """Return remaining fuel-slot counts for colors 0, 1, and 2 after excluding nonfuel ``layout``."""
        removed = set(layout)
        capacities = Counter(
            self.position_metadata[position].color
            for position in self.genome_sorter.included_positions
            if position not in removed
        )
        return {color: capacities[color] for color in range(3)}

    def _optimal_color_label_orders(
        self,
        layout: Iterable[str],
    ) -> tuple[tuple[str, str, str], ...]:
        """Return color-to-tertile label permutations minimizing spillover for nonfuel ``layout``."""
        layout_key = tuple(layout)
        if layout_key in self._color_layout_options:
            return self._color_layout_options[layout_key][1]
        return self._calculate_color_layout_options(layout_key)[1]

    def _calculate_color_layout_options(
        self,
        layout: Iterable[str],
    ) -> tuple[int, tuple[tuple[str, str, str], ...]]:
        """Return minimum spillover and optimal color-to-tertile mappings after excluding ``layout``."""
        capacities = self._color_capacities(layout)
        scores: dict[tuple[str, str, str], int] = {}
        for label_order in permutations(self.TERTILE_NAMES):
            scores[label_order] = sum(
                abs(capacities[color] - len(self.reactivity_tertiles[label_order[color]]))
                for color in range(3)
            )
        best_total_difference = min(scores.values())
        optimal_orders = tuple(
            label_order for label_order, score in scores.items() if score == best_total_difference
        )
        return best_total_difference // 2, optimal_orders

    def _minimum_color_mismatch(self, layout: Iterable[str]) -> int:
        """Return the minimum number of fuel genes outside their dominant color for nonfuel ``layout``."""
        layout_key = tuple(layout)
        if layout_key in self._color_layout_options:
            return self._color_layout_options[layout_key][0]
        return self._calculate_color_layout_options(layout_key)[0]

    def _assign_nonfuel_tokens(self, layout: Iterable[str], rng: random.Random) -> dict[str, Gene]:
        """Randomize nonfuel identities over ``layout`` with ``rng`` and return a slot-to-gene mapping."""
        positions = list(layout)
        genes = list(self.nonfuel_genes)
        if len(positions) != len(genes):
            raise ValueError("Jumlah posisi dan movable non-fuel harus sama.")
        rng.shuffle(genes)
        return dict(zip(positions, genes))

    def _build_nonfuel_assignment(self, rng: random.Random) -> dict[str, Gene]:
        """Use ``rng`` to select and populate a dispersed Ring-G layout; return its nonfuel mapping."""
        return self._assign_nonfuel_tokens(self._generate_dispersed_nonfuel_layout(rng), rng)

    def _fuel_positions_for_layout(self, nonfuel_assignment: Mapping[str, Gene]) -> tuple[str, ...]:
        """Return unoccupied fuel positions in included-position order after ``nonfuel_assignment``."""
        positions = tuple(
            position
            for position in self.genome_sorter.included_positions
            if position not in nonfuel_assignment
        )
        if len(positions) != len(self.fuel_genes):
            raise ValueError("Jumlah posisi fuel tidak sama dengan inventori fuel.")
        return positions

    def _order_positions_by_priority(
        self,
        positions: Iterable[str],
        priority_function: Callable[[str], float],
        rng: random.Random,
    ) -> tuple[str, ...]:
        """Return ``positions`` by decreasing ``priority_function`` value, shuffling equal bands with ``rng``."""
        bands: dict[float, list[str]] = defaultdict(list)
        for position in positions:
            bands[priority_function(position)].append(position)
        ordered: list[str] = []
        for priority in sorted(bands, reverse=True):
            band = bands[priority]
            rng.shuffle(band)
            ordered.extend(band)
        return tuple(ordered)

    def _build_balanced_group_quotas(
        self,
        groups: Mapping[object, list[str]],
        rng: random.Random,
    ) -> dict[object, dict[str, int]]:
        """Distribute fuel tertiles across ``groups`` using ``rng`` for ties; return quotas preserving global counts."""
        quotas: dict[object, dict[str, int]] = {}
        deficits = {label: len(self.reactivity_tertiles[label]) for label in self.TERTILE_NAMES}
        group_keys = list(groups)
        rng.shuffle(group_keys)

        for group in group_keys:
            base, remainder = divmod(len(groups[group]), 3)
            quotas[group] = {label: base for label in self.TERTILE_NAMES}
            for label in self.TERTILE_NAMES:
                deficits[label] -= base
            for _ in range(remainder):
                # Assign remaining slots to classes with the largest unmet global quotas.
                largest = max(
                    deficit for label, deficit in deficits.items() if quotas[group][label] == base
                )
                choices = [
                    label
                    for label, deficit in deficits.items()
                    if deficit == largest and quotas[group][label] == base
                ]
                chosen = rng.choice(choices)
                quotas[group][chosen] += 1
                deficits[chosen] -= 1

        if any(deficits.values()):
            raise RuntimeError(f"Kuota tertile global tidak seimbang: {deficits}.")
        return quotas

    @staticmethod
    def _solve_bipartite_allocation(
        cell_capacities: Mapping[tuple[str, int], int],
        ring_requirements: Mapping[str, int],
        sector_requirements: Mapping[int, int],
        rng: random.Random,
    ) -> dict[tuple[str, int], int] | None:
        """Return a feasible ring-sector flow allocation for one fuel tertile, or None.

        ``cell_capacities`` limits each physical intersection; ``ring_requirements``
        and ``sector_requirements`` prescribe marginal totals. ``rng`` varies the
        search order among equivalent solutions without changing capacities."""
        source = ("source", -1)
        sink = ("sink", -1)
        rings = list(ring_requirements)
        sectors = list(sector_requirements)
        rng.shuffle(rings)
        rng.shuffle(sectors)
        capacity: dict[tuple[object, object], int] = {}
        neighbors: dict[object, list[object]] = defaultdict(list)

        def add_edge(start: object, end: object, edge_capacity: int) -> None:
            """Add forward/reverse flow edges from ``start`` to ``end`` with ``edge_capacity``; update graph mappings and return None."""
            capacity[(start, end)] = edge_capacity
            capacity[(end, start)] = 0
            neighbors[start].append(end)
            neighbors[end].append(start)

        ring_nodes = {ring: ("ring", ring) for ring in rings}
        sector_nodes = {sector: ("sector", sector) for sector in sectors}
        for ring in rings:
            add_edge(source, ring_nodes[ring], ring_requirements[ring])
        cells = [cell for cell, count in cell_capacities.items() if count > 0]
        rng.shuffle(cells)
        for ring, sector in cells:
            add_edge(ring_nodes[ring], sector_nodes[sector], cell_capacities[(ring, sector)])
        for sector in sectors:
            add_edge(sector_nodes[sector], sink, sector_requirements[sector])

        total_flow = 0
        target_flow = sum(ring_requirements.values())
        while total_flow < target_flow:
            # Breadth-first search finds an augmenting path with remaining capacity.
            parent: dict[object, object | None] = {source: None}
            queue = deque([source])
            while queue and sink not in parent:
                node = queue.popleft()
                shuffled_neighbors = list(neighbors[node])
                rng.shuffle(shuffled_neighbors)
                for neighbor in shuffled_neighbors:
                    if neighbor not in parent and capacity[(node, neighbor)] > 0:
                        parent[neighbor] = node
                        queue.append(neighbor)
            if sink not in parent:
                return None
            path_capacity = target_flow - total_flow
            node = sink
            while parent[node] is not None:
                previous = parent[node]
                path_capacity = min(path_capacity, capacity[(previous, node)])
                node = previous
            node = sink
            while parent[node] is not None:
                previous = parent[node]
                capacity[(previous, node)] -= path_capacity
                capacity[(node, previous)] += path_capacity
                node = previous
            total_flow += path_capacity

        allocation: dict[tuple[str, int], int] = {}
        for ring, sector in cell_capacities:
            initial = cell_capacities[(ring, sector)]
            if initial == 0:
                allocation[(ring, sector)] = 0
            else:
                allocation[(ring, sector)] = (
                    initial - capacity[(ring_nodes[ring], sector_nodes[sector])]
                )
        return allocation

    def _build_ring_sector_class_assignment(
        self, fuel_positions: Iterable[str], rng: random.Random
    ) -> dict[str, str]:
        """Use ``rng`` to assign balanced LOW/MEDIUM/HIGH labels to ``fuel_positions``; return the slot-label mapping."""
        cells: dict[tuple[str, int], list[str]] = defaultdict(list)
        rings: dict[str, list[str]] = defaultdict(list)
        sectors: dict[int, list[str]] = defaultdict(list)
        for position in fuel_positions:
            item = self.position_metadata[position]
            cells[(item.ring, item.sector)].append(position)
            rings[item.ring].append(position)
            sectors[item.sector].append(position)

        for _ in range(100):
            ring_quotas = self._build_balanced_group_quotas(rings, rng)
            sector_quotas = self._build_balanced_group_quotas(sectors, rng)
            remaining = {cell: len(positions) for cell, positions in cells.items()}
            allocations = {cell: {label: 0 for label in self.TERTILE_NAMES} for cell in cells}
            label_order = list(self.TERTILE_NAMES)
            rng.shuffle(label_order)
            successful = True
            for label in label_order[:-1]:
                ring_requirements = {ring: quota[label] for ring, quota in ring_quotas.items()}
                sector_requirements = {
                    sector: quota[label] for sector, quota in sector_quotas.items()
                }
                allocation = self._solve_bipartite_allocation(
                    remaining, ring_requirements, sector_requirements, rng
                )
                if allocation is None:
                    successful = False
                    break
                for cell, count in allocation.items():
                    allocations[cell][label] = count
                    remaining[cell] -= count
            if not successful:
                continue

            final_label = label_order[-1]
            for cell, count in remaining.items():
                allocations[cell][final_label] = count
            if not self._marginal_allocations_match(allocations, ring_quotas, sector_quotas):
                continue

            class_assignment: dict[str, str] = {}
            for cell, positions in cells.items():
                positions = list(positions)
                rng.shuffle(positions)
                labels = [
                    label for label in self.TERTILE_NAMES for _ in range(allocations[cell][label])
                ]
                rng.shuffle(labels)
                class_assignment.update(zip(positions, labels))
            return class_assignment
        raise RuntimeError("Gagal membentuk assignment ring-sector yang seimbang.")

    def _marginal_allocations_match(
        self,
        cell_allocations: Mapping[tuple[str, int], Mapping[str, int]],
        ring_quotas: Mapping[str, Mapping[str, int]],
        sector_quotas: Mapping[int, Mapping[str, int]],
    ) -> bool:
        """Return whether ``cell_allocations`` satisfy all ``ring_quotas`` and ``sector_quotas``."""
        for ring, quotas in ring_quotas.items():
            for label, expected in quotas.items():
                actual = sum(
                    counts[label]
                    for (cell_ring, _), counts in cell_allocations.items()
                    if cell_ring == ring
                )
                if actual != expected:
                    return False
        for sector, quotas in sector_quotas.items():
            for label, expected in quotas.items():
                actual = sum(
                    counts[label]
                    for (_, cell_sector), counts in cell_allocations.items()
                    if cell_sector == sector
                )
                if actual != expected:
                    return False
        return True

    def _assign_fuels_within_classes(
        self,
        class_assignment: Mapping[str, str],
        rng: random.Random,
    ) -> dict[str, Gene]:
        """Shuffle fuel identities within ``class_assignment`` labels using ``rng``; return the slot-gene mapping."""
        if set(class_assignment.values()).difference(self.TERTILE_NAMES):
            raise ValueError("Class assignment memuat tertile yang tidak dikenal.")
        fuel_assignment: dict[str, Gene] = {}
        for label in self.TERTILE_NAMES:
            positions = [
                position
                for position in self.genome_sorter.included_positions
                if class_assignment.get(position) == label
            ]
            fuels = list(self.reactivity_tertiles[label])
            if len(positions) != len(fuels):
                raise ValueError(f"Jumlah posisi {label} tidak sama dengan jumlah fuel tertile.")
            rng.shuffle(fuels)
            fuel_assignment.update(zip(positions, fuels))
        return fuel_assignment

    def _assemble_genome(
        self,
        fuel_assignment: Mapping[str, Gene],
        nonfuel_assignment: Mapping[str, Gene],
    ) -> tuple[Gene, ...]:
        """Merge ``fuel_assignment`` and ``nonfuel_assignment`` into a genome tuple in included-position order."""
        overlap = set(fuel_assignment).intersection(nonfuel_assignment)
        if overlap:
            raise ValueError(f"Posisi fuel/non-fuel bertumpuk: {sorted(overlap)}.")
        assignments = dict(fuel_assignment)
        assignments.update(nonfuel_assignment)
        expected = set(self.genome_sorter.included_positions)
        if set(assignments) != expected:
            raise ValueError(
                f"Assignment belum tepat mencakup included positions; missing={sorted(expected - set(assignments))}, extra={sorted(set(assignments) - expected)}."
            )
        return tuple(assignments[position] for position in self.genome_sorter.included_positions)

    def _validate_candidate(self, genome: tuple[Gene, ...]) -> None:
        """Validate ``genome`` inventory, fixed-position exclusion, and nonfuel layout rules; return None."""
        if not isinstance(genome, tuple):
            raise TypeError("Generator strategi harus mengembalikan tuple genome.")
        self.mutable_genome_contract.validate(genome, "Candidate genome")

        complete = self.genome_sorter.construct_complete_genome(genome)
        if len(complete) != self.genome_sorter.complete_length:
            raise ValueError("Complete genome tidak sesuai panjang fisik.")
        for position, gene in zip(self.genome_sorter.slot_order, complete):
            if (
                position in self.genome_sorter.excluded_genes
                and gene != self.genome_sorter.excluded_genes[position]
            ):
                raise ValueError(f"Gene fixed pada {position} berubah.")

        movable_nonfuel = {
            position
            for position, gene in zip(self.genome_sorter.included_positions, genome)
            if gene[0] != "fuel"
        }
        if len(movable_nonfuel) != len(self.nonfuel_genes):
            raise ValueError("Jumlah posisi movable non-fuel tidak sesuai inventori.")
        if any(
            self.position_metadata[position].ring != self._outer_ring
            for position in movable_nonfuel
        ):
            raise ValueError("Seluruh movable non-fuel harus berada di Ring G.")
        sector_counts = Counter(
            self.position_metadata[position].sector for position in movable_nonfuel
        )
        sector_occupancies = [sector_counts[sector] for sector in range(6)]
        if max(sector_occupancies) - min(sector_occupancies) > 1:
            raise ValueError("Movable non-fuel belum tersebar pada enam sektor.")
        all_outer_nonfuel = movable_nonfuel.union(self._fixed_outer_nonfuel)
        for position in movable_nonfuel:
            if self.adjacency[position].intersection(all_outer_nonfuel.difference({position})):
                raise ValueError(
                    "Terdapat movable non-fuel Ring G yang adjacent dengan non-fuel lain."
                )

    def _allocate_strategy_counts(self) -> dict[str, int]:
        """Return integer strategy quotas using largest remainders to preserve the exact population size."""
        raw_counts = {
            strategy: self.population_size * percentage / 100.0
            for strategy, percentage in self.strategy_percentages.items()
        }
        counts = {strategy: math.floor(value) for strategy, value in raw_counts.items()}
        remaining = self.population_size - sum(counts.values())
        registry_order = {name: index for index, name in enumerate(self.STRATEGY_METHODS)}
        ranked = sorted(
            raw_counts,
            key=lambda strategy: (
                -(raw_counts[strategy] - counts[strategy]),
                registry_order[strategy],
            ),
        )
        for strategy in ranked[:remaining]:
            counts[strategy] += 1
        return {strategy: counts[strategy] for strategy in self.STRATEGY_METHODS}

    def _derive_strategy_rng(
        self, strategy: str, candidate_index: int, attempt: int
    ) -> random.Random:
        """Return a stable RNG for ``strategy``, ``candidate_index``, and retry ``attempt`` under the base seed."""
        payload = f"{self.random_seed}|{strategy}|{candidate_index}|{attempt}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:16], byteorder="big", signed=False)
        return random.Random(seed)

    def _get_strategy_generators(self) -> dict[str, Callable[[random.Random], tuple[Gene, ...]]]:
        """Return the seven strategy names mapped to their bound genome-generation methods."""
        generators: dict[str, Callable[[random.Random], tuple[Gene, ...]]] = {}
        for strategy, method_name in self.STRATEGY_METHODS.items():
            generator = getattr(self, method_name, None)
            if not callable(generator):
                raise RuntimeError(f"Method {method_name!r} untuk {strategy!r} tidak tersedia.")
            generators[strategy] = generator
        return generators


class GAEngine:
    """Reject the inactive compatibility facade; MainOptimizer owns the campaign lifecycle."""

    def __init__(self) -> None:
        """Raise NotImplementedError for the inactive facade; use MainOptimizer.run_optimization instead."""
        raise NotImplementedError(
            "GAEngine adalah facade legacy; gunakan MainOptimizer.run_optimization()."
        )
