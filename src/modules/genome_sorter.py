"""Map mutable genomes to the complete physical core slot order.

GenomeSorter extracts optimizable genes and inserts fixed genes to provide the
127-position genome required by OpenMC. Validation preserves deterministic
ordering and element counts. This module performs no I/O."""

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Generic, TypeVar


Gene = TypeVar("Gene")


class GenomeSorter(Generic[Gene]):
    """Keep excluded genes fixed in the complete physical core slot order."""

    def __init__(
        self,
        slot_order: Iterable[str],
        excluded_genes: Mapping[str, Gene],
    ) -> None:
        """Store physical ``slot_order`` and the ``excluded_genes`` fixed-position mapping.

        Derive the mutable included positions while retaining independent copies of the
        core contract. Return None."""
        ordered_slots = self._as_tuple(slot_order, "slot_order")

        if not ordered_slots:
            raise ValueError("slot_order tidak boleh kosong.")

        if any(not isinstance(position, str) or not position for position in ordered_slots):
            raise TypeError("Setiap posisi dalam slot_order harus berupa string non-kosong.")

        if len(set(ordered_slots)) != len(ordered_slots):
            raise ValueError("slot_order tidak boleh mengandung posisi duplikat.")

        if not isinstance(excluded_genes, Mapping):
            raise TypeError("excluded_genes harus berupa mapping posisi ke gene tetap.")

        unknown_positions = set(excluded_genes).difference(ordered_slots)
        if unknown_positions:
            unknown_text = ", ".join(sorted(map(str, unknown_positions)))
            raise ValueError(
                f"excluded_genes memuat posisi yang tidak ada dalam slot_order: {unknown_text}."
            )

        # Copy and freeze the fixed mapping so external mutation cannot change the contract.
        fixed_genes = dict(excluded_genes)
        self.slot_order = ordered_slots
        self.excluded_genes = MappingProxyType(fixed_genes)

        self.excluded_positions = tuple(
            position for position in ordered_slots if position in fixed_genes
        )

        self.included_positions = tuple(
            position for position in ordered_slots if position not in fixed_genes
        )

    @property
    def complete_length(self) -> int:
        """Return the number of physical core positions required by the model builder."""
        return len(self.slot_order)

    @property
    def included_length(self) -> int:
        """Return the number of mutable positions available to genetic operators."""
        return len(self.included_positions)

    def extract_included_genome(
        self,
        complete_genome: Iterable[Gene],
    ) -> tuple[Gene, ...]:
        """Return mutable genes from ``complete_genome`` in physical slot order.

        Require a complete genome ordered exactly as ``slot_order``; fixed-position
        genes are excluded from the returned tuple."""
        complete_genes = self._as_tuple(complete_genome, "complete_genome")
        self._validate_length(
            complete_genes,
            expected=self.complete_length,
            genome_name="complete_genome",
        )

        # Verify fixed genes before extracting the mutable portion of a complete loading.
        for position, gene in zip(self.slot_order, complete_genes):
            if position in self.excluded_genes and gene != self.excluded_genes[position]:
                raise ValueError(
                    f"Gene excluded pada {position} berubah: "
                    f"diharapkan {self.excluded_genes[position]!r}, menerima {gene!r}."
                )

        return tuple(
            gene
            for position, gene in zip(self.slot_order, complete_genes)
            if position not in self.excluded_genes
        )

    def construct_complete_genome(
        self,
        included_genome: Iterable[Gene],
    ) -> tuple[Gene, ...]:
        """Insert fixed genes around ``included_genome`` and return the complete loading.

        The mutable input follows ``included_positions``; the returned tuple follows
        ``slot_order`` and preserves all fixed-position genes."""
        included_genes = self._as_tuple(included_genome, "included_genome")
        self._validate_length(
            included_genes,
            expected=self.included_length,
            genome_name="included_genome",
        )

        complete_genome = []
        included_index = 0
        for position in self.slot_order:
            # Always restore configured fixed genes, consuming mutable genes only for included slots.
            if position in self.excluded_genes:
                complete_genome.append(self.excluded_genes[position])
            else:
                complete_genome.append(included_genes[included_index])
                included_index += 1

        return tuple(complete_genome)

    @staticmethod
    def _as_tuple(values: Iterable[Gene], value_name: str) -> tuple[Gene, ...]:
        """Return iterable ``values`` as an ordered tuple, using ``value_name`` in errors.

        Reject bare str and bytes values so text cannot be mistaken for individual genes."""
        # Bare text is iterable but would incorrectly become one gene per character.
        if isinstance(values, (str, bytes)):
            raise TypeError(f"{value_name} harus berupa iterable gene, bukan teks.")

        # Materialize generators so length and contents can be checked repeatedly.
        try:
            return tuple(values)
        except TypeError as exc:
            raise TypeError(f"{value_name} harus berupa iterable.") from exc

    @staticmethod
    def _validate_length(
        genome: tuple[Gene, ...],
        expected: int,
        genome_name: str,
    ) -> None:
        """Require ``genome`` to contain ``expected`` genes, labeled by ``genome_name``; return None."""
        # This helper validates length only; callers validate gene contents.
        if len(genome) != expected:
            raise ValueError(
                f"{genome_name} harus berisi {expected} gene, " f"tetapi menerima {len(genome)}."
            )
