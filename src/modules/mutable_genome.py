"""Normalize mutable genes and enforce their exact permutation inventory.

MutableGenomeContract stores allele counts and mutable genome length, builds
contracts from sorters or checkpoints, validates operator results, and serializes
inventory metadata. Fixed positions are outside this contract."""

# Inventory constrains allele identities and multiplicities, while allowing their order to change.
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any, TypeAlias


Gene: TypeAlias = tuple[str, str]


class MutableGenomeValidationError(ValueError):
    """Report a mutable genome that violates its configured allele inventory."""


def normalize_gene_pairs(genes: Any, value_name: str = "genome mutable") -> tuple[Gene, ...]:
    """Normalize iterable gene pairs while preserving their order and values.

    ``genes`` is a JSON-like iterable of ``(gene_type, gene_id)`` pairs and
    ``value_name`` identifies it in errors. The function returns immutable pairs."""
    if isinstance(genes, (str, bytes, Mapping)):
        raise MutableGenomeValidationError(f"{value_name} harus berupa iterable pasangan gene.")
    try:
        raw_genes = tuple(genes)
    except TypeError as error:
        raise MutableGenomeValidationError(
            f"{value_name} harus berupa iterable pasangan gene."
        ) from error

    normalized: list[Gene] = []
    for index, gene in enumerate(raw_genes):
        if isinstance(gene, (str, bytes, Mapping)):
            raise MutableGenomeValidationError(
                f"Gene indeks {index} pada {value_name} harus berupa pasangan (gene_type, gene_id)."
            )
        try:
            pair = tuple(gene)
        except TypeError as error:
            raise MutableGenomeValidationError(
                f"Gene indeks {index} pada {value_name} harus berupa pasangan (gene_type, gene_id)."
            ) from error
        if len(pair) != 2 or any(not isinstance(part, str) or not part.strip() for part in pair):
            raise MutableGenomeValidationError(
                f"Gene indeks {index} pada {value_name} harus berisi dua string non-kosong."
            )
        normalized.append((pair[0], pair[1]))
    return tuple(normalized)


class MutableGenomeContract:
    """Validate mutable genomes against one exact configured allele multiset."""

    def __init__(self, configured_alleles: Iterable[Iterable[str]], included_length: int) -> None:
        """Store configured alleles and their sorter-derived mutable cardinality.

        ``configured_alleles`` defines every allowed pair and ``included_length``
        is supplied by the position contract. The constructor returns no value."""
        if (
            isinstance(included_length, bool)
            or not isinstance(included_length, int)
            or included_length <= 0
        ):
            raise MutableGenomeValidationError("included_length harus berupa integer positif.")
        expected_alleles = normalize_gene_pairs(configured_alleles, "configured mutable alleles")
        if len(expected_alleles) != included_length:
            raise MutableGenomeValidationError(
                f"Configured mutable alleles harus berisi {included_length} gene, tetapi menerima {len(expected_alleles)}."
            )
        expected_inventory = Counter(expected_alleles)
        duplicate_alleles = sorted(
            allele for allele, count in expected_inventory.items() if count > 1
        )
        if duplicate_alleles:
            # PMX requires a one-to-one mapping from each allele to its locus.
            raise MutableGenomeValidationError(
                f"Configured mutable alleles tidak boleh memuat pasangan duplikat: {duplicate_alleles}."
            )
        self.expected_alleles = expected_alleles
        self.included_length = included_length
        self._expected_inventory = expected_inventory

    @classmethod
    def from_genome_sorter(
        cls, configured_alleles: Iterable[Iterable[str]], genome_sorter: Any
    ) -> "MutableGenomeContract":
        """Build a contract from configured alleles and a sorter's mutable length.

        ``configured_alleles`` supplies inventory and ``genome_sorter`` supplies
        ``included_length``. The method returns a validated domain contract."""
        included_length = getattr(genome_sorter, "included_length", None)
        return cls(configured_alleles, included_length)

    @classmethod
    def from_checkpoint_contract(
        cls, genome_contract: Mapping[str, Any]
    ) -> "MutableGenomeContract":
        """Rebuild the domain contract from a checkpoint genome contract.

        ``genome_contract`` must contain ``included_length`` and
        ``mutable_alleles``. The method returns a validator for resumed genomes."""
        if not isinstance(genome_contract, Mapping):
            raise MutableGenomeValidationError("genome_contract harus berupa mapping.")
        if "included_length" not in genome_contract or "mutable_alleles" not in genome_contract:
            raise MutableGenomeValidationError(
                "genome_contract harus memuat included_length dan mutable_alleles."
            )
        return cls(genome_contract["mutable_alleles"], genome_contract["included_length"])

    def validate(
        self, genome: Iterable[Iterable[str]], value_name: str = "Genome mutable"
    ) -> tuple[Gene, ...]:
        """Validate one genome and return its normalized immutable representation.

        ``genome`` is checked as an ordered sequence of allele pairs and
        ``value_name`` labels errors. The returned tuple preserves input order."""
        normalized = normalize_gene_pairs(genome, value_name)
        if len(normalized) != self.included_length:
            raise MutableGenomeValidationError(
                f"{value_name} harus berisi {self.included_length} gene, tetapi menerima {len(normalized)}."
            )
        observed_inventory = Counter(normalized)
        # Compare allele counts, not order: placement may change while inventory must remain exact.
        if observed_inventory != self._expected_inventory:
            missing = sorted((self._expected_inventory - observed_inventory).elements())
            extra = sorted((observed_inventory - self._expected_inventory).elements())
            # Validation rejects invalid inventory without repair or resampling.
            raise MutableGenomeValidationError(
                f"{value_name} tidak mempertahankan exact configured allele multiset; missing={missing}, extra={extra}."
            )
        # Preserve locus order; exact tuple equality defines a duplicate genome.
        return normalized

    def checkpoint_inventory(self) -> list[list[str]]:
        """Return the configured allele inventory in strict JSON-compatible form."""
        return [[gene_type, gene_id] for gene_type, gene_id in self.expected_alleles]
