"""Persist and restore the durable optimization campaign state.

CheckpointManager validates genomes and lifecycle transitions, records attempts,
results, and survivors, and writes checkpoints atomically. Loading verifies
input provenance and compatibility before migration or generation extension."""

from __future__ import annotations

import json
import hashlib
import logging
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, MutableMapping
from copy import deepcopy
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from modules.keff_constraint import KeffConstraint
from modules.nsga2 import extract_eligible_candidates

from modules.campaign_identity import validate_campaign_id
from modules.mutable_genome import MutableGenomeContract, MutableGenomeValidationError
from modules.input_provenance import (
    InputProvenanceError,
    legacy_config_matches_generation_extension,
    refresh_file_fingerprints,
    require_matching_input_provenance,
    validate_input_provenance,
)


PENDING_STATUS = "pending"
RUNNING_STATUS = "running"
COMPLETED_STATUS = "completed"
NOT_CONVERGED_STATUS = "not_converged"
FAILED_STATUS = "failed"
TERMINAL_EVALUATION_STATUSES = frozenset({COMPLETED_STATUS, NOT_CONVERGED_STATUS, FAILED_STATUS})
NSGA2_ELIGIBLE_STATUSES = frozenset({COMPLETED_STATUS})
RESULTS_EVALUATION_STATUSES = frozenset({COMPLETED_STATUS, NOT_CONVERGED_STATUS})


class CheckpointError(RuntimeError):
    """Base exception for checkpoint persistence failures."""


class CheckpointValidationError(CheckpointError):
    """Raised when a checkpoint state violates the expected schema."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when a checkpoint belongs to another campaign or configuration."""


class InsufficientEligibleParentsError(CheckpointValidationError):
    """Raised when completed parents cannot satisfy an algorithm requirement."""


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _to_json_compatible(value: Any, value_name: str = "value") -> Any:
    """Convert supported Python and NumPy values into strict JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CheckpointValidationError(f"{value_name} mengandung float non-finite: {value!r}.")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _to_json_compatible(value.item(), value_name)
    if isinstance(value, np.ndarray):
        return _to_json_compatible(value.tolist(), value_name)
    if isinstance(value, Mapping):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CheckpointValidationError(f"Semua key pada {value_name} harus berupa string.")
            converted[key] = _to_json_compatible(
                item,
                f"{value_name}.{key}",
            )
        return converted
    if isinstance(value, (list, tuple)):
        return [
            _to_json_compatible(item, f"{value_name}[{index}]") for index, item in enumerate(value)
        ]
    raise CheckpointValidationError(
        f"{value_name} bertipe {type(value).__name__} dan tidak dapat disimpan ke JSON."
    )


def _lists_to_tuples(value: Any) -> Any:
    """Recursively convert JSON lists back to tuples for Python RNG state."""
    if isinstance(value, list):
        return tuple(_lists_to_tuples(item) for item in value)
    if isinstance(value, dict):
        return {key: _lists_to_tuples(item) for key, item in value.items()}
    return value


def _normalize_string_ids(values: Iterable[str], value_name: str) -> list[str]:
    """Materialize unique, non-empty string identifiers without changing order."""
    if isinstance(values, (str, bytes)):
        raise CheckpointValidationError(f"{value_name} harus berupa iterable string.")
    normalized = []
    for item in values:
        if not isinstance(item, str) or not item.strip():
            raise CheckpointValidationError(
                f"Setiap item pada {value_name} harus berupa string non-kosong."
            )
        normalized.append(item.strip())
    if len(set(normalized)) != len(normalized):
        raise CheckpointValidationError(f"{value_name} mengandung ID duplikat.")
    return normalized


def _normalize_genome(genome: Iterable[Iterable[str]], expected_length: int) -> list[list[str]]:
    """Validate a genome of ``(gene_type, gene_id)`` pairs for JSON storage."""
    if isinstance(genome, (str, bytes)):
        raise CheckpointValidationError("genome harus berupa iterable pasangan gene.")
    try:
        genes = list(genome)
    except TypeError as exc:
        raise CheckpointValidationError("genome harus berupa iterable.") from exc

    if len(genes) != expected_length:
        raise CheckpointValidationError(
            f"Genome harus berisi {expected_length} gene, tetapi menerima {len(genes)}."
        )

    normalized = []
    for index, gene in enumerate(genes):
        if isinstance(gene, (str, bytes)):
            raise CheckpointValidationError(
                f"Gene indeks {index} harus berbentuk (gene_type, gene_id)."
            )
        try:
            gene_parts = list(gene)
        except TypeError as exc:
            raise CheckpointValidationError(
                f"Gene indeks {index} harus berupa iterable dua string."
            ) from exc
        if len(gene_parts) != 2 or any(
            not isinstance(part, str) or not part.strip() for part in gene_parts
        ):
            raise CheckpointValidationError(
                f"Gene indeks {index} harus berbentuk dua string non-kosong."
            )
        normalized.append([part.strip() for part in gene_parts])
    return normalized


def _normalize_integer(
    value: Any,
    value_name: str,
    minimum: int = 0,
) -> int:
    """Return a Python integer after validating NumPy or Python integer input.

    Parameters
    ----------
    value : object
        Candidate integer, including a possible NumPy integer scalar.
    value_name : str
        Parameter name used in the validation error.
    minimum : int, optional
        Smallest accepted value.

    Returns
    -------
    int
        Normalized built-in Python integer."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise CheckpointValidationError(
            f"{value_name} harus berupa integer dengan nilai minimum {minimum}."
        )
    normalized = int(value)
    if normalized < minimum:
        raise CheckpointValidationError(
            f"{value_name} harus berupa integer dengan nilai minimum {minimum}."
        )
    return normalized


class CheckpointManager:
    """Build, mutate, atomically save, and recover one GA campaign state."""

    SCHEMA_VERSION = 4
    LEGACY_SCHEMA_VERSION = 3
    TERMINAL_EVALUATION_STATUSES = TERMINAL_EVALUATION_STATUSES
    NSGA2_ELIGIBLE_STATUSES = NSGA2_ELIGIBLE_STATUSES
    INDIVIDUAL_STATUSES = frozenset({PENDING_STATUS, RUNNING_STATUS}) | TERMINAL_EVALUATION_STATUSES
    ATTEMPT_STATUSES = (
        frozenset({RUNNING_STATUS, FAILED_STATUS, "interrupted"}) | RESULTS_EVALUATION_STATUSES
    )
    PHASES = {
        "initialized",
        "evaluating_parents",
        "parents_evaluated",
        "creating_offspring",
        "offspring_registered",
        "evaluating_offspring",
        "offspring_evaluated",
        "environmental_selection",
        "generation_completed",
        "finished",
    }

    def __init__(
        self,
        checkpoint_path: str | Path,
        campaign_id: str,
        input_provenance: Mapping[str, Any],
        indent: int | None = 2,
        *,
        constraint_contract: KeffConstraint,
    ) -> None:
        """Initialize the manager with immutable scientific-input provenance.

        Parameters
        ----------
        checkpoint_path : str or pathlib.Path
            JSON file used as the authoritative resumable checkpoint.
        campaign_id : str
            Stable identifier that prevents loading another campaign by mistake.
        input_provenance : mapping
            Versioned hashes for configuration, inventory, materials, and genome contract.
        constraint_contract : KeffConstraint
            Configured BOC mean-keff threshold and tolerance, required for deterministic ranking.
        indent : int or None, optional
            JSON indentation; defaults to two spaces for a readable checkpoint.
            Use ``None`` only when compact one-line output is explicitly required.

        Returns
        -------
        None
            The manager stores the persistence and compatibility settings."""
        campaign_id = validate_campaign_id(campaign_id)
        if indent is not None and (
            isinstance(indent, bool) or not isinstance(indent, int) or indent < 0
        ):
            raise ValueError("indent harus berupa integer non-negatif atau None.")

        if not isinstance(checkpoint_path, (str, Path)) or not str(checkpoint_path).strip():
            raise ValueError("checkpoint_path harus berupa path non-kosong.")
        self.checkpoint_path = Path(checkpoint_path)
        if self.checkpoint_path.exists() and self.checkpoint_path.is_dir():
            raise IsADirectoryError(
                f"checkpoint_path harus menunjuk file, bukan directory: " f"{self.checkpoint_path}."
            )
        self.campaign_id = campaign_id
        self.indent = indent
        self.constraint_contract = constraint_contract
        normalized_provenance = _to_json_compatible(input_provenance, "input_provenance")
        validate_input_provenance(normalized_provenance)
        self.input_provenance = normalized_provenance

    @staticmethod
    def build_genome_contract(
        encoding: str,
        slot_order: Iterable[str],
        excluded_genes: Mapping[str, Iterable[str]],
        mutable_alleles: Iterable[Iterable[str]],
    ) -> dict[str, Any]:
        """Build the shared mutable/physical genome and allele contract.

        Parameters
        ----------
        encoding : str
            Human-readable name of the GA genome representation.
        slot_order : iterable of str
            Complete physical position order expected by the OpenMC builder.
        excluded_genes : mapping
            Fixed ``position -> (gene_type, gene_id)`` assignments.
        mutable_alleles : iterable
            Exact configured ``(gene_type, gene_id)`` inventory owned by the GA.

        Returns
        -------
        dict
            JSON-compatible genome contract used by every individual."""
        if not isinstance(encoding, str) or not encoding.strip():
            raise CheckpointValidationError("encoding harus berupa string non-kosong.")
        slots = _normalize_string_ids(slot_order, "slot_order")
        if not slots:
            raise CheckpointValidationError("slot_order tidak boleh kosong.")
        if not isinstance(excluded_genes, Mapping):
            raise CheckpointValidationError("excluded_genes harus berupa mapping.")

        normalized_excluded_input = {}
        for position, gene in excluded_genes.items():
            if not isinstance(position, str) or not position.strip():
                raise CheckpointValidationError(
                    "Setiap posisi excluded harus berupa string non-kosong."
                )
            normalized_position = position.strip()
            if normalized_position in normalized_excluded_input:
                raise CheckpointValidationError(
                    f"Posisi excluded {normalized_position!r} terdaftar lebih dari sekali."
                )
            normalized_excluded_input[normalized_position] = gene

        unknown_positions = set(normalized_excluded_input).difference(slots)
        if unknown_positions:
            unknown_text = ", ".join(sorted(unknown_positions))
            raise CheckpointValidationError(
                f"Posisi excluded tidak terdapat dalam slot_order: {unknown_text}."
            )

        normalized_excluded = {
            position: _normalize_genome([gene], 1)[0]
            for position, gene in normalized_excluded_input.items()
        }
        included_positions = [position for position in slots if position not in normalized_excluded]
        if not included_positions:
            raise CheckpointValidationError(
                "Genome GA harus memiliki sedikitnya satu included position."
            )
        try:
            mutable_contract = MutableGenomeContract(mutable_alleles, len(included_positions))
        except MutableGenomeValidationError as error:
            raise CheckpointValidationError(str(error)) from error
        return {
            "encoding": encoding.strip(),
            "included_length": len(included_positions),
            "complete_length": len(slots),
            "slot_order": slots,
            "included_positions": included_positions,
            "excluded_genes": normalized_excluded,
            "mutable_alleles": mutable_contract.checkpoint_inventory(),
        }

    def create_state(
        self,
        genome_contract: Mapping[str, Any],
        ga_settings: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        python_rng: Any | None = None,
        numpy_rng: Any | None = None,
    ) -> dict[str, Any]:
        """Create a new empty campaign state with genome and RNG contracts.

        Parameters
        ----------
        genome_contract : mapping
            Contract returned by :meth:`build_genome_contract`.
        ga_settings : mapping, optional
            GA parameters retained for audit, such as population size and seed.
        metadata : mapping, optional
            Additional JSON-compatible campaign information.
        python_rng, numpy_rng : optional
            Explicit RNG objects whose internal states are captured.

        Returns
        -------
        dict
            New state ready to receive individuals and generations."""
        contract = _to_json_compatible(genome_contract, "genome_contract")
        self._validate_genome_contract(contract)
        created_at = _utc_now()
        state = {
            "schema_version": self.SCHEMA_VERSION,
            "constraint_contract": self.constraint_contract.identity(),
            "campaign": {
                "campaign_id": self.campaign_id,
                "created_at": created_at,
                "updated_at": created_at,
            },
            "input_provenance": _to_json_compatible(self.input_provenance, "input_provenance"),
            "genome_contract": contract,
            "ga_settings": _to_json_compatible(ga_settings or {}, "ga_settings"),
            "metadata": _to_json_compatible(metadata or {}, "metadata"),
            "progress": {
                "generation": None,
                "phase": "initialized",
                "active_individual_id": None,
            },
            "rng_state": self.capture_rng_state(python_rng, numpy_rng),
            "individuals": {},
            "generations": [],
        }
        self.validate_state(state)
        return state

    @staticmethod
    def capture_rng_state(
        python_rng: Any | None = None,
        numpy_rng: Any | None = None,
    ) -> dict[str, Any]:
        """Capture explicit Python and NumPy RNG states for deterministic resume.

        Parameters
        ----------
        python_rng : random.Random, optional
            Python random-number generator exposing ``getstate()``.
        numpy_rng : numpy.random.Generator, optional
            NumPy generator exposing ``bit_generator.state``.

        Returns
        -------
        dict
            JSON-compatible RNG states; missing generators are stored as ``None``."""
        if python_rng is not None and not callable(getattr(python_rng, "getstate", None)):
            raise TypeError("python_rng harus menyediakan method getstate().")
        if numpy_rng is not None and not hasattr(numpy_rng, "bit_generator"):
            raise TypeError("numpy_rng harus berupa numpy.random.Generator.")

        return {
            "python": (
                _to_json_compatible(python_rng.getstate(), "python_rng_state")
                if python_rng is not None
                else None
            ),
            "numpy": (
                _to_json_compatible(
                    numpy_rng.bit_generator.state,
                    "numpy_rng_state",
                )
                if numpy_rng is not None
                else None
            ),
        }

    @staticmethod
    def restore_rng_state(
        state: Mapping[str, Any],
        python_rng: Any | None = None,
        numpy_rng: Any | None = None,
    ) -> None:
        """Restore supplied RNG objects from the checkpoint state.

        Parameters
        ----------
        state : mapping
            Loaded checkpoint state containing the ``rng_state`` section.
        python_rng, numpy_rng : optional
            Existing RNG objects that will receive the stored states.

        Returns
        -------
        None
            The supplied RNG objects are modified in place."""
        rng_state = state.get("rng_state")
        if not isinstance(rng_state, Mapping):
            raise CheckpointValidationError("rng_state tidak tersedia atau tidak valid.")

        if python_rng is not None and rng_state.get("python") is not None:
            if not callable(getattr(python_rng, "setstate", None)):
                raise TypeError("python_rng harus menyediakan method setstate().")
            python_rng.setstate(_lists_to_tuples(rng_state["python"]))
        if numpy_rng is not None and rng_state.get("numpy") is not None:
            if not hasattr(numpy_rng, "bit_generator"):
                raise TypeError("numpy_rng harus berupa numpy.random.Generator.")
            numpy_rng.bit_generator.state = rng_state["numpy"]

    def register_individual(
        self,
        state: MutableMapping[str, Any],
        individual_id: str,
        genome: Iterable[Iterable[str]],
        generation_created: int,
        origin: str,
        parent_ids: Iterable[str] = (),
        operator: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add one unique initial or offspring individual to the campaign state.

        Parameters
        ----------
        state : mutable mapping
            Campaign state created by :meth:`create_state`.
        individual_id : str
            Stable ID used by generation membership and OpenMC run records.
        genome : iterable
            Included GA genome ordered like ``included_positions``.
        generation_created : int
            Generation in which this individual first appeared.
        origin : {"initial", "offspring"}
            How the individual entered the campaign.
        parent_ids : iterable of str, optional
            Existing parent IDs for an offspring.
        operator : mapping, optional
            Crossover and mutation metadata used to create the individual.

        Returns
        -------
        dict
            Newly created mutable individual record."""
        self._require_non_empty_id(individual_id, "individual_id")
        generation_created = _normalize_integer(
            generation_created,
            "generation_created",
        )
        if origin not in {"initial", "offspring"}:
            raise CheckpointValidationError("origin harus 'initial' atau 'offspring'.")

        individuals = self._get_individuals(state)
        if individual_id in individuals:
            raise CheckpointValidationError(f"Individual {individual_id!r} sudah terdaftar.")
        parents = _normalize_string_ids(parent_ids, "parent_ids")
        missing_parents = [parent_id for parent_id in parents if parent_id not in individuals]
        if missing_parents:
            raise CheckpointValidationError(
                "Parent individual belum terdaftar: " + ", ".join(missing_parents) + "."
            )
        if origin == "initial" and parents:
            raise CheckpointValidationError("Individual initial tidak boleh memiliki parent_ids.")
        if origin == "offspring" and not parents:
            raise CheckpointValidationError("Individual offspring harus memiliki parent_ids.")
        if origin == "initial" and state["generations"]:
            raise CheckpointValidationError(
                "Individual initial hanya boleh didaftarkan sebelum generasi 0 dimulai."
            )
        if origin == "offspring":
            generation_record = self._find_generation(
                state,
                generation_created,
                required=False,
            )
            if (
                generation_record is None
                or generation_record["status"] != "running"
                or state["progress"].get("generation") != generation_created
                or state["progress"].get("phase")
                not in {"creating_offspring", "evaluating_offspring"}
            ):
                raise CheckpointValidationError(
                    "Individual offspring hanya boleh dibuat pada phase creating_offspring "
                    "atau evaluating_offspring pada generasi aktif."
                )
            ineligible_parents = [
                parent_id
                for parent_id in parents
                if individuals[parent_id]["status"] not in NSGA2_ELIGIBLE_STATUSES
            ]
            if ineligible_parents:
                raise CheckpointValidationError(
                    "Parent offspring harus completed dan eligible NSGA-II: "
                    + ", ".join(ineligible_parents)
                    + "."
                )

        mutable_contract = self._build_mutable_genome_contract(state["genome_contract"])
        try:
            normalized_genome = mutable_contract.validate(
                genome, f"Genome individual {individual_id!r}"
            )
        except MutableGenomeValidationError as error:
            raise CheckpointValidationError(str(error)) from error
        timestamp = _utc_now()
        individual = {
            "generation_created": generation_created,
            "origin": origin,
            "parent_ids": parents,
            "operator": _to_json_compatible(operator or {}, "operator"),
            "genome": [[gene_type, gene_id] for gene_type, gene_id in normalized_genome],
            "status": PENDING_STATUS,
            "attempts": [],
            "results": None,
            "nsga2": {
                "rank": None,
                "crowding_distance": None,
                "is_boundary": False,
            },
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        individuals[individual_id] = individual
        return individual

    @staticmethod
    def format_offspring_id(generation: int, offspring_index: int) -> str:
        """Return the stable ID for one offspring position in a generation.

        ``generation`` is the active generation record and ``offspring_index``
        is its zero-based deterministic position. The return value never
        depends on RNG state or on the order of a mapping."""
        normalized_generation = _normalize_integer(generation, "generation")
        normalized_index = _normalize_integer(offspring_index, "offspring_index")
        return f"G{normalized_generation:03d}-O{normalized_index:03d}"

    def register_generation_offspring(
        self,
        state: MutableMapping[str, Any],
        generation: int,
        offspring_records: Iterable[Mapping[str, Any]],
    ) -> tuple[str, ...]:
        """Register one complete offspring population and link it atomically.

        ``generation`` identifies the active parent generation. Each ordered
        record must contain ``genome``, ``parent_ids``, ``crossover``, and
        ``mutation``. The method validates the complete batch before changing
        ``state``; it never launches evaluation or transport. It returns the
        deterministic offspring IDs in registration order."""
        generation = _normalize_integer(generation, "generation")
        generation_record = self._find_generation(state, generation)
        if (
            generation_record["status"] != "running"
            or state["progress"].get("generation") != generation
        ):
            raise CheckpointValidationError(
                f"Generasi {generation} bukan generasi aktif untuk offspring."
            )
        if state["progress"].get("phase") != "creating_offspring":
            raise CheckpointValidationError(
                "Offspring batch hanya boleh didaftarkan pada phase creating_offspring."
            )
        if generation_record["offspring_ids"]:
            raise CheckpointValidationError(
                f"offspring_ids generasi {generation} sudah terdaftar; registrasi duplikat dilarang."
            )
        if isinstance(offspring_records, (str, bytes, Mapping)):
            raise CheckpointValidationError("offspring_records harus iterable record berurutan.")
        try:
            raw_records = list(offspring_records)
        except TypeError as error:
            raise CheckpointValidationError(
                "offspring_records harus iterable record berurutan."
            ) from error
        if not raw_records:
            raise CheckpointValidationError("offspring_records tidak boleh kosong.")

        individuals = self._get_individuals(state)
        orphan_ids = [
            individual_id
            for individual_id, record in individuals.items()
            if record.get("origin") == "offspring"
            and record.get("generation_created") == generation
        ]
        if orphan_ids:
            raise CheckpointValidationError(
                f"Registry offspring generasi {generation} terisi sebagian atau tidak terhubung: "
                + ", ".join(orphan_ids)
                + "."
            )

        normalized_records = []
        active_parent_ids = set(generation_record["parent_ids"])
        for offspring_index, raw_record in enumerate(raw_records):
            if not isinstance(raw_record, Mapping):
                raise CheckpointValidationError(
                    f"Record offspring indeks {offspring_index} harus berupa mapping."
                )
            required_keys = {"genome", "parent_ids", "crossover", "mutation"}
            missing_keys = required_keys.difference(raw_record)
            if missing_keys:
                raise CheckpointValidationError(
                    f"Record offspring indeks {offspring_index} kehilangan key: "
                    + ", ".join(sorted(missing_keys))
                    + "."
                )
            parent_ids = _normalize_string_ids(
                raw_record["parent_ids"], f"parent_ids offspring indeks {offspring_index}"
            )
            if len(parent_ids) != 2:
                raise CheckpointValidationError(
                    f"Offspring indeks {offspring_index} harus memiliki tepat dua parent berurutan."
                )
            foreign_parent_ids = [
                parent_id for parent_id in parent_ids if parent_id not in active_parent_ids
            ]
            if foreign_parent_ids:
                raise CheckpointValidationError(
                    f"Parent offspring indeks {offspring_index} bukan parent generation {generation}: "
                    + ", ".join(foreign_parent_ids)
                    + "."
                )
            ineligible_parent_ids = [
                parent_id
                for parent_id in parent_ids
                if individuals[parent_id]["status"] not in NSGA2_ELIGIBLE_STATUSES
            ]
            if ineligible_parent_ids:
                raise CheckpointValidationError(
                    f"Parent offspring harus completed dan eligible NSGA-II: "
                    + ", ".join(ineligible_parent_ids)
                    + "."
                )
            crossover_metadata = _to_json_compatible(
                raw_record["crossover"], f"crossover offspring indeks {offspring_index}"
            )
            mutation_metadata = _to_json_compatible(
                raw_record["mutation"], f"mutation offspring indeks {offspring_index}"
            )
            if not isinstance(crossover_metadata, Mapping) or not crossover_metadata:
                raise CheckpointValidationError(
                    f"Metadata crossover offspring indeks {offspring_index} harus mapping non-kosong."
                )
            if not isinstance(mutation_metadata, Mapping) or not mutation_metadata:
                raise CheckpointValidationError(
                    f"Metadata mutation offspring indeks {offspring_index} harus mapping non-kosong."
                )
            metadata_parent_ids = crossover_metadata.get("parent_ids")
            if metadata_parent_ids != parent_ids:
                raise CheckpointValidationError(
                    f"Metadata crossover offspring indeks {offspring_index} tidak cocok dengan parent_ids berurutan."
                )
            individual_id = self.format_offspring_id(generation, offspring_index)
            if individual_id in individuals:
                raise CheckpointValidationError(
                    f"ID offspring deterministik {individual_id!r} sudah terdaftar."
                )
            duplicate_handling = _to_json_compatible(
                raw_record.get(
                    "duplicate_handling",
                    {
                        "mode": "legacy_unspecified",
                        "duplicate_parent_id": None,
                        "fitness_source_parent_id": None,
                        "fitness_reused": False,
                    },
                ),
                f"duplicate_handling offspring indeks {offspring_index}",
            )
            if not isinstance(duplicate_handling, Mapping):
                raise CheckpointValidationError(
                    f"Metadata duplicate handling offspring indeks {offspring_index} harus berupa mapping."
                )
            normalized_records.append(
                (
                    individual_id,
                    raw_record["genome"],
                    parent_ids,
                    crossover_metadata,
                    mutation_metadata,
                    duplicate_handling,
                )
            )

        offspring_ids = tuple(record[0] for record in normalized_records)
        if len(set(offspring_ids)) != len(offspring_ids):
            raise CheckpointValidationError("ID offspring deterministik mengandung duplikat.")
        for (
            individual_id,
            genome,
            parent_ids,
            crossover_metadata,
            mutation_metadata,
            duplicate_handling,
        ) in normalized_records:
            self.register_individual(
                state,
                individual_id=individual_id,
                genome=genome,
                generation_created=generation,
                origin="offspring",
                parent_ids=parent_ids,
                operator={
                    "crossover": crossover_metadata,
                    "mutation": mutation_metadata,
                    "duplicate_handling": duplicate_handling,
                },
            )
        self.set_generation_offspring(state, generation, offspring_ids)
        return offspring_ids

    def begin_generation(
        self,
        state: MutableMapping[str, Any],
        generation: int,
        parent_ids: Iterable[str],
    ) -> dict[str, Any]:
        """Create one generation record and begin evaluation of its parents.

        Parameters
        ----------
        state : mutable mapping
            Campaign state containing all referenced parent individuals.
        generation : int
            Non-negative generation number, unique within the campaign.
        parent_ids : iterable of str
            Population evaluated as parents at this generation.

        Returns
        -------
        dict
            Newly appended generation record."""
        generation = _normalize_integer(generation, "generation")
        if self._find_generation(state, generation, required=False) is not None:
            raise CheckpointValidationError(f"Generasi {generation} sudah terdaftar.")

        parents = _normalize_string_ids(parent_ids, "parent_ids")
        if not parents:
            raise CheckpointValidationError("parent_ids tidak boleh kosong.")
        self._require_known_individuals(state, parents, "parent_ids")
        previous_generations = state["generations"]
        if not previous_generations and generation != 0:
            raise CheckpointValidationError("Campaign harus dimulai dari generasi 0.")
        if previous_generations:
            previous = previous_generations[-1]
            expected_generation = previous["generation"] + 1
            if previous.get("status") != "completed":
                raise CheckpointValidationError(
                    "Generasi sebelumnya harus completed sebelum generasi baru dimulai."
                )
            if generation != expected_generation:
                raise CheckpointValidationError(
                    f"Generasi berikutnya harus {expected_generation}, bukan {generation}."
                )
            if parents != previous["survivor_ids"]:
                raise CheckpointValidationError(
                    "parent_ids generasi baru harus sama dan seurut dengan survivor_ids "
                    "generasi sebelumnya."
                )
        timestamp = _utc_now()
        generation_record = {
            "generation": generation,
            "parent_ids": parents,
            "offspring_ids": [],
            "survivor_ids": [],
            "status": "running",
            "summary": {},
            "started_at": timestamp,
            "completed_at": None,
        }
        state["generations"].append(generation_record)
        self.set_progress(state, generation, "evaluating_parents")
        return generation_record

    def set_generation_offspring(
        self,
        state: MutableMapping[str, Any],
        generation: int,
        offspring_ids: Iterable[str],
    ) -> None:
        """Record offspring membership and stop at durable registration.

        Parameters
        ----------
        state : mutable mapping
            Campaign state containing the generation and offspring records.
        generation : int
            Generation receiving the offspring population.
        offspring_ids : iterable of str
            Registered offspring individuals; evaluation remains outside P2-3.

        Returns
        -------
        None
            The generation and progress sections are updated in place without
            enabling offspring evaluation."""
        generation = _normalize_integer(generation, "generation")
        if state["progress"].get("phase") not in {
            "creating_offspring",
            "offspring_registered",
        }:
            raise CheckpointValidationError(
                "Panggil begin_offspring_creation() sebelum menetapkan offspring."
            )
        offspring = _normalize_string_ids(offspring_ids, "offspring_ids")
        if not offspring:
            raise CheckpointValidationError("offspring_ids tidak boleh kosong.")
        self._require_known_individuals(state, offspring, "offspring_ids")
        generation_record = self._find_generation(state, generation)
        if generation_record["status"] != "running":
            raise CheckpointValidationError(
                f"Generasi {generation} sudah tidak dapat menerima offspring."
            )
        existing_offspring = generation_record["offspring_ids"]
        if existing_offspring and existing_offspring != offspring:
            raise CheckpointValidationError(
                f"offspring_ids generasi {generation} sudah ditetapkan dan tidak boleh ditimpa."
            )
        for individual_id in offspring:
            individual = state["individuals"][individual_id]
            if (
                individual["origin"] != "offspring"
                or individual["generation_created"] != generation
            ):
                raise CheckpointValidationError(
                    f"Individual {individual_id!r} bukan offspring generasi {generation}."
                )
        generation_record["offspring_ids"] = offspring
        self.set_progress(state, generation, "offspring_registered")

    def begin_offspring_creation(
        self,
        state: MutableMapping[str, Any],
        generation: int,
    ) -> None:
        """Move an active generation from parent evaluation to offspring creation.

        Parameters
        ----------
        state : mutable mapping
            Campaign state whose parent population has finished evaluation.
        generation : int
            Active generation that will create an offspring population.

        Returns
        -------
        None
            Progress changes to ``creating_offspring`` after validation."""
        generation = _normalize_integer(generation, "generation")
        generation_record = self._find_generation(state, generation)
        if generation_record["status"] != "running":
            raise CheckpointValidationError(f"Generasi {generation} sudah tidak aktif.")
        if state["progress"].get("active_individual_id") is not None:
            raise CheckpointValidationError(
                "Offspring tidak dapat dibuat ketika individual masih running."
            )
        nonterminal_parents = [
            individual_id
            for individual_id in generation_record["parent_ids"]
            if state["individuals"][individual_id]["status"] not in TERMINAL_EVALUATION_STATUSES
        ]
        if nonterminal_parents:
            raise CheckpointValidationError(
                "Semua parent harus terminal sebelum membuat offspring: "
                + ", ".join(nonterminal_parents)
                + "."
            )
        self.set_progress(state, generation, "creating_offspring")

    def begin_offspring_evaluation(self, state: MutableMapping[str, Any], generation: int) -> None:
        """Move a registered offspring population into its evaluation phase.

        ``state`` identifies the active campaign and ``generation`` identifies
        its already durable offspring registry. The method returns no value and
        never creates offspring, selects survivors, or starts another generation."""
        generation = _normalize_integer(generation, "generation")
        generation_record = self._find_generation(state, generation)
        phase = state["progress"].get("phase")
        if (
            generation_record["status"] != "running"
            or state["progress"].get("generation") != generation
        ):
            raise CheckpointValidationError(
                f"Generasi {generation} bukan generasi aktif untuk evaluasi offspring."
            )
        if phase not in {"offspring_registered", "evaluating_offspring"}:
            raise CheckpointValidationError(
                "Evaluasi offspring memerlukan phase offspring_registered atau evaluating_offspring."
            )
        if state["progress"].get("active_individual_id") is not None:
            raise CheckpointValidationError(
                "Evaluasi offspring tidak dapat dimulai ketika individual masih running."
            )
        offspring_ids = generation_record["offspring_ids"]
        if not offspring_ids:
            raise CheckpointValidationError(
                "Evaluasi offspring memerlukan offspring_ids yang sudah terdaftar."
            )
        invalid_statuses = [
            individual_id
            for individual_id in offspring_ids
            if state["individuals"][individual_id]["status"]
            not in TERMINAL_EVALUATION_STATUSES | {PENDING_STATUS}
        ]
        if invalid_statuses:
            raise CheckpointValidationError(
                "Status offspring tidak dapat dievaluasi ulang: "
                + ", ".join(invalid_statuses)
                + "."
            )
        self.set_progress(state, generation, "evaluating_offspring")

    def mark_offspring_evaluation_complete(
        self, state: MutableMapping[str, Any], generation: int
    ) -> None:
        """Close offspring evaluation after every registered child is terminal.

        ``state`` is the active campaign and ``generation`` is its active
        generation. The method returns no value and leaves replacement and
        future-generation orchestration to a later milestone."""
        generation = _normalize_integer(generation, "generation")
        generation_record = self._find_generation(state, generation)
        phase = state["progress"].get("phase")
        if (
            generation_record["status"] != "running"
            or state["progress"].get("generation") != generation
        ):
            raise CheckpointValidationError(
                f"Generasi {generation} bukan generasi aktif untuk menutup evaluasi offspring."
            )
        if phase not in {"evaluating_offspring", "offspring_evaluated"}:
            raise CheckpointValidationError(
                "Evaluasi offspring hanya dapat ditutup pada phase evaluating_offspring."
            )
        if state["progress"].get("active_individual_id") is not None:
            raise CheckpointValidationError(
                "Evaluasi offspring tidak dapat ditutup ketika individual masih running."
            )
        nonterminal_ids = [
            individual_id
            for individual_id in generation_record["offspring_ids"]
            if state["individuals"][individual_id]["status"] not in TERMINAL_EVALUATION_STATUSES
        ]
        if nonterminal_ids:
            raise CheckpointValidationError(
                "Semua offspring harus terminal sebelum evaluasi ditutup: "
                + ", ".join(nonterminal_ids)
                + "."
            )
        self.set_progress(state, generation, "offspring_evaluated")

    def begin_environmental_selection(
        self, state: MutableMapping[str, Any], generation: int
    ) -> None:
        """Move one fully evaluated generation into environmental replacement.

        ``state`` contains the active campaign and ``generation`` identifies its
        running generation. The method returns no value and changes only the
        durable lifecycle phase after all parent and offspring are terminal."""
        generation = _normalize_integer(generation, "generation")
        generation_record = self._find_generation(state, generation)
        phase = state["progress"].get("phase")
        if (
            generation_record["status"] != "running"
            or state["progress"].get("generation") != generation
        ):
            raise CheckpointValidationError(
                f"Generasi {generation} bukan generasi aktif untuk environmental selection."
            )
        if phase not in {"offspring_evaluated", "environmental_selection"}:
            raise CheckpointValidationError(
                "Environmental selection memerlukan phase offspring_evaluated atau environmental_selection."
            )
        if state["progress"].get("active_individual_id") is not None:
            raise CheckpointValidationError(
                "Environmental selection tidak dapat dimulai ketika individual masih running."
            )
        candidate_ids = generation_record["parent_ids"] + generation_record["offspring_ids"]
        if not generation_record["offspring_ids"]:
            raise CheckpointValidationError(
                "Environmental selection memerlukan offspring yang sudah dievaluasi."
            )
        nonterminal_ids = [
            individual_id
            for individual_id in candidate_ids
            if state["individuals"][individual_id]["status"] not in TERMINAL_EVALUATION_STATUSES
        ]
        if nonterminal_ids:
            raise CheckpointValidationError(
                "Environmental selection memerlukan seluruh kandidat terminal: "
                + ", ".join(nonterminal_ids)
                + "."
            )
        self.set_progress(state, generation, "environmental_selection")

    def mark_parent_evaluation_complete(
        self, state: MutableMapping[str, Any], generation: int
    ) -> None:
        """Mark all terminal parents ready for a later NSGA-II stage.

        ``state`` contains the active generation and ``generation`` identifies
        it. The method returns no value and does not create offspring, assign
        NSGA-II metrics, or complete the generation."""
        generation = _normalize_integer(generation, "generation")
        generation_record = self._find_generation(state, generation)
        if generation_record["status"] != "running":
            raise CheckpointValidationError(f"Generasi {generation} sudah tidak aktif.")
        if state["progress"].get("generation") != generation or state["progress"].get(
            "phase"
        ) not in {"evaluating_parents", "parents_evaluated"}:
            raise CheckpointValidationError(
                "Parent evaluation hanya dapat ditutup pada phase evaluating_parents generasi aktif."
            )
        if generation_record["offspring_ids"]:
            raise CheckpointValidationError(
                "Parent evaluation tidak dapat ditutup setelah offspring dibuat."
            )
        if state["progress"].get("active_individual_id") is not None:
            raise CheckpointValidationError(
                "Parent evaluation tidak dapat ditutup ketika individual masih running."
            )
        nonterminal_parents = [
            individual_id
            for individual_id in generation_record["parent_ids"]
            if state["individuals"][individual_id]["status"] not in TERMINAL_EVALUATION_STATUSES
        ]
        if nonterminal_parents:
            raise CheckpointValidationError(
                "Semua parent harus terminal sebelum parent evaluation ditutup: "
                + ", ".join(nonterminal_parents)
                + "."
            )
        self.set_progress(state, generation, "parents_evaluated")

    def get_nsga2_eligible_parent_ids(self, state: Mapping[str, Any], generation: int) -> list[str]:
        """Return completed parent IDs in their deterministic generation order.

        Parameters are the campaign state and generation number whose parent
        population is inspected. The returned list contains only parents eligible
        for NSGA-II rank, crowding, parent selection, or survival."""
        generation = _normalize_integer(generation, "generation")
        generation_record = self._find_generation(state, generation)
        return [
            individual_id
            for individual_id in generation_record["parent_ids"]
            if state["individuals"][individual_id]["status"] in NSGA2_ELIGIBLE_STATUSES
        ]

    def require_nsga2_eligible_parents(
        self, state: Mapping[str, Any], generation: int, required_count: int
    ) -> list[str]:
        """Require enough completed parents and return their IDs in parent order.

        ``state`` and ``generation`` identify the parent population, while
        ``required_count`` states the caller's algorithm requirement. The method
        returns eligible IDs or raises a domain error without changing campaign state."""
        generation = _normalize_integer(generation, "generation")
        required_count = _normalize_integer(required_count, "required_count", minimum=1)
        generation_record = self._find_generation(state, generation)
        eligible_parent_ids = self.get_nsga2_eligible_parent_ids(state, generation)
        if len(eligible_parent_ids) >= required_count:
            return eligible_parent_ids

        ineligible_details = [
            f"{individual_id}={state['individuals'][individual_id]['status']}"
            for individual_id in generation_record["parent_ids"]
            if individual_id not in eligible_parent_ids
        ]
        eligible_text = ", ".join(eligible_parent_ids) if eligible_parent_ids else "none"
        ineligible_text = ", ".join(ineligible_details) if ineligible_details else "none"
        raise InsufficientEligibleParentsError(
            f"Insufficient eligible parents pada generasi {generation}: diperlukan {required_count} completed, tersedia {len(eligible_parent_ids)}; eligible={eligible_text}; ineligible={ineligible_text}."
        )

    def set_progress(
        self,
        state: MutableMapping[str, Any],
        generation: int | None,
        phase: str,
        active_individual_id: str | None = None,
    ) -> None:
        """Update the resumable generation, lifecycle phase, and active individual.

        Parameters
        ----------
        state : mutable mapping
            Campaign state whose progress cursor will be updated.
        generation : int or None
            Current generation, or ``None`` before the first generation.
        phase : str
            Lifecycle phase listed in :attr:`PHASES`.
        active_individual_id : str, optional
            Individual currently running OpenMC.

        Returns
        -------
        None
            The progress cursor is modified in place."""
        if generation is not None:
            generation = _normalize_integer(generation, "generation progress")
        if phase not in self.PHASES:
            raise CheckpointValidationError(f"Phase checkpoint tidak dikenal: {phase!r}.")
        if active_individual_id is not None:
            self._require_known_individuals(
                state,
                [active_individual_id],
                "active_individual_id",
            )
        state["progress"] = {
            "generation": generation,
            "phase": phase,
            "active_individual_id": active_individual_id,
        }

    def mark_individual_running(
        self,
        state: MutableMapping[str, Any],
        individual_id: str,
        run_id: str,
        work_dir: str | Path,
        openmc_seed: int,
    ) -> dict[str, Any]:
        """Start a recorded OpenMC attempt before launching particle transport.

        Parameters
        ----------
        state : mutable mapping
            Campaign state containing a pending individual.
        individual_id : str
            Individual whose simulation will start.
        run_id : str
            Unique OpenMC run identifier for this attempt.
        work_dir : str or pathlib.Path
            Intended directory for the OpenMC output files.
        openmc_seed : int
            Explicit positive OpenMC random-number seed.

        Returns
        -------
        dict
            Newly appended running-attempt record."""
        self._require_non_empty_id(run_id, "run_id")
        if not isinstance(work_dir, (str, Path)) or not str(work_dir).strip():
            raise CheckpointValidationError("work_dir harus berupa path non-kosong.")
        openmc_seed = _normalize_integer(openmc_seed, "openmc_seed", minimum=1)
        individual = self._get_individual(state, individual_id)
        if individual["status"] != PENDING_STATUS:
            raise CheckpointValidationError(
                f"Individual {individual_id!r} harus berstatus pending sebelum dijalankan."
            )
        progress = state["progress"]
        generation_record = self._find_generation(
            state,
            progress.get("generation"),
            required=False,
        )
        population_key_by_phase = {
            "evaluating_parents": "parent_ids",
            "evaluating_offspring": "offspring_ids",
        }
        population_key = population_key_by_phase.get(progress.get("phase"))
        if (
            generation_record is None
            or population_key is None
            or individual_id not in generation_record[population_key]
        ):
            raise CheckpointValidationError(
                f"Individual {individual_id!r} bukan anggota populasi yang sedang dievaluasi."
            )

        active_id = state["progress"].get("active_individual_id")
        if active_id is not None:
            raise CheckpointValidationError(f"Individual {active_id!r} masih tercatat aktif.")
        for registered_individual in self._get_individuals(state).values():
            if any(
                attempt.get("run_id") == run_id
                for attempt in registered_individual.get("attempts", [])
            ):
                raise CheckpointValidationError(f"run_id {run_id!r} sudah pernah digunakan.")

        attempt = {
            "attempt_number": len(individual["attempts"]) + 1,
            "run_id": run_id,
            "work_dir": str(work_dir),
            "openmc_seed": openmc_seed,
            "status": RUNNING_STATUS,
            "statepoint_path": None,
            "statepoint_retained": False,
            "statepoint_removed_at": None,
            "retention_reason": None,
            "error": None,
            "started_at": _utc_now(),
            "finished_at": None,
        }
        individual["attempts"].append(attempt)
        individual["status"] = RUNNING_STATUS
        individual["updated_at"] = _utc_now()
        state["progress"]["active_individual_id"] = individual_id
        return attempt

    def mark_individual_completed(
        self,
        state: MutableMapping[str, Any],
        individual_id: str,
        results: Mapping[str, Any],
        statepoint_path: str | Path | None = None,
    ) -> None:
        """Commit evaluated physics results and complete the active attempt.

        Parameters
        ----------
        state : mutable mapping
            Campaign state containing the running individual.
        individual_id : str
            Individual whose OpenMC evaluation succeeded.
        results : mapping
            JSON-compatible keff, PPF, raw objectives, power, and diagnostics.
        statepoint_path : str or pathlib.Path, optional
            Produced statepoint path before any retention cleanup.

        Returns
        -------
        None
            The individual and active progress cursor are updated in place."""
        if not isinstance(results, Mapping):
            raise CheckpointValidationError("results harus berupa mapping.")
        if statepoint_path is not None and (
            not isinstance(statepoint_path, (str, Path)) or not str(statepoint_path).strip()
        ):
            raise CheckpointValidationError(
                "statepoint_path harus berupa path non-kosong atau None."
            )
        individual = self._get_individual(state, individual_id)
        attempt = self._get_running_attempt(individual, individual_id)
        self._require_constraint_contract(state)
        normalized_results = _to_json_compatible(results, "results")
        if "constraint" in normalized_results:
            self.constraint_contract.validate_results(normalized_results)
        normalized_results["constraint"] = self.constraint_contract.from_results(normalized_results)

        extract_eligible_candidates(
            {individual_id: {"status": COMPLETED_STATUS, "results": normalized_results}}
        )

        attempt["status"] = COMPLETED_STATUS
        attempt["statepoint_path"] = str(statepoint_path) if statepoint_path is not None else None
        attempt["statepoint_retained"] = statepoint_path is not None
        attempt["statepoint_removed_at"] = None
        attempt["retention_reason"] = None
        attempt["finished_at"] = _utc_now()
        individual["results"] = normalized_results
        individual["status"] = COMPLETED_STATUS
        individual["updated_at"] = _utc_now()
        self._clear_active_individual(state, individual_id)

    def reuse_parent_evaluation(
        self,
        state: MutableMapping[str, Any],
        individual_id: str,
        source_parent_id: str,
        evaluation_context: Mapping[str, Any],
    ) -> None:
        """Copy a completed active parent's results into a pending exact-genome
        offspring. ``evaluation_context`` must exactly match the context saved
        on the parent. The child is marked completed and retains no synthetic
        OpenMC attempt. This method returns ``None``."""
        if not isinstance(evaluation_context, Mapping):
            raise CheckpointValidationError("evaluation_context reuse harus berupa mapping.")
        individual = self._get_individual(state, individual_id)
        source = self._get_individual(state, source_parent_id)
        progress = state.get("progress", {})
        generation = self._find_generation(state, progress.get("generation"), required=False)
        if (
            generation is None
            or progress.get("phase") != "evaluating_offspring"
            or individual_id not in generation["offspring_ids"]
        ):
            raise CheckpointValidationError(
                "Reuse fitness hanya boleh dilakukan pada offspring yang sedang dievaluasi."
            )
        if source_parent_id not in generation["parent_ids"]:
            raise CheckpointValidationError(
                "Sumber reuse fitness harus parent aktif pada generasi yang sama."
            )
        if (
            individual.get("status") != PENDING_STATUS
            or individual.get("attempts")
            or individual.get("results") is not None
        ):
            raise CheckpointValidationError(
                "Offspring reuse harus pending dan belum memiliki attempt atau results."
            )
        if source.get("status") != COMPLETED_STATUS or not isinstance(
            source.get("results"), Mapping
        ):
            raise CheckpointValidationError(
                "Sumber reuse fitness harus parent completed dengan results valid."
            )
        mutable_contract = self._build_mutable_genome_contract(state["genome_contract"])
        child_genome = mutable_contract.validate(
            individual["genome"], f"Genome offspring reuse {individual_id!r}"
        )
        source_genome = mutable_contract.validate(
            source["genome"], f"Genome parent reuse {source_parent_id!r}"
        )
        if child_genome != source_genome:
            raise CheckpointValidationError(
                "Reuse fitness memerlukan genome kanonik offspring dan parent yang identik."
            )
        source_operator = source.get("operator")
        source_context = (
            source_operator.get("evaluation_context")
            if isinstance(source_operator, Mapping)
            else None
        )
        normalized_context = _to_json_compatible(evaluation_context, "evaluation_context reuse")
        if source_context != normalized_context:
            raise CheckpointValidationError(
                "Reuse fitness ditolak karena evaluation context parent tidak identik."
            )
        operator = individual.get("operator")
        duplicate_metadata = (
            operator.get("duplicate_handling") if isinstance(operator, MutableMapping) else None
        )
        if (
            not isinstance(duplicate_metadata, MutableMapping)
            or duplicate_metadata.get("mode") != "reuse_parent_fitness"
            or duplicate_metadata.get("duplicate_parent_id") != source_parent_id
        ):
            raise CheckpointValidationError(
                "Metadata duplicate handling offspring tidak mengizinkan reuse parent ini."
            )

        individual["results"] = deepcopy(source["results"])
        operator["evaluation_context"] = deepcopy(normalized_context)
        duplicate_metadata["fitness_source_parent_id"] = source_parent_id
        duplicate_metadata["fitness_reused"] = True
        duplicate_metadata["reuse_reason"] = "identical_genome_and_evaluation_context"
        individual["status"] = COMPLETED_STATUS
        individual["updated_at"] = _utc_now()

    def reuse_cached_evaluation(
        self,
        state: MutableMapping[str, Any],
        individual_id: str,
        results: Mapping[str, Any],
        evaluation_context: Mapping[str, Any],
        cache_provenance: Mapping[str, Any],
    ) -> None:
        """Complete a pending individual with validated cached ``results``.

        ``evaluation_context`` describes the active scientific contract, while
        ``cache_provenance`` identifies the source checkpoint and individual. Update
        the supplied state without creating a synthetic transport attempt; return None."""
        individual = self._get_individual(state, individual_id)
        if (
            individual.get("status") != PENDING_STATUS
            or individual.get("attempts")
            or individual.get("results") is not None
        ):
            raise CheckpointValidationError(
                "Reuse cache hanya menerima individu pending tanpa attempt/results."
            )
        progress = state.get("progress", {})
        generation = self._find_generation(state, progress.get("generation"), required=False)
        if generation is None:
            raise CheckpointValidationError("Reuse cache memerlukan generation aktif.")
        phase = progress.get("phase")
        valid_member = (
            phase == "evaluating_parents" and individual_id in generation["parent_ids"]
        ) or (phase == "evaluating_offspring" and individual_id in generation["offspring_ids"])
        if not valid_member:
            raise CheckpointValidationError(
                "Reuse cache hanya boleh dilakukan pada kandidat yang sedang dievaluasi."
            )
        normalized_results = _to_json_compatible(results, "cached results")
        self.constraint_contract.validate_results(normalized_results)
        normalized_results["constraint"] = self.constraint_contract.from_results(normalized_results)
        extract_eligible_candidates(
            {individual_id: {"status": COMPLETED_STATUS, "results": normalized_results}}
        )
        normalized_context = _to_json_compatible(evaluation_context, "cached evaluation_context")
        normalized_provenance = _to_json_compatible(cache_provenance, "cache_provenance")
        required_provenance = {
            "lookup",
            "genome_hash",
            "context_hash",
            "source_path",
            "source_checkpoint_sha256",
            "source_campaign_id",
            "source_individual_id",
            "validation_mode",
        }
        if not isinstance(normalized_provenance, Mapping) or required_provenance.difference(
            normalized_provenance
        ):
            raise CheckpointValidationError("Provenance cache hit tidak lengkap.")
        valid_cache_modes = {
            "strict",
            "legacy",
            "strict-compatible-openmc-version",
            "legacy-compatible-openmc-version",
        }
        if (
            normalized_provenance.get("lookup") != "hit"
            or normalized_provenance.get("validation_mode") not in valid_cache_modes
        ):
            raise CheckpointValidationError("Provenance cache hit memiliki mode tidak valid.")
        if normalized_provenance.get("validation_mode", "").endswith(
            "compatible-openmc-version"
        ) and (
            not isinstance(normalized_provenance.get("source_openmc_version"), str)
            or not isinstance(normalized_provenance.get("target_openmc_version"), str)
        ):
            raise CheckpointValidationError("Provenance cache lintas versi OpenMC tidak lengkap.")
        operator = individual.get("operator")
        if not isinstance(operator, MutableMapping):
            raise CheckpointValidationError(
                "Metadata operator individual cache harus mutable mapping."
            )
        operator["evaluation_context"] = normalized_context
        operator["evaluation_cache"] = normalized_provenance
        individual["results"] = normalized_results
        individual["status"] = COMPLETED_STATUS
        individual["updated_at"] = _utc_now()

    def set_statepoint_retention(
        self,
        state: MutableMapping[str, Any],
        individual_id: str,
        retained: bool,
        reason: str | None = None,
    ) -> None:
        """Update metadata after retaining or removing a completed statepoint.

        Parameters
        ----------
        state : mutable mapping
            Campaign state containing a completed OpenMC attempt.
        individual_id : str
            Individual whose latest completed statepoint policy changed.
        retained : bool
            Whether the recorded statepoint is still retained on disk.
        reason : str, optional
            Retention or removal reason, such as ``final_pareto`` or ``rejected``.

        Returns
        -------
        None
            Only artifact metadata is changed; this method never deletes a file."""
        if not isinstance(retained, bool):
            raise TypeError("retained harus berupa bool.")
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise CheckpointValidationError("reason harus berupa string non-kosong atau None.")
        individual = self._get_individual(state, individual_id)
        evaluated_attempts = [
            attempt
            for attempt in individual["attempts"]
            if attempt.get("status") in RESULTS_EVALUATION_STATUSES
        ]
        if individual["status"] not in RESULTS_EVALUATION_STATUSES or not evaluated_attempts:
            raise CheckpointValidationError(
                "Retensi statepoint hanya dapat diubah untuk individual yang sudah dievaluasi."
            )
        attempt = evaluated_attempts[-1]
        if attempt.get("statepoint_path") is None:
            raise CheckpointValidationError(
                "Retensi tidak dapat diubah karena statepoint_path tidak tercatat."
            )
        attempt["statepoint_retained"] = retained
        attempt["statepoint_removed_at"] = None if retained else _utc_now()
        attempt["retention_reason"] = reason.strip() if reason is not None else None
        individual["updated_at"] = _utc_now()

    def mark_individual_not_converged(
        self,
        state: MutableMapping[str, Any],
        individual_id: str,
        results: Mapping[str, Any],
        statepoint_path: str | Path,
    ) -> None:
        """Commit valid transport results that fail the statistical gate.

        Parameters are the campaign state, active individual ID, complete
        evaluator/gate results, and retained statepoint path. The method returns
        no value and creates a terminal non-retryable ``not_converged`` outcome."""
        if not isinstance(results, Mapping):
            raise CheckpointValidationError("results harus berupa mapping.")
        if not isinstance(statepoint_path, (str, Path)) or not str(statepoint_path).strip():
            raise CheckpointValidationError("statepoint_path harus berupa path non-kosong.")
        individual = self._get_individual(state, individual_id)
        attempt = self._get_running_attempt(individual, individual_id)
        self._require_constraint_contract(state)
        normalized_results = _to_json_compatible(results, "results")
        if "constraint" in normalized_results:
            self.constraint_contract.validate_results(normalized_results)
        normalized_results["constraint"] = self.constraint_contract.from_results(normalized_results)

        attempt["status"] = NOT_CONVERGED_STATUS
        attempt["statepoint_path"] = str(statepoint_path)
        attempt["statepoint_retained"] = True
        attempt["statepoint_removed_at"] = None
        attempt["retention_reason"] = "statistical_non_convergence"
        attempt["finished_at"] = _utc_now()
        individual["results"] = normalized_results
        individual["status"] = NOT_CONVERGED_STATUS
        individual["updated_at"] = _utc_now()
        self._clear_active_individual(state, individual_id)

    def mark_individual_failed(
        self,
        state: MutableMapping[str, Any],
        individual_id: str,
        error: str | BaseException,
        will_retry: bool,
    ) -> None:
        """Record a real OpenMC failure and choose pending or terminal status.

        Parameters
        ----------
        state : mutable mapping
            Campaign state containing the running individual.
        individual_id : str
            Individual whose current attempt failed.
        error : str or BaseException
            Concise failure information retained for diagnosis.
        will_retry : bool
            Whether the GA may launch another attempt for this individual.

        Returns
        -------
        None
            The attempt becomes failed and the individual becomes pending or failed."""
        if not isinstance(will_retry, bool):
            raise TypeError("will_retry harus berupa bool.")
        individual = self._get_individual(state, individual_id)
        attempt = self._get_running_attempt(individual, individual_id)
        error_text = str(error).strip() or type(error).__name__

        attempt["status"] = FAILED_STATUS
        attempt["error"] = error_text
        attempt["finished_at"] = _utc_now()
        individual["status"] = PENDING_STATUS if will_retry else FAILED_STATUS
        individual["updated_at"] = _utc_now()
        self._clear_active_individual(state, individual_id)

    def recover_interrupted_individuals(
        self,
        state: MutableMapping[str, Any],
        reason: str = "Program berhenti sebelum attempt selesai.",
    ) -> list[str]:
        """Convert unfinished running attempts to interrupted and retryable pending.

        Parameters
        ----------
        state : mutable mapping
            State loaded after a crash, power loss, or forced termination.
        reason : str, optional
            Diagnostic text stored on every recovered attempt.

        Returns
        -------
        list of str
            IDs reset from ``running`` to ``pending`` without counting a failure."""
        if not isinstance(reason, str) or not reason.strip():
            raise CheckpointValidationError("reason interupsi harus string non-kosong.")
        recovered = []
        for individual_id, individual in self._get_individuals(state).items():
            if individual.get("status") != "running":
                continue
            attempt = self._get_running_attempt(individual, individual_id)
            attempt["status"] = "interrupted"
            attempt["error"] = reason
            attempt["finished_at"] = _utc_now()
            individual["status"] = "pending"
            individual["updated_at"] = _utc_now()
            recovered.append(individual_id)

        if recovered:
            state["progress"]["active_individual_id"] = None
        return recovered

    def set_nsga2_metrics(
        self,
        state: MutableMapping[str, Any],
        individual_id: str,
        rank: int,
        crowding_distance: float | None,
        is_boundary: bool = False,
    ) -> None:
        """Store nondominated rank and JSON-safe crowding information.

        Parameters
        ----------
        state : mutable mapping
            Campaign state containing a completed individual.
        individual_id : str
            Individual receiving NSGA-II metadata.
        rank : int
            Non-negative nondominated-front rank.
        crowding_distance : float or None
            Finite distance; use ``None`` for an infinite boundary distance.
        is_boundary : bool, optional
            Whether the individual lies on a crowding-distance boundary.

        Returns
        -------
        None
            The NSGA-II metadata is updated in place."""
        rank = _normalize_integer(rank, "rank")
        if not isinstance(is_boundary, bool):
            raise TypeError("is_boundary harus berupa bool.")
        if crowding_distance is not None:
            crowding_distance = float(crowding_distance)
            if math.isinf(crowding_distance):
                crowding_distance = None
                is_boundary = True
            elif math.isnan(crowding_distance) or crowding_distance < 0.0:
                raise CheckpointValidationError("crowding_distance harus finite dan non-negatif.")

        individual = self._get_individual(state, individual_id)
        if individual["status"] not in NSGA2_ELIGIBLE_STATUSES:
            raise CheckpointValidationError(
                "Metrik NSGA-II hanya boleh diberikan kepada individual completed."
            )
        individual["nsga2"] = {
            "rank": rank,
            "crowding_distance": crowding_distance,
            "is_boundary": is_boundary,
        }
        individual["updated_at"] = _utc_now()

    def complete_generation(
        self,
        state: MutableMapping[str, Any],
        generation: int,
        survivor_ids: Iterable[str],
        summary: Mapping[str, Any] | None = None,
    ) -> None:
        """Finalize NSGA-II replacement and record the next parent population.

        Parameters
        ----------
        state : mutable mapping
            Campaign state containing the evaluated generation.
        generation : int
            Generation whose environmental selection has finished.
        survivor_ids : iterable of str
            Completed individuals selected from parents plus offspring.
        summary : mapping, optional
            Compact generation statistics for reporting.

        Returns
        -------
        None
            The generation is marked completed and progress advances to a safe point."""
        generation = _normalize_integer(generation, "generation")
        generation_record = self._find_generation(state, generation)
        if generation_record["status"] != "running":
            raise CheckpointValidationError(
                f"Generasi {generation} sudah completed dan tidak boleh ditulis ulang."
            )
        if state["progress"].get("active_individual_id") is not None:
            raise CheckpointValidationError(
                "Generasi tidak dapat diselesaikan ketika individual masih running."
            )
        survivors = _normalize_string_ids(survivor_ids, "survivor_ids")
        if not survivors:
            raise CheckpointValidationError("survivor_ids tidak boleh kosong.")
        self._require_known_individuals(state, survivors, "survivor_ids")
        candidate_ids = set(generation_record["parent_ids"]) | set(
            generation_record["offspring_ids"]
        )
        invalid_survivors = [item for item in survivors if item not in candidate_ids]
        if invalid_survivors:
            raise CheckpointValidationError(
                "Survivor bukan anggota parent/offspring: " + ", ".join(invalid_survivors) + "."
            )
        nonterminal_candidates = [
            item
            for item in candidate_ids
            if state["individuals"][item]["status"] not in TERMINAL_EVALUATION_STATUSES
        ]
        if nonterminal_candidates:
            raise CheckpointValidationError(
                "Semua kandidat harus terminal sebelum replacement: "
                + ", ".join(sorted(nonterminal_candidates))
                + "."
            )
        incomplete_survivors = [
            item
            for item in survivors
            if state["individuals"][item]["status"] not in NSGA2_ELIGIBLE_STATUSES
        ]
        if incomplete_survivors:
            raise CheckpointValidationError(
                "Survivor belum completed: " + ", ".join(incomplete_survivors) + "."
            )

        generation_record["survivor_ids"] = survivors
        generation_record["summary"] = _to_json_compatible(summary or {}, "summary")
        generation_record["status"] = "completed"
        generation_record["completed_at"] = _utc_now()
        self.set_progress(state, generation, "generation_completed")

    def get_pending_individual_ids(
        self,
        state: Mapping[str, Any],
        generation: int,
        population: str,
    ) -> list[str]:
        """Return pending parent or offspring IDs in their population order.

        Parameters
        ----------
        state : mapping
            Loaded or in-memory campaign state.
        generation : int
            Generation whose population will be inspected.
        population : {"parents", "offspring"}
            Population group used to choose the ordered ID list.

        Returns
        -------
        list of str
            IDs whose current status is ``pending``."""
        if population not in {"parents", "offspring"}:
            raise ValueError("population harus 'parents' atau 'offspring'.")
        generation = _normalize_integer(generation, "generation")
        generation_record = self._find_generation(state, generation)
        population_key = "parent_ids" if population == "parents" else "offspring_ids"
        member_ids = generation_record[population_key]
        return [
            individual_id
            for individual_id in member_ids
            if state["individuals"][individual_id]["status"] == "pending"
        ]

    def mark_finished(self, state: MutableMapping[str, Any]) -> None:
        """Mark the whole campaign finished while retaining its final generation."""
        if state["progress"].get("active_individual_id") is not None:
            raise CheckpointValidationError(
                "Campaign tidak dapat finished ketika individual masih running."
            )
        if not state["generations"]:
            raise CheckpointValidationError(
                "Campaign belum memiliki generasi yang dapat diselesaikan."
            )
        if state["generations"][-1]["status"] != "completed":
            raise CheckpointValidationError(
                "Generasi terakhir harus completed sebelum campaign finished."
            )
        generation = state["progress"].get("generation")
        self.set_progress(state, generation, "finished")

    def exists(self) -> bool:
        """Return whether the configured checkpoint JSON currently exists."""
        return self.checkpoint_path.is_file()

    def save(
        self,
        state: MutableMapping[str, Any],
        python_rng: Any | None = None,
        numpy_rng: Any | None = None,
    ) -> None:
        """Validate and atomically replace the authoritative checkpoint JSON.

        Parameters
        ----------
        state : mutable mapping
            Complete in-memory campaign state.
        python_rng, numpy_rng : optional
            Current RNG objects captured immediately before persistence.

        Returns
        -------
        None
            The checkpoint is durable when this method returns successfully."""
        if python_rng is not None or numpy_rng is not None:
            captured_rng = self.capture_rng_state(python_rng, numpy_rng)
            if python_rng is not None:
                state["rng_state"]["python"] = captured_rng["python"]
            if numpy_rng is not None:
                state["rng_state"]["numpy"] = captured_rng["numpy"]
        state["campaign"]["updated_at"] = _utc_now()
        self.validate_compatibility(state)
        self.validate_state(state)
        payload = _to_json_compatible(state, "state")

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.checkpoint_path.name}.",
            suffix=".tmp",
            dir=self.checkpoint_path.parent,
            text=True,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(
                    payload,
                    stream,
                    indent=self.indent,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.checkpoint_path)
            self._fsync_directory(self.checkpoint_path.parent)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    def load(
        self,
        expected_genome_contract: Mapping[str, Any] | None = None,
        current_ga_settings: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Load, validate, and compatibility-check the latest checkpoint.

        ``expected_genome_contract`` optionally supplies the contract assembled
        from the current configuration. ``current_ga_settings`` enables a
        monotonic generation-target extension and controlled schema-v3
        migration. The method returns resumable state only when stored
        structure, campaign identity, and contracts match.

        Returns
        -------
        dict
            Mutable campaign state ready for recovery or continuation."""
        if not self.exists():
            raise FileNotFoundError(f"Checkpoint tidak ditemukan: {self.checkpoint_path}.")
        try:
            checkpoint_bytes = self.checkpoint_path.read_bytes()
            state = json.loads(checkpoint_bytes.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise CheckpointValidationError(
                f"Checkpoint JSON bukan UTF-8 valid: {self.checkpoint_path}."
            ) from exc
        except json.JSONDecodeError as exc:
            raise CheckpointValidationError(
                f"Checkpoint JSON rusak: {self.checkpoint_path}."
            ) from exc

        observed_schema = state.get("schema_version") if isinstance(state, Mapping) else None
        is_legacy = observed_schema == self.LEGACY_SCHEMA_VERSION
        if is_legacy:
            if current_ga_settings is None:
                raise CheckpointCompatibilityError(
                    "Migrasi checkpoint schema v3 memerlukan ga_settings runtime."
                )
            self._validate_legacy_compatibility(state, current_ga_settings)
            self.validate_state(state)
        else:
            self.validate_compatibility(state)
            self.validate_state(state)
        if expected_genome_contract is not None:
            expected_contract = _to_json_compatible(
                expected_genome_contract, "expected_genome_contract"
            )
            self._validate_genome_contract(expected_contract)
            if state["genome_contract"] != expected_contract:
                raise CheckpointCompatibilityError(
                    "genome_contract checkpoint tidak sama dengan konfigurasi runtime. Mulai campaign baru atau lakukan migrasi checkpoint secara eksplisit."
                )

        changed = False
        previous_target = (
            state.get("ga_settings", {}).get("num_generations")
            if isinstance(state.get("ga_settings"), Mapping)
            else None
        )
        if is_legacy:
            self._ensure_legacy_backup(checkpoint_bytes)
            state["schema_version"] = self.SCHEMA_VERSION
            state["input_provenance"] = _to_json_compatible(
                self.input_provenance, "input_provenance"
            )
            metadata = state.get("metadata")
            if not isinstance(metadata, MutableMapping):
                raise CheckpointValidationError("metadata checkpoint harus berupa mutable mapping.")
            metadata.setdefault("generation_limit_extensions", [])
            changed = True
        target_extended = False
        if current_ga_settings is not None:
            target_extended = self._reconcile_generation_target(state, current_ga_settings)
            changed = target_extended or changed
        if changed:
            self.save(state)
            if is_legacy:
                backup_path = self.checkpoint_path.with_name(
                    f"{self.checkpoint_path.stem}.v3.backup{self.checkpoint_path.suffix}"
                )
                logging.warning(
                    "Checkpoint schema v3 dimigrasikan ke v%s; backup byte-identik disimpan di %s.",
                    self.SCHEMA_VERSION,
                    backup_path,
                )
            if target_extended:
                logging.warning(
                    "Target total generasi campaign dinaikkan dari %s menjadi %s; resume dimulai setelah generation %s.",
                    previous_target,
                    current_ga_settings["num_generations"],
                    state["progress"].get("generation"),
                )
        return state

    def _validate_legacy_compatibility(
        self, state: Mapping[str, Any], current_ga_settings: Mapping[str, Any]
    ) -> None:
        """Validate schema-v3 state and prove that only its generation target changed."""
        self._require_constraint_contract(state)
        campaign = state.get("campaign")
        if not isinstance(campaign, Mapping) or campaign.get("campaign_id") != self.campaign_id:
            raise CheckpointCompatibilityError(
                "campaign_id checkpoint legacy tidak sama dengan runtime."
            )
        stored_provenance = state.get("input_provenance")
        try:
            validate_input_provenance(stored_provenance, allow_legacy=True)
        except InputProvenanceError as error:
            raise CheckpointCompatibilityError(str(error)) from error
        if stored_provenance.get("schema_version") != 1:
            raise CheckpointCompatibilityError(
                "Checkpoint schema v3 harus memakai input provenance schema v1."
            )

        stored_ga_settings = state.get("ga_settings")
        stored_target, current_target = self._require_compatible_ga_settings(
            stored_ga_settings, current_ga_settings
        )
        if current_target < stored_target:
            raise CheckpointCompatibilityError(
                f"ga.num_generations tidak boleh diturunkan dari {stored_target} menjadi {current_target} pada campaign yang sama."
            )
        stored_inputs = stored_provenance["inputs"]
        current_inputs = self.input_provenance["inputs"]
        changed_non_config = [
            name
            for name in ("fuel_inventory.json", "materials.xml", "canonical_genome_contract")
            if stored_inputs[name]["sha256"] != current_inputs[name]["sha256"]
        ]
        if changed_non_config:
            raise CheckpointCompatibilityError(
                "Fingerprint input ilmiah berubah: "
                + ", ".join(changed_non_config)
                + ". Mulai campaign baru."
            )

        stored_config_sha = stored_inputs["config.ini"]["sha256"]
        current_config_record = current_inputs["config.ini"]
        current_config_path = current_config_record.get("path")
        if current_target == stored_target:
            config_matches = current_config_record["sha256"] == stored_config_sha
        else:
            try:
                config_matches = legacy_config_matches_generation_extension(
                    current_config_path, stored_config_sha, stored_target, current_target
                )
            except (FileNotFoundError, InputProvenanceError) as error:
                raise CheckpointCompatibilityError(str(error)) from error
        if not config_matches:
            raise CheckpointCompatibilityError(
                "config.ini legacy berubah selain nilai ga.num_generations; campaign tidak dapat dimigrasikan."
            )

    def _require_compatible_ga_settings(
        self, stored: Any, current: Mapping[str, Any]
    ) -> tuple[int, int]:
        """Validate immutable GA settings and return stored/current generation targets."""
        if not isinstance(stored, Mapping) or not isinstance(current, Mapping):
            raise CheckpointCompatibilityError(
                "ga_settings checkpoint/runtime harus berupa mapping."
            )
        stored_target = stored.get("num_generations")
        current_target = current.get("num_generations")
        for name, value in (("checkpoint", stored_target), ("runtime", current_target)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise CheckpointCompatibilityError(
                    f"num_generations {name} harus berupa integer positif."
                )
        stored_immutable = {key: value for key, value in stored.items() if key != "num_generations"}
        current_immutable = {
            key: value for key, value in current.items() if key != "num_generations"
        }
        if stored_immutable != current_immutable:
            raise CheckpointCompatibilityError(
                "ga_settings berubah selain num_generations; mulai campaign baru."
            )
        return int(stored_target), int(current_target)

    def _reconcile_generation_target(
        self, state: MutableMapping[str, Any], current_ga_settings: Mapping[str, Any]
    ) -> bool:
        """Apply one monotonic generation-target increase and return whether state changed."""
        stored_target, current_target = self._require_compatible_ga_settings(
            state.get("ga_settings"), current_ga_settings
        )
        if current_target < stored_target:
            raise CheckpointCompatibilityError(
                f"ga.num_generations tidak boleh diturunkan dari {stored_target} menjadi {current_target} pada campaign yang sama."
            )
        if current_target == stored_target:
            return False

        generations = state.get("generations")
        progress = state.get("progress")
        if not isinstance(generations, list) or not isinstance(progress, MutableMapping):
            raise CheckpointValidationError(
                "Checkpoint tidak memiliki generations/progress yang dapat diperpanjang."
            )
        completed_generation = progress.get("generation")
        if progress.get("phase") == "finished":
            if (
                len(generations) != stored_target
                or completed_generation != stored_target - 1
                or not generations
                or generations[-1].get("status") != "completed"
            ):
                raise CheckpointValidationError(
                    "Campaign finished tidak konsisten dengan target generasi checkpoint."
                )
            progress["phase"] = "generation_completed"

        metadata = state.get("metadata")
        if not isinstance(metadata, MutableMapping):
            raise CheckpointValidationError("metadata checkpoint harus berupa mutable mapping.")
        extension_history = metadata.setdefault("generation_limit_extensions", [])
        if not isinstance(extension_history, list):
            raise CheckpointValidationError(
                "metadata.generation_limit_extensions harus berupa list."
            )
        extension_history.append(
            {
                "from": stored_target,
                "to": current_target,
                "extended_at": _utc_now(),
                "completed_generation": completed_generation,
            }
        )
        state["ga_settings"] = _to_json_compatible(current_ga_settings, "current_ga_settings")
        state["input_provenance"] = _to_json_compatible(self.input_provenance, "input_provenance")
        return True

    def _ensure_legacy_backup(self, checkpoint_bytes: bytes) -> None:
        """Create one durable byte-identical schema-v3 backup without overwriting."""
        backup_path = self.checkpoint_path.with_name(
            f"{self.checkpoint_path.stem}.v3.backup{self.checkpoint_path.suffix}"
        )
        if backup_path.exists():
            if not backup_path.is_file() or backup_path.read_bytes() != checkpoint_bytes:
                raise CheckpointCompatibilityError(
                    f"Backup legacy sudah ada tetapi berbeda: {backup_path}."
                )
            return

        backup_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{backup_path.name}.", suffix=".tmp", dir=backup_path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(checkpoint_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary_path, backup_path)
            except FileExistsError:
                if not backup_path.is_file() or backup_path.read_bytes() != checkpoint_bytes:
                    raise CheckpointCompatibilityError(
                        f"Backup legacy sudah ada tetapi berbeda: {backup_path}."
                    )
            self._fsync_directory(backup_path.parent)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _require_constraint_contract(self, state: Mapping[str, Any]) -> None:
        """Reject missing or changed ranking contracts in ``state`` before any resume mutation."""
        if state.get("constraint_contract") != self.constraint_contract.identity():
            raise CheckpointCompatibilityError(
                "Kontrak constraint checkpoint hilang/berubah (keff_min, feasibility_tolerance, contract_id); rank lama tidak kompatibel. Mulai campaign baru; migrasi otomatis tidak tersedia."
            )

    def validate_compatibility(self, state: Mapping[str, Any]) -> None:
        """Reject schema, campaign, or configuration mismatches during resume.

        Parameters
        ----------
        state : mapping
            Loaded checkpoint payload to compare with this manager.

        Returns
        -------
        None
            The method returns normally only for a compatible checkpoint."""
        self._require_constraint_contract(state)
        if state.get("schema_version") != self.SCHEMA_VERSION:
            observed_version = state.get("schema_version")
            raise CheckpointCompatibilityError(
                f"Checkpoint schema v{observed_version} tidak kompatibel dengan v{self.SCHEMA_VERSION}; "
                "hanya schema v3 yang dapat dimigrasikan otomatis melalui validasi extension num_generations."
            )
        campaign = state.get("campaign")
        if not isinstance(campaign, Mapping):
            raise CheckpointCompatibilityError("Bagian campaign checkpoint tidak valid.")
        if campaign.get("campaign_id") != self.campaign_id:
            raise CheckpointCompatibilityError(
                f"campaign_id checkpoint {campaign.get('campaign_id')!r} tidak sama "
                f"dengan {self.campaign_id!r}."
            )
        try:
            current_provenance = refresh_file_fingerprints(self.input_provenance)
            require_matching_input_provenance(state.get("input_provenance"), current_provenance)
        except (FileNotFoundError, InputProvenanceError) as error:
            raise CheckpointCompatibilityError(str(error)) from error

    def validate_state(self, state: Mapping[str, Any]) -> None:
        """Validate structural invariants required for deterministic resume.

        Parameters
        ----------
        state : mapping
            Full in-memory or deserialized campaign state.

        Returns
        -------
        None
            Invalid state raises :class:`CheckpointValidationError`."""
        self._require_constraint_contract(state)
        required_keys = {
            "schema_version",
            "campaign",
            "input_provenance",
            "genome_contract",
            "ga_settings",
            "metadata",
            "progress",
            "rng_state",
            "individuals",
            "generations",
        }
        missing = required_keys.difference(state)
        if missing:
            raise CheckpointValidationError(
                "Checkpoint kehilangan key: " + ", ".join(sorted(missing)) + "."
            )
        campaign = state["campaign"]
        if not isinstance(campaign, Mapping):
            raise CheckpointValidationError("campaign harus berupa mapping.")
        metadata = state["metadata"]
        if not isinstance(metadata, Mapping):
            raise CheckpointValidationError("metadata harus berupa mapping.")
        if state.get("schema_version") == self.SCHEMA_VERSION:
            self._validate_generation_extension_history(metadata, state.get("ga_settings"))
        rng_state = state["rng_state"]
        if not isinstance(rng_state, Mapping) or not {"python", "numpy"}.issubset(rng_state):
            raise CheckpointValidationError("rng_state harus memuat key python dan numpy.")
        self._validate_genome_contract(state["genome_contract"])
        individuals = self._get_individuals(state)
        mutable_contract = self._build_mutable_genome_contract(state["genome_contract"])

        running_ids = []
        registered_run_ids = set()
        for individual_id, individual in individuals.items():
            self._require_non_empty_id(individual_id, "individual_id")
            if not isinstance(individual, Mapping):
                raise CheckpointValidationError(
                    f"Record individual {individual_id!r} harus berupa mapping."
                )
            required_individual_keys = {
                "generation_created",
                "origin",
                "parent_ids",
                "operator",
                "genome",
                "status",
                "attempts",
                "results",
                "nsga2",
                "created_at",
                "updated_at",
            }
            if required_individual_keys.difference(individual):
                raise CheckpointValidationError(
                    f"Record individual {individual_id!r} belum lengkap."
                )
            _normalize_integer(
                individual["generation_created"],
                f"generation_created {individual_id}",
            )
            if individual["origin"] not in {"initial", "offspring"}:
                raise CheckpointValidationError(f"origin individual {individual_id!r} tidak valid.")
            parents = _normalize_string_ids(
                individual["parent_ids"],
                f"parent_ids {individual_id}",
            )
            self._require_known_individuals(
                state,
                parents,
                f"parent_ids {individual_id}",
            )
            if individual["origin"] == "initial" and parents:
                raise CheckpointValidationError(
                    f"Individual initial {individual_id!r} tidak boleh memiliki parent."
                )
            if individual["origin"] == "offspring" and not parents:
                raise CheckpointValidationError(
                    f"Individual offspring {individual_id!r} harus memiliki parent."
                )
            ineligible_parents = [
                parent_id
                for parent_id in parents
                if individuals[parent_id]["status"] not in NSGA2_ELIGIBLE_STATUSES
            ]
            if individual["origin"] == "offspring" and ineligible_parents:
                raise CheckpointValidationError(
                    f"Parent offspring {individual_id!r} tidak eligible NSGA-II: "
                    + ", ".join(ineligible_parents)
                    + "."
                )
            if not isinstance(individual["operator"], Mapping):
                raise CheckpointValidationError(
                    f"operator individual {individual_id!r} harus berupa mapping."
                )
            try:
                mutable_contract.validate(
                    individual.get("genome", ()), f"Genome tersimpan individual {individual_id!r}"
                )
            except MutableGenomeValidationError as error:
                raise CheckpointValidationError(str(error)) from error
            if individual.get("status") not in self.INDIVIDUAL_STATUSES:
                raise CheckpointValidationError(f"Status individual {individual_id!r} tidak valid.")
            attempts = individual.get("attempts")
            if not isinstance(attempts, list):
                raise CheckpointValidationError(
                    f"attempts individual {individual_id!r} harus berupa list."
                )
            for attempt_index, attempt in enumerate(attempts, start=1):
                if not isinstance(attempt, Mapping):
                    raise CheckpointValidationError(
                        f"Attempt individual {individual_id!r} harus berupa mapping."
                    )
                if attempt.get("attempt_number") != attempt_index:
                    raise CheckpointValidationError(
                        f"Nomor attempt individual {individual_id!r} tidak berurutan."
                    )
                self._require_non_empty_id(
                    attempt.get("run_id"),
                    f"run_id attempt {individual_id}",
                )
                if attempt["run_id"] in registered_run_ids:
                    raise CheckpointValidationError(
                        f"run_id {attempt['run_id']!r} digunakan lebih dari sekali."
                    )
                registered_run_ids.add(attempt["run_id"])
                _normalize_integer(
                    attempt.get("openmc_seed"),
                    f"openmc_seed attempt {individual_id}",
                    minimum=1,
                )
                if not isinstance(attempt.get("statepoint_retained"), bool):
                    raise CheckpointValidationError(
                        f"statepoint_retained attempt {individual_id!r} harus bool."
                    )
                if attempt["statepoint_retained"] and attempt.get("statepoint_path") is None:
                    raise CheckpointValidationError(
                        f"Attempt {individual_id!r} retained tanpa statepoint_path."
                    )
            running_attempts = [
                attempt
                for attempt in attempts
                if isinstance(attempt, Mapping) and attempt.get("status") == "running"
            ]
            invalid_attempts = [
                attempt
                for attempt in attempts
                if not isinstance(attempt, Mapping)
                or attempt.get("status") not in self.ATTEMPT_STATUSES
            ]
            if invalid_attempts or len(running_attempts) > 1:
                raise CheckpointValidationError(
                    f"Riwayat attempt individual {individual_id!r} tidak valid."
                )
            if individual["status"] == "running":
                if len(running_attempts) != 1:
                    raise CheckpointValidationError(
                        f"Individual running {individual_id!r} harus memiliki "
                        "satu attempt running."
                    )
                running_ids.append(individual_id)
            elif running_attempts:
                raise CheckpointValidationError(
                    f"Individual {individual_id!r} bukan running tetapi attempt masih running."
                )
            evaluation_status = individual["status"]
            if evaluation_status in RESULTS_EVALUATION_STATUSES:
                if not isinstance(individual["results"], Mapping):
                    raise CheckpointValidationError(
                        f"Individual evaluated {individual_id!r} harus memiliki results."
                    )
                self.constraint_contract.validate_results(individual["results"])
                if evaluation_status == COMPLETED_STATUS:
                    extract_eligible_candidates({individual_id: individual})
                duplicate_metadata = individual["operator"].get("duplicate_handling")
                fitness_reused = (
                    isinstance(duplicate_metadata, Mapping)
                    and duplicate_metadata.get("fitness_reused") is True
                )
                cache_metadata = individual["operator"].get("evaluation_cache")
                cache_reused = (
                    isinstance(cache_metadata, Mapping) and cache_metadata.get("lookup") == "hit"
                )
                if cache_reused:
                    required_cache_keys = {
                        "genome_hash",
                        "context_hash",
                        "source_path",
                        "source_checkpoint_sha256",
                        "source_campaign_id",
                        "source_individual_id",
                        "validation_mode",
                    }
                    evaluation_context = individual["operator"].get("evaluation_context")
                    genome_payload = json.dumps(
                        individual["genome"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                    context_payload = (
                        json.dumps(
                            evaluation_context,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ).encode("utf-8")
                        if isinstance(evaluation_context, Mapping)
                        else b""
                    )
                    if (
                        evaluation_status != COMPLETED_STATUS
                        or attempts
                        or required_cache_keys.difference(cache_metadata)
                        or cache_metadata.get("validation_mode")
                        not in {
                            "strict",
                            "legacy",
                            "strict-compatible-openmc-version",
                            "legacy-compatible-openmc-version",
                        }
                        or (
                            str(cache_metadata.get("validation_mode", "")).endswith(
                                "compatible-openmc-version"
                            )
                            and (
                                not isinstance(cache_metadata.get("source_openmc_version"), str)
                                or not isinstance(cache_metadata.get("target_openmc_version"), str)
                            )
                        )
                        or not isinstance(evaluation_context, Mapping)
                        or cache_metadata.get("genome_hash")
                        != hashlib.sha256(genome_payload).hexdigest()
                        or cache_metadata.get("context_hash")
                        != hashlib.sha256(context_payload).hexdigest()
                    ):
                        raise CheckpointValidationError(
                            f"Metadata evaluation cache individual {individual_id!r} tidak konsisten."
                        )
                elif fitness_reused:
                    source_parent_id = duplicate_metadata.get("fitness_source_parent_id")
                    source = (
                        individuals.get(source_parent_id)
                        if isinstance(source_parent_id, str)
                        else None
                    )
                    generation_record = self._find_generation(
                        state, individual["generation_created"], required=False
                    )
                    source_context = (
                        source.get("operator", {}).get("evaluation_context")
                        if isinstance(source, Mapping)
                        else None
                    )
                    child_context = individual["operator"].get("evaluation_context")
                    try:
                        source_genome = (
                            mutable_contract.validate(source.get("genome", ()))
                            if isinstance(source, Mapping)
                            else None
                        )
                        child_genome = mutable_contract.validate(individual["genome"])
                    except MutableGenomeValidationError as error:
                        raise CheckpointValidationError(str(error)) from error
                    if (
                        evaluation_status != COMPLETED_STATUS
                        or attempts
                        or duplicate_metadata.get("mode") != "reuse_parent_fitness"
                        or duplicate_metadata.get("duplicate_parent_id") != source_parent_id
                        or source is None
                        or source.get("status") != COMPLETED_STATUS
                        or generation_record is None
                        or source_parent_id not in generation_record["parent_ids"]
                        or source_genome != child_genome
                        or source_context is None
                        or source_context != child_context
                        or source.get("results") != individual["results"]
                    ):
                        raise CheckpointValidationError(
                            f"Metadata reuse fitness individual {individual_id!r} tidak konsisten."
                        )
                elif not attempts or attempts[-1]["status"] != evaluation_status:
                    raise CheckpointValidationError(
                        f"Individual {evaluation_status} {individual_id!r} tidak memiliki attempt yang sesuai."
                    )
            elif individual["results"] is not None:
                raise CheckpointValidationError(
                    f"Individual {individual_id!r} belum completed tetapi memiliki results."
                )
            if individual["status"] == "failed" and (
                not attempts or attempts[-1]["status"] != "failed"
            ):
                raise CheckpointValidationError(
                    f"Individual failed {individual_id!r} tidak memiliki attempt failed."
                )
            nsga2 = individual["nsga2"]
            if not isinstance(nsga2, Mapping) or not {
                "rank",
                "crowding_distance",
                "is_boundary",
            }.issubset(nsga2):
                raise CheckpointValidationError(
                    f"Metrik NSGA-II individual {individual_id!r} tidak lengkap."
                )
            has_nsga2_metrics = (
                nsga2["rank"] is not None
                or nsga2["crowding_distance"] is not None
                or nsga2["is_boundary"] is not False
            )
            if individual["status"] not in NSGA2_ELIGIBLE_STATUSES and has_nsga2_metrics:
                raise CheckpointValidationError(
                    f"Individual {individual_id!r} tidak eligible menerima metrik NSGA-II."
                )

        if len(running_ids) > 1:
            raise CheckpointValidationError(
                "Checkpoint sequential hanya boleh memiliki satu individual running."
            )

        progress = state["progress"]
        if not isinstance(progress, Mapping) or progress.get("phase") not in self.PHASES:
            raise CheckpointValidationError("Bagian progress checkpoint tidak valid.")
        progress_generation = progress.get("generation")
        if progress_generation is not None:
            _normalize_integer(progress_generation, "progress generation")
        active_id = progress.get("active_individual_id")
        if active_id is not None and active_id not in running_ids:
            raise CheckpointValidationError(
                "active_individual_id harus menunjuk individual berstatus running."
            )
        if running_ids and active_id != running_ids[0]:
            raise CheckpointValidationError(
                "Individual running tidak sama dengan active_individual_id."
            )

        generations = state["generations"]
        if not isinstance(generations, list):
            raise CheckpointValidationError("generations harus berupa list.")
        ga_settings = state.get("ga_settings")
        configured_target = (
            ga_settings.get("num_generations") if isinstance(ga_settings, Mapping) else None
        )
        if configured_target is not None:
            configured_target = _normalize_integer(
                configured_target, "ga_settings.num_generations", minimum=1
            )
            if len(generations) > configured_target:
                raise CheckpointValidationError(
                    "Jumlah generation record melebihi ga_settings.num_generations."
                )
        generation_numbers = []
        for record in generations:
            if not isinstance(record, Mapping):
                raise CheckpointValidationError("Setiap generation record harus mapping.")
            generation_number = _normalize_integer(
                record.get("generation"),
                "nomor generasi",
            )
            generation_numbers.append(generation_number)
            if record.get("status") not in {"running", "completed"}:
                raise CheckpointValidationError(f"Status generasi {generation_number} tidak valid.")
            for key in ("parent_ids", "offspring_ids", "survivor_ids"):
                member_ids = _normalize_string_ids(record.get(key, ()), key)
                self._require_known_individuals(state, member_ids, key)
            candidates = set(record["parent_ids"]) | set(record["offspring_ids"])
            if not set(record["survivor_ids"]).issubset(candidates):
                raise CheckpointValidationError(
                    f"survivor_ids generasi {generation_number} bukan subset kandidat."
                )
            if record["status"] == "completed" and not record["survivor_ids"]:
                raise CheckpointValidationError(
                    f"Generasi completed {generation_number} harus memiliki survivor."
                )
        expected_generation_numbers = list(range(len(generations)))
        if generation_numbers != expected_generation_numbers:
            raise CheckpointValidationError("Generasi harus tersimpan berurutan mulai dari 0.")
        for index in range(1, len(generations)):
            if generations[index]["parent_ids"] != generations[index - 1]["survivor_ids"]:
                raise CheckpointValidationError(
                    f"parent_ids generasi {index} tidak sama dengan survivor sebelumnya."
                )
        running_generations = [
            record["generation"] for record in generations if record["status"] == "running"
        ]
        if len(running_generations) > 1 or (
            running_generations and running_generations[0] != generations[-1]["generation"]
        ):
            raise CheckpointValidationError("Hanya generasi terakhir yang boleh berstatus running.")
        if generations and progress_generation != generations[-1]["generation"]:
            raise CheckpointValidationError("progress generation harus menunjuk generasi terakhir.")
        if not generations and progress_generation is not None:
            raise CheckpointValidationError(
                "progress generation harus None sebelum generasi pertama."
            )
        if not generations and progress["phase"] != "initialized":
            raise CheckpointValidationError("Phase harus initialized sebelum generasi pertama.")
        if (
            generations
            and generations[-1]["status"] == "completed"
            and progress["phase"] not in {"generation_completed", "finished"}
        ):
            raise CheckpointValidationError(
                "Phase harus generation_completed atau finished setelah generasi selesai."
            )
        if (
            generations
            and generations[-1]["status"] == "running"
            and progress["phase"] in {"initialized", "generation_completed", "finished"}
        ):
            raise CheckpointValidationError(
                "Phase progress tidak sesuai dengan generasi yang masih running."
            )
        if generations and progress["phase"] == "parents_evaluated":
            current_generation = generations[-1]
            nonterminal_parents = [
                individual_id
                for individual_id in current_generation["parent_ids"]
                if individuals[individual_id]["status"] not in TERMINAL_EVALUATION_STATUSES
            ]
            if (
                current_generation["status"] != "running"
                or current_generation["offspring_ids"]
                or active_id is not None
                or nonterminal_parents
            ):
                raise CheckpointValidationError(
                    "Phase parents_evaluated memerlukan seluruh parent terminal tanpa offspring atau individual aktif."
                )
        if generations and progress["phase"] == "offspring_registered":
            current_generation = generations[-1]
            offspring_ids = current_generation["offspring_ids"]
            invalid_offspring_ids = [
                individual_id
                for individual_id in offspring_ids
                if individuals[individual_id]["status"] != PENDING_STATUS
            ]
            if (
                current_generation["status"] != "running"
                or not offspring_ids
                or active_id is not None
                or invalid_offspring_ids
            ):
                raise CheckpointValidationError(
                    "Phase offspring_registered memerlukan seluruh offspring pending, tanpa individual aktif."
                )
        if generations and progress["phase"] == "evaluating_offspring":
            current_generation = generations[-1]
            offspring_ids = current_generation["offspring_ids"]
            if current_generation["status"] != "running" or not offspring_ids:
                raise CheckpointValidationError(
                    "Phase evaluating_offspring memerlukan offspring pada generasi running."
                )
            invalid_offspring_ids = [
                individual_id
                for individual_id in offspring_ids
                if individuals[individual_id]["status"]
                not in TERMINAL_EVALUATION_STATUSES | {PENDING_STATUS, RUNNING_STATUS}
            ]
            if invalid_offspring_ids:
                raise CheckpointValidationError(
                    "Phase evaluating_offspring memiliki status offspring tidak valid."
                )
        if generations and progress["phase"] == "offspring_evaluated":
            current_generation = generations[-1]
            offspring_ids = current_generation["offspring_ids"]
            nonterminal_ids = [
                individual_id
                for individual_id in offspring_ids
                if individuals[individual_id]["status"] not in TERMINAL_EVALUATION_STATUSES
            ]
            if (
                current_generation["status"] != "running"
                or not offspring_ids
                or active_id is not None
                or nonterminal_ids
            ):
                raise CheckpointValidationError(
                    "Phase offspring_evaluated memerlukan seluruh offspring terminal tanpa individual aktif."
                )
        if generations and progress["phase"] == "environmental_selection":
            current_generation = generations[-1]
            candidate_ids = current_generation["parent_ids"] + current_generation["offspring_ids"]
            nonterminal_ids = [
                individual_id
                for individual_id in candidate_ids
                if individuals[individual_id]["status"] not in TERMINAL_EVALUATION_STATUSES
            ]
            if (
                current_generation["status"] != "running"
                or not current_generation["offspring_ids"]
                or active_id is not None
                or nonterminal_ids
            ):
                raise CheckpointValidationError(
                    "Phase environmental_selection memerlukan seluruh parent dan offspring terminal tanpa individual aktif."
                )
        if progress["phase"] == "finished" and configured_target is not None:
            if (
                len(generations) != configured_target
                or progress_generation != configured_target - 1
            ):
                raise CheckpointValidationError(
                    "Campaign finished harus tepat mencapai ga_settings.num_generations."
                )

        _to_json_compatible(state, "state")

    @staticmethod
    def _validate_generation_extension_history(
        metadata: Mapping[str, Any], ga_settings: Any
    ) -> None:
        """Validate monotonic extension audit entries against the stored GA target."""
        history = metadata.get("generation_limit_extensions", [])
        if not isinstance(history, list):
            raise CheckpointValidationError(
                "metadata.generation_limit_extensions harus berupa list."
            )
        previous_target = None
        for index, entry in enumerate(history):
            if not isinstance(entry, Mapping):
                raise CheckpointValidationError(
                    f"Riwayat extension indeks {index} harus berupa mapping."
                )
            old_target = _normalize_integer(entry.get("from"), f"extension {index} from", minimum=1)
            new_target = _normalize_integer(entry.get("to"), f"extension {index} to", minimum=1)
            if new_target <= old_target:
                raise CheckpointValidationError(
                    f"Riwayat extension indeks {index} tidak menaikkan target."
                )
            if previous_target is not None and old_target != previous_target:
                raise CheckpointValidationError(
                    "Riwayat extension tidak membentuk rantai target yang berurutan."
                )
            if not isinstance(entry.get("extended_at"), str) or not entry["extended_at"].strip():
                raise CheckpointValidationError(
                    f"Riwayat extension indeks {index} tidak memiliki extended_at."
                )
            completed_generation = entry.get("completed_generation")
            if completed_generation is not None:
                _normalize_integer(completed_generation, f"extension {index} completed_generation")
            previous_target = new_target
        stored_target = (
            ga_settings.get("num_generations") if isinstance(ga_settings, Mapping) else None
        )
        if previous_target is not None and previous_target != stored_target:
            raise CheckpointValidationError(
                "Target terakhir riwayat extension tidak sama dengan ga_settings.num_generations."
            )

    @staticmethod
    def _validate_genome_contract(contract: Mapping[str, Any]) -> None:
        """Validate lengths, positions, and fixed genes in a genome contract."""
        if not isinstance(contract, Mapping):
            raise CheckpointValidationError("genome_contract harus berupa mapping.")
        required = {
            "encoding",
            "included_length",
            "complete_length",
            "slot_order",
            "included_positions",
            "excluded_genes",
            "mutable_alleles",
        }
        if required.difference(contract):
            raise CheckpointValidationError("genome_contract belum lengkap.")
        if not isinstance(contract["encoding"], str) or not contract["encoding"].strip():
            raise CheckpointValidationError("encoding genome contract tidak valid.")
        slots = _normalize_string_ids(contract["slot_order"], "slot_order")
        included = _normalize_string_ids(
            contract["included_positions"],
            "included_positions",
        )
        excluded = contract["excluded_genes"]
        if not isinstance(excluded, Mapping):
            raise CheckpointValidationError("excluded_genes harus berupa mapping.")
        if not included:
            raise CheckpointValidationError(
                "genome_contract harus memiliki sedikitnya satu included position."
            )
        unknown_positions = set(included).union(excluded).difference(slots)
        if unknown_positions:
            raise CheckpointValidationError("Genome contract memuat posisi tidak dikenal.")
        expected_included = [position for position in slots if position not in excluded]
        if included != expected_included:
            raise CheckpointValidationError(
                "included_positions harus mengikuti slot_order tanpa excluded positions."
            )
        if contract["included_length"] != len(included):
            raise CheckpointValidationError("included_length tidak sesuai contract.")
        if contract["complete_length"] != len(slots):
            raise CheckpointValidationError("complete_length tidak sesuai slot_order.")
        for position, gene in excluded.items():
            _normalize_genome([gene], 1)
            if position in included:
                raise CheckpointValidationError(
                    f"Posisi {position!r} tidak boleh included sekaligus excluded."
                )

        CheckpointManager._build_mutable_genome_contract(contract)

    @staticmethod
    def _build_mutable_genome_contract(contract: Mapping[str, Any]) -> MutableGenomeContract:
        """Build the reusable validator from a checkpoint genome contract.

        ``contract`` supplies mutable inventory and cardinality. The method
        returns a validator or raises a checkpoint-specific validation error."""
        try:
            return MutableGenomeContract.from_checkpoint_contract(contract)
        except MutableGenomeValidationError as error:
            raise CheckpointValidationError(str(error)) from error

    @staticmethod
    def _get_individuals(state: Mapping[str, Any]) -> MutableMapping[str, Any]:
        """Return the mutable individual registry after validating its type."""
        individuals = state.get("individuals")
        if not isinstance(individuals, MutableMapping):
            raise CheckpointValidationError("individuals harus berupa mutable mapping.")
        return individuals

    def _get_individual(
        self,
        state: Mapping[str, Any],
        individual_id: str,
    ) -> MutableMapping[str, Any]:
        """Return one registered individual or raise a validation error."""
        individuals = self._get_individuals(state)
        if individual_id not in individuals:
            raise CheckpointValidationError(f"Individual {individual_id!r} belum terdaftar.")
        individual = individuals[individual_id]
        if not isinstance(individual, MutableMapping):
            raise CheckpointValidationError(f"Record individual {individual_id!r} tidak mutable.")
        return individual

    def _find_generation(
        self,
        state: Mapping[str, Any],
        generation: int,
        required: bool = True,
    ) -> MutableMapping[str, Any] | None:
        """Return the record for ``generation`` in checkpoint ``state``.

        If ``required`` is true, raise for a missing record; otherwise return None."""
        generations = state.get("generations")
        if not isinstance(generations, list):
            raise CheckpointValidationError("generations harus berupa list.")
        for record in generations:
            if record.get("generation") == generation:
                return record
        if required:
            raise CheckpointValidationError(f"Generasi {generation} belum terdaftar.")
        return None

    def _require_known_individuals(
        self,
        state: Mapping[str, Any],
        individual_ids: Iterable[str],
        value_name: str,
    ) -> None:
        """Check that all ``individual_ids`` exist in ``state``; return None.

        Use ``value_name`` to identify the invalid registry field in error messages."""
        individuals = self._get_individuals(state)
        missing = [item for item in individual_ids if item not in individuals]
        if missing:
            raise CheckpointValidationError(
                f"{value_name} memuat individual belum terdaftar: " + ", ".join(missing) + "."
            )

    @staticmethod
    def _require_non_empty_id(value: str, value_name: str) -> None:
        """Validate an identifier as a non-empty string."""
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise CheckpointValidationError(f"{value_name} harus berupa string non-kosong.")

    @staticmethod
    def _get_running_attempt(
        individual: MutableMapping[str, Any],
        individual_id: str,
    ) -> MutableMapping[str, Any]:
        """Return the single running attempt of an active individual."""
        if individual.get("status") != "running":
            raise CheckpointValidationError(f"Individual {individual_id!r} tidak sedang running.")
        running_attempts = [
            attempt
            for attempt in individual.get("attempts", [])
            if attempt.get("status") == "running"
        ]
        if len(running_attempts) != 1:
            raise CheckpointValidationError(
                f"Individual {individual_id!r} harus memiliki satu attempt running."
            )
        return running_attempts[0]

    @staticmethod
    def _clear_active_individual(
        state: MutableMapping[str, Any],
        individual_id: str,
    ) -> None:
        """Clear the active cursor only when it points at ``individual_id``."""
        active_id = state["progress"].get("active_individual_id")
        if active_id != individual_id:
            raise CheckpointValidationError(
                f"active_individual_id {active_id!r} bukan {individual_id!r}."
            )
        state["progress"]["active_individual_id"] = None

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        """Flush a directory entry after atomic replacement when supported."""
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(directory, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
