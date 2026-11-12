"""Fingerprint scientific inputs and verify resume compatibility.

Canonical serializers identify configuration and genome contracts. File hashes
and source revisions record provenance; validators reject incompatible resumes.
Fingerprint refresh helpers record explicitly changed input files."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import configparser
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


PROVENANCE_SCHEMA_VERSION = 2
LEGACY_PROVENANCE_SCHEMA_VERSION = 1
HASH_ALGORITHM = "sha256"
CANONICAL_GENOME_FORMAT = "json_utf8_sorted_keys_compact_v1"
CONFIG_COMPATIBILITY_FORMAT = "configparser_strings_without_ga.num_generations_v1"
PROTECTED_INPUT_NAMES = (
    "config.ini",
    "fuel_inventory.json",
    "materials.xml",
    "canonical_genome_contract",
)


class InputProvenanceError(ValueError):
    """Report malformed provenance or incompatible scientific inputs."""


def sha256_file(file_path: str | Path) -> str:
    """Hash one existing file by bytes and return a lowercase SHA-256 digest."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input provenance tidak ditemukan: {path}.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_config_contract_bytes(config_path: str | Path) -> bytes:
    """Serialize effective INI values except mutable generation and cache-policy controls.

    ``config_path`` identifies the campaign INI. The returned canonical UTF-8
    JSON bytes excludes ``ga.num_generations`` and the audited operational
    ``evaluation_cache.compatible_openmc_version_pairs`` option."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input provenance tidak ditemukan: {path}.")
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    try:
        with path.open("r", encoding="utf-8") as stream:
            parser.read_file(stream)
    except (configparser.Error, UnicodeDecodeError) as error:
        raise InputProvenanceError(
            f"config.ini tidak dapat dibuat kontrak kompatibilitas: {path}."
        ) from error

    canonical_sections: dict[str, dict[str, str]] = {}
    for section in parser.sections():
        canonical_options: dict[str, str] = {}
        for option, value in parser.items(section, raw=True):
            if section.casefold() == "ga" and option.casefold() == "num_generations":
                continue
            if (
                section.casefold() == "evaluation_cache"
                and option.casefold() == "compatible_openmc_version_pairs"
            ):
                # This operational reuse policy is audited in cache metadata;
                # it does not alter model inputs, transport, or GA ranking rules.
                continue
            canonical_options[option] = value
        canonical_sections[section] = canonical_options
    serialized = json.dumps(
        canonical_sections,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return serialized.encode("utf-8")


def canonical_config_contract_sha256(config_path: str | Path) -> str:
    """Hash effective scientific settings while excluding mutable operational controls."""
    return hashlib.sha256(canonical_config_contract_bytes(config_path)).hexdigest()


def legacy_config_matches_generation_extension(
    config_path: str | Path,
    stored_sha256: str,
    stored_num_generations: int,
    current_num_generations: int,
) -> bool:
    """Return whether a legacy INI differs only by an increased generation value.

    ``config_path`` is the current INI, ``stored_sha256`` is the schema-v1 raw
    fingerprint, and the two integer parameters are the checkpoint and current
    total generation targets. The current numeric token is replaced in memory;
    no file is modified."""
    path = Path(config_path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise InputProvenanceError(f"config.ini legacy bukan UTF-8 valid: {path}.") from error

    section_pattern = re.compile(r"^\s*\[([^]]+)\]\s*(?:[#;].*)?$")
    option_pattern = re.compile(
        r"^(\s*num_generations\s*[=:]\s*)([+-]?\d+)([ \t]*(?:\r?\n)?)$", re.IGNORECASE
    )
    in_ga_section = False
    replaced_lines: list[str] = []
    match_count = 0
    for line in text.splitlines(keepends=True):
        section_match = section_pattern.match(line.rstrip("\r\n"))
        if section_match is not None:
            in_ga_section = section_match.group(1).strip().casefold() == "ga"
        option_match = option_pattern.match(line) if in_ga_section else None
        if option_match is None:
            replaced_lines.append(line)
            continue
        if int(option_match.group(2)) != current_num_generations:
            return False
        match_count += 1
        replaced_lines.append(
            f"{option_match.group(1)}{stored_num_generations}{option_match.group(3)}"
        )
    if match_count != 1:
        return False
    reconstructed = "".join(replaced_lines).encode("utf-8")
    return hashlib.sha256(reconstructed).hexdigest() == stored_sha256


def canonical_genome_contract_bytes(genome_contract: Mapping[str, Any]) -> bytes:
    """Serialize slot order, fixed genes, and mutable inventory into canonical bytes."""
    if not isinstance(genome_contract, Mapping):
        raise InputProvenanceError("genome_contract provenance harus berupa mapping.")
    required_keys = ("slot_order", "excluded_genes", "mutable_alleles")
    missing_keys = [key for key in required_keys if key not in genome_contract]
    if missing_keys:
        raise InputProvenanceError(
            "genome_contract provenance kehilangan key: " + ", ".join(missing_keys) + "."
        )
    protected_contract = {key: genome_contract[key] for key in required_keys}
    try:
        serialized = json.dumps(
            protected_contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise InputProvenanceError(
            "genome_contract provenance tidak dapat diserialisasi secara kanonik."
        ) from error
    return serialized.encode("utf-8")


def canonical_genome_contract_sha256(genome_contract: Mapping[str, Any]) -> str:
    """Hash the canonical protected genome contract and return its SHA-256 digest."""
    return hashlib.sha256(canonical_genome_contract_bytes(genome_contract)).hexdigest()


def capture_source_revision(source_root: str | Path | None) -> dict[str, Any]:
    """Read an optional Git commit and dirty flag without changing repository state."""
    if source_root is None:
        return {"available": False, "commit": None, "dirty": None}
    root = Path(source_root)
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    command_options = {
        "cwd": root,
        "capture_output": True,
        "text": True,
        "check": False,
        "env": environment,
    }
    try:
        revision = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], **command_options)
        if revision.returncode != 0:
            return {"available": False, "commit": None, "dirty": None}
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"], **command_options
        )
    except OSError:
        return {"available": False, "commit": None, "dirty": None}
    return {
        "available": status.returncode == 0,
        "commit": revision.stdout.strip(),
        "dirty": bool(status.stdout) if status.returncode == 0 else None,
    }


def build_input_provenance(
    config_path: str | Path,
    fuel_inventory_path: str | Path,
    materials_path: str | Path,
    genome_contract: Mapping[str, Any],
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return fingerprints for the three input paths and ``genome_contract``.

    ``config_path``, ``fuel_inventory_path``, and ``materials_path`` identify the
    campaign inputs. ``source_root`` locates source-revision evidence. The returned
    manifest records file hashes and the canonical physical-genome contract."""
    paths = {
        "config.ini": Path(config_path),
        "fuel_inventory.json": Path(fuel_inventory_path),
        "materials.xml": Path(materials_path),
    }
    inputs = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    inputs["config.ini"].update(
        {
            "compatibility_sha256": canonical_config_contract_sha256(paths["config.ini"]),
            "compatibility_format": CONFIG_COMPATIBILITY_FORMAT,
        }
    )
    inputs["canonical_genome_contract"] = {
        "path": None,
        "sha256": canonical_genome_contract_sha256(genome_contract),
        "canonical_format": CANONICAL_GENOME_FORMAT,
    }
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "hash_algorithm": HASH_ALGORITHM,
        "inputs": inputs,
        "source_revision": capture_source_revision(source_root),
    }


def validate_input_provenance(provenance: Mapping[str, Any], *, allow_legacy: bool = False) -> None:
    """Validate a provenance manifest, optionally accepting legacy schema v1."""
    if not isinstance(provenance, Mapping):
        raise InputProvenanceError("Input provenance harus berupa mapping.")
    schema_version = provenance.get("schema_version")
    supported_versions = (
        {PROVENANCE_SCHEMA_VERSION, LEGACY_PROVENANCE_SCHEMA_VERSION}
        if allow_legacy
        else {PROVENANCE_SCHEMA_VERSION}
    )
    if schema_version not in supported_versions:
        raise InputProvenanceError(
            f"Input provenance schema v{provenance.get('schema_version')} tidak didukung; diperlukan v{PROVENANCE_SCHEMA_VERSION}."
        )
    if provenance.get("hash_algorithm") != HASH_ALGORITHM:
        raise InputProvenanceError(f"Algoritma hash provenance harus {HASH_ALGORITHM}.")
    inputs = provenance.get("inputs")
    if not isinstance(inputs, Mapping):
        raise InputProvenanceError("Input provenance tidak memiliki mapping inputs.")
    missing_inputs = [name for name in PROTECTED_INPUT_NAMES if name not in inputs]
    if missing_inputs:
        raise InputProvenanceError(
            "Input provenance kehilangan fingerprint: " + ", ".join(missing_inputs) + "."
        )
    for name in PROTECTED_INPUT_NAMES:
        record = inputs[name]
        digest = record.get("sha256") if isinstance(record, Mapping) else None
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise InputProvenanceError(f"Fingerprint {name} bukan SHA-256 heksadesimal valid.")
    if schema_version == PROVENANCE_SCHEMA_VERSION:
        config_record = inputs["config.ini"]
        compatibility_digest = (
            config_record.get("compatibility_sha256")
            if isinstance(config_record, Mapping)
            else None
        )
        if (
            not isinstance(compatibility_digest, str)
            or len(compatibility_digest) != 64
            or any(character not in "0123456789abcdef" for character in compatibility_digest)
        ):
            raise InputProvenanceError(
                "Fingerprint kompatibilitas config.ini bukan SHA-256 heksadesimal valid."
            )
        if config_record.get("compatibility_format") != CONFIG_COMPATIBILITY_FORMAT:
            raise InputProvenanceError(
                f"Format kompatibilitas config.ini harus {CONFIG_COMPATIBILITY_FORMAT}."
            )


def require_matching_input_provenance(
    stored: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    """Reject changed immutable inputs while allowing only generation-target edits."""
    validate_input_provenance(stored)
    validate_input_provenance(current)
    changed_inputs = [
        name
        for name in PROTECTED_INPUT_NAMES
        if name != "config.ini"
        and stored["inputs"][name]["sha256"] != current["inputs"][name]["sha256"]
    ]
    if (
        stored["inputs"]["config.ini"]["compatibility_sha256"]
        != current["inputs"]["config.ini"]["compatibility_sha256"]
    ):
        changed_inputs.insert(0, "config.ini")
    if changed_inputs:
        raise InputProvenanceError(
            "Fingerprint input ilmiah berubah: "
            + ", ".join(changed_inputs)
            + ". Mulai campaign/calibration baru; migrasi otomatis tidak tersedia."
        )


def refresh_file_fingerprints(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Re-hash protected files at recorded paths and return a refreshed manifest copy."""
    validate_input_provenance(provenance)
    refreshed = deepcopy(dict(provenance))
    for name in ("config.ini", "fuel_inventory.json", "materials.xml"):
        path = refreshed["inputs"][name].get("path")
        if not isinstance(path, str) or not path.strip():
            raise InputProvenanceError(
                f"Path audit {name} tidak tersedia untuk refresh fingerprint."
            )
        refreshed["inputs"][name]["sha256"] = sha256_file(path)
        if name == "config.ini":
            refreshed["inputs"][name]["compatibility_sha256"] = canonical_config_contract_sha256(
                path
            )
    return refreshed
