"""Index compatible OpenMC results in a campaign-owned SQLite catalog.

EvaluationCache snapshots source checkpoints, including active campaigns, and
indexes completed results by canonical genome. Compatible hits complete pending
candidates without new transport. Source checkpoints are always read-only."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from copy import deepcopy
import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any
import zlib

from modules.mutable_genome import MutableGenomeContract, MutableGenomeValidationError
from modules.keff_constraint import KeffConstraint
from modules.checkpoint import CheckpointManager


class EvaluationCacheError(RuntimeError):
    """Report an incompatible cache source or an invalid catalog."""


class EvaluationCache:
    """Build and query a compatible OpenMC result cache for one target campaign."""

    SCHEMA_VERSION = 2
    CATALOG_NAME = "evaluation_cache.sqlite3"

    def __init__(
        self,
        *,
        enabled: bool,
        source_checkpoints: Sequence[str | Path],
        allow_legacy_context: bool,
        compatible_openmc_version_pairs: Sequence[Sequence[str]] = (),
        project_root: str | Path,
        campaign_root: str | Path,
        target_checkpoint: str | Path,
        evaluation_context: Mapping[str, Any],
        genome_contract: Mapping[str, Any],
        state: MutableMapping[str, Any],
        checkpoint_manager: Any,
    ) -> None:
        """Connect a source catalog and cache statistics to the target campaign state.

        ``source_checkpoints`` are ordered by priority. ``evaluation_context`` and
        ``genome_contract`` define the target scientific contract. Build the catalog
        when required without modifying source checkpoints; return None."""
        if not isinstance(enabled, bool) or not isinstance(allow_legacy_context, bool):
            raise TypeError("enabled dan allow_legacy_context cache harus bool.")
        if not isinstance(state, MutableMapping):
            raise TypeError("state cache harus mutable mapping.")
        self.enabled = enabled
        self.allow_legacy_context = allow_legacy_context
        self.compatible_openmc_version_pairs = self._normalize_version_pairs(
            compatible_openmc_version_pairs
        )
        self.version_policy_hash = self._hash_json(self.compatible_openmc_version_pairs)
        self.project_root = Path(project_root).resolve()
        self.campaign_root = Path(campaign_root).resolve()
        self.target_checkpoint = Path(target_checkpoint).resolve()
        self.catalog_path = self.campaign_root / self.CATALOG_NAME
        self.evaluation_context = self._json_value(evaluation_context, "evaluation_context")
        self.context_hash = self._hash_json(self.evaluation_context)
        self.genome_contract = self._json_value(genome_contract, "genome_contract")
        self.state = state
        self.checkpoint_manager = checkpoint_manager
        self.source_paths = tuple(self._resolve_source(path) for path in source_checkpoints)
        if len(set(self.source_paths)) != len(self.source_paths):
            raise EvaluationCacheError(
                "Daftar checkpoint cache mengandung path duplikat setelah resolusi."
            )
        if self.target_checkpoint in self.source_paths:
            raise EvaluationCacheError(
                "Checkpoint target tidak boleh menjadi sumber cache dirinya sendiri."
            )
        self._connection: sqlite3.Connection | None = None
        if not self.enabled:
            return
        if not self.source_paths:
            raise EvaluationCacheError("Cache aktif memerlukan sedikitnya satu checkpoint sumber.")
        self._initialize_catalog()

    @staticmethod
    def _normalize_version_pairs(
        pairs: Sequence[Sequence[str]],
    ) -> tuple[tuple[str, str], ...]:
        """Validate directed ``pairs`` of source/target versions and return canonical tuples."""
        if isinstance(pairs, (str, bytes)):
            raise TypeError("Pasangan kompatibilitas OpenMC harus berupa sequence pasangan.")
        normalized: list[tuple[str, str]] = []
        for pair in pairs:
            if isinstance(pair, (str, bytes)):
                raise TypeError("Setiap kompatibilitas OpenMC harus berupa pasangan versi.")
            try:
                source_version, target_version = pair
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "Setiap kompatibilitas OpenMC harus berisi versi sumber dan target."
                ) from error
            if not all(
                isinstance(version, str) and version.strip()
                for version in (source_version, target_version)
            ):
                raise TypeError("Versi OpenMC pada kebijakan kompatibilitas tidak boleh kosong.")
            normalized_pair = (source_version.strip(), target_version.strip())
            if normalized_pair[0] == normalized_pair[1]:
                raise ValueError("Pasangan versi OpenMC identik tidak perlu dikonfigurasi.")
            normalized.append(normalized_pair)
        if len(set(normalized)) != len(normalized):
            raise ValueError("Kebijakan kompatibilitas OpenMC mengandung pasangan duplikat.")
        return tuple(normalized)

    @staticmethod
    def _json_value(value: Any, name: str) -> Any:
        """Return a strict-JSON copy of ``value``, using ``name`` to label invalid values."""
        try:
            return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise EvaluationCacheError(
                f"{name} tidak dapat diserialisasi sebagai JSON ketat."
            ) from error

    def _resolve_source(self, configured_path: str | Path) -> Path:
        """Resolve ``configured_path`` against the project root and return its absolute path."""
        if not isinstance(configured_path, (str, Path)) or not str(configured_path).strip():
            raise EvaluationCacheError("Path checkpoint cache harus berupa path non-kosong.")
        path = Path(configured_path)
        return (path if path.is_absolute() else self.project_root / path).resolve()

    @staticmethod
    def _hash_bytes(payload: bytes) -> str:
        """Return the hexadecimal SHA-256 digest of byte ``payload``."""
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def _hash_json(cls, value: Any) -> str:
        """Serialize ``value`` canonically and return its SHA-256 digest."""
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return cls._hash_bytes(payload)

    @classmethod
    def genome_identity(cls, genome: Any) -> tuple[str, str]:
        """Return the SHA-256 digest and canonical JSON of ordered gene pairs in ``genome``."""
        if isinstance(genome, (str, bytes, Mapping)):
            raise EvaluationCacheError("Genome cache harus berupa urutan pasangan gene.")
        try:
            genes = list(genome)
        except TypeError as error:
            raise EvaluationCacheError("Genome cache tidak berbentuk pasangan gene.") from error
        normalized = []
        for gene in genes:
            if isinstance(gene, (str, bytes)):
                raise EvaluationCacheError("Genome cache mengandung gene bukan pasangan.")
            try:
                parts = list(gene)
            except TypeError as error:
                raise EvaluationCacheError(
                    "Genome cache mengandung gene bukan pasangan."
                ) from error
            if len(parts) != 2 or any(not isinstance(part, str) or not part for part in parts):
                raise EvaluationCacheError("Genome cache mengandung pasangan gene tidak valid.")
            normalized.append(parts)
        genome_json = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
        return cls._hash_bytes(genome_json.encode("utf-8")), genome_json

    @staticmethod
    def _sha256_file(path: Path) -> str:
        """Stream file ``path`` without modification and return its SHA-256 digest."""
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _initialize_catalog(self) -> None:
        """Reuse a verified catalog or atomically build a new source snapshot; return None."""
        recorded = self.state.get("metadata", {}).get("evaluation_cache")
        if isinstance(recorded, Mapping) and self.catalog_path.is_file():
            observed_hash = self._sha256_file(self.catalog_path)
            if (
                recorded.get("schema_version") == self.SCHEMA_VERSION
                and recorded.get("context_hash") == self.context_hash
                and recorded.get("compatible_openmc_version_pairs")
                == [list(pair) for pair in self.compatible_openmc_version_pairs]
                and recorded.get("catalog_sha256") == observed_hash
            ):
                self._open_catalog()
                return
        manifest = self._build_catalog()
        metadata = self.state.setdefault("metadata", {})
        metadata["evaluation_cache"] = manifest
        self.checkpoint_manager.save(self.state)
        self._open_catalog()

    def _open_catalog(self) -> None:
        """Open the SQLite catalog read-only and verify its identity; return None."""
        uri = f"file:{self.catalog_path.as_posix()}?mode=ro"
        try:
            connection = sqlite3.connect(uri, uri=True)
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'context_hash'"
            ).fetchone()
            policy_row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'version_policy_hash'"
            ).fetchone()
        except sqlite3.Error as error:
            raise EvaluationCacheError(
                f"Katalog cache tidak dapat dibuka: {self.catalog_path}."
            ) from error
        if (
            row is None
            or row[0] != self.context_hash
            or policy_row is None
            or policy_row[0] != self.version_policy_hash
        ):
            connection.close()
            raise EvaluationCacheError(
                "Katalog cache memiliki evaluation context atau kebijakan versi yang berbeda."
            )
        self._connection = connection

    def _build_catalog(self) -> dict[str, Any]:
        """Validate sources in priority order, atomically write SQLite, and return catalog metadata."""
        self.campaign_root.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.CATALOG_NAME}.", suffix=".tmp", dir=self.campaign_root
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        source_records: list[dict[str, Any]] = []
        inserted_count = 0
        conflict_count = 0
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(temporary_path)
            connection.executescript(
                """
                PRAGMA journal_mode=DELETE;
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE entries (
                    genome_hash TEXT PRIMARY KEY,
                    genome_json TEXT NOT NULL,
                    result_blob BLOB NOT NULL,
                    source_priority INTEGER NOT NULL,
                    source_path TEXT NOT NULL,
                    source_checkpoint_sha256 TEXT NOT NULL,
                    source_campaign_id TEXT NOT NULL,
                    source_individual_id TEXT NOT NULL,
                    validation_mode TEXT NOT NULL,
                    context_hash TEXT NOT NULL
                );
                """
            )
            connection.execute(
                """INSERT INTO metadata(key, value)
                   VALUES ('schema_version', ?), ('context_hash', ?), ('version_policy_hash', ?)""",
                (str(self.SCHEMA_VERSION), self.context_hash, self.version_policy_hash),
            )
            for priority, source_path in enumerate(self.source_paths):
                source_record, source_inserted, source_conflicts = self._ingest_source(
                    connection, source_path, priority
                )
                source_records.append(source_record)
                inserted_count += source_inserted
                conflict_count += source_conflicts
            connection.commit()
            connection.execute("VACUUM")
            connection.close()
            os.replace(temporary_path, self.catalog_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        finally:
            if connection is not None:
                connection.close()
        catalog_sha256 = self._sha256_file(self.catalog_path)
        logging.info(
            "Katalog evaluation cache dibangun: sources=%s entries=%s conflicts=%s path=%s.",
            len(source_records),
            inserted_count,
            conflict_count,
            self.catalog_path,
        )
        return {
            "schema_version": self.SCHEMA_VERSION,
            "enabled": True,
            "allow_legacy_context": self.allow_legacy_context,
            "compatible_openmc_version_pairs": [
                list(pair) for pair in self.compatible_openmc_version_pairs
            ],
            "context_hash": self.context_hash,
            "catalog_path": str(self.catalog_path),
            "catalog_sha256": catalog_sha256,
            "sources": source_records,
            "entry_count": inserted_count,
            "priority_conflicts": conflict_count,
            "stats": {"hits": 0, "misses": 0, "simulations_avoided": 0},
        }

    def _load_source(self, source_path: Path) -> tuple[dict[str, Any], str]:
        """Read and validate checkpoint ``source_path``; return its JSON payload and byte hash."""
        if not source_path.is_file():
            raise EvaluationCacheError(f"Checkpoint sumber cache tidak ditemukan: {source_path}.")
        try:
            source_bytes = source_path.read_bytes()
            source = json.loads(source_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise EvaluationCacheError(
                f"Checkpoint sumber bukan JSON UTF-8 valid: {source_path}."
            ) from error
        if not isinstance(source, dict):
            raise EvaluationCacheError(
                f"Checkpoint sumber harus berupa object JSON: {source_path}."
            )
        return source, self._hash_bytes(source_bytes)

    def _validate_source_header(self, source: Mapping[str, Any], source_path: Path) -> str:
        """Validate ``source`` phase and scientific contract at ``source_path``; return its campaign ID."""
        progress = source.get("progress")
        if (
            not isinstance(progress, Mapping)
            or progress.get("phase") not in CheckpointManager.PHASES
        ):
            raise EvaluationCacheError(f"Phase checkpoint sumber tidak valid: {source_path}.")
        campaign = source.get("campaign")
        campaign_id = campaign.get("campaign_id") if isinstance(campaign, Mapping) else None
        if not isinstance(campaign_id, str) or not campaign_id.strip():
            raise EvaluationCacheError(f"Campaign ID sumber cache tidak valid: {source_path}.")
        source_contract = source.get("genome_contract")
        if self._hash_json(source_contract) != self._hash_json(self.genome_contract):
            raise EvaluationCacheError(f"Kontrak genome sumber cache berbeda: {source_path}.")
        source_constraint = source.get("constraint_contract")
        target_constraint = self.evaluation_context.get("evaluation_config", {}).get("constraint")
        if source_constraint != target_constraint:
            raise EvaluationCacheError(f"Kontrak constraint sumber cache berbeda: {source_path}.")
        source_inputs = source.get("input_provenance", {}).get("inputs", {})
        target_inputs = self.evaluation_context.get("scientific_inputs", {})
        for name in ("fuel_inventory.json", "materials.xml", "canonical_genome_contract"):
            record = source_inputs.get(name) if isinstance(source_inputs, Mapping) else None
            if not isinstance(record, Mapping) or record.get("sha256") != target_inputs.get(name):
                raise EvaluationCacheError(
                    f"Fingerprint {name} sumber cache berbeda: {source_path}."
                )
        return campaign_id.strip()

    def _ingest_source(
        self, connection: sqlite3.Connection, source_path: Path, priority: int
    ) -> tuple[dict[str, Any], int, int]:
        """Index compatible results from ``source_path`` using its configured ``priority``.

        Write through SQLite ``connection`` and return source metadata, accepted result
        count, and duplicate result count."""
        source, checkpoint_sha256 = self._load_source(source_path)
        campaign_id = self._validate_source_header(source, source_path)
        individuals = source.get("individuals")
        if not isinstance(individuals, Mapping):
            raise EvaluationCacheError(
                f"Checkpoint sumber tidak memiliki individuals: {source_path}."
            )
        inserted = 0
        conflicts = 0
        eligible = 0
        validation_modes: set[str] = set()
        mutable_contract = MutableGenomeContract.from_checkpoint_contract(self.genome_contract)
        for individual_id, individual in individuals.items():
            if not isinstance(individual, Mapping) or individual.get("status") != "completed":
                continue
            results = individual.get("results")
            if not isinstance(results, Mapping):
                raise EvaluationCacheError(
                    f"Hasil completed {individual_id} tidak valid pada {source_path}."
                )
            mode = self._validation_mode(individual, results, source_path, str(individual_id))
            eligible += 1
            validation_modes.add(mode)
            try:
                validated_genome = mutable_contract.validate(
                    individual.get("genome"), f"Genome cache {individual_id}"
                )
            except MutableGenomeValidationError as error:
                raise EvaluationCacheError(str(error)) from error
            genome_hash, genome_json = self.genome_identity(validated_genome)
            result_bytes = json.dumps(
                results, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            cursor = connection.execute(
                """INSERT OR IGNORE INTO entries
                   (genome_hash, genome_json, result_blob, source_priority, source_path,
                    source_checkpoint_sha256, source_campaign_id, source_individual_id,
                    validation_mode, context_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    genome_hash,
                    genome_json,
                    sqlite3.Binary(zlib.compress(result_bytes, 6)),
                    priority,
                    str(source_path),
                    checkpoint_sha256,
                    campaign_id,
                    str(individual_id),
                    mode,
                    self.context_hash,
                ),
            )
            if cursor.rowcount == 1:
                inserted += 1
            else:
                conflicts += 1
        return (
            {
                "priority": priority,
                "path": str(source_path),
                "checkpoint_sha256": checkpoint_sha256,
                "campaign_id": campaign_id,
                "snapshot_phase": source["progress"]["phase"],
                "snapshot_updated_at": source["campaign"].get("updated_at"),
                "eligible_completed": eligible,
                "inserted": inserted,
                "priority_conflicts": conflicts,
                "validation_modes": sorted(validation_modes),
            },
            inserted,
            conflicts,
        )

    def _validation_mode(
        self,
        individual: Mapping[str, Any],
        results: Mapping[str, Any],
        source_path: Path,
        individual_id: str,
    ) -> str:
        """Validate ``individual`` and ``results`` from ``source_path`` for ``individual_id``.

        Return the strict or explicitly permitted legacy validation mode."""
        source_context = individual.get("operator", {}).get("evaluation_context")
        if isinstance(source_context, Mapping):
            source_transport = source_context.get("openmc_parameters")
            target_transport = self.evaluation_context["openmc_parameters"]
            for parameter in ("particles", "inactive_batches", "active_batches"):
                if (
                    not isinstance(source_transport, Mapping)
                    or source_transport.get(parameter) != target_transport[parameter]
                ):
                    raise EvaluationCacheError(
                        f"Parameter {parameter} berbeda untuk {individual_id}: {source_path}."
                    )
        if source_context == self.evaluation_context:
            self._validate_common_results(results, source_path, individual_id)
            return "strict"
        if self._contexts_differ_only_by_compatible_openmc_version(source_context):
            self._validate_common_results(results, source_path, individual_id)
            observed_version = self._result_openmc_version(results)
            expected_source_version = str(source_context["openmc_version"])
            if observed_version != expected_source_version:
                raise EvaluationCacheError(
                    f"Versi OpenMC hasil/context berbeda untuk {individual_id}: {source_path}."
                )
            return "strict-compatible-openmc-version"
        if not self.allow_legacy_context:
            raise EvaluationCacheError(
                f"Evaluation context {individual_id} hilang/berbeda dan legacy dinonaktifkan: {source_path}."
            )
        exact_version = self._validate_legacy_results(results, source_path, individual_id)
        return "legacy" if exact_version else "legacy-compatible-openmc-version"

    @staticmethod
    def _result_openmc_version(results: Mapping[str, Any]) -> str | None:
        """Return the recorded statepoint OpenMC version in ``results``, or None if absent."""
        statistics = results.get("statistics")
        statepoint = statistics.get("statepoint") if isinstance(statistics, Mapping) else None
        version = statepoint.get("openmc_version") if isinstance(statepoint, Mapping) else None
        return str(version) if version is not None else None

    def _openmc_versions_compatible(self, source_version: str, target_version: str) -> bool:
        """Return whether ``source_version`` equals ``target_version`` or is explicitly allowed."""
        return (
            source_version == target_version
            or (
                source_version,
                target_version,
            )
            in self.compatible_openmc_version_pairs
        )

    def _contexts_differ_only_by_compatible_openmc_version(self, source_context: Any) -> bool:
        """Return whether ``source_context`` matches the target except for an allowed OpenMC version pair."""
        if not isinstance(source_context, Mapping):
            return False
        source_version = str(source_context.get("openmc_version"))
        target_version = str(self.evaluation_context.get("openmc_version"))
        if source_version == target_version or not self._openmc_versions_compatible(
            source_version, target_version
        ):
            return False
        source_without_version = dict(source_context)
        target_without_version = dict(self.evaluation_context)
        source_without_version.pop("openmc_version", None)
        target_without_version.pop("openmc_version", None)
        return source_without_version == target_without_version

    def _validate_common_results(
        self, results: Mapping[str, Any], source_path: Path, individual_id: str
    ) -> None:
        """Validate objectives, constraints, and convergence in ``results`` for the labeled source; return None."""
        convergence = results.get("convergence")
        if not isinstance(convergence, Mapping) or convergence.get("passed") is not True:
            raise EvaluationCacheError(
                f"Hasil {individual_id} tidak lulus convergence: {source_path}."
            )
        objectives = results.get("objectives")
        keff = results.get("keff")
        if (
            not isinstance(objectives, list)
            or len(objectives) != 2
            or not isinstance(keff, Mapping)
            or objectives[0] != -keff.get("mean")
            or objectives[1] != results.get("ppf_3d")
        ):
            raise EvaluationCacheError(
                f"Objective hasil {individual_id} tidak konsisten: {source_path}."
            )
        if results.get("constraint") is None:
            raise EvaluationCacheError(
                f"Constraint hasil {individual_id} tidak tersedia: {source_path}."
            )
        constraint_identity = self.evaluation_context["evaluation_config"]["constraint"]
        try:
            constraint = KeffConstraint(
                constraint_identity["keff_min"],
                constraint_identity["feasibility_tolerance"],
            )
            constraint.validate_results(results)
        except (KeyError, TypeError, ValueError) as error:
            raise EvaluationCacheError(
                f"Constraint hasil {individual_id} tidak konsisten: {source_path}."
            ) from error

    def _validate_legacy_results(
        self, results: Mapping[str, Any], source_path: Path, individual_id: str
    ) -> bool:
        """Validate legacy ``results`` and source labels; return whether the OpenMC version matches exactly."""
        self._validate_common_results(results, source_path, individual_id)
        expected_eval = self.evaluation_context["evaluation_config"]
        expected_transport = self.evaluation_context["openmc_parameters"]
        metadata = results.get("metadata")
        statistics = results.get("statistics")
        statepoint = statistics.get("statepoint") if isinstance(statistics, Mapping) else None
        if not isinstance(metadata, Mapping) or not isinstance(statepoint, Mapping):
            raise EvaluationCacheError(
                f"Metadata legacy {individual_id} tidak lengkap: {source_path}."
            )
        required_metadata = {
            "tally_name": expected_eval["tally_name"],
            "tally_score": expected_eval["tally_score"],
            "objective_names": ["negative_keff", "ppf_3d"],
            "objective_orientation": "minimize",
            "ppf_3d_aggregation": "max_segment_score_over_mean_segment_score",
        }
        for key, expected in required_metadata.items():
            if metadata.get(key) != expected:
                raise EvaluationCacheError(
                    f"Metadata legacy {key} berbeda untuk {individual_id}: {source_path}."
                )
        source_version = str(statepoint.get("openmc_version"))
        target_version = str(self.evaluation_context["openmc_version"])
        if not self._openmc_versions_compatible(source_version, target_version):
            raise EvaluationCacheError(
                f"Versi OpenMC legacy {source_version} -> {target_version} tidak diizinkan "
                f"untuk {individual_id}: {source_path}."
            )
        if statepoint.get("particles_per_generation") != expected_transport["particles"]:
            raise EvaluationCacheError(
                f"Jumlah partikel legacy berbeda untuk {individual_id}: {source_path}."
            )
        if statepoint.get("inactive_batches") != expected_transport["inactive_batches"]:
            raise EvaluationCacheError(
                f"Inactive batch legacy berbeda untuk {individual_id}: {source_path}."
            )
        if statepoint.get("active_batches_completed") != expected_transport["active_batches"]:
            raise EvaluationCacheError(
                f"Active batch legacy berbeda untuk {individual_id}: {source_path}."
            )
        expected_convergence = expected_eval.get("convergence", {})
        observed_convergence = results.get("convergence", {})
        if observed_convergence.get("enabled") != expected_convergence.get("enabled"):
            raise EvaluationCacheError(
                f"Mode convergence legacy berbeda untuk {individual_id}: {source_path}."
            )
        criteria = observed_convergence.get("criteria")
        if not isinstance(criteria, Mapping):
            raise EvaluationCacheError(
                f"Kriteria convergence legacy hilang untuk {individual_id}: {source_path}."
            )
        limits = {
            "keff_std_dev_pcm": expected_convergence.get("max_keff_std_dev_pcm"),
            "tally_mean_relative_std_dev": expected_convergence.get(
                "max_tally_mean_relative_std_dev"
            ),
            "tally_p95_relative_std_dev": expected_convergence.get(
                "max_tally_p95_relative_std_dev"
            ),
            "tally_max_relative_std_dev": expected_convergence.get(
                "max_tally_max_relative_std_dev"
            ),
        }
        for criterion, limit in limits.items():
            record = criteria.get(criterion)
            if not isinstance(record, Mapping) or record.get("observed", float("inf")) > limit:
                raise EvaluationCacheError(
                    f"Kriteria {criterion} legacy gagal untuk {individual_id}: {source_path}."
                )
        zero_record = criteria.get("zero_mean_bins")
        if not isinstance(zero_record, Mapping):
            raise EvaluationCacheError(
                f"Kriteria zero bin legacy hilang untuk {individual_id}: {source_path}."
            )
        allow_zero = expected_convergence.get("allow_zero_mean_bins")
        if not allow_zero and zero_record.get("observed") != 0:
            raise EvaluationCacheError(
                f"Zero mean bin legacy tidak diizinkan untuk {individual_id}: {source_path}."
            )
        return source_version == target_version

    def try_reuse(self, state: MutableMapping[str, Any], individual_id: str) -> bool:
        """Commit compatible cached results for ``individual_id`` in ``state`` and return whether a hit occurred."""
        if not self.enabled:
            return False
        individual = state["individuals"][individual_id]
        existing = individual.get("operator", {}).get("evaluation_cache")
        if isinstance(existing, Mapping) and existing.get("lookup") == "miss":
            return False
        genome_hash, genome_json = self.genome_identity(individual["genome"])
        if self._connection is None:
            raise EvaluationCacheError("Katalog cache belum dibuka.")
        row = self._connection.execute(
            """SELECT genome_json, result_blob, source_path, source_checkpoint_sha256,
                      source_campaign_id, source_individual_id, validation_mode
               FROM entries WHERE genome_hash = ?""",
            (genome_hash,),
        ).fetchone()
        if row is None or row[0] != genome_json:
            self._record_miss(individual)
            return False
        results = json.loads(zlib.decompress(row[1]).decode("utf-8"))
        provenance = {
            "lookup": "hit",
            "genome_hash": genome_hash,
            "context_hash": self.context_hash,
            "source_path": row[2],
            "source_checkpoint_sha256": row[3],
            "source_campaign_id": row[4],
            "source_individual_id": row[5],
            "validation_mode": row[6],
            "source_openmc_version": self._result_openmc_version(results),
            "target_openmc_version": self.evaluation_context["openmc_version"],
        }
        self.checkpoint_manager.reuse_cached_evaluation(
            state,
            individual_id,
            deepcopy(results),
            self.evaluation_context,
            provenance,
        )
        stats = state["metadata"]["evaluation_cache"]["stats"]
        stats["hits"] += 1
        stats["simulations_avoided"] += 1
        logging.info(
            "Evaluation cache hit: individual=%s source=%s/%s mode=%s.",
            individual_id,
            row[4],
            row[5],
            row[6],
        )
        return True

    def _record_miss(self, individual: MutableMapping[str, Any]) -> None:
        """Record a cache miss once for ``individual`` and update campaign statistics; return None."""
        operator = individual.setdefault("operator", {})
        existing = operator.get("evaluation_cache")
        if isinstance(existing, Mapping) and existing.get("lookup") == "miss":
            return
        genome_hash, _ = self.genome_identity(individual["genome"])
        operator["evaluation_cache"] = {
            "lookup": "miss",
            "genome_hash": genome_hash,
            "context_hash": self.context_hash,
        }
        self.state["metadata"]["evaluation_cache"]["stats"]["misses"] += 1

    def summary(self) -> dict[str, Any]:
        """Return public cache statistics from the target checkpoint metadata."""
        if not self.enabled:
            return {
                "enabled": False,
                "source_count": 0,
                "entry_count": 0,
                "hits": 0,
                "misses": 0,
                "simulations_avoided": 0,
            }
        metadata = self.state["metadata"]["evaluation_cache"]
        stats = metadata["stats"]
        return {
            "enabled": True,
            "source_count": len(metadata["sources"]),
            "entry_count": metadata["entry_count"],
            "priority_conflicts": metadata["priority_conflicts"],
            "hits": stats["hits"],
            "misses": stats["misses"],
            "simulations_avoided": stats["simulations_avoided"],
        }
