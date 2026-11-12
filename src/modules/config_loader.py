"""Load typed INI parameters and validate the runtime configuration.

ConfigLoader checks genome contracts, operators, population strategies,
transport parameters, and convergence limits. get_params returns an independent
copy of the validated configuration."""

from __future__ import annotations

import configparser
from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Callable

from modules.campaign_identity import validate_campaign_id

from modules.crossover import (
    IMPLEMENTED_CROSSOVER_METHODS,
    TRIGA_HCMX_NONFUEL_POLICY,
    TRIGA_HCMX_POWER_MASK_METHOD,
)
from modules.mutation import IMPLEMENTED_MUTATION_METHODS
from modules.keff_constraint import KeffConstraint


class ConfigLoader:
    """Read an INI configuration and expose validated Python parameters."""

    SUPPORTED_TALLY_TRIGGER_TYPES = frozenset({"rel_err", "std_dev", "variance"})
    OBSOLETE_OBJECTIVE_SECTIONS = frozenset(
        {"fitness_keff", "fitness_ppf", "fitness_combined", "constraints"}
    )
    OBSOLETE_OPENMC_OPTIONS = frozenset({"final_statepoint_batch"})
    CROSSOVER_STRATEGY_SCHEMAS: dict[str, dict[str, Callable[[str], Any]]] = {
        TRIGA_HCMX_POWER_MASK_METHOD: {
            "mask_size_min": int,
            "mask_size_max": int,
            "selection_alpha": float,
            "epsilon": float,
            "nonfuel_policy": str,
            "require_parent_power_tally": bool,
            "require_convergence_passed": bool,
        },
    }

    _SECTION_SCHEMA: dict[str, dict[str, Callable[[str], Any]]] = {
        "campaign": {"campaign_id": str, "run_root": str},
        "paths": {"fuel_inventory_file": str, "materials_file": str, "openmc_executable": str},
        "ga": {
            "algorithm": str,
            "keff_min": float,
            "feasibility_tolerance": float,
            "population_size": int,
            "num_generations": int,
            "max_reproduction_attempts": int,
            "resample_duplicate_offspring": bool,
            "random_seed": int,
        },
        "selection": {"method": str, "tournament_size": int},
        "crossover": {"method": str, "probability": float},
        "mutation": {
            "method": str,
            "probability_per_individual": float,
            "swaps_per_mutated_individual": int,
        },
        "replacement": {"method": str},
        "openmc": {
            "particles": int,
            "batches": int,
            "inactive": int,
            "threads": int,
            "geometry_debug": bool,
            "keff_trigger_enabled": bool,
            "keff_trigger_stdev_pcm": float,
            "trigger_batch_interval": int,
            "trigger_max_batches": int,
            "tally_trigger_enabled": bool,
            "tally_trigger_type": str,
            "tally_trigger_threshold": float,
            "tally_trigger_ignore_zeros": bool,
            "fission_tally_name": str,
            "fission_tally_score": str,
        },
        "convergence": {
            "enabled": bool,
            "max_keff_std_dev_pcm": float,
            "max_tally_mean_relative_std_dev": float,
            "max_tally_p95_relative_std_dev": float,
            "max_tally_max_relative_std_dev": float,
            "allow_zero_mean_bins": bool,
        },
        "retry": {
            "max_attempts": int,
            "interrupted_counts_as_failed_attempt": bool,
            "on_max_attempts": str,
        },
    }
    _OPTION_DEFAULTS: dict[tuple[str, str], Any] = {
        ("ga", "max_reproduction_attempts"): 10000,
        ("ga", "resample_duplicate_offspring"): True,
        ("ga", "keff_min"): 1.0,
        ("ga", "feasibility_tolerance"): 0.0,
        ("paths", "openmc_executable"): None,
        ("openmc", "trigger_batch_interval"): 5,
        ("openmc", "trigger_max_batches"): 300,
        ("openmc", "tally_trigger_enabled"): False,
        ("openmc", "tally_trigger_type"): "rel_err",
        ("openmc", "tally_trigger_threshold"): 0.10,
        ("openmc", "tally_trigger_ignore_zeros"): False,
        ("convergence", "enabled"): False,
        ("convergence", "max_keff_std_dev_pcm"): 50.0,
        ("convergence", "max_tally_mean_relative_std_dev"): 0.05,
        ("convergence", "max_tally_p95_relative_std_dev"): 0.10,
        ("convergence", "max_tally_max_relative_std_dev"): 0.20,
        ("convergence", "allow_zero_mean_bins"): False,
    }
    _REQUIRED_SECTIONS = frozenset(
        {
            *(_SECTION_SCHEMA.keys() - {"convergence"}),
            "initial_population_percentages",
            "chromosome",
            "excluded_genes",
        }
    )

    def __init__(self, config_path: str | Path) -> None:
        """Read ``config_path``, validate its settings, and store typed parameters; return None."""
        self.config_path = Path(config_path)
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        self.params: dict[str, dict[str, Any]] = {}

        if not self.config_path.is_file():
            raise FileNotFoundError(f"Configuration file tidak ditemukan: {self.config_path}.")

        try:
            with self.config_path.open("r", encoding="utf-8") as stream:
                self.config.read_file(stream)
        except configparser.Error as exc:
            raise ValueError(f"Format file konfigurasi tidak valid: {self.config_path}.") from exc

        self._load_all_configs()
        self._validate_configs()

    def _load_all_configs(self) -> None:
        """Parse all configured sections into ``self.params``; return None."""
        self._validate_required_sections()

        for section, option_schema in self._SECTION_SCHEMA.items():
            self.params[section] = {
                option: self._get_typed_value(section, option, value_type)
                for option, value_type in option_schema.items()
            }
        self._load_crossover_strategy_config()
        self._load_evaluation_cache_config()

        self.params["initial_population_percentages"] = {
            strategy: self._get_typed_value("initial_population_percentages", strategy, float)
            for strategy in self.config.options("initial_population_percentages")
        }
        self.params["chromosome"] = self._load_chromosome_config()

    def _load_evaluation_cache_config(self) -> None:
        """The optional section accepts one source path per continuation line (or
        comma-separated paths). OpenMC compatibility entries use the directed
        ``source_version -> target_version`` form."""
        section = "evaluation_cache"
        if not self.config.has_section(section):
            self.params[section] = {
                "enabled": False,
                "source_checkpoints": (),
                "allow_legacy_context": False,
                "compatible_openmc_version_pairs": (),
            }
            return
        allowed_options = {
            "enabled",
            "source_checkpoints",
            "allow_legacy_context",
            "compatible_openmc_version_pairs",
        }
        unknown_options = set(self.config.options(section)).difference(allowed_options)
        if unknown_options:
            raise ValueError(
                "Option evaluation_cache tidak dikenal: " + ", ".join(sorted(unknown_options)) + "."
            )
        try:
            enabled = self.config.getboolean(section, "enabled", fallback=False)
            allow_legacy = self.config.getboolean(section, "allow_legacy_context", fallback=False)
        except ValueError as error:
            raise ValueError(
                "evaluation_cache.enabled dan allow_legacy_context harus boolean."
            ) from error
        raw_sources = self.config.get(section, "source_checkpoints", fallback="")
        sources = tuple(
            token.strip()
            for line in raw_sources.splitlines()
            for token in line.split(",")
            if token.strip()
        )
        if len(set(sources)) != len(sources):
            raise ValueError("evaluation_cache.source_checkpoints mengandung path duplikat.")
        if enabled and not sources:
            raise ValueError("evaluation_cache.source_checkpoints wajib diisi ketika cache aktif.")
        raw_version_pairs = self.config.get(section, "compatible_openmc_version_pairs", fallback="")
        version_pairs: list[tuple[str, str]] = []
        for entry in (
            token.strip()
            for line in raw_version_pairs.splitlines()
            for token in line.split(",")
            if token.strip()
        ):
            parts = [part.strip() for part in entry.split("->")]
            if len(parts) != 2 or not all(parts):
                raise ValueError(
                    "evaluation_cache.compatible_openmc_version_pairs harus memakai "
                    "format 'versi_sumber -> versi_target'."
                )
            pair = (parts[0], parts[1])
            if pair[0] == pair[1]:
                raise ValueError("Pasangan versi OpenMC identik tidak perlu dikonfigurasi.")
            version_pairs.append(pair)
        if len(set(version_pairs)) != len(version_pairs):
            raise ValueError(
                "evaluation_cache.compatible_openmc_version_pairs mengandung pasangan duplikat."
            )
        self.params[section] = {
            "enabled": enabled,
            "source_checkpoints": sources,
            "allow_legacy_context": allow_legacy,
            "compatible_openmc_version_pairs": tuple(version_pairs),
        }

    def _load_crossover_strategy_config(self) -> None:
        """Load only options owned by the selected implemented crossover strategy."""
        method = self.params["crossover"]["method"]
        strategy_schema = self.CROSSOVER_STRATEGY_SCHEMAS.get(method, {})
        if method not in IMPLEMENTED_CROSSOVER_METHODS:
            return
        allowed_options = set(self._SECTION_SCHEMA["crossover"]).union(strategy_schema)
        incompatible_options = set(self.config.options("crossover")).difference(allowed_options)
        if incompatible_options:
            raise ValueError(
                f"Option crossover tidak berlaku untuk method {method!r}: {', '.join(sorted(incompatible_options))}."
            )
        for option, value_type in strategy_schema.items():
            self.params["crossover"][option] = self._get_typed_value(
                "crossover", option, value_type
            )

    def _load_chromosome_config(self) -> dict[str, Any]:
        """Return chromosome settings, mutable allele inventory, tokens, and fixed-gene mapping."""
        chromosome: dict[str, Any] = {
            "encoding": self._get_typed_value("chromosome", "encoding", str),
            "genome_length": self._get_typed_value("chromosome", "genome_length", int),
            "complete_genome_length": self._get_typed_value(
                "chromosome", "complete_genome_length", int
            ),
            "movable_pin_count": self._get_typed_value("chromosome", "movable_pin_count", int),
            "fuel_pin_ids": self._get_csv("chromosome", "fuel_pin_ids"),
            "excluded_positions": self._get_csv("chromosome", "excluded_positions"),
            "movable_irradiation_tube_count": self._get_typed_value(
                "chromosome", "movable_irradiation_tube_count", int
            ),
            "empty_hole_count": self._get_typed_value("chromosome", "empty_hole_count", int),
            "empty_hole_tokens": self._get_csv_allow_empty("chromosome", "empty_hole_tokens"),
            "movable_irradiation_tube_tokens": self._get_csv_allow_empty(
                "chromosome", "movable_irradiation_tube_tokens"
            ),
        }

        excluded_genes: dict[str, tuple[str, str]] = {}
        for position in chromosome["excluded_positions"]:
            gene_parts = self._get_csv("excluded_genes", position)
            if len(gene_parts) != 2:
                raise ValueError(
                    f"excluded_genes.{position} harus berbentuk " "'jenis_gene, id_gene'."
                )
            excluded_genes[position] = (gene_parts[0], gene_parts[1])

        fuel_genes = tuple(("fuel", fuel_id) for fuel_id in chromosome["fuel_pin_ids"])
        empty_genes = tuple(("empty_hole", token) for token in chromosome["empty_hole_tokens"])
        irradiation_genes = tuple(
            ("irradiation_tube", token) for token in chromosome["movable_irradiation_tube_tokens"]
        )
        chromosome["excluded_genes"] = excluded_genes
        chromosome["mutable_genes"] = fuel_genes + empty_genes + irradiation_genes
        return chromosome

    def _get_typed_value(
        self,
        section: str,
        option: str,
        value_type: Callable[[str], Any],
    ) -> Any:
        """Read ``option`` in INI ``section``, convert with ``value_type``, and return its value."""
        try:
            raw_value = self.config.get(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError) as exc:
            if (section, option) in self._OPTION_DEFAULTS:
                return self._OPTION_DEFAULTS[(section, option)]
            raise ValueError(f"Konfigurasi wajib {section}.{option} belum tersedia.") from exc

        try:
            if value_type is bool:
                value = self.config.getboolean(section, option)
            elif value_type is int:
                value = self.config.getint(section, option)
            elif value_type is float:
                value = self.config.getfloat(section, option)
            elif value_type is str:
                value = raw_value.strip()
            else:
                value = value_type(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Nilai {section}.{option} tidak valid untuk tipe "
                f"{value_type.__name__}: {raw_value!r}."
            ) from exc

        if isinstance(value, str) and not value:
            raise ValueError(f"Nilai {section}.{option} tidak boleh kosong.")
        return value

    def _get_csv(self, section: str, option: str) -> tuple[str, ...]:
        """Return stripped comma-separated tokens from ``section`` and ``option``."""
        raw_value = self._get_typed_value(section, option, str)
        tokens = tuple(part.strip() for part in raw_value.split(",") if part.strip())
        if not tokens:
            raise ValueError(f"Daftar {section}.{option} tidak boleh kosong.")
        return tokens

    def _get_csv_allow_empty(self, section: str, option: str) -> tuple[str, ...]:
        """Return stripped CSV tokens for ``section``/``option``, allowing an empty tuple."""
        try:
            raw_value = self.config.get(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError) as exc:
            raise ValueError(f"Konfigurasi wajib {section}.{option} belum tersedia.") from exc
        return tuple(part.strip() for part in raw_value.split(",") if part.strip())

    def _validate_required_sections(self) -> None:
        """Require every application-defined INI section; return None or raise on missing sections."""
        obsolete_sections = self.OBSOLETE_OBJECTIVE_SECTIONS.intersection(self.config.sections())
        if obsolete_sections:
            raise ValueError(
                "Section objective lama tidak lagi didukung: "
                + ", ".join(sorted(obsolete_sections))
                + "."
            )
        obsolete_openmc_options = (
            self.OBSOLETE_OPENMC_OPTIONS.intersection(self.config.options("openmc"))
            if self.config.has_section("openmc")
            else set()
        )
        if obsolete_openmc_options:
            raise ValueError(
                "Option OpenMC tanpa runtime owner tidak lagi didukung: "
                + ", ".join(sorted(obsolete_openmc_options))
                + "."
            )
        configured_crossover_options = (
            set(self.config.options("crossover")) if self.config.has_section("crossover") else set()
        )
        known_strategy_options = set().union(
            *(schema.keys() for schema in self.CROSSOVER_STRATEGY_SCHEMAS.values())
        )
        unknown_crossover_options = configured_crossover_options.difference(
            self._SECTION_SCHEMA["crossover"]
        ).difference(known_strategy_options)
        if unknown_crossover_options:
            raise ValueError(
                "Option crossover tidak dikenal atau belum memiliki runtime owner: "
                + ", ".join(sorted(unknown_crossover_options))
                + "."
            )
        missing_sections = self._REQUIRED_SECTIONS.difference(self.config.sections())
        if missing_sections:
            raise ValueError(
                "Section konfigurasi belum tersedia: " + ", ".join(sorted(missing_sections)) + "."
            )

    def _validate_configs(self) -> None:
        """Validate GA, genome, and transport consistency; return None or raise ValueError."""
        self.params["campaign"]["campaign_id"] = validate_campaign_id(
            self.params["campaign"]["campaign_id"]
        )
        chromosome = self.params["chromosome"]
        self._validate_unique_tokens("chromosome.fuel_pin_ids", chromosome["fuel_pin_ids"])
        self._validate_unique_tokens(
            "chromosome.empty_hole_tokens", chromosome["empty_hole_tokens"]
        )
        self._validate_unique_tokens(
            "chromosome.movable_irradiation_tube_tokens",
            chromosome["movable_irradiation_tube_tokens"],
        )
        self._validate_unique_tokens(
            "chromosome.excluded_positions", chromosome["excluded_positions"]
        )
        fixed_mutable_overlap = set(chromosome["mutable_genes"]).intersection(
            chromosome["excluded_genes"].values()
        )
        if fixed_mutable_overlap:
            raise ValueError(
                f"Gene tidak boleh sekaligus fixed dan mutable: {sorted(fixed_mutable_overlap)}."
            )

        self._validate_declared_count("movable_pin_count", len(chromosome["fuel_pin_ids"]))
        self._validate_declared_count("empty_hole_count", len(chromosome["empty_hole_tokens"]))
        self._validate_declared_count(
            "movable_irradiation_tube_count",
            len(chromosome["movable_irradiation_tube_tokens"]),
        )

        mutable_count = len(chromosome["mutable_genes"])
        if chromosome["genome_length"] != mutable_count:
            raise ValueError(
                "chromosome.genome_length tidak sama dengan jumlah gene mutable: "
                f"{chromosome['genome_length']} != {mutable_count}."
            )
        expected_complete_length = mutable_count + len(chromosome["excluded_genes"])
        if chromosome["complete_genome_length"] != expected_complete_length:
            raise ValueError(
                "chromosome.complete_genome_length tidak sama dengan gene mutable "
                f"+ gene excluded: {chromosome['complete_genome_length']} != "
                f"{expected_complete_length}."
            )

        configured_excluded = set(self.config.options("excluded_genes"))
        declared_excluded = set(chromosome["excluded_positions"])
        if configured_excluded != declared_excluded:
            missing = sorted(declared_excluded - configured_excluded)
            extra = sorted(configured_excluded - declared_excluded)
            raise ValueError(
                "Option [excluded_genes] harus tepat sama dengan excluded_positions; "
                f"missing={missing}, extra={extra}."
            )

        self._validate_ga_parameters()
        self._validate_openmc_parameters()
        self._validate_convergence_parameters()

    def _validate_declared_count(self, option: str, actual_count: int) -> None:
        """Require chromosome ``option`` to equal nonnegative ``actual_count``; return None."""
        declared_count = self.params["chromosome"][option]
        if declared_count < 0:
            raise ValueError(f"chromosome.{option} tidak boleh negatif.")
        if declared_count != actual_count:
            raise ValueError(
                f"chromosome.{option} tidak sama dengan jumlah token: "
                f"{declared_count} != {actual_count}."
            )

    @staticmethod
    def _validate_unique_tokens(name: str, tokens: tuple[str, ...]) -> None:
        """Reject duplicates in ``tokens`` using ``name`` as the error label; return None."""
        seen: set[str] = set()
        duplicates: set[str] = set()
        for token in tokens:
            if token in seen:
                duplicates.add(token)
            seen.add(token)
        if duplicates:
            raise ValueError(f"{name} mengandung token duplikat: {sorted(duplicates)}.")

    def _validate_ga_parameters(self) -> None:
        """Validate population size, reproduction limits, and operator probabilities; return None."""
        ga = self.params["ga"]
        KeffConstraint(ga["keff_min"], ga["feasibility_tolerance"])
        selection = self.params["selection"]
        crossover = self.params["crossover"]
        mutation = self.params["mutation"]

        self._require_positive_int("ga.population_size", ga["population_size"])
        self._require_positive_int("ga.num_generations", ga["num_generations"])
        self._require_positive_int("ga.max_reproduction_attempts", ga["max_reproduction_attempts"])
        if selection["method"] != "crowded_binary_tournament":
            raise ValueError("selection.method harus 'crowded_binary_tournament'.")
        self._require_positive_int("selection.tournament_size", selection["tournament_size"])
        if selection["tournament_size"] > ga["population_size"]:
            raise ValueError("selection.tournament_size tidak boleh melebihi ga.population_size.")

        if crossover["method"] not in IMPLEMENTED_CROSSOVER_METHODS:
            raise ValueError(
                f"crossover.method {crossover['method']!r} belum diimplementasikan; tersedia={sorted(IMPLEMENTED_CROSSOVER_METHODS)}."
            )
        self._require_probability("crossover.probability", crossover["probability"])
        if crossover["method"] == TRIGA_HCMX_POWER_MASK_METHOD:
            self._validate_triga_hcmx_power_mask(crossover)
        if mutation["method"] not in IMPLEMENTED_MUTATION_METHODS:
            raise ValueError(
                f"mutation.method {mutation['method']!r} belum diimplementasikan; tersedia={sorted(IMPLEMENTED_MUTATION_METHODS)}."
            )
        self._require_probability(
            "mutation.probability_per_individual",
            mutation["probability_per_individual"],
        )
        self._require_positive_int(
            "mutation.swaps_per_mutated_individual",
            mutation["swaps_per_mutated_individual"],
        )

        percentages = self.params["initial_population_percentages"]
        if not percentages:
            raise ValueError("initial_population_percentages tidak boleh kosong.")
        invalid = {
            name: value
            for name, value in percentages.items()
            if not math.isfinite(value) or value < 0
        }
        if invalid:
            raise ValueError("Persentase strategi harus finite dan tidak negatif: " f"{invalid}.")
        total = math.fsum(percentages.values())
        if not math.isclose(total, 100.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Jumlah initial_population_percentages harus 100, " f"bukan {total}.")

    def _validate_triga_hcmx_power_mask(self, crossover: dict[str, Any]) -> None:
        """Validate HCMX mask parameters against the active mutable fuel inventory."""
        self._require_positive_int("crossover.mask_size_min", crossover["mask_size_min"])
        self._require_positive_int("crossover.mask_size_max", crossover["mask_size_max"])
        if crossover["mask_size_min"] > crossover["mask_size_max"]:
            raise ValueError("crossover.mask_size_min tidak boleh melebihi mask_size_max.")
        mutable_fuel_count = sum(
            gene_type == "fuel" for gene_type, _ in self.params["chromosome"]["mutable_genes"]
        )
        if crossover["mask_size_max"] > mutable_fuel_count:
            raise ValueError(
                f"crossover.mask_size_max tidak boleh melebihi jumlah mutable fuel loci ({mutable_fuel_count})."
            )
        if not math.isfinite(crossover["selection_alpha"]) or crossover["selection_alpha"] <= 0.0:
            raise ValueError("crossover.selection_alpha harus finite dan positif.")
        if not math.isfinite(crossover["epsilon"]) or crossover["epsilon"] <= 0.0:
            raise ValueError("crossover.epsilon harus finite dan positif.")
        if crossover["nonfuel_policy"] != TRIGA_HCMX_NONFUEL_POLICY:
            raise ValueError(f"crossover.nonfuel_policy harus {TRIGA_HCMX_NONFUEL_POLICY!r}.")
        if crossover["require_parent_power_tally"] is not True:
            raise ValueError(
                "crossover.require_parent_power_tally harus true untuk triga_hcmx_power_mask."
            )
        if crossover["require_convergence_passed"] is not True:
            raise ValueError(
                "crossover.require_convergence_passed harus true untuk triga_hcmx_power_mask."
            )

    def _validate_openmc_parameters(self) -> None:
        """Validate transport batch counts, triggers, tally identity, and retry policy; return None."""
        openmc = self.params["openmc"]
        retry = self.params["retry"]
        self._require_positive_int("openmc.particles", openmc["particles"])
        self._require_positive_int("openmc.batches", openmc["batches"])
        self._require_positive_int("openmc.threads", openmc["threads"])
        if openmc["inactive"] < 0 or openmc["inactive"] >= openmc["batches"]:
            raise ValueError("openmc.inactive harus non-negatif dan lebih kecil dari batches.")
        if (
            not math.isfinite(openmc["keff_trigger_stdev_pcm"])
            or openmc["keff_trigger_stdev_pcm"] <= 0
        ):
            raise ValueError("openmc.keff_trigger_stdev_pcm harus positif.")
        self._require_positive_int(
            "openmc.trigger_batch_interval", openmc["trigger_batch_interval"]
        )
        self._require_positive_int("openmc.trigger_max_batches", openmc["trigger_max_batches"])
        if openmc["trigger_max_batches"] < openmc["batches"]:
            raise ValueError(
                "openmc.trigger_max_batches tidak boleh lebih kecil dari openmc.batches."
            )
        if openmc["tally_trigger_type"] not in self.SUPPORTED_TALLY_TRIGGER_TYPES:
            raise ValueError(
                f"openmc.tally_trigger_type harus salah satu {sorted(self.SUPPORTED_TALLY_TRIGGER_TYPES)}."
            )
        if (
            not math.isfinite(openmc["tally_trigger_threshold"])
            or openmc["tally_trigger_threshold"] <= 0.0
        ):
            raise ValueError("openmc.tally_trigger_threshold harus finite dan positif.")
        self._require_positive_int("retry.max_attempts", retry["max_attempts"])

    def _validate_convergence_parameters(self) -> None:
        """Validate statistical convergence-gate limits; return None or raise on invalid limits."""
        convergence = self.params["convergence"]
        threshold_names = (
            "max_keff_std_dev_pcm",
            "max_tally_mean_relative_std_dev",
            "max_tally_p95_relative_std_dev",
            "max_tally_max_relative_std_dev",
        )
        for option in threshold_names:
            value = convergence[option]
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"convergence.{option} harus finite dan non-negatif.")

    @staticmethod
    def _require_positive_int(name: str, value: int) -> None:
        """Require positive integer ``value`` and label errors with ``name``; return None."""
        if value <= 0:
            raise ValueError(f"{name} harus lebih besar dari nol.")

    @staticmethod
    def _require_probability(name: str, value: float) -> None:
        """Require finite ``value`` in [0, 1] and label errors with ``name``; return None."""
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} harus berada dalam rentang [0, 1].")

    def get_params(self) -> dict[str, dict[str, Any]]:
        """Return an independent copy of all parsed and validated configuration sections."""
        return deepcopy(self.params)
