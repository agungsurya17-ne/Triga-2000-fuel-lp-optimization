"""Compose the NSGA-II core and OpenMC campaign runtime.

MainOptimizer loads configuration, builds the 108-mutable/19-fixed/127-position
genome contract, restores or creates checkpoints, and connects operators,
schedulers, the transport pipeline, and reporting. Core operators remain in
their own modules; the optimization loop owns the campaign lifecycle."""

import logging
import random
from pathlib import Path

from modules.checkpoint import CheckpointManager, TERMINAL_EVALUATION_STATUSES
from modules.campaign_identity import validate_campaign_id
from modules.config_loader import ConfigLoader
from modules.keff_constraint import KeffConstraint
from modules.convergence_gate import ConvergenceGate
from modules.crossover import CrossoverDispatcher, CrossoverResult, build_default_crossover_registry
from modules.fuel_inventory_loader import FuelInventoryLoader
from modules.evaluation_cache import EvaluationCache
from modules.ga_engine import PopulationInitializer
from modules.genome_sorter import GenomeSorter
from modules.generation_transition_controller import (
    GenerationTransitionController,
    GenerationTransitionResult,
)
from modules.input_provenance import build_input_provenance
from modules.initial_population_scheduler import (
    InitialPopulationScheduleResult,
    InitialPopulationScheduler,
)
from modules.OpenMCRunner import OpenMCRunner
from modules.OpenMCResultEvaluator import OpenMCResultEvaluator
from modules.openmc_model_builder import OpenMCModelBuilder
from modules.optimization_reporting import OptimizationReporter, OptimizationReportingResult
from modules.nsga2 import (
    EnvironmentalSelectionResult,
    environmental_selection,
    extract_eligible_candidates,
    rank_candidates,
)
from modules.mutation import MutationDispatcher, MutationResult, build_default_mutation_registry
from modules.openmc_executable import resolve_openmc_executable
from modules.offspring_creation_scheduler import (
    OffspringCreationScheduler,
    OffspringRegistrationResult,
)
from modules.offspring_evaluation_scheduler import (
    OffspringEvaluationScheduleResult,
    OffspringEvaluationScheduler,
)
from modules.optimization_loop_controller import OptimizationLoopController, OptimizationRunResult
from modules.parent_selection import CrowdedTournamentSelector
from modules.single_candidate_pipeline import SingleCandidateOutcome, SingleCandidatePipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.ini"


class CampaignRootReuseError(RuntimeError):
    """Raised when a campaign directory has artifacts but no checkpoint."""


def _resolve_project_path(configured_path: str | Path) -> Path:
    """Resolve a configured input path against the project root.

    ``configured_path`` may be absolute or project-relative. The return value
    is the path that composition-root components should use."""
    path = Path(configured_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


class MainOptimizer:
    """Coordinate configuration, checkpoint state, and the NSGA-II/OpenMC campaign."""

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        *,
        campaign_id_override: str | None = None,
        run_root_override: str | Path | None = None,
        particles_override: int | None = None,
        inactive_batches_override: int | None = None,
        active_batches_override: int | None = None,
        keff_trigger_enabled_override: bool | None = None,
        keff_trigger_stdev_pcm_override: float | None = None,
        trigger_batch_interval_override: int | None = None,
        trigger_max_batches_override: int | None = None,
        tally_trigger_enabled_override: bool | None = None,
        tally_trigger_type_override: str | None = None,
        tally_trigger_threshold_override: float | None = None,
        tally_trigger_ignore_zeros_override: bool | None = None,
        convergence_enabled_override: bool | None = None,
        openmc_executable_override: str | Path | None = None,
        openmc_seed: int | None = None,
        threads_override: int | None = None,
        geometry_debug_override: bool | None = None,
        evaluation_cache_enabled_override: bool | None = None,
    ) -> None:
        """Compose an optimizer from ``config_path`` and optional runtime overrides.

        Campaign ID and run-root overrides isolate output artifacts. Particle, batch,
        trigger, convergence, thread, geometry-debug, cache, and executable overrides
        replace the corresponding configured settings. ``openmc_seed`` is the base
        transport seed; later attempts use sequential seeds. Executable validation
        precedes checkpoint creation or modification. Return None after storing the
        validated configuration and restored or newly initialized campaign state."""
        self.config_path = Path(config_path)

        try:
            self.config_loader = ConfigLoader(self.config_path)
            self.config = self.config_loader.get_params()
        except (FileNotFoundError, ValueError) as error:
            logging.error("Gagal memuat konfigurasi: %s", error)
            raise SystemExit("Program dihentikan karena konfigurasi tidak valid.") from error

        campaign_config = self.config["campaign"]
        paths_config = self.config["paths"]
        ga_config = self.config["ga"]
        selection_config = self.config["selection"]
        crossover_config = self.config["crossover"]
        chromosome_config = self.config["chromosome"]
        openmc_config = self.config["openmc"]
        retry_config = self.config["retry"]
        convergence_config = self.config["convergence"]
        strategy_percentages = self.config["initial_population_percentages"]
        evaluation_cache_config = self.config["evaluation_cache"]

        self.openmc_executable = resolve_openmc_executable(
            openmc_executable_override=openmc_executable_override,
            configured_openmc_executable=paths_config["openmc_executable"],
            project_root=PROJECT_ROOT,
        )

        self.result_evaluator = OpenMCResultEvaluator(
            tally_name=openmc_config["fission_tally_name"],
            tally_score=openmc_config["fission_tally_score"],
        )

        self.fuel_inventory_path = _resolve_project_path(paths_config["fuel_inventory_file"])
        self.materials_path = _resolve_project_path(paths_config["materials_file"])

        self.genome_sorter = GenomeSorter(
            slot_order=OpenMCModelBuilder.SLOT_ORDER,
            excluded_genes=chromosome_config["excluded_genes"],
        )

        # U-235 mass is an initialization heuristic, not a keff calculation.
        fuel_ids = tuple(
            gene_id
            for gene_type, gene_id in chromosome_config["mutable_genes"]
            if gene_type == "fuel"
        )
        reactivity_proxy = FuelInventoryLoader.load_reactivity_proxy(
            self.fuel_inventory_path, fuel_ids
        )

        self.population_initializer = PopulationInitializer(
            population_size=ga_config["population_size"],
            mutable_genes=chromosome_config["mutable_genes"],
            genome_sorter=self.genome_sorter,
            reactivity_proxy=reactivity_proxy,
            strategy_percentages=strategy_percentages,
            random_seed=ga_config["random_seed"],
        )

        # Share one RNG across selection and reproduction so checkpoints can restore draw order.
        self.selection_rng = random.Random(ga_config["random_seed"])
        self.parent_selector = CrowdedTournamentSelector(
            tournament_size=selection_config["tournament_size"],
            rng=self.selection_rng,
            avoid_self_pairing=True,
        )
        self.crossover_registry = build_default_crossover_registry()
        crossover_operator_options = {
            name: value
            for name, value in crossover_config.items()
            if name not in {"method", "probability"}
        }
        # HCMX preflight needs tally identity; PMX does not use parent power tallies.
        crossover_operator_options.update(
            {
                "expected_tally_name": openmc_config["fission_tally_name"],
                "expected_tally_score": openmc_config["fission_tally_score"],
            }
        )
        self.crossover_dispatcher = CrossoverDispatcher(
            method=crossover_config["method"],
            probability=crossover_config["probability"],
            registry=self.crossover_registry,
            mutable_genome_contract=self.population_initializer.mutable_genome_contract,
            genome_sorter=self.genome_sorter,
            rng=self.selection_rng,
            operator_options=crossover_operator_options,
        )
        mutation_config = self.config["mutation"]
        self.mutation_registry = build_default_mutation_registry()
        self.mutation_dispatcher = MutationDispatcher(
            method=mutation_config["method"],
            probability_per_individual=mutation_config["probability_per_individual"],
            swaps_per_mutated_individual=mutation_config["swaps_per_mutated_individual"],
            registry=self.mutation_registry,
            mutable_genome_contract=self.population_initializer.mutable_genome_contract,
            genome_sorter=self.genome_sorter,
            rng=self.selection_rng,
        )

        selected_campaign_id = (
            campaign_config["campaign_id"] if campaign_id_override is None else campaign_id_override
        )
        self.campaign_id = validate_campaign_id(selected_campaign_id)

        configured_run_root = Path(
            campaign_config["run_root"] if run_root_override is None else run_root_override
        )
        self.run_root = (
            configured_run_root
            if configured_run_root.is_absolute()
            else PROJECT_ROOT / configured_run_root
        )
        self.campaign_root = self.run_root / self.campaign_id
        self.openmc_work_root = self.campaign_root / "openmc_runs"
        self.report_root = self.campaign_root / "reports"
        self.optimization_reporter = OptimizationReporter(self.report_root)
        checkpoint_path = self.campaign_root / "checkpoint.json"
        self._require_reusable_campaign_root(self.campaign_root, checkpoint_path)

        configured_active_batches = openmc_config["batches"] - openmc_config["inactive"]
        self.particles = (
            openmc_config["particles"] if particles_override is None else particles_override
        )
        self.inactive_batches = (
            openmc_config["inactive"]
            if inactive_batches_override is None
            else inactive_batches_override
        )
        self.active_batches = (
            configured_active_batches
            if active_batches_override is None
            else active_batches_override
        )
        self.keff_trigger_enabled = (
            openmc_config["keff_trigger_enabled"]
            if keff_trigger_enabled_override is None
            else keff_trigger_enabled_override
        )
        self.keff_trigger_stdev_pcm = (
            openmc_config["keff_trigger_stdev_pcm"]
            if keff_trigger_stdev_pcm_override is None
            else keff_trigger_stdev_pcm_override
        )
        self.trigger_batch_interval = (
            openmc_config["trigger_batch_interval"]
            if trigger_batch_interval_override is None
            else trigger_batch_interval_override
        )
        self.trigger_max_batches = (
            openmc_config["trigger_max_batches"]
            if trigger_max_batches_override is None
            else trigger_max_batches_override
        )
        self.tally_trigger_enabled = (
            openmc_config["tally_trigger_enabled"]
            if tally_trigger_enabled_override is None
            else tally_trigger_enabled_override
        )
        self.tally_trigger_type = (
            openmc_config["tally_trigger_type"]
            if tally_trigger_type_override is None
            else tally_trigger_type_override
        )
        self.tally_trigger_threshold = (
            openmc_config["tally_trigger_threshold"]
            if tally_trigger_threshold_override is None
            else tally_trigger_threshold_override
        )
        self.tally_trigger_ignore_zeros = (
            openmc_config["tally_trigger_ignore_zeros"]
            if tally_trigger_ignore_zeros_override is None
            else tally_trigger_ignore_zeros_override
        )
        self.convergence_enabled = (
            convergence_config["enabled"]
            if convergence_enabled_override is None
            else convergence_enabled_override
        )
        self.openmc_seed = ga_config["random_seed"] if openmc_seed is None else openmc_seed
        self.threads = openmc_config["threads"] if threads_override is None else threads_override
        self.geometry_debug = (
            openmc_config["geometry_debug"]
            if geometry_debug_override is None
            else geometry_debug_override
        )
        if isinstance(self.threads, bool) or not isinstance(self.threads, int) or self.threads <= 0:
            raise ValueError("threads_override harus berupa integer positif.")
        if not isinstance(self.geometry_debug, bool):
            raise TypeError("geometry_debug_override harus berupa bool.")

        self.convergence_gate = ConvergenceGate(
            enabled=self.convergence_enabled,
            max_keff_std_dev_pcm=convergence_config["max_keff_std_dev_pcm"],
            max_tally_mean_relative_std_dev=convergence_config["max_tally_mean_relative_std_dev"],
            max_tally_p95_relative_std_dev=convergence_config["max_tally_p95_relative_std_dev"],
            max_tally_max_relative_std_dev=convergence_config["max_tally_max_relative_std_dev"],
            allow_zero_mean_bins=convergence_config["allow_zero_mean_bins"],
        )

        self.genome_contract = CheckpointManager.build_genome_contract(
            encoding=chromosome_config["encoding"],
            slot_order=OpenMCModelBuilder.SLOT_ORDER,
            excluded_genes=chromosome_config["excluded_genes"],
            mutable_alleles=chromosome_config["mutable_genes"],
        )
        self.input_provenance = build_input_provenance(
            self.config_path,
            self.fuel_inventory_path,
            self.materials_path,
            self.genome_contract,
            PROJECT_ROOT,
        )
        self.checkpoint_manager = CheckpointManager(
            checkpoint_path=checkpoint_path,
            campaign_id=self.campaign_id,
            input_provenance=self.input_provenance,
            constraint_contract=KeffConstraint(
                ga_config["keff_min"], ga_config["feasibility_tolerance"]
            ),
        )

        self.openmc_runner = OpenMCRunner(
            work_root=self.openmc_work_root,
            threads=self.threads,
            geometry_debug=self.geometry_debug,
            openmc_exec=self.openmc_executable,
        )
        self.single_candidate_pipeline = SingleCandidatePipeline(
            genome_sorter=self.genome_sorter,
            builder_factory=OpenMCModelBuilder,
            runner=self.openmc_runner,
            result_evaluator=self.result_evaluator,
            checkpoint_manager=self.checkpoint_manager,
            material_file=self.materials_path,
            work_root=self.openmc_work_root,
            tally_name=openmc_config["fission_tally_name"],
            tally_score=openmc_config["fission_tally_score"],
            particles=self.particles,
            inactive_batches=self.inactive_batches,
            active_batches=self.active_batches,
            keff_trigger_enabled=self.keff_trigger_enabled,
            keff_trigger_stdev_pcm=self.keff_trigger_stdev_pcm,
            trigger_batch_interval=self.trigger_batch_interval,
            trigger_max_batches=self.trigger_max_batches,
            tally_trigger_enabled=self.tally_trigger_enabled,
            tally_trigger_type=self.tally_trigger_type,
            tally_trigger_threshold=self.tally_trigger_threshold,
            tally_trigger_ignore_zeros=self.tally_trigger_ignore_zeros,
            convergence_gate=self.convergence_gate,
            openmc_seed=self.openmc_seed,
            max_attempts=retry_config["max_attempts"],
            interrupted_counts_as_failed_attempt=retry_config[
                "interrupted_counts_as_failed_attempt"
            ],
            on_max_attempts=retry_config["on_max_attempts"],
        )

        if self.checkpoint_manager.exists():
            self.state = self.checkpoint_manager.load(
                expected_genome_contract=self.genome_contract, current_ga_settings=ga_config
            )
            self.checkpoint_manager.restore_rng_state(self.state, python_rng=self.selection_rng)
            recovered_ids = self.checkpoint_manager.recover_interrupted_individuals(self.state)
            if recovered_ids:
                self.checkpoint_manager.save(self.state, python_rng=self.selection_rng)
            logging.info("Melanjutkan campaign dari %s.", checkpoint_path)
        else:
            self.state = self.checkpoint_manager.create_state(
                genome_contract=self.genome_contract,
                ga_settings=ga_config,
                metadata={"algorithm": ga_config["algorithm"]},
                python_rng=self.selection_rng,
            )
            self.checkpoint_manager.save(self.state, python_rng=self.selection_rng)
            logging.info("Campaign baru dibuat pada %s.", checkpoint_path)

        self.optimization_reporter.generate(self.state)

        cache_enabled = (
            evaluation_cache_config["enabled"]
            if evaluation_cache_enabled_override is None
            else evaluation_cache_enabled_override
        )
        if not isinstance(cache_enabled, bool):
            raise TypeError("evaluation_cache_enabled_override harus bool atau None.")
        self.evaluation_cache = EvaluationCache(
            enabled=cache_enabled,
            source_checkpoints=evaluation_cache_config["source_checkpoints"],
            allow_legacy_context=evaluation_cache_config["allow_legacy_context"],
            compatible_openmc_version_pairs=evaluation_cache_config[
                "compatible_openmc_version_pairs"
            ],
            project_root=PROJECT_ROOT,
            campaign_root=self.campaign_root,
            target_checkpoint=checkpoint_path,
            evaluation_context=self.single_candidate_pipeline.evaluation_context(),
            genome_contract=self.genome_contract,
            state=self.state,
            checkpoint_manager=self.checkpoint_manager,
        )

        self.initial_population_scheduler = InitialPopulationScheduler(
            population_initializer=self.population_initializer,
            candidate_pipeline=self.single_candidate_pipeline,
            checkpoint_manager=self.checkpoint_manager,
            state=self.state,
            campaign_id=self.campaign_id,
            evaluation_cache=self.evaluation_cache,
        )
        self.offspring_creation_scheduler = OffspringCreationScheduler(
            checkpoint_manager=self.checkpoint_manager,
            state=self.state,
            # The target is N offspring per generation, rather than per crossover pair.
            population_size=ga_config["population_size"],
            # Defer RNG persistence until the complete offspring registry is durable.
            select_parent_ids=lambda parent_count: self.select_parent_ids(
                parent_count, persist_rng=False
            ),
            crossover_dispatcher=self.crossover_dispatcher,
            mutation_dispatcher=self.mutation_dispatcher,
            rng=self.selection_rng,
            max_reproduction_attempts=ga_config["max_reproduction_attempts"],
            resample_duplicate_offspring=ga_config["resample_duplicate_offspring"],
        )
        self.offspring_evaluation_scheduler = OffspringEvaluationScheduler(
            candidate_pipeline=self.single_candidate_pipeline,
            checkpoint_manager=self.checkpoint_manager,
            state=self.state,
            campaign_id=self.campaign_id,
            resample_duplicate_offspring=ga_config["resample_duplicate_offspring"],
            evaluation_cache=self.evaluation_cache,
        )
        self.generation_transition_controller = GenerationTransitionController(
            checkpoint_manager=self.checkpoint_manager,
            state=self.state,
            population_size=ga_config["population_size"],
            num_generations=ga_config["num_generations"],
            offspring_registration_scheduler=self.offspring_creation_scheduler,
            offspring_evaluation_scheduler=self.offspring_evaluation_scheduler,
            complete_generation=self.complete_active_generation,
            rng=self.selection_rng,
            boundary_callback=self._handle_generation_boundary,
        )
        self.optimization_loop_controller = OptimizationLoopController(
            checkpoint_manager=self.checkpoint_manager,
            state=self.state,
            num_generations=ga_config["num_generations"],
            initial_population_scheduler=self.initial_population_scheduler,
            generation_transition_controller=self.generation_transition_controller,
            rng=self.selection_rng,
        )

    @staticmethod
    def _require_reusable_campaign_root(campaign_root: Path, checkpoint_path: Path) -> None:
        """Reject artifact-bearing campaign roots that lack their checkpoint.

        ``campaign_root`` is the isolated directory for one campaign and
        ``checkpoint_path`` is its only resume authority. An empty or new root
        is allowed; an existing checkpoint is validated later by
        :class:`CheckpointManager` before any scheduler can resume it."""
        if not campaign_root.exists():
            return
        if not campaign_root.is_dir():
            raise CampaignRootReuseError(f"Campaign root harus berupa directory: {campaign_root}.")
        if checkpoint_path.exists():
            return
        if any(campaign_root.iterdir()):
            raise CampaignRootReuseError(
                f"Campaign root {campaign_root} sudah berisi artefak tetapi checkpoint.json tidak ada. Gunakan --campaign-id atau --run-root baru; artefak lama tidak akan dicampur dengan campaign baru."
            )

    def run_initial_population(
        self, registration_only: bool = False
    ) -> InitialPopulationScheduleResult:
        """Register and optionally evaluate only the generation-zero parent population.

        ``registration_only`` stops after all parents and generation membership
        are durable. The method returns checkpoint phase, evaluated IDs, and
        status counts without creating offspring or invoking NSGA-II operators."""
        return self.initial_population_scheduler.run(registration_only=registration_only)

    def select_parent_ids(self, parent_count: int = 2, persist_rng: bool = True) -> tuple[str, ...]:
        """Rank active completed parents and durably select parent IDs.

        ``parent_count`` is the number of IDs requested; consecutive IDs form
        self-pair-free pairs when at least two eligible parents exist. The method
        returns IDs only. ``persist_rng`` keeps the standalone P1 API durable;
        the offspring registrar sets it false and persists the shared RNG only
        after its complete offspring registry has been formed."""
        if not isinstance(persist_rng, bool):
            raise TypeError("persist_rng harus berupa bool.")

        generation = self.state["progress"].get("generation")
        if generation is None or self.state["progress"].get("phase") not in {
            "parents_evaluated",
            "creating_offspring",
        }:
            raise RuntimeError(
                "Parent selection memerlukan phase parents_evaluated atau creating_offspring pada generasi aktif."
            )

        eligible_ids = self.checkpoint_manager.get_nsga2_eligible_parent_ids(self.state, generation)
        eligible_records = {
            individual_id: self.state["individuals"][individual_id]
            for individual_id in eligible_ids
        }

        # Recompute constrained rank and crowding on the active parent pool.
        ranking = rank_candidates(
            extract_eligible_candidates(eligible_records, "active_parent_individuals")
        )
        for candidate in ranking.ranked_candidates:
            self.checkpoint_manager.set_nsga2_metrics(
                self.state,
                candidate.individual_id,
                candidate.rank,
                candidate.crowding_distance,
                candidate.is_boundary,
            )

        selected_ids = self.parent_selector.select_parent_ids(eligible_records, parent_count)

        # Standalone selection persists its RNG; batch reproduction commits it with the registry.
        if persist_rng:
            self.checkpoint_manager.save(self.state, python_rng=self.selection_rng)
        return selected_ids

    def register_offspring_population(self) -> OffspringRegistrationResult:
        """Create or resume one complete offspring registry without evaluation.

        The active generation must be ready for offspring creation. The return
        value contains ordered deterministic IDs and whether an existing durable
        registry was resumed. This method never invokes OpenMC, an evaluator,
        environmental selection, or a later generation."""
        return self.offspring_creation_scheduler.run()

    def evaluate_offspring_population(self) -> OffspringEvaluationScheduleResult:
        """Evaluate only registered offspring and stop before replacement.

        The method resumes durable child genomes from the active generation and
        returns their lifecycle summary. It does not rerun parents, select
        survivors, or start another generation."""
        return self.offspring_evaluation_scheduler.run()

    def run_generation_transition(self, target_generation: int) -> GenerationTransitionResult:
        """Run or idempotently resume exactly one configured generation target.

        ``target_generation`` is an explicit idempotency key bounded by
        ``ga.num_generations``. The returned object summarizes durable parents,
        offspring, survivors, evaluation activity, and phase boundaries."""
        return self.generation_transition_controller.run(target_generation)

    def run_optimization(self) -> OptimizationRunResult:
        """Run or resume NSGA-II through exactly ``ga.num_generations`` transitions.

        Generation 0 starts with initial-population evaluation, then performs
        the first reproduction and replacement. The method returns a durable
        completion summary and treats an already finished checkpoint as a no-op."""
        return self.optimization_loop_controller.run()

    def refresh_reports(self) -> OptimizationReportingResult:
        """Regenerate reports from durable completed generations and return their file paths."""
        return self.optimization_reporter.generate(self.state)

    def _handle_generation_boundary(self, phase: str) -> None:
        """Generate reports after durable ``generation_completed`` and return no value."""
        if phase != "generation_completed":
            return
        report = self.refresh_reports()
        generation = self.state["progress"]["generation"]
        logging.info(
            "Report generation %s diperbarui pada %s (%s generation tersedia).",
            generation,
            self.report_root,
            len(report.completed_generations),
        )

    def complete_active_generation(self) -> EnvironmentalSelectionResult:
        """Select and durably finalize survivors for one evaluated active generation.

        The active checkpoint must be at ``offspring_evaluated`` or its durable
        ``environmental_selection`` boundary. The method ranks only completed
        parent and offspring records by raw minimization objectives
        ``(-keff, ppf_3d)``, persists their NSGA-II metrics, and closes the
        same generation. It never begins or evaluates another generation."""
        generation = self.state["progress"].get("generation")
        phase = self.state["progress"].get("phase")
        if (
            not isinstance(generation, int)
            or isinstance(generation, bool)
            or phase not in {"offspring_evaluated", "environmental_selection"}
        ):
            raise RuntimeError(
                "Environmental replacement memerlukan phase offspring_evaluated atau environmental_selection pada generasi aktif."
            )

        generation_record = self.state["generations"][-1] if self.state.get("generations") else None
        if (
            not isinstance(generation_record, dict)
            or generation_record.get("generation") != generation
            or generation_record.get("status") != "running"
        ):
            raise RuntimeError(
                "Environmental replacement memerlukan generation aktif yang masih running."
            )
        if self.state["progress"].get("active_individual_id") is not None:
            raise RuntimeError(
                "Environmental replacement tidak dapat berjalan ketika individual masih running."
            )
        if phase == "offspring_evaluated":
            self.checkpoint_manager.begin_environmental_selection(self.state, generation)
            self.checkpoint_manager.save(self.state, python_rng=self.selection_rng)

        # Restrict replacement to the active parent-offspring union.
        parent_ids = tuple(generation_record["parent_ids"])
        offspring_ids = tuple(generation_record["offspring_ids"])
        candidate_ids = parent_ids + offspring_ids
        missing_candidate_ids = [
            individual_id
            for individual_id in candidate_ids
            if individual_id not in self.state["individuals"]
        ]
        if missing_candidate_ids:
            raise RuntimeError(
                "Environmental replacement menemukan kandidat yang tidak terdaftar: "
                + ", ".join(missing_candidate_ids)
                + "."
            )
        # Failed terminal individuals are excluded later by the completed-result filter.
        nonterminal_ids = [
            individual_id
            for individual_id in candidate_ids
            if self.state["individuals"][individual_id]["status"]
            not in TERMINAL_EVALUATION_STATUSES
        ]
        if nonterminal_ids:
            raise RuntimeError(
                "Environmental replacement memerlukan seluruh parent dan offspring terminal: "
                + ", ".join(nonterminal_ids)
                + "."
            )
        parent_records = {
            individual_id: self.state["individuals"][individual_id] for individual_id in parent_ids
        }
        offspring_records = {
            individual_id: self.state["individuals"][individual_id]
            for individual_id in offspring_ids
        }
        target_population_size = self.config["ga"]["population_size"]
        selection = environmental_selection(
            parent_records, offspring_records, target_population_size
        )

        for candidate in selection.ranking.ranked_candidates:
            # Combined-pool rank and crowding may differ from the parent-only metrics.
            self.checkpoint_manager.set_nsga2_metrics(
                self.state,
                candidate.individual_id,
                candidate.rank,
                candidate.crowding_distance,
                candidate.is_boundary,
            )

        summary = {
            "replacement": {
                # Record the unweighted minimization objective order: (-keff, ppf_3d).
                "objectives": ["-keff", "ppf_3d"],
                "constraint_contract": self.state["constraint_contract"],
                "eligible_parent_ids": list(selection.eligible_parent_ids),
                "eligible_offspring_ids": list(selection.eligible_offspring_ids),
                "fronts": [list(front) for front in selection.ranking.fronts],
                "survivor_ids": list(selection.survivor_ids),
            }
        }
        self.checkpoint_manager.complete_generation(
            self.state, generation, selection.survivor_ids, summary
        )
        self.checkpoint_manager.save(self.state, python_rng=self.selection_rng)
        return selection

    def crossover_parent_ids(self, parent_ids: tuple[str, str]) -> CrossoverResult:
        """Resolve checkpoint parent IDs and run the configured crossover method.

        ``parent_ids`` must identify two completed checkpoint records. The method
        returns child genomes plus JSON-safe metadata without registering offspring,
        saving RNG state, constructing a generation, or invoking OpenMC."""
        return self.crossover_dispatcher.crossover(parent_ids, self.state["individuals"])

    def mutate_child_genome(self, child_genome: tuple[tuple[str, str], ...]) -> MutationResult:
        """Apply configured mutation to one mutable child without persistence or transport.

        ``child_genome`` is a child produced by an earlier operator in mutable
        position order. The return value contains its immutable mutated genome and
        JSON-safe metadata; it does not register an offspring or run OpenMC."""
        return self.mutation_dispatcher.mutate(child_genome)

    def run_single_candidate(
        self,
        candidate_index: int = 0,
        individual_id: str = "G000-I000",
        run_id_prefix: str | None = None,
    ) -> SingleCandidateOutcome:
        """Evaluate one reproducibly initialized candidate and return its successful outcome.

        ``candidate_index`` selects the initializer candidate, ``individual_id`` is its
        checkpoint key, and ``run_id_prefix`` names the associated OpenMC attempts."""
        if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
            raise TypeError("candidate_index harus berupa integer dan bukan bool.")
        candidates = self.population_initializer.initialize_population()
        if candidate_index < 0 or candidate_index >= len(candidates):
            raise IndexError("candidate_index berada di luar initial population.")
        candidate = candidates[candidate_index]

        if individual_id not in self.state["individuals"]:
            self.checkpoint_manager.register_individual(
                self.state,
                individual_id=individual_id,
                genome=candidate.genome,
                generation_created=0,
                origin="initial",
                operator={"initialization_strategy": candidate.strategy},
            )
            if not self.state["generations"]:
                self.checkpoint_manager.begin_generation(
                    self.state, generation=0, parent_ids=(individual_id,)
                )
            self.checkpoint_manager.save(self.state)

        selected_run_id_prefix = (
            f"{self.campaign_id}-{individual_id}" if run_id_prefix is None else run_id_prefix
        )
        return self.single_candidate_pipeline.evaluate(
            candidate, self.state, individual_id, selected_run_id_prefix
        )
