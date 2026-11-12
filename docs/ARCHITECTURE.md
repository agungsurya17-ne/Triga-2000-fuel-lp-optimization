# Runtime architecture

The launcher delegates to `MainOptimizer` in `src/main.py`. The composition root
connects pure genetic operators to durable campaign services.

| Layer | Main components | Contract |
|---|---|---|
| Genome | `mutable_genome`, `genome_sorter`, `fuel_inventory_loader` | Exact 108-allele inventory; insert 19 fixed genes only at the physical-model boundary |
| Initialization | `ga_engine.PopulationInitializer` | Seven strategies; U-235 grams are a heuristic proxy |
| Operators | `parent_selection`, `crossover`, `mutation`, `nsga2`, `keff_constraint` | Constrained tournament, PMX, swap mutation, elitist survivor selection |
| Physics | `openmc_model_builder`, `OpenMCRunner`, `OpenMCResultEvaluator`, `convergence_gate` | Geometry v2, isolated attempts, ordered tally mapping, statistical acceptance |
| Persistence | `checkpoint`, `input_provenance`, `evaluation_cache` | Atomic checkpoint state, scientific fingerprints, compatible-result reuse |
| Lifecycle | Initial/offspring schedulers, generation and optimization controllers | Persist complete registries and RNG state; resume from durable phases |
| Reporting | `optimization_reporting` | CSV and PNG from completed checkpoint generations |

The production configuration uses PMX. The optional power-mask crossover is a
separate heuristic and is not silently enabled. Feasibility compares mean BOC
keff with the configured lower bound and tolerance before Pareto objectives.
Raw objectives remain `(-keff, ppf_3d)` without weighting or objective penalties.

The regular-fuel optimization domain contains 1,620 axial segments. The separate
final-evaluation model can include 1,665 segments with FFCR fuel followers.
Geometry dimensions are in cm, U-235 proxy masses in g, and keff/PPF are
dimensionless. See `MODEL_BUILDER.md` for the domain distinction.

Reevaluation source is isolated in `reevaluation_iv_4_5_2/`. SDM extends the
nominal builder in `src/final_evaluation/`; neither workflow replaces NSGA-II
selection or retroactively changes stored objective values. Their public
cohort/input contracts are described in `WORKFLOWS.md`.

`public_inputs.py` validates public reference loadings and completed cohorts.
Material fingerprints prevent selection of real-inventory or incompatible runs.
