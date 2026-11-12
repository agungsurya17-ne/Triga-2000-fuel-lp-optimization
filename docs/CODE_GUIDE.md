# Reading the code

Start with the optimization launcher. The reevaluation and SDM workflows use
the same fuel inventory and nominal geometry, but have separate run protocols.

## Follow one optimization run

1. `run_optimization.py` reads command-line options. With `--dry-run`, it checks
   inputs and component setup without evaluating a fuel loading.
2. `MainOptimizer` in `src/main.py` loads the INI file and connects the operators,
   model builder, evaluation pipeline, and checkpoint manager.
3. `PopulationInitializer` in `ga_engine.py` creates the initial permutations of
   108 mutable fuel genes. U-235 mass, in grams, guides some starting strategies;
   it is not used as a substitute for an OpenMC eigenvalue calculation.
4. The initial-population scheduler evaluates those candidates. Later generations
   select parent IDs, apply crossover and mutation, and evaluate the offspring.
5. `nsga2.py` ranks the combined parent/offspring population and retains the next
   population. The optimization loop repeats until the generation target is met.
6. Checkpoints save the population, results, generation phase, and random-number
   state. Reports are produced from completed generations.

For the genetic operators, read `mutable_genome.py`, `genome_sorter.py`,
`crossover.py`, `mutation.py`, and then `nsga2.py`. For transport, start with
`single_candidate_pipeline.py`, followed by the model builder, runner,
result evaluator, and convergence gate.

## From a fuel permutation to an objective value

A gene is a pair such as `("fuel", "fuel_900001")`. Its position in the mutable
genome determines which mutable core slot receives that fuel element. PMX and
swap mutation change the ordering while preserving the exact allele inventory.

`GenomeSorter.construct_complete_genome()` inserts the 19 fixed genes and
returns the complete 127-position loading. The OpenMC builder creates a separate
slot universe for each element and records the fuel-segment cell IDs. The
evaluator checks tally-bin order against those records before calculating PPF.

The objectives are `(-keff, ppf_3d)`, both minimized. The negative sign therefore
favors higher keff. Feasibility under the configured mean-keff lower bound is
compared before objective dominance. The regular-fuel PPF uses 1,620 segments;
reevaluation and nominal final evaluation also report a 1,665-segment domain
including the three FFCR fuel followers. Keep those domains separate.

## Python conventions used here

| Syntax | Meaning in this project |
|---|---|
| `name: float`, `tuple[Gene, ...]` | Type hints describe expected values. The code still validates scientific inputs explicitly. |
| `Path \| None` | A path or no override. `None` usually means use the configured default. |
| `def f(a, *, seed=1)` | Arguments after `*` must be named, for example `seed=123`. |
| `self` | The current object, which holds settings or campaign state between method calls. |
| `@dataclass(frozen=True)` | A record whose fields cannot be reassigned after construction. Useful for operator and ranking results. |
| `@property` | A method read as an attribute, such as `sorter.included_length`. |
| `@staticmethod` | A helper that does not need object state. |
| `@classmethod` | A method that receives the class, often to build an object from a stored contract. |
| `[value for ...]`, `{key: value for ...}` | Comprehensions construct a list or mapping from existing data. |
| `left:right` | A half-open slice: include `left`, exclude `right`. PMX uses this convention for cut segments. |
| `with ... as ...` | A context manager closes files or releases a lock even if an exception occurs. |
| `raise Error(...) from error` | Report a more specific problem while preserving its underlying cause. |

`rng` is a `random.Random` object shared by selection, crossover, and mutation.
Saving its state allows an interrupted run to continue its draw sequence.
Operator probabilities do not guarantee a changed genome: crossover can return
a parent ordering, and repeated swaps can undo each other.

A callback is a function passed to another component. For example, the
offspring scheduler calls the configured parent-selection callback rather than
implementing tournament selection itself. This lets the same scheduler manage
registration and resume without owning the genetic operator.

## OpenMC region syntax

Region expressions use overloaded Python operators:

```python
fuel_region = +lower_plane & -upper_plane & -outer_cylinder
```

`+lower_plane` selects the positive half-space above the lower axial boundary;
`-upper_plane` selects the space below the upper boundary. For a cylinder,
the negative half-space is its interior. `&` intersects regions, `|` joins them,
and `~` takes their complement. These are geometry operations rather than
ordinary arithmetic. Coordinates and dimensions are in cm.

Comments such as `MCNP surface 8` identify the corresponding entry in the
reference geometry deck. They do not prescribe the object IDs assigned by
OpenMC during model construction.

## Saved state and validation

Candidate IDs identify individuals across generations; they are not fuel IDs.
The checkpoint phase indicates which operation was last saved. A pending
offspring is evaluated from its stored genome instead of drawing a new one.
Completed results are reused only when the required context matches.

Checkpoint writes use a temporary file and replacement so a reader sees a
complete saved file. Scientific fingerprints cover inputs and evaluation
context. Keep the original inputs with each campaign and use a new campaign ID
when changing the model, materials, tally, or transport settings.

The tests use small permutations, known arithmetic examples, geometry probes,
and mocked transport. Read a test beside the function it exercises. A skipped
optional campaign test means its transport/checkpoint artifacts are unavailable;
it does not establish a physics result.

For the public reevaluation and SDM input requirements, see
[WORKFLOWS.md](WORKFLOWS.md).
