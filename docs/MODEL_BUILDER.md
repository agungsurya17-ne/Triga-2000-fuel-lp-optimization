# TRIGA 2000 model builder

The active runtime has one nominal geometry implementation:
[`OpenMCModelBuilder`](../src/modules/openmc_model_builder.py). Its evaluation
identity is `triga2000_openmc_model/v2`. Runtime code does not offer a geometry
revision selector. Keep this identity in evaluation metadata and checkpoints.

## Inputs and workflow

1. `GenomeSorter.construct_complete_genome()` inserts the 19 fixed genes into
   the 108 mutable genes, producing the physical 127-position loading ordered
   from A1 through G36. Genetic operators act only on the mutable genes.
2. The builder reads the configured material definitions from `materials.xml`.
   Material identities, composition, temperature, and the cross-section index
   remain input data rather than documentation settings.
3. Slot-specific builders construct fuel pins, fuel-followed control rods
   (FFCRs), control rods, irradiation tubes, support structures, and empty holes.
   The seven-ring hexagonal lattice is assembled inside the fixed reactor
   geometry. Coordinates, radii, and lengths are in **cm**.
4. `build_settings()` applies explicit particle, batch, seed, and trigger
   settings. Inactive and active counts refer to batches. The caller controls
   thread count and geometry-debug execution.
5. `build_model()` returns the OpenMC model and metadata linking tally bins to
   physical fuel segments. The evaluator uses this ordering to compute keff,
   relative-power diagnostics, and the minimization objectives `(-keff, ppf_3d)`.

## Tally domains

The optimization builder scores `fission-q-recoverable` in **1,620 regular-fuel
segments**: 108 pins with 15 axial segments each. Its 3-D PPF is the maximum
segment score divided by the mean segment score; equal regular-segment volumes
make this equivalent to the corresponding power-density ratio. This PPF is
dimensionless and does not convert the tally to reactor power in watts.

[`FinalModelBuilder`](../src/final_evaluation/model.py) extends the same nominal
geometry with independent rod states and a tally covering **1,665 segments**,
including 45 FFCR follower segments. Final evaluation uses fuel-volume-weighted
power-density metrics. The SDM extension has its own revision identifier; this
does not select a different nominal geometry. Its identity includes the base
geometry identity.

Do not interchange PPF values from different fuel domains. Archived results
retain the model identity and input provenance under which they were produced.
Editing documentation does not recalculate or validate those results.

## Verification

Run the existing regression suite in the project OpenMC Python environment:

```bash
python -B -m unittest discover -s tests -q
```

[`test_openmc_model_builder_geometry.py`](../tests/test_openmc_model_builder_geometry.py)
checks material assignments at audited points, nearby perturbations, grid
plates and rod extensions, all 1,665 fuel-segment material assignments and
dimensions, the regular-fuel tally domain, and inherited model identities.
It reads the compact loading JSON and probe/segment-mapping CSV fixtures; those files are
test data, not alternative builder implementations.

These tests do not run neutron transport. They verify software and geometry
regressions, not statistical convergence or agreement with measured reactor data.

Source and configuration hashes may change when documentation is edited even
if executable logic and physical settings remain identical. Preserve original
campaign archives and review provenance before attempting to resume them; do
not rewrite stored hashes or model identities to bypass compatibility checks.
