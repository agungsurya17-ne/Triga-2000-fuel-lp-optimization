# TRIGA 2000 fuel-loading optimization

**Author:** Anak Agung Gde Surya Ananda Putra  
**Version:** 1.1.0 — 13 November 2026  
**Nominal geometry:** `triga2000_openmc_model/v2`

Source code accompanying my undergraduate thesis on fuel-loading optimization
with NSGA-II and OpenMC. It includes optimization, nominal reevaluation, and
shutdown-margin (SDM) calculations. Flux-analysis workflows are outside this package.

**This public version uses a fictional fresh-fuel inventory.** Real fuel serial
numbers, irradiated compositions, and the actual reference loading are excluded.
It demonstrates the calculation workflow and cannot reproduce the numerical
results of the confidential thesis inventory.

## Public inputs

There are 108 movable pins: 36 each of M8, M9, and M10. Three fixed FFCR
followers use one type each. IDs `fuel_900001` through `fuel_900111` are invented;
they are not aliases for real serial numbers. Each pin has 15 identical fresh
axial compositions. The reference loading is shuffled with a public random seed.

| Fuel | Density (g/cm³) | Uranium enrichment by mass |
|---|---:|---:|
| M8: U-ZrH 8.5–20 | 5.79875 | 20% |
| M9: U-ZrH 12–20 | 5.94499 | 20% |
| M10: U-ZrH-Er 20–20 | 6.40732 | 20% |

Fuel recipes and densities were supplied by the author. Structural materials
use documented generic assumptions. Read [PUBLIC_INPUTS.md](docs/PUBLIC_INPUTS.md)
before interpreting any calculated response.

## Install and check

The environment targets Linux and OpenMC 0.15.3. From the repository root:

```bash
python tools/verify_release.py
conda env create --file environment.yml
conda activate ta-nsga2-openmc
python -B -m unittest discover -s tests -v
(cd reevaluation_iv_4_5_2 && python -B -m unittest test_pipeline -v)
```

Tests check software, arithmetic, and geometry without neutron transport.
Tests that need a completed campaign or transport artifacts explicitly skip
when those files do not exist. See [validation](validation/README.md).

Obtain a compatible nuclear library separately, including H in ZrH, H in water,
graphite thermal scattering, and all listed isotopes. Configure a new working copy:

```bash
python tools/configure_local.py --cross-sections /absolute/path/to/cross_sections.xml
python run_optimization.py --dry-run
python examples/build_reference_model.py
```

The setup changes local paths and therefore release checksums. Verify the
unmodified copy first. The model-export example writes XML without transport.
For notebook use, install JupyterLab/ipykernel in this environment and open the
cleared `run_optimization.ipynb`; its initial mode is dry-run.

## Run an optimization

The 108 movable genes and 19 fixed genes form 127 physical positions. NSGA-II
minimizes `(-keff, ppf_3d)` using the configured mean-keff feasibility constraint.
The optimization tally covers 1,620 regular-fuel segments. Lengths are in cm,
U-235 initialization proxies in g, and keff/PPF are dimensionless.

```bash
# Low-statistics software smoke run; launches OpenMC.
python run_optimization.py --config config/smoke_generation0.ini

# Full demonstration campaign; launches many expensive transport jobs.
python run_optimization.py
```

The main INI retains population 100, 50 generation records, seed 12345,
100,000 particles/batch, 45 inactive/150 nominal total batches, trigger extension
to 300 batches, and 32 OpenMP threads. Review the computing budget before running.
The fresh core has not been calibrated to measured criticality or power data.
Smoke results are insufficient for physics conclusions.

Use a new campaign ID when changing scientific inputs. Resume an interrupted
campaign with the same inputs and command; Ctrl+C preserves checkpoint state.
Cache reuse is disabled by default. Real-inventory checkpoints are incompatible.
Permutations between identical pins of the same fresh type are physically
equivalent, even though their fictional IDs remain unique for PMX crossover.

## Reevaluation and SDM

Both workflows select the saved first front from a **finished public campaign**.
Selection checks material/inventory fingerprints, fixed genes, permutations, and
constraint-based dominance. Front size follows the actual saved run.

```bash
python reevaluation_iv_4_5_2/reevaluate_final_front.py --dry-run
PYTHONPATH=src python -m final_evaluation.sdm_campaign prepare results/sdm_public
PYTHONPATH=src python -m final_evaluation.sdm_campaign preflight results/sdm_public
```

These commands prepare/check inputs without transport. Follow
[WORKFLOWS.md](docs/WORKFLOWS.md) to run, resume, or analyze them. Reevaluation
includes the public reference and separately reports 1,620 regular and 1,665
total fuel segments. SDM uses five stuck-rod cases, each with one rod OUT and
four IN. Its fixed beta_eff is an illustrative dollar conversion, not a beta
calculation for this fresh core; outputs are methodological demonstrations.

## Code and citation

Start with the [code guide](docs/CODE_GUIDE.md), [architecture](docs/ARCHITECTURE.md),
[builder](docs/MODEL_BUILDER.md), and [reproduction scope](docs/REPRODUCIBILITY.md).
`config/public_fuel_definition.json` and `tools/generate_public_inputs.py` make
the public material/inventory generation inspectable. `PACKAGE_MANIFEST.sha256`
records distributed file checksums. Runtime results and nuclear libraries are excluded.

Please cite this code using `CITATION.cff`. A reuse license has not yet been
selected; see [NOTICE.md](NOTICE.md).
