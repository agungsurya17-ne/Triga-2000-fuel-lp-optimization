# Reevaluation and SDM

Start by completing an optimization with the public fresh-fuel inputs.
The default checkpoint is
`results/TRIGA2000_PUBLIC_FRESH_DEMO/checkpoint.json`. Results from a real
inventory or changed material file are rejected. A run is selected only after
its phase is `finished`; the last completed generation supplies the survivor
front. Candidate IDs and front size follow the saved checkpoint.

## Nominal reevaluation

From the project root:

```bash
python reevaluation_iv_4_5_2/reevaluate_final_front.py --dry-run
python reevaluation_iv_4_5_2/reevaluate_final_front.py --run
python reevaluation_iv_4_5_2/reevaluate_final_front.py --resume
python reevaluation_iv_4_5_2/analyze_reevaluation.py
```

Use `--checkpoint /absolute/path/to/checkpoint.json` for another compatible
public campaign. Preserve that argument when resuming. `--output` must be a
new subdirectory of `reevaluation_iv_4_5_2/`; use separate destinations for
different cohorts or input contexts. `--threads` defaults to 16.

`--dry-run` exports/validates every input and never launches transport. `--run`
evaluates the saved cohort plus the independent public reference at 100,000
particles, 45 inactive and 500 total batches, with triggers disabled.
`--smoke-test` runs only the first candidate and reference at low statistics;
it is a software check. Analysis reads saved verified results without transport.

The track-length fission-q tally covers 111 elements and 1,665 segments.
Volume-weighted PPF uses integrated segment response divided by segment volume.
The regular-only 108-element metrics are reported separately for comparison to
the optimizer objectives. PPF covariance and maximum-selection uncertainty are
not estimated. Tables and plots describe the evaluated public cohort only.

## Shutdown margin

Edit `src/final_evaluation/sdm_final.json` to select another public checkpoint
or thread count. Paths are relative to the repository root unless absolute.
Use an activated environment whose `python` and `openmc` expose version 0.15.3.

```bash
PYTHONPATH=src python -m final_evaluation.sdm_campaign prepare results/sdm_public
PYTHONPATH=src python -m final_evaluation.sdm_campaign preflight results/sdm_public
PYTHONPATH=src python -m final_evaluation.sdm_campaign run results/sdm_public --dry-run
PYTHONPATH=src python -m final_evaluation.sdm_campaign run results/sdm_public
PYTHONPATH=src python -m final_evaluation.sdm_campaign resume results/sdm_public
PYTHONPATH=src python -m final_evaluation.sdm_campaign analyze results/sdm_public
```

Only run/resume without `--dry-run` launches transport. Each selected candidate
has five cases: one stuck OUT rod, with the other four IN. Each case uses
100,000 particles, 45 inactive/200 total batches, no trigger, and a deterministic
distinct seed. Cases run sequentially. Separate attempts preserve failures;
`resume --retry-failed` explicitly retries failed attempts.

Preparation checks the CSG partition, rod motion and source/entropy coverage.
The public package has no independent private MCNP deck benchmark. Hashes bind
the checkpoint, inputs, code, executable and used nuclear data to each campaign.

The arithmetic uses `rho=(k-1)/k`, `k_upper=k+1.645*sigma_k`, and
`SDM_lower=(1/k_upper-1)/beta_eff`. Candidate SDM is the minimum of all five
valid case bounds. The fixed beta_eff=0.007294 and 0.5-dollar comparison are
illustrative protocol values, not a calculated beta or validated safety limit
for this fresh-fuel demonstration. A valid numerical bound does not establish
source convergence. Precision, diagnostics, and source review remain separate.

No original thesis results, cohort sizes, private reference inputs, or flux
analyses are required or included.
