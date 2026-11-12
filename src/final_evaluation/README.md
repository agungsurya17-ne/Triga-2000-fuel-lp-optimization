# Public shutdown-margin workflow

`model.py` extends geometry v2 with independent rod states. `evaluator.py`
contains volume-weighted power metrics and SDM arithmetic. `sdm_contract.py`,
`sdm_campaign.py`, and `sdm_results.py` implement selection, provenance,
sequential attempts, restart checks and reports. `geometry_checks.py` provides CSG diagnostics; `sdm_results.py` checks saved
source snapshots and entropy histories.

The cohort comes from a finished public optimizer checkpoint selected in
`sdm_final.json`; its size is not fixed. All five stuck-rod cases are required
for each candidate summary. The fixed beta_eff=0.007294 is illustrative and
has not been computed for the demonstration core. Preparation checks CSG
consistency without claiming independent MCNP validation.

See [WORKFLOWS.md](../../docs/WORKFLOWS.md) for commands and interpretation.
