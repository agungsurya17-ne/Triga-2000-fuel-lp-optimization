# Nominal reevaluation

This pipeline reevaluates the saved final first front of a completed public
fresh-fuel optimization, together with a separately shuffled public reference.
The directory name follows the thesis section; inputs and cohort membership
belong to the public demonstration.

`inputs.py` verifies material/inventory identity, permutations and constrained
front selection through `modules.public_inputs`. `model.py` builds geometry v2
and a 1,665-segment tally. `reevaluate_final_front.py` handles isolated attempts,
checkpoints and restart. `analyze_reevaluation.py` writes response comparisons,
tables and plots. No confidential reference deck is required.

See [WORKFLOWS.md](../docs/WORKFLOWS.md) for dry-run, run, resume and analysis
commands. `--checkpoint` selects another compatible completed public campaign.
