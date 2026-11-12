# Checks for public release 1.1.0

Checked on 13 November 2026 using the existing OpenMC 0.15.3 environment.
No new neutron-transport calculation was launched.

- Main suite: 88 tests, 85 passed, 3 skipped (`unittest.log`).
- Reevaluation suite: 12 test methods, 9 passed, 3 skipped, plus one
  artifact-test class skipped (`reevaluation_unittest.log`).
- Fresh cards: normalized mass fractions, supplied densities, uniform axial
  compositions, fictional assignments and U-235 gram inventory checked.
- Public cohort selection: constrained dominance, variable front size, fixed
  genes, permutations and rejection of changed material fingerprints tested.
- Public workflow integration: genuine CSG models exported for nominal
  reevaluation and all five SDM rod cases from clearly synthetic selection
  records; transport calls forbidden. No synthetic physics results are distributed.
- Public input generator: byte-identical output for all six generated inputs
  in another directory. Local setup, optimizer preflight, model export and
  SDM environment provenance verified in a temporary copy.
- Nuclear index: all 50 required isotope/thermal-law names are available in
  the local ENDF/B-VII.1-based library. The library itself is excluded.

`public_input_checks.json` records isolated setup/export results. Optional
checks requiring a completed public optimizer run or real smoke statepoints
remain skipped. The source-comparison test also skips because the standalone
package excludes the parent research workspace.

All Python functions have English docstrings. Inline comments are reserved
for non-obvious logic and physical/model choices. Formatting uses Black
with a 100-character line length. `public_tree_audit.json` records the
working-tree publication audit; it does not audit GitHub history or remote caches.

These checks establish software/input consistency. They do not establish
fresh-core criticality, source convergence, agreement with measurements or
facility shutdown capability. SDM dollar conversion uses an illustrative beta.
