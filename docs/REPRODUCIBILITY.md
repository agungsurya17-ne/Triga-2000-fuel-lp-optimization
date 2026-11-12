# Reproduction scope

Version 1.1.0 distributes code and a fresh-fuel demonstration dataset. It
supports reproduction of the public workflow after installing OpenMC and
providing compatible nuclear data. It does not reproduce the thesis numerical
results, which used a confidential irradiated inventory and reference loading.

The public cards and densities, generic structural assumptions, invented pin
assignments, reference seed and U-235 mass calculation are documented in
[PUBLIC_INPUTS.md](PUBLIC_INPUTS.md). Geometry uses the corrected v2 builder.
Source/model identity and material bytes together determine the physical
evaluation context. Keep each campaign with its original inputs and hashes.

The release includes no completed demonstration transport campaign. Software
tests, material consistency, geometry probes, input exports, and dry-run
preflight checks are distinct from a converged neutron-transport calculation.
No fresh-core criticality, power, SDM or source-convergence conclusion is claimed.

The environment recipe pins the tested Python/API packages. Nuclear HDF5 data
must be obtained separately. Changing the library, temperature, transport
parameters or material composition changes the scientific context. Configure
local paths in a fresh working copy after verifying distributed checksums.

Random seeds support reproducible initialization and distinct transport jobs.
Bitwise transport identity may depend on OpenMC build, threading and hardware.
SDM dollars use an illustrative fixed beta, and PPF maximum uncertainty is not
estimated. See [WORKFLOWS.md](WORKFLOWS.md) for the supported protocols.
