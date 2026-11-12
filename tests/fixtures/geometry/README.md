# Public geometry regression fixtures

`reference_genome.json` contains invented public fuel IDs in a reference
permutation shuffled with seed 731942. It is unrelated to the facility loading.

`cell_mapping_verified.csv` retains nominal geometric coordinates, annular fuel
bounds and volumes in cm/cm³. Material names follow the invented reference.
Original serials, material numbers, densities and MCNP source mappings have
been removed. `geometry_probe_differences.csv` contains only coordinates for
48 regression probes around the corrected aluminum regions.

Tests check CSG consistency and the 1,620/1,665 segment contracts without
transport. These fixtures do not validate the demonstration compositions
against measurements or establish a fresh-core MCNP benchmark.
