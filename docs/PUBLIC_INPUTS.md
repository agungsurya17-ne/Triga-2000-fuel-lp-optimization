# Public fresh-fuel inputs

This dataset is independent of the confidential irradiated inventory. Its
serial numbers and type assignments are invented. There is no published
original-to-fictional mapping and no original loading pattern.

## Fuel recipes

The source definition is `config/public_fuel_definition.json`. MCNP negative
material-card entries are interpreted as relative mass weights; each recipe is
normalized by its own total before writing OpenMC `wo` values. Cell densities
supplied with a minus sign are represented as positive `g/cm3` values in OpenMC.

| Component | M8 | M9 | M10 |
|---|---:|---:|---:|
| U235 | 38 | 55 | 99 |
| U238 | 152 | 220 | 396 |
| Natural Zr | 2009.761910 | 1981.631800 | 1934.171720 |
| H1 | 35.532200 | 35.034870 | 34.195780 |
| Er166 | 0 | 0 | 3.855640 |
| Er167 | 0 | 0 | 2.663350 |
| Density (g/cm³) | 5.79875 | 5.94499 | 6.40732 |

Natural zirconium is expanded using OpenMC natural isotopic abundances and
atomic masses. Er166 and Er167 are used exactly as supplied; other erbium
isotopes are not added. Each fuel material uses H in ZrH thermal scattering.
ACE suffixes such as `.60C` are not OpenMC nuclide names; a separate HDF5 library
supplies evaluated cross sections. Temperature is an explicit 300 K assumption.

Each recipe has 20% U235 in its uranium mass. The exact card normalization gives
approximately 8.5%, 12%, and 20.0414% uranium in total fuel mass, respectively;
the M10 label is retained without silently changing the supplied weights.

## Inventory and initialization

Regular IDs 900001–900036 use M8, 900037–900072 use M9, and 900073–900108 use M10.
Follower IDs 900109, 900110, and 900111 use M8, M9, and M10 at D2, D13, and D16.
These equal counts are a demonstration choice, not the reactor inventory.
The reference shuffles regular IDs with seed 731942 in the standard slot order.
Its ordering is unrelated to the real reference loading.

Regular fuel meat has inner radius 0.3175 cm, outer radius 1.82245 cm and active
length 38.1 cm. Follower fuel has the same inner radius and length, and outer
radius 1.67005 cm. Each active length is divided into 15 layers of 2.54 cm.

For each regular pin, the inventory proxy is
`m_U235 = density * pi * (r_outer² - r_inner²) * length * w_U235`, in grams.
Fresh and current masses are equal. This proxy guides initial loading heuristics;
it is not a neutronic worth calculation. Followers are fixed and excluded from
the 108-entry mutable inventory.

Within one type, pins and all axial layers have identical fresh compositions.
Exchanging same-type IDs does not change the physical model. Unique IDs remain
necessary for the permutation operators; different-ID permutations can therefore
represent equivalent physical loadings.

## Structural assumptions

All structural definitions are generic demonstration recipes at 300 K. They were
built independently rather than copied from the confidential material XML.
The JSON records every component weight and density. Water is H1/O16 at
0.997 g/cm³, graphite is elemental carbon at 1.7, SS304 is an Fe/Cr/Ni/Mn
70/19/10/1 mass mixture at 8.0, and steel is Fe/C 99/1 at 7.85.
Aluminum, zirconium, B4C, concrete, air, and lead also use explicit public
assumptions. Structural oxygen uses O16 and carbon uses the elemental C0
evaluation for compatibility with the target ENDF/B-VII.1 library.

Geometry v2 remains the common nominal builder. Geometry regression fixtures
retain geometric coordinates and bounds, with all fuel/material identities
replaced. They contain no old densities, MCNP material numbers, or original
serials. This checks implementation consistency and does not establish agreement
of the demonstration compositions with facility measurements.

## Regenerate

In a fresh copy without campaign outputs:

```bash
python tools/generate_public_inputs.py
```

This reconstructs material XML, the mass inventory, public assignments, INI
fuel IDs, and reference JSON from public definitions. It resets the XML data
index to its portable placeholder. Run local setup afterward. Do not regenerate
inputs in a campaign that you intend to resume.

OpenMC reference: [Material compositions](https://docs.openmc.org/en/stable/usersguide/materials.html).
