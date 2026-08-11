# Input data

Two CSV files, both read by `hipo.core.build_model()`. Nothing else in the
repository hardcodes model parameters.

## `torsional_scans.csv`

Relative Boltzmann weights *P(φ | T)* for the four tautomers of the
imidazopyridine dimer (**IPD**), on a 15° grid through a full 360° rotation
about the bridging C–C bond (25 points, with both −180° and +180° present).

| | |
|---|---|
| Method | DFT, relaxed torsional scan |
| Functional / basis | B3LYP / 6-31G(d) |
| Frontier orbitals | B3LYP / 6-311G(d,p) |
| Optimisation | `opt + freq` at every 15° step |
| Phase | gas phase |
| Software | Gaussian 16 |
| Side chains | methyl (simplification; see manuscript) |
| Reference state | T_a at φ = 0° taken as 0.0 kcal/mol |

Values are **unnormalised**. `build_model()` interpolates each column onto a
uniform 1° grid over [−180°, +180°) with `period=360` and normalises so that
Σ P·dφ = 1 for each tautomer.

> **To add before release:** the Gaussian 16 output files (or at minimum the
> optimised geometries and total energies for all four tautomers) belong in
> `data/gaussian/`. The manuscript SI currently reports coordinates for T_a
> only.

## `tautomer_parameters.csv`

- `P_T` — dimer tautomer equilibrium populations, obtained by periodic-trapezoid
  integration of each raw curve and normalising across tautomers.
- `T_fwd_from_*` — the forward transition matrix, row *i* = P(T_{k+1} | T_k = i).
  Allowed transitions follow the shared-monomer compatibility rule:
  T_a, T_b → T_a or T_c; T_c, T_d → T_b or T_d.

`T_rev` is **not** stored. It is derived in `core.py` from `T_fwd` and `P_T` by
Bayes' theorem at a single step. That derivation is exact only for a single step; see the README.
