# HIPO conformation model

Probabilistic model of backbone conformation in regioregular hydrogen-bonded
imidazopyridine oligomers (**HIPO**), supporting *Novel Imidazopyridine Oligomers with
Manipulable Backbone Conformation Evaluated Under a Probabilistic Bayesian Framework*.

Each bridging bond between adjacent monomers is a **dimer** carrying a tautomeric state
T ∈ {T_a, T_b, T_c, T_d} and a torsion angle φ. Because adjacent dimers share a monomer,
the tautomer of dimer *k* constrains dimer *k*±1 — a topological compatibility rule that
becomes a transition matrix. DFT-derived P(φ|T) distributions are propagated along the
chain by belief propagation, giving conformational distributions at any chain length.

The model is a tautomer-hidden-state relative of Flory's rotational isomeric state theory:
the hidden variable is the **tautomer** rather than a rotational isomer, and the state
space is constrained by topological compatibility between overlapping dimers rather than
by first-neighbour steric weights.

## Install and run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd src
PYTHONPATH=. python -m hipo.plots      # writes figures to ../figures/
PYTHONPATH=. python ../tests/test_model.py
```

Runtime is a few seconds; no GPU required.

## Layout

```
data/          DFT inputs as CSV + provenance   (nothing is hardcoded)
src/hipo/
  core.py      the model — no plotting, no file writing
  plots.py     every figure, all drawn from core
tests/         regression tests + addendum reconciliation
figures/       output (git-ignored)
```

`core.py` is deliberately import-only. Earlier versions of this analysis lived in four
separate scripts, each with its own copy of the constants and its own propagation loop;
they drifted. One importable core makes that impossible.

## What `core.py` provides

| Function | Returns |
|---|---|
| `build_model(data_dir=None)` | `ChainModel` from the CSVs; `MODEL` is the default instance |
| `belief_at(m, steps, direction)` | tautomer belief after *steps* propagations |
| `marginal(m, steps, direction)` | P(φ) at one dimer |
| `avg_marginal(m, n, k)` | orientation-averaged P(φ) at dimer *k* (manuscript eq 6) |
| `pairwise_joint(m, i, j, direction, normalise)` | P(φ_i, φ_j), correlation preserved |
| `anchor_phirest_joint(m, n, anchor, direction)` | P(φ_anchor, Φ_rest) |
| `cis_trans_profile(m, n)` | (trans, cis) probability at every dimer |

## Conventions

1. **Chain direction.** The paper reads the chain from the T_d-rich end toward the T_a-rich
   end, which is the opposite of the internal head→tail propagation. `LABEL` in `core.py`
   declares this once and is used by both the console output and the figure titles. The
   mathematics is unaffected — it is a choice of which end is called k₁.
2. **The reverse kernel.** `T_rev` is derived from `T_fwd` and `P(T)` by Bayes at a single
   step. P(T) is *not* stationary under `T_fwd`, so `T_rev**s` is not the exact time-reverse
   for s ≥ 2. `tests/test_model.py` prints the divergence rather than asserting it away.

## Reproducibility notes

Float64 throughout (`jax_enable_x64`). Angles on a uniform 360-point grid over
[−180°, +180°), dφ = 1°. Joint distributions normalised so Σ P·dφ² = 1. Integration is the
periodic trapezoid rule, which on a uniform periodic grid coincides with the rectangle rule.

`tests/test_addenda_part{1,2,3}.py` reproduce every published number

## Citation

See `CITATION.cff`. Please cite both the software (Zenodo DOI) and the paper.

## Licence

MIT — see `LICENSE`.
