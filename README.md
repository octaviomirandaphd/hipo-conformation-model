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
PYTHONPATH=. python -m hipo.plots      # writes the published figures to ../figures/
PYTHONPATH=. python ../tests/test_marginalisation.py
PYTHONPATH=. python ../tests/test_model.py
```

Runtime is a few seconds; no GPU required.

Or run the single-file version, which needs nothing from `src/` or `data/`:

```bash
python hipo_model_standalone.py checks    # marginalisation invariants
python hipo_model_standalone.py tables    # every published table
python hipo_model_standalone.py figures   # writes figures/*.png
python hipo_model_standalone.py all
```

`hipo_model_standalone.py` embeds the DFT input and derives every model
quantity in one readable pass. It is bit-identical to the packaged model
(verified to 0.0e+00 on `P_phi_T`, `P_T`, `T_fwd`, `T_rev` and on the joints)
and is the file to hand to a reviewer or auditor.

## Layout

```
hipo_model_standalone.py   the whole model in one file — start here
FIGURE_MAP.md              which published figure comes from which call
HIPO_model_theory.docx     the theory document (NOT the manuscript SI)
AUDIT.md                   independent audit: method, findings, what was fixed
data/          DFT inputs as CSV + provenance   (nothing is hardcoded)
src/hipo/
  core.py      the model — no plotting, no file writing
  plots.py     the published figure suite (fig_1c, fig_3a, fig_4abc,
               fig_4de, fig_4f, fig_S1, fig_S3, fig_S4, fig_S5)
  _figblock.py the figure bodies — shared verbatim with the standalone file
tests/         regression tests: marginalisation invariants + published values
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
| `pairwise_joint(m, i, j, direction, normalise)` | P(φ_i, φ_j) for one reading direction, correlation preserved |
| `orientation_averaged_joint(m, i, j, normalise)` | P(φ_i, φ_j) averaged over both reading directions (eq 6 axis convention) |
| `anchor_phirest_joint(m, n, anchor, direction)` | P(φ_anchor, Φ_rest) |
| `cis_trans_profile(m, n)` | (trans, cis) probability at every dimer |
| `taut_profile(m, n)` | orientation-averaged tautomer belief at every dimer |
| `tc_bridge_region(m, n, delta)` | span of the T_c conflict zone, manuscript eq 21 |
| `k_star(m, n)` | crossover position and its chain-length-independent fraction |
| `escape_probabilities(m)` | p_F and p_R in the manuscript's reading order |
| `global_marginal(m)` | P(φ) before any propagation, manuscript eq 4 |

## Conventions

Two points matter before interpreting any figure:

1. **Chain direction.** The paper reads the chain from the T_d-rich end toward the T_a-rich
   end, which is the opposite of the internal head→tail propagation. `LABEL` in `core.py`
   declares this once and is used by both the console output and the figure titles. The
   mathematics is unaffected — it is a choice of which end is called k₁.
2. **Reading direction.** The manuscript, `HIPO_model_theory.docx` and every figure read
   the chain from the T_d-rich end; the code propagates from the T_a-rich end. The two
   directional kernels therefore exchange names: published `[T_F]` is `T_rev` here and
   published `[T_R]` is `T_fwd` here, and dimer positions map as `k_paper = N - k_model`.
   Read with *code* labels, `T_fwd[Td→Tc]` and `T_rev[Ta→Tc]` are both exactly zero — that
   is the labelling, not an error in the paper.

   **`manuscript_view(m)` applies the whole mapping in one place.** It returns `T_F`, `T_R`,
   `lam2_F`, `lam2_R`, `p_F` and `p_R` already relabelled, so the theory document can be read
   against the code without holding the inversion in your head. `to_paper_index` /
   `to_model_index` convert dimer numbering and `paper_profile(m, n)` returns the profile
   arrays in printed order. `LABEL` applies the inversion to figure titles; the profile
   figures apply it to the chain axis.
3. **The reverse kernel.** Both matrices are primary: each is a conditional probability
   read directly off the trimer topology, obtained by renormalising the same equilibrium
   tautomer populations over that direction's permitted-successor set. They satisfy the
   one-step Bayes relation exactly (an algebraic identity, verified in
   `tests/test_marginalisation.py`), which is why `core.py` constructs `T_rev` from `T_fwd`
   and `P(T)` rather than reading it in separately. P(T) is *not* stationary under `T_fwd`,
   so `T_rev` is **not** the time-reverse of `T_fwd` — but it does not need to be, because
   it is defined independently. `tests/test_model.py` prints the divergence rather than
   asserting it away.
4. **Axis convention in the orientation-averaged joint.** Because `T_rev` is not the
   time-reverse of `T_fwd`, the reverse joint is *not* the transpose of the forward joint
   (they differ by ~75% of the peak). `orientation_averaged_joint` therefore transposes the
   reverse term before averaging, so both terms refer to the same physical dimer on each
   axis. Averaging without the transpose puts the prior on axis 0 of both terms, the mixture
   factorises, and the mutual information of the averaged joint collapses to ~1e-06. The
   invariant that pins this down — the averaged joint must marginalise to `avg_marginal` —
   is asserted in `tests/test_marginalisation.py`.

## Reproducibility notes

Float64 throughout (`jax_enable_x64`). Angles on a uniform 360-point grid over
[−180°, +180°), dφ = 1°. Joint distributions normalised so Σ P·dφ² = 1. Integration is the
periodic trapezoid rule, which on a uniform periodic grid coincides with the rectangle rule.

`tests/test_manuscript_convention.py` checks every number in `HIPO_model_theory.docx`
against the code through `manuscript_view` — the two transition matrices, eigenvalues,
escape probabilities, the divergence table, T_c populations, Φ_rest, belief crossings, k*
and the joint entropies. The document was for a while internally mixed, with `[T_F]` meaning
different matrices in different tables; nothing caught it because no test compared the
document to the code. Now one does.

`tests/test_figure_parity.py` asserts the figure code in `src/hipo/_figblock.py` and the
copy embedded in `hipo_model_standalone.py` are byte-identical — two copies of anything is
how the original four analysis scripts drifted apart. Both paths produce PNGs with matching
hashes.

`tests/test_marginalisation.py` checks that every joint the model produces marginalises to
the corresponding marginal computed by an independent code path. This is the cheapest test
that catches axis-convention errors and should be run first.

`tests/test_addenda_part{1,2,3}.py` reproduce the published model values — tautomer
populations, transition-matrix eigenvalues, belief crossings, T_c populations, joint
entropies, and the cumulative-twist distributions — directly from the DFT inputs in
`data/`.

## Audit

`AUDIT.md` records an independent audit of this repository — the two-pass method used, every
finding, what was fixed in response, and what remains open. Read it before trusting any
number here.

## Citation

See `CITATION.cff`. Please cite both the software (Zenodo DOI) and the paper.

## Licence

MIT — see `LICENSE`.
