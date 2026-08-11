# Independent audit of this repository

This file records how the model was audited, what the audit found, what was changed in
response, and what is still open. It is here so a referee does not have to take the code's
own word for anything — including this file's.


---

## 1. Method

A fresh LLM session is the same model that wrote the code; handing it the conclusions
produces agreement, not evidence. The code is also heavily commented, and several comments
*state* the answers, so an auditor could return a confident, correct-sounding report by
paraphrasing. The audit was designed around both problems and run in two passes, in separate
sessions.

**Pass 1 — blind reconstruction.** The auditor received only the primary material: the two
DFT spreadsheets, `HIPO_model_theory.docx`, and the manuscript's modelling sections. *Not*
the code. It was asked to build the model from scratch and derive the conventions — in
particular the axis convention of the orientation-averaged joint — from first principles
*before* writing any code. Its numbers were then compared against this repository.

**Pass 2 — adversarial read.** A separate session received the full repository plus the
spreadsheets and manuscript, with the explicit ground rule that *comments and docstrings are
claims, not evidence*: where a docstring states a number, recompute it; where it states a
reason, check the reasoning.

**Mutation testing.** The sharpest question in Pass 2 was not "do the tests pass" but "can
you break them". The auditor applied single-token edits to the source, re-ran the suite, and
classified each survivor as either a genuine gap in the tests or an equivalent mutant. This
is what exposed most of the test-suite defects below.

Every finding was then re-verified independently before being accepted — in several cases by
brute-force enumeration over all tautomer paths, which shares no code with the model.

---

## 2. Findings and disposition

### A — `anchor_phirest_joint(anchor="last")` was not the joint it claimed to be — **FIXED**

The `"last"` branch seeded with the terminal forward belief and walked *backwards* using
`T_rev`. But `T_rev` is the Bayes reverse of `T_fwd` **at the first step only**; P(T) is not
stationary under `T_fwd`, so iterating it does not give the chain's backward conditionals.
The file said as much about itself two hundred lines earlier.

Symptom: applying `T_rev` to the belief at dimer N−1 should recover the belief at N−2.

```
n = 3   2.8e-17        n = 4   0.107        n = 10   0.358
```

Exact at n = 3, where the only backward step *is* the first step — which is why every
existing test passed. The single case checked was the single case that worked.

Verified against brute-force enumeration over all 4^N tautomer paths: the shipped
construction deviated from the true joint by **5–16 % of peak height** for n ≥ 4.

**Fix.** The backward walk is gone. `_accumulate_last` folds the N−1 non-anchor angles
forward from the prior and attaches the anchor at the final step, so no backward kernel is
ever needed. It is also cheaper — a 1-D running sum rather than a 2-D accumulator. Agreement
with brute force: **1.9 × 10⁻¹⁹**.

**Consequences.** Theory-doc Table 9 changed; Figures 4d, 4e and S3 panels d/e/f were
regenerated. P(cis) moves 12–17 % relative (0.2522 → 0.2960 at n = 10); the entropy moves
0.1–0.6 %.

**New invariant, requiring no second implementation.** Both anchors describe the same chain,
so convolving either joint's two axes must give the same total twist φ₁+…+φ_N. Residual
**6 × 10⁻¹⁸** corrected, **2–8 × 10⁻⁴** before, and exact at n = 3 in both — the signature of
the bug. `tests/test_marginalisation.py` §6b.

### B — the interpolation convention — **RESOLVED: convention kept, reporting clarified**

`build_model` interpolates the *Boltzmann weights* linearly from the 15° DFT scan onto the 1°
grid. The audit noted that this draws a chord across an exponential, and that manuscript
Table 3's ⟨cos²φ⟩ is not computed from the distribution the model uses:

| method | ⟨cos²⟩ T_a |
|---|---|
| periodic trapezoid on the 15° scan grid — **the spreadsheet, and Table 3** | 0.951481 |
| the model's interpolated 1° grid | 0.941305 |
| linear in ΔG on the 1° grid | 0.951437 |

**The interpolant criticism is correct.** Cross-validation — hold out every other scan point,
predict it back from 30° spacing — gives a median relative error of 7–13 for linear-in-P
against 0.5–1.8 for linear-in-ΔG, on every tautomer.

**The convention was nonetheless kept, for two reasons.**

*The propagation layer is indifferent to it.* P(T) is the integral of P alone, and the
trapezoid rule is the exact integral of a linear interpolant — so the tabulated populations
and both transition matrices come out identical on either grid, to 4 × 10⁻¹⁰. The parameter
spreadsheet and the shipped P(φ|T) are one self-consistent set. Switching to ΔG-space would
move P(T) (T_a 0.5295 → 0.5338, T_c 0.1906 → 0.1825) and therefore both matrices
(T_fwd[T_a→T_c] 0.26470 → 0.25475) and p_F/p_R as quoted in the manuscript — breaking the
one artefact a reader can use to validate the first step by hand.

*⟨cos²φ⟩ feeds nothing.* It is a single-dimer descriptor of coplanarity. No part of the chain
model reads it, so the ~1% grid dependence propagates nowhere. It appears only in Table 1 /
Table 3.

k\* is 7.694 under either convention, so nothing rhetorical hangs on the choice.

**What changed instead.** `core.mean_cos2(m, on="scan"|"grid")` computes both and documents
why the scan-grid value is the one reported; `tests/test_manuscript_convention.py` pins the
published values to 1 × 10⁻⁶ *and* asserts that P(T) reconstructed from the 1° grid matches
the tabulated populations, so the indifference claim is checked rather than asserted. Table 1
of the theory document now carries a footnote naming the integration method and quoting the
1°-grid values alongside. **The manuscript needs the same one-line footnote on Table 3.**

**A claim was withdrawn.** §11 previously asserted that reported quantities "were checked to
be insensitive" to the interpolation choice, and named the 60°–120° region as where it
mattered. That check was never run and the statement is false — the largest change for T_a is
at the peak. §11 now states what is actually true, including the cross-validation result and
the reason the weaker interpolant was retained.

### C — Figure 4a–c panel order and axis orientation — **FIXED**

`pairwise_joint` always places the walk's prior on axis 0, so axis 0 is a different physical
dimer in each direction. Two consequences:

- **Panels a and b were swapped.** The manuscript captions 4a as the reverse direction, which
  in its own naming is [T_R] = code `T_fwd`, the T_d-funnelling joint. The function had that
  in panel b.
- **Panels a and c needed transposing** so that every panel plots x = paper φ₁ and y = paper
  φ_N. Panel b was already in that orientation.

Both had been corrected by hand on the published figure — a transpose is a horizontal flip
plus a 90° rotation — so the paper was right and the code was not. The code now reproduces
the manuscript's own description: the prior spread appears along the φ_N axis in 4a and along
the φ₁ axis in 4b.

Also fixed: Figures 4d and 4e printed the same Φ definition, though panel e's Φ excludes φ_N,
not φ₁; and both x axes read "φ_anchor" rather than naming the anchor.

### D — annotations that never rendered — **FIXED**

The "head (T_d/cis)" and "tail (T_a/trans)" labels on Figure 4f and all six S4 panels used
`ax.annotate` with a dummy anchor at `xy=(0,0)` in *data* coordinates, outside
`xlim = (0.5, N+0.5)`. matplotlib clips annotations whose anchor is off-axes, so both artists
existed in `ax.texts` and appeared in no output figure.

Resolved by removing them: the legend already gives T_a/T_d and cis/trans, and the curves
identify the ends, so the labels were redundant.

### E — surviving mutants — **FIXED (three of four)**

| mutant | effect | status |
|---|---|---|
| reverse `_paper_positions` in both copies | mirrors the chain axis of Fig 4f and all of S4; T_d and T_a ends swap, k\* moves to its mirror | killed — the plotted axis is now asserted directly, including that the T_d peak lands at paper k = 1 |
| swap the eigenvalues in `k_star` | moves the published k\* marker from 7.7 to 11.3 | killed — `k_star()` had **zero** references in `tests/`; the convention test re-derived the formula inline instead of calling it. It is now called |
| `to_paper_index` returns anything | dead code: no test, no caller anywhere | killed — `_paper_positions` now routes through it, so the conversion has one definition |
| `0.5 → 0.6` in `orientation_averaged_joint` | none on any normalised output | **equivalent mutant**, left alone. A uniform scale factor cannot be detected by a test of a normalised distribution and should not be |

Note on `test_figure_parity`: it detects *divergence* between the two copies of the figure
code, not error in either. Editing both copies identically restores parity and the kill
disappears. It is a consistency test and is not correctness coverage.

### Test-suite defects found and corrected

- The Bayes check in `test_marginalisation` §1 is near-circular — `T_rev` is *computed* from
  the relation being checked, so it can only return exactly 0. It is now labelled as such and
  followed by the non-circular version: rebuilding `T_rev` from the trimer populations
  (relative agreement 1.3 × 10⁻¹⁶) and `T_fwd` likewise (8 × 10⁻¹⁰ absolute, limited by the
  8 significant figures stored in the CSV, which is the true precision floor of the parameter
  chain).
- `escape_probabilities()` was checked only through `manuscript_view`, a separate code path;
  swapping its two return values went undetected. Now checked directly, with an assertion on
  the ordering.
- Theory-doc Table 9 was pinned to values read out of the code, which pinned the Finding-A
  error in place. The values are updated, but the invariant in §6b is what now protects that
  axis; the goldens only document it.
- `test_model.py` and `test_addenda_part{1,2,3}.py` print PASS/FAIL and always exit 0. They
  are reproduction reports, not gates. Treat only `test_marginalisation`,
  `test_manuscript_convention` and `test_figure_parity` as gating.

---

## 3. What the audit established as correct

Recomputed, not read off the comments:

- **Transition matrices.** `T_rev` rebuilt from the trimer populations independently of the
  Bayes construction: relative agreement 1.3 × 10⁻¹⁶. `T_fwd` likewise, 8 × 10⁻¹⁰ absolute.
  Row sums exact.
- **`pairwise_joint`.** Marginalises to `marginal()` on both axes, both directions, n = 3…20:
  worst residual ~1 × 10⁻¹⁷. Mutual information stays non-zero, so the outer-product failure
  mode is genuinely absent.
- **The transpose in `orientation_averaged_joint`.** Both quoted numbers reproduced: residual
  against eq 6 is 5.25 × 10⁻³ without the transpose and ~5 × 10⁻¹⁸ with it; I(avg) 1.11 × 10⁻²
  with, ~1 × 10⁻⁶ without.
- **`anchor_phirest_joint(anchor="first")`.** Reproduced to 0.0 with an independently written
  accumulator, and to 1.9 × 10⁻¹⁹ against brute-force path enumeration.
- **The terminus mapping.** Verified by matrix powers (`T_fwd`⁴⁰⁰ → T_d, `T_rev`⁴⁰⁰ → T_a), by
  the profile values at both ends, and by the identity k\*_model + k\*_paper = N + 1.
- **The manuscript's p_F and p_R are correct as written**, and only under the swapped reading:
  p_F = [T_F](T_c|T_d) = 0.40507, p_R = [T_R](T_c|T_a) = 0.26470. Read with code labels the
  two quoted elements are exactly zero — that is the labelling, not an error in the paper.
- **k\*.** λ₂ = 1 − p exactly for both kernels, so manuscript eq 10 and the eigenvalue formula
  in `k_star()` are the same equation; both give 7.7 for the 20-mer in paper coordinates.
- **Numerical hygiene.** float64 confirmed in effect; all distributions normalise to ≤ 2 × 10⁻¹⁶;
  the `jnp.maximum(..., 0)` clips never fire; FFT round-trip drift ~1 × 10⁻¹⁵ after 49
  convolutions. The small denominator in the `T_b` row of `T_rev` cancels algebraically.
- **Figure provenance.** All eleven PNGs regenerated from both code paths; md5 identical.

---

## 4. Still open

1. **Manuscript Table 3 needs a footnote** naming the integration method for ⟨cos²φ⟩ — the
   same one now beneath Table 1 of the theory document. No value changes.
2. **The cis/trans windows are 61° wide, not 60°.** `(φ ≥ −30) & (φ ≤ 30)` selects 61 grid
   cells; under the rectangle rule that integrates a 61° window while every caption says ±30°.
   Effect: +0.2 % to +0.5 % relative on every reported P(trans) and P(cis).
3. **The Gaussian 16 output is not in the repository.** The chain currently starts at Boltzmann
   weights someone else computed. `data/README.md` already flags this.
4. **The negative half of the torsional scan does not exist** — φ = −180°…−15° carry no
   Hartree energies in the spreadsheet; the profile is mirrored from 0°…180°. That is a
   substantive assumption (the model cannot express torsional handedness) and it is stated
   nowhere.
5. **The Hartree → kcal/mol → Boltzmann step is not executable code.** Reconstructing it from
   the spreadsheet gives RT = 0.59179075 kcal/mol, i.e. 298 K with the International Table
   calorie; the convention is nowhere stated, and the thermochemical calorie would shift
   P(φ|T) by up to 2.5 × 10⁻⁴ relative.
6. **`requirements.txt` is unpinned.** The PNG-hash claim is matplotlib-version-dependent.
7. **`Transition Matrix flip.xlsx`** labels both matrix blocks "Reverse"; the second should
   read "Forward". It is cited as the authority for the naming convention.
8. **Figure S2** (frontier orbitals) is cited in the manuscript but has no function here and
   no row in `FIGURE_MAP.md`. It is not a model output.

---

## 5. Honest limits of this audit

Two runs of the same model share blind spots, and the auditor was the same model family that
wrote the code. Pass 1 mitigates this by withholding the implementation, and mutation testing
mitigates it by asking a question that cannot be answered from the comments — but neither
eliminates it.

The check still worth more than either pass is a human in the group cloning the repository,
running `tests/`, and regenerating the figures on their own machine. That catches environment
assumptions and version drift that no amount of reading will, and it is exactly what a
referee can do.
