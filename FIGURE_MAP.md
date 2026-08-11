# Figure-to-function map

Which published item comes from which call. Without this an auditor can verify the model
but not that the figures came from it.

Every function below exists twice, identically: in `src/hipo/plots.py` (via
`src/hipo/_figblock.py`) and inside `hipo_model_standalone.py`.
`tests/test_figure_parity.py` asserts the two copies are byte-identical, and the PNGs they
produce hash the same.

---

## Main text

| Figure | Content | Call |
|---|---|---|
| **1c** | ΔG torsional profile, four tautomers + Boltzmann average | `fig_1c(m)` |
| **3a** | Conditional P(φ\|T) for T_a–T_d plus the global P(φ) of eq 4 | `fig_3a(m)` |
| **4a** | Pairwise joint P(φ₁,φ_N), reverse direction (T_d-funnelling) | `fig_4abc(m, n=20)` panel a |
| **4b** | Pairwise joint P(φ₁,φ_N), forward direction (T_a-funnelling) | `fig_4abc(m, n=20)` panel b |
| **4c** | Pairwise joint P(φ₁,φ_N), orientation-averaged | `fig_4abc(m, n=20)` panel c |
| **4d** | Cumulative torsional joint P(φ₁, Φ), forward, φ₁ anchor | `fig_4de(m, n=20)` panel d |
| **4e** | Cumulative torsional joint P(φ_N, Φ), reverse, φ_N anchor | `fig_4de(m, n=20)` panel e |
| **4f** | 20-mer tautomer + cis/trans profile, k\* and T_c bridge | `fig_4f(m, n=20)` |

`fig_4abc(..., ln=True)` and `fig_4de(..., ln=True)` give the log-colour variants;
`fig_4abc(..., normalised=False)` gives the unnormalised joint.

## Supporting Information

| Figure | Content | Call |
|---|---|---|
| **S1** | ΔG and P(φ\|T) per tautomer, panels a–d | `fig_S1(m)` |
| **S3** | Marginals across chain lengths, 6 panels | `fig_S3(m, n_detail=20)` |
| **S4** | Tautomer + cis/trans profiles for n = 3, 4, 5, 6, 10, 15, with k\* and T_c bridge | `fig_S4(m)` |
| **S5** | a) chain-averaged cis/trans convergence, b) terminal tautomer belief saturation | `fig_S5(m, n_max=50)` |

S3 panel breakdown: **a)** P(φ₁) at the T_d/cis head, **b)** P(φ_N) at the T_a/trans tail,
**c)** orientation-averaged at both ends, **d)** Φ forward (k₁ anchor), **e)** Φ reverse
(k_{n−1} anchor), **f)** anchors vs cumulative Φ for the 20-mer.

**Figure 4a–c axis orientation (v1.4).** `pairwise_joint` always puts the walk's prior on
axis 0, so axis 0 is a *different physical dimer* in each direction: for the T_d-funnelling
joint it is paper φ_N, for the T_a-funnelling joint it is paper φ₁. Two consequences, both
fixed in v1.4 and both previously patched by hand on the published figure:

- **Panels a and b were swapped.** The manuscript captions 4a as the reverse direction, which
  in its own naming is [T_R] = code `T_fwd` — the T_d-funnelling joint. The function had that
  in panel b.
- **Panels a and c are transposed** so that every panel plots x = paper φ₁ and y = paper φ_N.
  Panel b was already in that orientation. A transpose is a horizontal flip plus a 90°
  rotation, which is exactly the manual edit that used to be applied to 4a and 4c.

With this, the code reproduces the manuscript's own description: the prior spread appears
along the φ_N axis in 4a and along the φ₁ axis in 4b.

**Panel c carries two curves per chain length, and that is correct.** The orientation-averaged
marginal differs at the two ends, because avg(0) = ½[prior + rev→T_a] while
avg(N−1) = ½[fwd→T_d + prior]. They differ by 1.15 × 10⁻² at n = 20. More generally the
averaged profile is *not* symmetric under k → N−1−k, since [T_F] is not the time-reverse of
[T_R] — the same fact that forces the transpose in the averaged joint. Panels a and b each
show one end and need one curve each; c is the averaged counterpart of both, so it needs two.
Solid is the head, dashed the tail.

**Profile figures (4f, S4)** no longer carry "head"/"tail" annotations: the legend gives
T_a/T_d and cis/trans, and the curves themselves identify the ends, so the labels were
redundant. (They had also never rendered — see AUDIT.md, Finding D.)

**Legend convention.** In every figure where the curves are *chain lengths* — all six S3
panels — the legend entries read `n=3, n=4, n=5, n=10, n=15, n=20`. `φ`-indexed legends
appear only where the curves are individual dimers of a single chain, which in the
published set is nowhere; the per-dimer overlay in `fig_pairwise` is a repository
diagnostic and is not a manuscript figure.

## HIPO model theory document

`HIPO_model_theory.docx`, shipped in this repository only. **This is not the manuscript
Supporting Information** — most of its tables are internal derivation and validation
material that will not appear in the SI submitted with the paper.

| Table | Content | Call |
|---|---|---|
| 1, 3 | P(T), [T_F], [T_R] | `manuscript_view(m).T_F`, `.T_R` |
| 4 | Eigenvalues and limiting distributions | `manuscript_view(m).lam2_F`, `.lam2_R` |
| 5 | Escape probabilities p_F, p_R | `manuscript_view(m).p_F`, `.p_R` |
| 6 | Divergence of [T_R]ˢ from the exact reverse of [T_F]ˢ | `run_tables()` → TABLE 4 block |
| 7 | T_c populations, mixture vs product rule | `run_tables()` → TABLE 7 |
| 8 | Chain-averaged cis/trans composition | `cis_trans_profile(m, n).mean()` → TABLE 8 |
| 9 | Φ_rest entropy and cis probability | `anchor_phirest_joint(m, n, "last", "forward")` + `entropy_1d` → TABLE 9 |
| 10 | Crossover position k\* | `k_star(m, n)` → TABLE 10 |
| 11 | Joint entropies S(φ₁,φ_N) | `entropy_2d` of the three joints → TABLE 11 |

---

## Conventions that will trip an auditor

**Indexing.** `n` = number of monomers, `N = n − 1` = number of dimers. Function arguments
`i, j, k` are **0-based dimer indices**; figure and table labels φ₁ … φ_N are **1-based**.
The terminal pair of a 20-mer is `pairwise_joint(m, 0, 18, …)`, labelled P(φ₁, φ₁₉).

**Reading direction.** The paper reads the chain from the T_d-rich end; the code propagates
from the T_a-rich end. The two directional kernels therefore swap names between the two
conventions:

```
manuscript [T_F]  ==  code T_rev          manuscript [T_R]  ==  code T_fwd
```

so the manuscript's escape probabilities are correct as written:
p_F = [T_F](T_c|T_d) = 0.40507 and p_R = [T_R](T_c|T_a) = 0.26470. Read with *code* labels,
`T_fwd[Td→Tc]` and `T_rev[Ta→Tc]` are both exactly zero — that is the labelling, not an
error in the paper. `Transition Matrix flip.xlsx` tabulates both matrices in the
manuscript's reading order and is the reference for this.

`HIPO_model_theory.docx` now uses this same published convention throughout, so the
document, the manuscript and every figure agree. `core.manuscript_view(m)` returns the
directional quantities already relabelled, and `tests/test_manuscript_convention.py` checks
every table in the document against it.

The inversion is applied in two places: `LABEL` flips it in plot *titles*, and the profile
figures flip it on the *chain axis*, `k_paper = N − k_model`, so that k = 1 is the T_d/cis
head and k = N is the T_a/trans tail.

**k\* is quoted from both ends.** `k_star()` returns 0.6281 in *model* coordinates. Under
`manuscript_view` the same formula gives 0.3719, which is the value the theory document and
the manuscript quote as primary; 0.6281 is then "measured from the opposite terminus". Both
are correct and sum to 1.0000. The profile figures plot `k*_paper = N − k*_model + 1`,
which is 7.7 for the 20-mer.

**Φ_rest, not Φ_sum.** Φ_rest holds the anchor angle on its own axis and sums the other
N−1 angles. Φ_sum, which included the anchor in the sum, has been removed. The old Φ_sum
entropies (5.226 / 5.45 / 5.575 / 5.81 / 5.865 / 5.875) are not the Φ_rest values.

---

## Correction in v1.4 — Figures 4d/4e and Table 9 must be regenerated

`anchor_phirest_joint(anchor="last")` was built by seeding with the terminal forward belief
and walking back with the opposite kernel. That is exact only at n = 3: `T_rev` is the Bayes
reverse of `T_fwd` at the *first* step, and P(T) is not stationary, so iterating it does not
give the chain's backward conditionals. Measured against brute-force enumeration over all
tautomer paths, the error was 5–16% of peak height for n ≥ 4.

It is now accumulated forward — the N−1 non-anchor angles folded in from the prior, the
anchor attached at the final step — which needs no backward kernel at all. Verified against
brute force to 1.9 × 10⁻¹⁹.

**Any Figure 4d, 4e, or S3 panel d/e/f generated before v1.4 must be regenerated**, and
theory-doc Table 9 changes: P(cis) moves 12–17% relative (e.g. 0.2522 → 0.2960 at n = 10),
the entropy by 0.1–0.6%. Figures 4a–4c, 4f, S1, S4 and S5 are unaffected.

The invariant that now protects this — and which needs no second implementation — is that
both anchors describe the same chain, so convolving either joint's two axes must give the
same total twist φ₁+…+φ_N. Residual 6 × 10⁻¹⁸ corrected, 2–8 × 10⁻⁴ before.

## Two corrections applied while building this map

**Figure 4c must be regenerated.** Any orientation-averaged joint made before the transpose
fix used `0.5*(jf + jr)`, which does not marginalise to the profile in 4e/4f and whose
mutual information at n = 20 is 7.4 × 10⁻⁷ instead of 1.1 × 10⁻². `fig_4abc` now calls
`orientation_averaged_joint`. Figures 4a, 4b, 4d, 4e and 4f are unaffected.

**Figure S3 panel c had the same class of error, in the marginal.** The standalone script
that produced S3 averaged the two directions at the *same* dimer index rather than the
mirrored index of eq 6. One consequence is visible: `½[P_fwd(0) + P_rev(0)]` is exactly the
prior for every chain length, so an entire family of curves collapsed onto a single
n-independent line. Residual against eq 6 was 5.3 × 10⁻³. `fig_S3` now uses `avg_marginal`.
