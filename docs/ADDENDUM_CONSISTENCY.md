# Addendum consistency audit

Every quantitative claim in the eight methods / interpretation documents, checked against
`hipo.core` in float64. Reproduced by `tests/test_addenda_part{1,2,3}.py`.

Documents audited: `methods_paragraph`, `methods_addendum`,
`Methods_Addendum_orientation_averaging_kstar`, `topology_addendum`,
`cis_trans_interpretation`, `Claude_polymer_physical_interpretation`,
`entropy_correction_addendum_v2`, `phisum_addendum_v2`.

**Convention note.** The addenda read the chain from the T_d end toward the T_a end, so the
document's [T_F] is the code's `T_rev` and vice versa. Once that reflection is applied,
most tables reproduce exactly. The audit below states results in the documents' own
convention.

---

## A. Reproduces exactly

**Joint entropy S(φ₁, φ_N)** — `entropy_correction_addendum_v2`. All 21 values, 4 d.p.:

| n | S_fwd | claim | S_rev | claim | S_avg | claim |
|---|---|---|---|---|---|---|
| 3 | 9.7021 | 9.7021 | 9.6748 | 9.6748 | 9.7319 | 9.7319 |
| 4 | 9.6515 | 9.6515 | 9.6429 | 9.6429 | 9.7346 | 9.7346 |
| 5 | 9.5686 | 9.5686 | 9.6100 | 9.6100 | 9.7247 | 9.7247 |
| 6 | 9.4801 | 9.4801 | 9.5855 | 9.5855 | 9.7121 | 9.7121 |
| 10 | 9.2092 | 9.2092 | 9.5484 | 9.5484 | 9.6755 | 9.6755 |
| 15 | 9.0650 | 9.0650 | 9.5430 | 9.5430 | 9.6605 | 9.6605 |
| 20 | 9.0206 | 9.0206 | 9.5426 | 9.5426 | 9.6571 | 9.6571 |

The non-monotonic S_avg uptick at n = 4 is reproduced. This confirms the corrected
(correlation-preserving) pairwise joint is the one those numbers came from.

**Belief crossings** — `Methods_Addendum_orientation_averaging_kstar`. Mixture and product
rules, every chain length, exact:

| n | mixture | claim | (k−1)/(N−1) | claim | product | claim |
|---|---|---|---|---|---|---|
| 6 | 1.885 | 1.885 | 0.221 | 0.221 | 2.354 | 2.354 |
| 10 | 3.381 | 3.381 | 0.298 | 0.298 | 3.906 | 3.906 |
| 15 | 5.234 | 5.234 | 0.326 | 0.326 | 5.805 | 5.805 |
| 20 | 7.083 | 7.083 | 0.338 | 0.338 | 7.680 | 7.680 |
| 30 | 10.815 | 10.815 | 0.351 | 0.351 | 11.414 | 11.414 |
| 50 | 18.227 | 18.227 | 0.359 | 0.359 | 18.834 | 18.834 |

**T_c populations** — same document, exact:

| n | terminus | claim | interior min | claim | mean (mix) | claim | mean (prod) | claim |
|---|---|---|---|---|---|---|---|---|
| 6 | 0.1232 | 0.123 | 0.0853 | 0.085 | 0.1001 | 0.100 | 0.0457 | 0.046 |
| 10 | 0.1034 | 0.103 | 0.0398 | 0.040 | 0.0634 | 0.063 | 0.0145 | 0.015 |
| 15 | 0.0970 | 0.097 | 0.0152 | 0.015 | 0.0422 | 0.042 | 0.0040 | 0.004 |
| 20 | 0.0957 | 0.096 | 0.0058 | 0.006 | 0.0313 | 0.031 | 0.0012 | 0.001 |

γ₁(T_c) = 0.0957, γ_N(T_c) = 0.0953, against the closed form ½P(T_c) = 0.0953. ✓

**Other exact matches**

- λ₂(T_fwd) = 0.735304, λ₂(T_rev) = 0.594933, λ₃ = λ₄ = 0 ✓
- `T_rev` matrix reproduces the `cis_trans_interpretation` §2 table entry-for-entry
- k* fractions: 0.6281 and 0.3719, summing to 1.0000 ✓
- Unnormalised peak growth k=0 → k=9: **2.324×** (claim ~2.3×) ✓
- cis/trans chain averages for n = 5, 10, 20: 0.490/0.454, 0.443/0.511, 0.405/0.555
  against 0.485/0.471, 0.444/0.511, 0.405/0.555 ✓
- T_c bridge width at n = 20, δ = 0.10: **10 dimers** ✓
- Chain-end saturation: T_a end plateaus by n ≈ 12–15, T_d end by n ≈ 18–20 ✓

---

## B. Discrepancies requiring a decision

### B1. Two addenda disagree on the stationary distributions

`cis_trans_interpretation` §2:

> **Stationary distribution:** Both networks converge to the same equilibrium
> {P(T_a)=0.529, P(T_c)=0.191, P(T_d)=0.280}.

`Claude_polymer_physical_interpretation` §5:

> The stationary distribution of the forward transition matrix converges to P(T_d) ≈ 1.0.

Computed:

```
T_fwd  stationary → [1.03e-06, 3.71e-07, 3.71e-07, 0.999998]   (T_d)
T_rev  stationary → [0.999999, 1.96e-07, 1.96e-07, 2.88e-07]   (T_a)
```

`Claude_polymer` is correct. The `cis_trans_interpretation` sentence should be struck — the
two networks converge to *opposite* absorbing states, which is the entire funnelling result.
{0.529, 0.191, 0.280} is the **prior** P(T), not a stationary distribution of either matrix.

### B2. The escape probabilities are attributed to the wrong matrices

`Methods_Addendum_orientation_averaging_kstar` and the **manuscript** both state:

> p_F = [T_F][T_d→T_c] = 0.405 … p_R = [T_R][T_a→T_c] = 0.265

Computed, in code labelling:

```
T_fwd[Ta→Tc] = 0.26470      T_fwd[Td→Tc] = 0.00000
T_rev[Ta→Tc] = 0.00000      T_rev[Td→Tc] = 0.40507
```

The two quoted entries are **exactly zero** in the matrices they are attributed to. 0.405 is
T_rev[T_d→T_c] and 0.265 is T_fwd[T_a→T_c] — the labels are swapped.

`methods_paragraph` and `cis_trans_interpretation` describe this correctly in words
("p_fwd = 0.26: probability that a T_a-region dimer converts toward T_d via T_c in the
forward direction"). The numerical values are right everywhere; only the matrix attribution
is wrong, and only in the two places above. **The manuscript sentence needs fixing.**

### B3. k* is reported in three mutually incompatible conventions

All describe the same physical crossing:

| Source | Definition | n = 20 |
|---|---|---|
| `cis_trans_interpretation` §4 | k* = 0.629 × N | **11.95** |
| `Methods_Addendum…kstar` | (k*−1)/(N−1) = 0.372 | **7.694** |
| Manuscript eq | closed form | **7.70** |

Verified: 0.6281 + 0.3719 = 1.0000, so 11.95 and 7.69 are the same point measured from
opposite ends — but they also use different affine conventions (k*/N versus
(k*−1)/(N−1)). Four combinations are in circulation across the documents. Pick one, state
which end k₁ is, and convert the rest.

### B4. `methods_paragraph`'s T_rev formula is not a stochastic matrix

As written:

> T_rev[T_i, T_j] = T_fwd[T_j, T_i] · P(T_i) / P(T_j)

Evaluated literally with the equilibrium P(T) in the denominator, the row sums are
**[0.735, 1.681, 0.735, 1.681]** — not a transition matrix. The code divides by
P_next = P(T)·T_fwd instead, giving row sums of exactly 1. The two agree only where P(T) is
stationary under T_fwd, which it is not (max deviation 0.191).

This is the same issue as the manuscript's eq 4 and should be fixed in both places
simultaneously.

### B5. ~~Φ_sum table does not reproduce~~ — RETRACTED, this was an error in the audit

**The Φ_sum table in `phisum_addendum_v2` is correct.** An earlier revision of this
document reported a discrepancy; that was a bug in the audit script, not in the addendum.

The audit had seeded the accumulator with a delta at Φ = 0 and then run **N**
transition-and-convolve steps. Addendum eq 26a seeds with the first angle already attached,

```
m_1(T_1, s) = P(T_1) · P(φ_1 = s | T_1)
```

and then runs **N − 1** steps. Both fold in N angles, but the audit version inserted one
extra transition *before* the first angle, pushing the chain one step further toward the
T_d absorbing state — which inflates cis and lowers entropy, exactly the direction of the
apparent discrepancy.

With eq 26a seeding, the addendum reproduces:

| n | S(Φ_sum) computed | claim | P(Φ_sum cis) computed | claim |
|---|---|---|---|---|
| 3 | 5.226 | 5.226 | 0.525 | 0.525 |
| 4 | 5.445 | 5.45 | 0.405 | 0.41 |
| 5 | 5.559 | 5.56–5.59 | 0.389 | 0.35–0.39 |
| 10 | 5.804 | 5.80–5.82 | 0.283 | 0.24–0.28 |
| 15 | 5.859 | 5.86–5.87 | 0.239 | 0.21–0.24 |
| 20 | 5.875 | 5.87–5.88 | 0.215 | 0.19–0.22 |

The tetramer cross-table reproduces to all four quoted decimals, both anchors:

| anchor | (trans,trans) | (trans,cis) | (cis,trans) | (cis,cis) |
|---|---|---|---|---|
| φ₁ computed | 0.2457 | 0.1392 | 0.0456 | 0.2581 |
| φ₁ claimed | 0.2457 | 0.1392 | 0.0456 | 0.2581 |
| φ₃ computed | 0.1400 | 0.0829 | 0.1490 | 0.3133 |
| φ₃ claimed | 0.1400 | 0.0829 | 0.1490 | 0.3133 |

φ₃'s own marginal cis probability is 0.656 against the claimed 65.6%.

**Structural check.** At n = 2 (N = 1, zero transitions) Φ_sum must reduce to P(φ₁).
Eq-26a seeding gives max deviation 5.2e-18; the delta-seeded version does not. This is the
test that catches the error, and it is now in `tests/test_addenda_part2.py`.

**Also retracted:** the claim that Φ_rest(n) ≡ Φ_sum(n−1). They fold in the same number of
angles but Φ_rest's angle chain begins one transition further along, so they differ:

| n | S[Φ_rest(n)] | S[Φ_sum(n−1)] | max abs difference |
|---|---|---|---|
| 4 | 5.1212 | 5.2260 | 2.12e-03 |
| 5 | 5.3643 | 5.4449 | 2.05e-03 |
| 10 | 5.7158 | 5.7809 | 1.17e-03 |
| 20 | 5.8378 | 5.8731 | 7.26e-04 |

**What still stands.** Φ_sum and Φ_rest remain *different observables*, and the shipped
`nmer_anchor_phirest_joint_flip.py` `anchor="first"` branch does over-propagate by one
transition. That finding was verified independently of any FFT accumulator — against
`P(φ_i,φ_j)` built from matrix powers, and against a brute-force triple sum over
(T₁,T₂,T₃) at n = 4 — and both showed the shipped code matching the two-transition result
to ~1e-19 instead of the correct one-transition result. `anchor="last"` is correct.

So the decision here is only which observable the paper reports, not whether the addendum's
numbers are right. They are.

### B6. The joint-entropy trend is inverted relative to the interpretation document

`Claude_polymer_physical_interpretation` §7 and §8:

> The joint entropy S(φ₁, φ_N)/k_B **increases** with chain length n as the two chain ends
> become decorrelated.

Corrected values: S_fwd = 9.7021 (n=3) → 9.2092 (n=10) → 9.0206 (n=20). It **decreases**
monotonically. `entropy_correction_addendum_v2` acknowledges this and argues the corrected
trend strengthens the physical argument; `Claude_polymer` predates the correction and still
carries the old claim.

Both documents cannot go into the SI as they stand. §7–§8 of the interpretation document
need rewriting to match the corrected table, and §9's polymerisation-entropy argument
should be re-examined since it rests on the sharpening/decorrelation picture.

### B7. Smaller items

- **T_c bridge extent, n = 20.** `methods_addendum` says "k = 4 to k = 14 (10 dimers wide)".
  Computed span at δ = 0.10 is **k = 4 to k = 13**, which *is* 10 dimers. The width is right;
  the upper bound is off by one (k=4…14 would be 11).
- **T_c bridge chain-length threshold.** `methods_addendum` requires n ≥ 15; the SI
  Figure S4 caption says "T_c bridge only shown for chains with ≥10 monomers". Pick one.
- **cis/trans limits.** Claimed ⟨trans⟩ → 0.380, ⟨cis⟩ → 0.583. Computed at n = 201:
  **0.369 / 0.596**. The n = 50 row (0.380/0.583 computed vs 0.385/0.578 claimed) suggests
  the published limit was extrapolated from n ≈ 50 rather than converged.
- **n = 3 chain averages.** Computed 0.518/0.418 against claimed 0.534/0.396.
- **Integration rule** is described three ways: "rectangle rule" (`methods_paragraph`),
  "periodic trapezoidal rule" (`Claude_polymer` §3), "Reimann sums" (SI). On a uniform
  periodic grid these coincide, so it is a wording problem, but the SI spelling also needs
  fixing (Riemann).
- **The ~10⁻¹⁷ / ~10⁻¹⁸ precision claims** hold for the float64 scripts. They do not hold
  for the float32 `polymer_conformation_V1/V2` scripts. This repository is float64
  throughout, so the claim is now supportable everywhere.

---

## C. Suggested order

1. **B2** — swap the p_F / p_R matrix attribution in the manuscript. One sentence.
2. **B4** — fix the T_rev formula in `methods_paragraph` and manuscript eq 4 together.
3. **B1** — strike the stationary-distribution sentence in `cis_trans_interpretation`.
4. **B3** — choose one k* convention and convert every document to it.
5. **B6** — rewrite `Claude_polymer` §7–§8 (and re-examine §9) against the corrected table.
6. **B5** — decide Φ_sum versus Φ_rest, then regenerate that table and the tetramer cells.
7. **B7** — the small ones.
