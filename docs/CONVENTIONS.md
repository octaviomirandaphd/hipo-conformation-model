# Conventions

Read this before interpreting any figure or number.

## 1. Indexing

- `n` — monomers in the chain
- `N = n − 1` — dimers, and therefore torsion angles
- Dimers are `k = 1 … N` in prose and the documents; **0-based in code**

Adjacent dimers overlap by one monomer, so the tautomer at dimer *k* constrains dimers
*k*±1 through the requirement that the shared monomer adopt one consistent tautomeric form.
That constraint is what the transition matrices encode.

## 2. Chain direction and the `LABEL` map

The manuscript reads the chain from the T_d-rich end toward the T_a-rich end. Internally,
`T_fwd` propagates head→tail and funnels toward T_d; `T_rev` funnels toward T_a. So the
paper's reading direction is the reverse of the internal one.

`core.LABEL` declares this once:

```python
LABEL = {"forward": "Reverse (tail→head)",
         "reverse": "Forward (head→tail)"}
```

Both the console output and the figure titles use it. Earlier versions applied the flip to
the figures but not the printed output, so a single run reported `T_fwd` as "forward" in the
terminal and "Reverse" in the PNGs.

**This is a labelling choice, not a physical one.** The mathematics is unchanged; only which
end is called k₁ differs. The invariant claim is that the two matrices funnel to *opposite*
chain ends:

```
T_fwd stationary → T_d      T_rev stationary → T_a
```

## 3. The reverse kernel and its limitation

`T_rev` is derived from `T_fwd` and `P(T)` by Bayes' theorem at a single step:

```
T_rev[j, i] = T_fwd[i, j] · P(T)[i] / (P(T) @ T_fwd)[j]
```

Note the denominator is **P(T) @ T_fwd**, not P(T). Those coincide only if P(T) is
stationary under T_fwd, and it is not:

```
P(T) @ T_fwd = [0.3893, 1.75e-07, 0.1401, 0.4705]
P(T)         = [0.5295, 1.04e-07, 0.1906, 0.2799]     max deviation 0.191
```

**Consequence.** The Bayes relation is exact for one step (verified to 0.000e+00 in float64)
but `T_rev**s` is *not* the exact time-reverse of `T_fwd**s` for s ≥ 2:

| step | max abs deviation |
|---|---|
| 1 | 0.000e+00 |
| 2 | 1.756e-01 |
| 3 | 3.053e-01 |
| 5 | 4.314e-01 |

`T_rev` remains a valid row-stochastic kernel funnelling to T_a. What it is *not* is a
position-independent time-reverse. `tests/test_model.py` prints this table rather than
asserting it away, so the limitation is visible in test output.

Any manuscript sentence presenting the Bayes relation as *defining* [T_R] globally should
instead present the two matrices as independently valid directional kernels that coincide
with the Bayes pair at the first step.

## 4. Orientation averaging

Manuscript eq 6, mixture rule, with the reverse belief read at the **mirrored** index:

```
P_avg(T, k) = ½ [ P_fwd(T, k) + P_rev(T, N − 1 − k) ]
```

`avg_marginal()` implements this. Earlier `polymer_conformation_V1/V2` scripts used the
*same* index k for both directions, which differs by up to 5.3e-03 at the chain ends and
vanishes at the midpoint.

The mixture rule (rather than the product / forward–backward rule) is adopted because the
spectroscopic observables are direction-agnostic bulk averages, and because the product rule
suppresses T_c — the obligatory T_a↔T_d intermediate — by a factor of ~26 at n = 20. See
the orientation-averaging addendum.

## 5. Grid and numerics

- 360 points over [−180°, +180°), dφ = 1°, periodic
- float64 throughout (`jax_enable_x64`)
- 1-D distributions normalised to Σ P·dφ = 1; joints to Σ P·dφ² = 1
- Periodic trapezoid integration (identical to the rectangle rule on this grid)
- cis window φ ∈ 180° ± 30°; trans window φ ∈ 0° ± 30°

## 6. Pairwise joints must not be outer products

```
P(φ_i, φ_j) = Σ_{Ti,Tj} P(T_i) [T^(j−i)]_{Ti,Tj} P(φ_i|T_i) P(φ_j|T_j)
```

The transition matrix is contracted **between** the two P(φ|T) factors. Building the joint
as `outer(marginal_i, marginal_j)` forces I(φ_i; φ_j) = 0 for all values and discards the
correlation carried by the intermediate tautomers — which is the entire coupling mechanism.
`pairwise_joint(..., normalise=False)` gives the unnormalised joint from the correct
construction; the mutual information is identical normalised or not.

## 7. Φ_rest versus Φ_sum

Seeding differs: Φ_rest places the anchor on its own axis with the running sum starting at
zero; Φ_sum seeds on the diagonal because at k = 1 the running sum *is* the anchor angle.
Both then run N−1 transition-and-convolve steps.

`anchor_phirest_joint` computes **Φ_rest**, the circular sum of the N−1 angles that are
*not* the anchor, so the two axes never share a term. An earlier construction, Φ_sum,
included the anchor in its own sum and required primed notation to disambiguate.

Both fold in the same number of angles for Φ_rest(n) and Φ_sum(n−1), but Φ_rest's angle
chain begins one transition further along, so the two are close but not identical
(≤ 2.1e-03 in probability, ≤ 0.11 nats in entropy). They are different observables.

Documents quoting Φ_sum numbers are describing the earlier observable. See
`docs/ADDENDUM_CONSISTENCY.md` §B5.
