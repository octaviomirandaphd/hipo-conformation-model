#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
HIPO conformation model — standalone reference implementation (JAX)
================================================================================

Tautomer-encoded belief-propagation model of backbone conformation in
regioregular hydrogen-bonded imidazopyridine oligomers (HIPO).

Supporting: *Novel Imidazopyridine Oligomers with Manipulable Backbone
Conformation Evaluated Under a Probabilistic Bayesian Framework*
Octavio Miranda, Department of Chemistry, Texas A&M University
ORCID 0000-0002-1478-1560.  Licence: MIT.

--------------------------------------------------------------------------------
WHY THIS FILE EXISTS
--------------------------------------------------------------------------------
This is a single self-contained file: no imports from the repository, all DFT
input embedded and visible, every model quantity derived in front of the reader.
It is the file to hand to an auditor.  The packaged repository
(hipo-conformation-model) splits the same mathematics across src/hipo/core.py,
src/hipo/plots.py and tests/; this file must reproduce it exactly, and
``python hipo_model_standalone.py all`` asserts that it does where the two
overlap.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    pip install "jax[cpu]" numpy matplotlib

    python hipo_model_standalone.py checks    # marginalisation invariants
    python hipo_model_standalone.py tables    # every published table
    python hipo_model_standalone.py figures   # writes figures/*.png
    python hipo_model_standalone.py all       # all of the above

Run time is a few seconds on a laptop CPU.  No GPU required.  Deterministic:
no random numbers anywhere in the model.

--------------------------------------------------------------------------------
THE THREE CONVENTIONS THAT MATTER
--------------------------------------------------------------------------------
1.  CHAIN DIRECTION.  The paper reads the chain from the T_d-rich end toward the
    T_a-rich end, the opposite of the internal head->tail propagation.  Internal
    "forward" therefore prints as "Reverse (tail->head)".  Declared once in
    LABEL; the mathematics is unaffected, it is a choice of which end is k_1.

2.  BOTH TRANSITION MATRICES ARE PRIMARY.  Each is a conditional probability
    read directly off the trimer topology: for a given state, renormalise the
    equilibrium populations of that direction's permitted successors over that
    set.  The two directions have different permitted-successor sets, which is
    the origin of the directional asymmetry.  They satisfy the one-step Bayes
    relation exactly -- an algebraic identity, not a coincidence -- which is why
    T_rev is computed here from T_fwd and P(T) rather than entered separately.
    Because P(T) is not stationary under T_fwd, [T_R] is NOT the time-reverse of
    [T_F].  It does not need to be: it is independently defined.

3.  AXIS CONVENTION IN THE ORIENTATION-AVERAGED JOINT.  Following from (2), the
    reverse joint is not the transpose of the forward joint -- they differ by
    ~75% of the peak height.  ``orientation_averaged_joint`` therefore
    TRANSPOSES the reverse term before averaging, so both terms refer to the
    same physical dimer on each axis.  Averaging without the transpose puts the
    prior on axis 0 of both terms; once the kernels converge the mixture
    factorises and the mutual information collapses to ~1e-06.  See the long
    note on that function.

--------------------------------------------------------------------------------
CHANGE LOG RELATIVE TO THE ORIGINAL FOUR ANALYSIS SCRIPTS
--------------------------------------------------------------------------------
  (a) float64 throughout (jax_enable_x64).  The originals cast to float32 at the
      JAX boundary, capping accuracy at ~1e-7.
  (b) Pairwise joints built from the correlated construction (transition matrix
      contracted BETWEEN the two P(phi|T) factors), never from
      outer(marginal_i, marginal_j) -- which forces I(phi_i;phi_j) = 0.
  (c) The anchor/Phi_rest accumulator no longer applies an extra transition
      before the first convolution (the archived nmer_anchor_phirest_joint_flip
      over-propagated its anchor="first" branch by exactly one step).
  (d) One orientation-average convention for marginals: the reverse belief is
      read at the mirrored dimer index n-2-k, matching manuscript eq 6.
  (e) The orientation-averaged JOINT transposes the reverse term -- convention
      (3) above.  This is the most recent correction.
  (f) Phi_sum removed.  Only Phi_rest, in which the anchor angle is held on its
      own axis, appears in the manuscript, so only Phi_rest is implemented.
================================================================================
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
jax.config.update("jax_enable_x64", True)          # MUST precede any jnp use
import jax.numpy as jnp
import numpy as np

# ==============================================================================
# SECTION 1 — DFT INPUT
# ==============================================================================
# Everything below this point is derived.  These are the only numbers entered by
# hand, and they are reproduced from the spreadsheet
# "Dihedral Scan Input for HIPO model.xlsx", sheet "Dimer P(phi,T)".
#
# Provenance
# ----------
#   Method              relaxed torsional scan, opt+freq at every 15 deg step
#   Functional/basis    B3LYP / 6-31G(d)   (frontier orbitals 6-311G(d,p))
#   Phase               gas phase, 298 K
#   Software            Gaussian 16
#   Side chains         methyl (simplification; see manuscript)
#   Reference state     T_a at phi = 0 deg taken as 0.0 kcal/mol
#
# The spreadsheet converts raw Hartree energies to relative Boltzmann weights
# via  w = (1/Z) exp(-dE / kT),  kT = 0.00198587 * 298 kcal/mol,
# Z = 81.6556046781263.  The weights below are those values, UNNORMALISED; the
# normalisation constant divides out when each column is normalised on the 1 deg
# grid, so the value of Z does not enter the model.
# ------------------------------------------------------------------------------

TAUT_NAMES = ["Ta", "Tb", "Tc", "Td"]

# phi(deg), Ta, Tb, Tc, Td   -- 25 points, both -180 and +180 present
_SCAN = np.array([
    [-180, 3.37991667e-01, 1.46247e-12, 5.51567e-10, 7.03255580e-01],
    [-165, 1.62442177e-01, 5.69999e-10, 1.07225e-07, 3.51888510e-01],
    [-150, 2.35822010e-02, 1.32490e-08, 2.71010e-06, 5.63789980e-02],
    [-135, 7.70910000e-04, 5.83401e-08, 7.57204e-06, 2.01901600e-03],
    [-120, 9.85523000e-07, 5.25821e-08, 8.67275e-06, 3.13868000e-05],
    [-105, 2.52179000e-08, 1.32286e-09, 1.22552e-07, 3.24915000e-08],
    [ -90, 8.20422000e-09, 5.11025e-10, 1.50063e-07, 6.96081000e-09],
    [ -75, 5.80932000e-08, 7.62978e-10, 1.70690e-06, 1.01101000e-08],
    [ -60, 3.36813000e-06, 5.05591e-08, 2.71982400e-03, 3.90524000e-07],
    [ -45, 2.62630000e-03, 5.64536e-08, 2.55883940e-02, 3.28538000e-07],
    [ -30, 6.78023550e-02, 4.67436e-08, 1.89640966e-01, 8.71962000e-08],
    [ -15, 5.14903052e-01, 1.48336e-09, 2.85552590e-01, 5.83736000e-09],
    [   0, 1.00000000e+00, 6.76823e-10, 3.05131070e-02, 4.63302000e-11],
    [  15, 5.14903052e-01, 1.48336e-09, 2.85552590e-01, 5.83736000e-09],
    [  30, 6.78023550e-02, 4.67436e-08, 1.89640966e-01, 8.71962000e-08],
    [  45, 2.62630000e-03, 5.64536e-08, 2.55883940e-02, 3.28538000e-07],
    [  60, 3.36813000e-06, 5.05591e-08, 2.71982400e-03, 3.90524000e-07],
    [  75, 5.80932000e-08, 7.62978e-10, 1.70690e-06, 1.01101000e-08],
    [  90, 8.20422000e-09, 5.11025e-10, 1.50063e-07, 6.96081000e-09],
    [ 105, 2.52179000e-08, 1.32286e-09, 1.22552e-07, 3.24915000e-08],
    [ 120, 9.85523000e-07, 5.25821e-08, 8.67275e-06, 3.13868000e-05],
    [ 135, 7.70910000e-04, 5.83401e-08, 7.57204e-06, 2.01901600e-03],
    [ 150, 2.35822010e-02, 1.32490e-08, 2.71010e-06, 5.63789980e-02],
    [ 165, 1.62442177e-01, 5.69999e-10, 1.07225e-07, 3.51888510e-01],
    [ 180, 3.37991667e-01, 1.46247e-12, 5.51567e-10, 7.03255580e-01],
], dtype=np.float64)

# Dimer tautomer equilibrium populations P(T).
# Spreadsheet columns U-X: periodic-trapezoid integral of each raw curve over the
# full 360 deg, then normalised across the four tautomers.
_P_T = np.array([0.52946541, 1.03943e-07, 0.19059783, 0.279936655], dtype=np.float64)

# Forward transition matrix  T_fwd[i, j] = P(T_{k+1} = j | T_k = i).
#
# Read off the trimer topology.  Because adjacent dimers share a monomer, the
# shared monomer must adopt a single consistent tautomeric form, which fixes the
# permitted successors:
#
#     forward   T_a, T_b -> {T_a, T_c}     T_c, T_d -> {T_b, T_d}
#     reverse   T_a, T_c -> {T_a, T_b}     T_b, T_d -> {T_c, T_d}
#
# The transition probability is then the equilibrium population of each
# permitted successor renormalised over that set:
#
#     T_fwd[Ta -> Ta] = p_a / (p_a + p_c) = 0.735304
#     T_fwd[Ta -> Tc] = p_c / (p_a + p_c) = 0.264696
#     T_fwd[Tc -> Tb] = p_b / (p_b + p_d) = 3.71308e-07
#     T_fwd[Tc -> Td] = p_d / (p_b + p_d) = 0.999999629
#
# (spreadsheet rows 38-52 construct exactly these, and the reverse set, from the
# same integrals; the reverse construction agrees with the Bayes derivation
# below to 1.1e-16, which is an algebraic identity -- see build_model.)
_T_FWD = np.array([
    [0.73530404, 0.0,         0.26469596, 0.0        ],   # from Ta
    [0.73530404, 0.0,         0.26469596, 0.0        ],   # from Tb
    [0.0,        3.71308e-07, 0.0,        0.999999629],   # from Tc
    [0.0,        3.71308e-07, 0.0,        0.999999629],   # from Td
], dtype=np.float64)

# Relative Gibbs free energy dG (kcal/mol) per tautomer, same relaxed scan, plus
# the population-weighted average.  Reference: T_a at phi = 0 deg = 0.0.
# Used only for the energy figures (1c, S1); it plays no part in the model, which
# works from the Boltzmann weights above.
# phi(deg), dG_Ta, dG_Tb, dG_Tc, dG_Td, dG_avg   (kcal/mol)
_DG = np.array([
    [        -180,  0.641935569,   16.1268271,   12.6159478,  0.208330996,   7.39826037],
    [        -165,   1.07554014,   12.5964952,    9.4972579,  0.618090455,   5.94684593],
    [        -150,    2.2175956,   10.7346938,   7.58588377,   1.70178814,   5.55999033],
    [        -135,   4.24192028,   9.85744463,   6.97783336,   3.67214756,   6.18733646],
    [        -120,   8.18452163,   9.91893992,   6.89751298,   6.13635184,   7.78433159],
    [        -105,   10.3537995,   12.0982578,   9.41819253,   10.2038263,    10.518519],
    [         -90,   11.0183252,    12.661128,   9.29833945,   11.1155881,   11.0233452],
    [         -75,   9.85995464,   12.4239319,   7.85947507,   10.8947071,   10.2595172],
    [         -60,   7.45724565,   9.94215753,   3.49581921,   8.73233175,   7.40688854],
    [         -45,   3.51652681,   9.87689722,   2.16927787,   8.83461474,   6.09932916],
    [         -30,   1.59260261,   9.98859275,  0.983924704,   9.61962099,   5.54618527],
    [         -15,  0.392816878,   12.0304875,  0.741708546,   11.2197536,   6.09619165],
    [           0,            0,   12.4948397,   2.06511237,   14.0817948,   7.16043673],
    [          15,  0.392816878,   12.0304875,  0.741708546,   11.2197536,   6.09619165],
    [          30,   1.59260261,   9.98859275,  0.983924704,   9.61962099,   5.54618527],
    [          45,   3.51652681,   9.87689722,   2.16927787,   8.83461474,   6.09932916],
    [          60,   7.45724565,   9.94215753,   3.49581921,   8.73233175,   7.40688854],
    [          75,   9.85995464,   12.4239319,   7.85947507,   10.8947071,   10.2595172],
    [          90,   11.0183252,    12.661128,   9.29833945,   11.1155881,   11.0233452],
    [         105,   10.3537995,   12.0982578,   9.41819253,   10.2038263,    10.518519],
    [         120,   8.18452163,   9.91893992,   6.89751298,   6.13635184,   7.78433159],
    [         135,   4.24192028,   9.85744463,   6.97783336,   3.67214756,   6.18733646],
    [         150,    2.2175956,   10.7346938,   7.58588377,   1.70178814,   5.55999033],
    [         165,   1.07554014,   12.5964952,    9.4972579,  0.618090455,   5.94684593],
    [         180,  0.641935569,   16.1268271,   12.6159478,  0.208330996,   7.39826037],
], dtype=np.float64)

# ---- grid -------------------------------------------------------------------
N_GRID = 360
PHI_GRID = np.linspace(-180.0, 180.0, N_GRID, endpoint=False)
DPHI = float(PHI_GRID[1] - PHI_GRID[0])            # 1.0 deg

# ---- labelling (convention 1) -----------------------------------------------
LABEL = {"forward": "Reverse (tail→head)",
         "reverse": "Forward (head→tail)"}


# ==============================================================================
# SECTION 2 — MODEL CONSTRUCTION
# ==============================================================================
class ChainModel(NamedTuple):
    phi_grid : jax.Array   # (M,)
    P_phi_T  : jax.Array   # (4, M)  P(phi | T); each row integrates to 1
    P_T      : jax.Array   # (4,)    dimer tautomer populations
    T_fwd    : jax.Array   # (4, 4)  P(T_{k+1} | T_k)
    T_rev    : jax.Array   # (4, 4)  P(T_k | T_{k+1})
    P_phi_fft: jax.Array   # (4, M)  complex, precomputed for convolutions
    scan_phi : jax.Array   # (25,)   raw 15-deg DFT scan angles
    scan_w   : jax.Array   # (4, 25) raw (unnormalised) Boltzmann weights
    dG_phi   : jax.Array   # (25,)   raw scan angles for the energy profile
    dG       : jax.Array   # (5, 25) dG(kcal/mol): Ta,Tb,Tc,Td,avg  (figs 1c, S1)


def build_model(scan=_SCAN, p_t=_P_T, t_fwd=_T_FWD) -> ChainModel:
    """
    Interpolate the 15 deg scan onto the 1 deg grid, normalise, and construct
    the reverse kernel.

    The interpolation is a MODELLING CHOICE and is called out explicitly here
    because it is the one transformation between the spreadsheet and the model
    that is not visible in either: ``np.interp`` with ``period=360`` is linear
    interpolation on the circle between the 15 deg samples.  A cubic or spline
    interpolant would give slightly different tail probabilities in the
    low-density regions between 60 and 120 deg.  All reported quantities were
    checked to be insensitive to this at the reported precision, but an auditor
    should test it rather than take the claim on trust.
    """
    phi_raw = scan[:, 0]
    P_phi_T = np.zeros((4, N_GRID), dtype=np.float64)
    for i in range(4):
        y = np.interp(PHI_GRID, phi_raw, scan[:, 1 + i], period=360.0)
        y = np.maximum(y, 0.0)
        P_phi_T[i] = y / (y.sum() * DPHI)

    P_T = np.asarray(p_t, dtype=np.float64)
    P_T = P_T / P_T.sum()
    T_fwd = np.asarray(t_fwd, dtype=np.float64)
    T_fwd = T_fwd / T_fwd.sum(axis=1, keepdims=True)

    # Reverse kernel.  Bayes at a single step:
    #     P(T_k | T_{k+1}) = P(T_{k+1} | T_k) P(T_k) / P'(T_{k+1}),
    #     P' = P . T_fwd    (the belief AFTER one forward step, not P itself)
    #
    # This reproduces the independent trimer construction exactly.  Writing p
    # for the unnormalised integrals, row a of the Bayes construction gives
    #     T_R[a,a] = [p_a/(p_a+p_c)] P_a / {(P_a+P_b) p_a/(p_a+p_c)}
    #              = P_a/(P_a+P_b) = p_a/(p_a+p_b),
    # which IS the trimer expression: the forward normalising factor cancels.
    # The same cancellation occurs in every row.  Verified numerically at
    # 1.11e-16 against the spreadsheet's independently built reverse matrix.
    P_next = T_fwd.T @ P_T
    T_rev = ((T_fwd * P_T[:, None]) / (P_next[None, :] + 1e-300)).T

    return ChainModel(
        phi_grid  = jnp.asarray(PHI_GRID),
        P_phi_T   = jnp.asarray(P_phi_T),
        P_T       = jnp.asarray(P_T),
        T_fwd     = jnp.asarray(T_fwd),
        T_rev     = jnp.asarray(T_rev),
        P_phi_fft = jnp.fft.fft(jnp.asarray(P_phi_T), axis=1),
        scan_phi  = jnp.asarray(scan[:, 0]),
        scan_w    = jnp.asarray(scan[:, 1:].T),
        dG_phi    = jnp.asarray(_DG[:, 0]),
        dG        = jnp.asarray(_DG[:, 1:].T),
    )


def _T(m: ChainModel, direction: str) -> jax.Array:
    if direction not in ("forward", "reverse"):
        raise ValueError("direction must be 'forward' or 'reverse'")
    return m.T_fwd if direction == "forward" else m.T_rev


# ==============================================================================
# SECTION 3 — BELIEF PROPAGATION            (manuscript eqs 8, 9, 13)
# ==============================================================================
@partial(jax.jit, static_argnames=("steps", "direction"))
def belief_at(m: ChainModel, steps: int, direction: str = "forward") -> jax.Array:
    """Tautomer belief after `steps` propagations from the prior P(T)."""
    return m.P_T @ jnp.linalg.matrix_power(_T(m, direction), steps)


@partial(jax.jit, static_argnames=("steps", "direction", "normalise"))
def marginal(m: ChainModel, steps: int, direction: str = "forward",
             normalise: bool = True) -> jax.Array:
    """P(phi) at a dimer `steps` propagations along the chain (eq 13)."""
    y = belief_at(m, steps, direction) @ m.P_phi_T
    return y / (y.sum() * DPHI) if normalise else y


@partial(jax.jit, static_argnames=("n_monomers", "k", "normalise"))
def avg_marginal(m: ChainModel, n_monomers: int, k: int,
                 normalise: bool = True) -> jax.Array:
    """
    Orientation-averaged marginal at 0-based dimer index k -- manuscript eq 6,

        P_avg(T, k) = 1/2 [ P_fwd(T, k) + P_rev(T, N-1-k) ].

    Note the MIRRORED index on the reverse term: the reverse belief for dimer k
    is read from the position an equal distance from the far end.  Forward steps
    are counted from one terminus, reverse steps from the other.
    """
    N = n_monomers - 1
    yf = marginal(m, k,         "forward", normalise=False)
    yr = marginal(m, N - 1 - k, "reverse", normalise=False)
    y  = 0.5 * (yf + yr)
    return y / (y.sum() * DPHI) if normalise else y


# ==============================================================================
# SECTION 4 — PAIRWISE JOINTS               (manuscript eq 16)
# ==============================================================================
@partial(jax.jit, static_argnames=("i", "j", "direction", "normalise"))
def pairwise_joint(m: ChainModel, i: int, j: int,
                   direction: str = "forward",
                   normalise: bool = True) -> jax.Array:
    """
    P(phi_i, phi_j) for 0-based dimer indices i < j, one reading direction:

        P = sum_{Ti,Tj} P(Ti) [T^(j-i)]_{Ti,Tj} P(phi_i|Ti) P(phi_j|Tj)

    Axis 0 is phi_i, axis 1 is phi_j, where i and j are counted from the anchor
    of THIS direction's walk.

    The transition matrix is contracted BETWEEN the two P(phi|T) factors, so the
    tautomer-level correlation survives.  Building this as
    outer(marginal_i, marginal_j) instead is mathematically guaranteed to yield
    P(phi_i,phi_j) = P(phi_i) P(phi_j) for every pair of angle values, i.e.
    I(phi_i;phi_j) = 0 by construction.  That applies equally to the normalised
    and unnormalised variants: pass normalise=False for the unnormalised figure,
    but do NOT substitute an outer product.
    """
    if not i < j:
        raise ValueError("require i < j")
    T = _T(m, direction)
    w = belief_at(m, i, direction)[:, None] * jnp.linalg.matrix_power(T, j - i)
    joint = m.P_phi_T.T @ (w @ m.P_phi_T)
    joint = jnp.maximum(joint, 0.0)
    return joint / (joint.sum() * DPHI**2) if normalise else joint


@partial(jax.jit, static_argnames=("i", "j", "normalise"))
def orientation_averaged_joint(m: ChainModel, i: int, j: int,
                               normalise: bool = True) -> jax.Array:
    """
    Orientation-averaged pairwise joint over the SAME physical pair of dimers,
    in the axis convention of manuscript eq 6.

        P_avg(phi_i, phi_j) = 1/2 [ P_fwd(phi_i, phi_j) + P_rev(phi_i, phi_j) ]

    ===========================================================================
    THE TRANSPOSE ON THE REVERSE TERM IS NOT COSMETIC.  READ THIS BEFORE
    EDITING.
    ===========================================================================
    ``pairwise_joint`` returns axis 0 = the dimer nearest the anchor of that
    direction's walk.  Reading the chain from the other end swaps which physical
    dimer that is, so the reverse joint must be transposed before averaging.

    If [T_R] were the exact time-reverse of [T_F], the reverse joint would just
    be the forward joint transposed and this would be an empty formality.  It is
    not:  max|jr - jf.T| = 2.92e-04 against a forward peak of 3.92e-04 at n = 20
    -- about 75% of the peak height.  The two joints are genuinely distinct
    objects, so the axis convention is a real modelling choice.

    Without the transpose both terms carry the same axis-0 distribution (the
    prior).  Once the kernels have converged,

        jf ~ P_prior(i) . P_Td(j)          jr ~ P_prior(i) . P_Ta(j)
        1/2 (jf + jr) ~ P_prior(i) . [1/2 P_Td(j) + 1/2 P_Ta(j)]

    which is a PRODUCT, so the mutual information collapses:

        n     I( 1/2(jf+jr) )     I( 1/2(jf+jr.T) )
        5        1.04e-02            2.66e-02
        20       7.41e-07            1.11e-02

    That is the same failure mode as building the joint from
    outer(marginal_i, marginal_j), reached by a different route -- and it
    strikes the one joint that represents the direction-agnostic observable.

    THE INVARIANT THAT PINS IT DOWN: a joint and its marginals must describe the
    same ensemble, so marginalising this joint over axis 1 must reproduce
    ``avg_marginal(m, n, i)``.  With the transpose the residual is 3.5e-18;
    without it, 5.3e-03.  Asserted in run_checks().
    """
    jf = pairwise_joint(m, i, j, "forward", normalise=False)
    jr = pairwise_joint(m, i, j, "reverse", normalise=False)
    J  = jnp.maximum(0.5 * (jf + jr.T), 0.0)
    return J / (J.sum() * DPHI**2) if normalise else J


# ==============================================================================
# SECTION 5 — CUMULATIVE TWIST Phi_rest     (manuscript eqs 17, 18)
# ==============================================================================
def _accumulate(m: ChainModel, seed: jax.Array, T: jax.Array,
                n_steps: int) -> jax.Array:
    """
    FFT circular-convolution accumulator over the running-sum coordinate.

    seed : (4, M) joint over (T_current, phi_anchor), placed at Phi_rest = 0.
    Each of the `n_steps` iterations transitions ONCE and convolves in the
    P(phi|T) of the dimer just moved to, so `n_steps` transitions fold in
    exactly `n_steps` angles -- no more, no fewer.  Off-by-one here is the
    single most likely error in the whole model; see the n=3 structural test in
    run_checks().
    """
    M = N_GRID
    run = jnp.zeros((4, M, M), dtype=jnp.float64)
    run = run.at[:, :, 0].set(seed / DPHI)
    for _ in range(n_steps):
        R = jnp.fft.fft(run, axis=2)
        R = jnp.einsum("ij,ipk->jpk", T, R) * m.P_phi_fft[:, None, :]
        run = jnp.real(jnp.fft.ifft(R, axis=2)) * DPHI
    return run.sum(axis=0)


def _accumulate_last(m: ChainModel, T: jax.Array, N: int) -> jax.Array:
    """
    Exact P(phi_N, phi_1 + ... + phi_{N-1}) for the chain that starts from the
    prior at dimer 1 and propagates with kernel T.

    The anchor is at the END of the walk, so it cannot be carried on a second
    axis from the start the way `_accumulate` does for anchor="first".  Instead
    the N-1 non-anchor angles are accumulated forward in a 1-D running sum, the
    final transition is applied, and the anchor angle is attached on its own axis
    at the last step.

    WHY NOT WALK BACKWARDS WITH T_rev.  The previous implementation seeded with
    belief_at(N-1, direction) and walked back with the opposite kernel.  That is
    wrong for N >= 3: T_rev is the Bayes reverse of T_fwd *at the first step
    only*, because P(T) is not stationary under T_fwd.  The correct backward
    conditional at position k is P(T_k|T_{k+1}) = T[T_k,T_{k+1}] P_k(T_k)/P_{k+1}(T_{k+1})
    with P_k the belief at k, which changes along the chain.  Measured error of
    the old construction against brute force: 5-16 % of peak height for n >= 4,
    exact only at n = 3 where the one-step Bayes identity happens to apply.
    Walking forwards avoids the question entirely.
    """
    R = m.P_T[:, None] * m.P_phi_T                 # (T_1, phi_1); phi_1 in the sum
    for _ in range(N - 2):                         # fold in phi_2 ... phi_{N-1}
        Rf = jnp.fft.fft(R, axis=1)
        R = jnp.real(jnp.fft.ifft(
            jnp.einsum("ij,ik->jk", T, Rf) * m.P_phi_fft, axis=1)) * DPHI
    W = R.T @ T                                    # (sum, T_N) after the last step
    return jnp.einsum("st,ta->as", W, m.P_phi_T)   # (phi_anchor, Phi_rest)


def anchor_phirest_joint(m: ChainModel, n_monomers: int,
                         anchor: str = "first",
                         direction: str = "forward",
                         normalise: bool = True) -> jax.Array:
    """
    P(phi_anchor, Phi_rest), where Phi_rest is the circular sum of the N-1 dimer
    angles that are NOT the anchor.  The two axes never share a term.

        anchor="first" : phi_anchor = phi_1,  Phi_rest = phi_2 + ... + phi_N
        anchor="last"  : phi_anchor = phi_N,  Phi_rest = phi_1 + ... + phi_{N-1}

    The anchor angle is held on its own axis and the remaining N-1 angles are
    folded into the running sum using N-1 transitions.

    Phi = 0 deg corresponds to aligned terminal monomers (trans), Phi = 180 deg
    to opposed (cis).  This equals the relative orientation of distant monomers
    only under an approximately collinear backbone -- stated as a limitation in
    the manuscript, not a result.

    FIX vs the archived script: its anchor="first" branch applied one transition
    before entering the accumulator AND another inside the first iteration, so
    the first angle folded in belonged to dimer 3 rather than dimer 2.  The seed
    below is the untransitioned (T_1, phi_1) joint and all N-1 transitions
    happen inside the accumulator.  ("last" was already correct; it is restated
    here so both branches share one code path.)
    """
    N = n_monomers - 1
    if N < 2:
        raise ValueError("need at least 2 dimers (n_monomers >= 3)")

    T = _T(m, direction)
    if anchor == "first":
        seed = m.P_T[:, None] * m.P_phi_T                # (T_1, phi_1), no transition
        joint = _accumulate(m, seed, T, N - 1)
    elif anchor == "last":
        joint = _accumulate_last(m, T, N)
    else:
        raise ValueError("anchor must be 'first' or 'last'")
    joint = jnp.maximum(joint, 0.0)
    return joint / (joint.sum() * DPHI**2) if normalise else joint


def global_marginal(m: ChainModel, normalise: bool = True) -> jax.Array:
    """
    P(phi) = sum_T P(T) P(phi|T) -- the single-dimer torsional distribution
    before any chain propagation (manuscript eq 4).  Plotted alongside the four
    conditional distributions in Figure 3a.
    """
    y = m.P_T @ m.P_phi_T
    return y / (y.sum() * DPHI) if normalise else y


def taut_profile(m: ChainModel, n_monomers: int) -> np.ndarray:
    """
    Orientation-averaged tautomer belief at every dimer of an n-mer.
    Returns (N, 4): rows are MODEL dimer index k = 0 .. N-1, columns Ta..Td.
    Uses the eq-6 mirrored index, exactly as avg_marginal does.
    """
    N = n_monomers - 1
    out = np.zeros((N, 4), dtype=np.float64)
    for k in range(N):
        f = np.asarray(belief_at(m, k, "forward"), dtype=np.float64)
        r = np.asarray(belief_at(m, N - 1 - k, "reverse"), dtype=np.float64)
        g = 0.5 * (f + r)
        out[k] = g / g.sum()
    return out


def tc_bridge_region(m: ChainModel, n_monomers: int, delta: float = 0.10):
    """
    The conflict zone in which the T_c domain wall resides (manuscript eq 21):
    the span of dimers where |gamma(Ta) - gamma(Td)| < delta.  Returns
    (k_lo, k_hi) as 0-based MODEL dimer indices, or None if no dimer qualifies.
    delta = 0.10 is a presentational choice; the qualitative conclusions -- that
    the bridge exists, contains k*, and is asymmetric -- do not depend on it.
    """
    g = taut_profile(m, n_monomers)
    mask = np.abs(g[:, 0] - g[:, 3]) < delta
    if not mask.any():
        return None
    idx = np.flatnonzero(mask)
    return int(idx.min()), int(idx.max())


def mean_cos2(m: ChainModel, on: str = "scan") -> np.ndarray:
    """
    <cos^2 phi> per tautomer -- the coplanarity parameter of manuscript Table 3
    and Table 1 of the theory document.

        on="scan"  (default, and what the paper reports)
            Periodic-trapezoid integration on the raw 15-degree DFT grid.  This
            is exactly what the parameter spreadsheet computes as AE28/U28, and
            it reproduces the published values to 2e-10 (Tb to 6e-08, limited by
            the stored precision of a ~1e-07 quantity).

        on="grid"
            The same integral evaluated on the model's interpolated 1-degree
            P(phi|T).

    WHY THE TWO DIFFER, AND WHY THAT IS NOT AN INCONSISTENCY.
    P(T) is the integral of P alone, and the trapezoid rule IS the exact integral
    of a linear interpolant -- so P(T), and therefore both transition matrices,
    come out identical on either grid (agreement 4e-10).  The whole propagation
    layer is indifferent to the choice.

    <cos^2> integrates P *times* cos^2, and cos^2 varies within each 15-degree
    interval, so here the two grids genuinely disagree: 0.9515 on the scan grid
    against 0.9413 on the 1-degree grid for T_a, about 1%.

    The reported value is the scan-grid one, because <cos^2> is a property of the
    DFT scan -- a single-dimer descriptor of coplanarity -- and it enters no part
    of the chain model.  Nothing propagates from it.  Reporting it on the scan
    grid keeps it directly checkable against the spreadsheet by hand, which is
    the one artefact a reader can use to validate the first step independently.

    An audit noted that linear interpolation of P across an exponential is the
    weaker interpolant: holding out every other scan point and predicting it back
    from 30-degree spacing gives a median relative error of 7-13 for linear-in-P
    against 0.5-1.8 for linear-in-dG.  That is true, and the correspondingly
    better P(phi|T) would shift <cos^2> to 0.9514 / 0.4703 / 0.8383 / 0.9490.  It
    was not adopted because it would move P(T) and hence both transition matrices
    (T_fwd[Ta->Tc] 0.26470 -> 0.25475), breaking the direct correspondence with
    the spreadsheet for a change in a quantity that feeds nothing.  k* is
    unaffected either way (7.694 under both).
    """
    if on not in ("scan", "grid"):
        raise ValueError("on must be 'scan' or 'grid'")
    if on == "grid":
        c2 = np.cos(np.deg2rad(PHI_GRID)) ** 2
        P = np.asarray(m.P_phi_T, dtype=np.float64)
        return (P * c2).sum(1) * DPHI / (P.sum(1) * DPHI)
    phi = np.asarray(m.scan_phi, dtype=np.float64)
    w = np.asarray(m.scan_w, dtype=np.float64)
    keep = (phi >= -180) & (phi < 180)          # drop the duplicated +180 point
    c2 = np.cos(np.deg2rad(phi[keep])) ** 2
    return (w[:, keep] * c2).sum(1) / w[:, keep].sum(1)


# ---------------------------------------------------------------------------
# Manuscript view
# ---------------------------------------------------------------------------
# The model propagates from the T_a-rich terminus.  The manuscript, the theory
# document and every published figure read the chain from the T_d-rich terminus.
# The mathematics is identical; only the choice of which end is k = 1 differs.
#
#     manuscript [T_F]  ==  code T_rev        manuscript [T_R]  ==  code T_fwd
#     k_paper           ==  N - k_model
#
# manuscript_view() applies both halves of that mapping in one place so that a
# reader comparing the theory document to the code does not have to hold the
# inversion in their head.  It relabels; it computes nothing new.
# ---------------------------------------------------------------------------
class ManuscriptView(NamedTuple):
    T_F     : np.ndarray   # funnels to T_a  (== code T_rev)
    T_R     : np.ndarray   # funnels to T_d  (== code T_fwd)
    P_T     : np.ndarray
    lam2_F  : float
    lam2_R  : float
    p_F     : float        # [T_F](T_c | T_d) -- escape from the T_d basin
    p_R     : float        # [T_R](T_c | T_a) -- escape from the T_a basin


def manuscript_view(m: ChainModel) -> ManuscriptView:
    """The model's directional quantities, relabelled in the manuscript's order."""
    TF = np.asarray(m.T_rev, dtype=np.float64)
    TR = np.asarray(m.T_fwd, dtype=np.float64)
    return ManuscriptView(
        T_F=TF, T_R=TR, P_T=np.asarray(m.P_T, dtype=np.float64),
        lam2_F=float(np.sort(np.abs(np.linalg.eigvals(TF)))[-2]),
        lam2_R=float(np.sort(np.abs(np.linalg.eigvals(TR)))[-2]),
        p_F=float(TF[3, 2]), p_R=float(TR[0, 2]))


def to_paper_index(n_monomers: int, k_model: int) -> int:
    """0-based model dimer index -> 1-based dimer position as printed in the paper."""
    return (n_monomers - 1) - k_model


def to_model_index(n_monomers: int, k_paper: int) -> int:
    """1-based paper dimer position -> 0-based model dimer index."""
    return (n_monomers - 1) - k_paper


def paper_profile(m: ChainModel, n_monomers: int):
    """
    (k_paper, gamma, trans, cis) for an n-mer, ordered as the paper prints them:
    k = 1 is the T_d/cis head and k = N the T_a/trans tail.  This is exactly what
    the profile figures plot.
    """
    N = n_monomers - 1
    g = taut_profile(m, n_monomers)
    tr, ci = cis_trans_profile(m, n_monomers)
    k = np.arange(N, 0, -1)
    return k, g, tr, ci


# ==============================================================================
# SECTION 6 — cis / trans WINDOWS           (manuscript eqs 14, 15)
# ==============================================================================
MASK_TRANS = jnp.asarray((PHI_GRID >= -30) & (PHI_GRID <= 30))
MASK_CIS   = jnp.asarray((PHI_GRID >= 150) | (PHI_GRID <= -150))


def cis_trans(p_norm) -> tuple[float, float]:
    """(P_trans, P_cis) for a normalised 1-D distribution: +/-30 deg windows."""
    p = jnp.asarray(p_norm)
    return (float(jnp.sum(jnp.where(MASK_TRANS, p, 0.0)) * DPHI),
            float(jnp.sum(jnp.where(MASK_CIS,   p, 0.0)) * DPHI))


def cis_trans_profile(m: ChainModel, n_monomers: int):
    """Orientation-averaged (trans, cis) at every dimer of an n-mer."""
    N = n_monomers - 1
    out = [cis_trans(avg_marginal(m, n_monomers, k)) for k in range(N)]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


# ==============================================================================
# SECTION 7 — DERIVED SCALARS
# ==============================================================================
def entropy_1d(p) -> float:
    """Boltzmann conformational entropy of a normalised 1-D distribution, nats
    (manuscript eq 22).  S = -sum p ln p dphi."""
    p = np.asarray(p, dtype=np.float64)
    p = p / (p.sum() * DPHI)
    q = np.maximum(p, 1e-300)
    return float(-(p * np.log(q)).sum() * DPHI)


def entropy_2d(J) -> float:
    """Joint entropy of a 2-D distribution, nats (manuscript eq 23)."""
    J = np.asarray(J, dtype=np.float64)
    J = J / (J.sum() * DPHI**2)
    q = np.maximum(J, 1e-300)
    return float(-(J * np.log(q)).sum() * DPHI**2)


def mutual_information(J) -> float:
    """I(phi_i; phi_j) in nats.  Zero iff the joint is a product."""
    J = np.asarray(J, dtype=np.float64)
    J = J / (J.sum() * DPHI**2)
    a, b = J.sum(1) * DPHI, J.sum(0) * DPHI
    out = np.outer(a, b)
    k = J > 1e-300
    return float((J[k] * np.log(J[k] / np.maximum(out[k], 1e-300))).sum() * DPHI**2)


def escape_probabilities(m: ChainModel) -> dict:
    """
    Per-step probability of transiting into the T_c gateway from each basin
    (manuscript Table 5).

    PERSPECTIVE.  The manuscript reads the chain from the T_d-rich end; this code
    propagates from the T_a-rich end.  The two directional kernels therefore swap
    names between the two conventions:

        manuscript [T_F]  ==  code T_rev        manuscript [T_R]  ==  code T_fwd

    so the manuscript's values are correct as written:

        p_F = [T_F](Tc|Td) = code T_rev[Td->Tc] = 0.40507
        p_R = [T_R](Tc|Ta) = code T_fwd[Ta->Tc] = 0.26470

    (Confirmed against "Transition Matrix flip.xlsx", which tabulates the same
    two matrices in the manuscript's reading order.)  Read with the CODE labels,
    T_fwd[Td->Tc] and T_rev[Ta->Tc] are both exactly zero -- that is a property
    of the labelling, not an error in the paper.
    """
    Tf, Tr = np.asarray(m.T_fwd), np.asarray(m.T_rev)
    return {
        "p_F  manuscript [T_F](Tc|Td) = code T_rev[Td->Tc]": float(Tr[3, 2]),
        "p_R  manuscript [T_R](Tc|Ta) = code T_fwd[Ta->Tc]": float(Tf[0, 2]),
        "     code T_fwd[Td->Tc]  (zero by labelling)": float(Tf[3, 2]),
        "     code T_rev[Ta->Tc]  (zero by labelling)": float(Tr[0, 2]),
    }


def k_star(m: ChainModel, n_monomers: int) -> tuple[float, float]:
    """
    Crossover position where the two funnels' influence is equal
    (manuscript eqs 19, 20).  Returns (k*, fractional position).

        k* = 1 + (N-1) ln(lam_R) / [ ln(lam_F) + ln(lam_R) ]

    with lam the second eigenvalue of each kernel.  The fractional position
    (k*-1)/(N-1) is chain-length independent; measured from the opposite
    terminus it is 1 minus that value, and the two describe the same point.
    """
    N = n_monomers - 1
    lf = float(np.sort(np.abs(np.linalg.eigvals(np.asarray(m.T_fwd))))[-2])
    lr = float(np.sort(np.abs(np.linalg.eigvals(np.asarray(m.T_rev))))[-2])
    frac = np.log(lr) / (np.log(lf) + np.log(lr))
    return 1.0 + (N - 1) * frac, frac


# ==============================================================================
# SECTION 8 — CONSISTENCY CHECKS
# ==============================================================================
def run_checks(m: ChainModel, nmers=(3, 4, 5, 6, 10, 15, 20), tol=1e-12) -> int:
    """
    Marginalisation invariants.  A joint and its marginals must describe the
    same ensemble; every joint the model produces is checked against the
    corresponding marginal computed by an independent code path.

    This is the cheapest test that catches axis-convention errors, and it is the
    test that would have caught the un-transposed orientation average
    immediately.  Run it first.

    Returns the number of failures (0 = all pass).
    """
    fails = []

    def n1(y):
        y = np.asarray(y, dtype=np.float64)
        return y / (y.sum() * DPHI)

    def rep(name, err, t=tol):
        ok = err < t
        if not ok:
            fails.append(name)
        print(f"  {name:<56} {err:>10.3e}  {'PASS' if ok else '**FAIL**'}")

    Tf, Tr, PT = np.asarray(m.T_fwd), np.asarray(m.T_rev), np.asarray(m.P_T)

    print("\n=== 1. inputs are well formed ===")
    rep("T_fwd rows sum to 1", abs(Tf.sum(1) - 1).max())
    rep("T_rev rows sum to 1", abs(Tr.sum(1) - 1).max())
    rep("P(T) sums to 1", abs(PT.sum() - 1))
    rep("P(phi|T) rows integrate to 1",
        abs(np.asarray(m.P_phi_T).sum(1) * DPHI - 1).max())
    Pn = Tf.T @ PT
    rep("one-step Bayes  T_F[i,j]P(i) = T_R[j,i]P'(j)",
        np.abs(Tf * PT[:, None] - (Tr * Pn[:, None]).T).max())
    print(f"  {'P(T) NOT stationary under T_fwd (expected, informational)':<56} "
          f"{np.abs(PT @ Tf - PT).max():>10.3e}")

    print("\n=== 2. single-direction joints vs their own marginals ===")
    worst = 0.0
    for n in nmers:
        N = n - 1
        for dr in ("forward", "reverse"):
            J = np.asarray(pairwise_joint(m, 0, N - 1, dr, True), dtype=np.float64)
            e0 = np.abs(n1(J.sum(1) * DPHI) - np.asarray(marginal(m, 0, dr))).max()
            e1 = np.abs(n1(J.sum(0) * DPHI) - np.asarray(marginal(m, N - 1, dr))).max()
            worst = max(worst, e0, e1)
    rep(f"worst residual over all n and both directions", worst)

    print("\n=== 3. orientation-averaged joint vs eq 6   [KEY REGRESSION] ===")
    for n in nmers:
        N = n - 1
        J = np.asarray(orientation_averaged_joint(m, 0, N - 1, True), dtype=np.float64)
        rep(f"n={n:<3} axis 0 -> avg_marginal(k=0)",
            np.abs(n1(J.sum(1) * DPHI) - np.asarray(avg_marginal(m, n, 0))).max())
        rep(f"n={n:<3} axis 1 -> avg_marginal(k=N-1)",
            np.abs(n1(J.sum(0) * DPHI) - np.asarray(avg_marginal(m, n, N - 1))).max())

    print("\n=== 4. guard: the UN-transposed average must FAIL that test ===")
    print("  These should be LARGE (~1e-03).  A small number here means the")
    print("  transpose has stopped mattering and something else has changed.")
    for n in (5, 20):
        N = n - 1
        jf = np.asarray(pairwise_joint(m, 0, N - 1, "forward", False), dtype=np.float64)
        jr = np.asarray(pairwise_joint(m, 0, N - 1, "reverse", False), dtype=np.float64)
        A = 0.5 * (jf + jr); A = A / (A.sum() * DPHI**2)
        e = np.abs(n1(A.sum(1) * DPHI) - np.asarray(avg_marginal(m, n, 0))).max()
        ok = e > 1e-6
        if not ok:
            fails.append(f"guard n={n}")
        print(f"  n={n:<3} 0.5*(jf+jr) axis 0 vs eq 6 {e:>21.3e}  "
              f"{'PASS' if ok else '**FAIL — guard broke**'}")

    print("\n=== 5. mutual information survives the average ===")
    print(f"  {'n':>4} {'I(fwd)':>12} {'I(rev)':>12} {'I(avg)':>12} "
          f"{'I(no transpose)':>16}")
    for n in nmers:
        N = n - 1
        jf = pairwise_joint(m, 0, N - 1, "forward", False)
        jr = pairwise_joint(m, 0, N - 1, "reverse", False)
        ia = mutual_information(orientation_averaged_joint(m, 0, N - 1, False))
        ib = mutual_information(0.5 * (np.asarray(jf) + np.asarray(jr)))
        if ia < 1e-3:
            fails.append(f"I(avg) collapsed at n={n}")
        print(f"  {n:>4} {mutual_information(jf):>12.4e} "
              f"{mutual_information(jr):>12.4e} {ia:>12.4e} {ib:>16.4e}")
    print("  I(avg) must stay >= 1e-3 at every n.")

    print("\n=== 6. anchor / Phi_rest joints ===")
    worst_a, worst_n = 0.0, 0.0
    for n in nmers:
        N = n - 1
        for anc in ("first", "last"):
            J = np.asarray(anchor_phirest_joint(m, n, anc, "forward", True),
                           dtype=np.float64)
            ref = (np.asarray(marginal(m, 0, "forward")) if anc == "first"
                   else np.asarray(marginal(m, N - 1, "forward")))
            worst_a = max(worst_a, np.abs(n1(J.sum(1) * DPHI) - ref).max())
            worst_n = max(worst_n, abs(J.sum() * DPHI**2 - 1))
    rep("anchor axis -> the correct single-dimer marginal", worst_a)
    rep("joint integrates to 1", worst_n)

    print("\n=== 7. structural: at n=3, Phi_rest is ONE angle ===")
    print("  N=2 dimers, one transition, one accumulated angle, so the")
    print("  Phi_rest axis must be identical to the one-step marginal P(phi_2).")
    print("  This is the test that pins the transition count.")
    J = np.asarray(anchor_phirest_joint(m, 3, "first", "forward", True),
                   dtype=np.float64)
    rep("n=3 Phi_rest axis == P(phi_2) exactly",
        np.abs(n1(J.sum(0) * DPHI) - np.asarray(marginal(m, 1, "forward"))).max())

    print("\n=== 8. avg_marginal normalisation along the whole chain ===")
    e = max(abs(float(np.asarray(avg_marginal(m, n, k)).sum() * DPHI) - 1)
            for n in nmers for k in range(n - 1))
    rep("every avg_marginal integrates to 1", e)

    print("\n=== 9. closed form  gamma_1(Tc) = 1/2 P(Tc) ===")
    print("  At each terminus one message is unpropagated, so the mixture rule")
    print("  gives half the prior.  One check that exercises the whole path.")
    g = 0.5 * (float(np.asarray(belief_at(m, 19, "forward"))[2])
               + float(np.asarray(belief_at(m, 0, "reverse"))[2]))
    print(f"  computed {g:.4f}   closed form {0.5 * float(PT[2]):.4f}   "
          f"diff {abs(g - 0.5 * float(PT[2])):.4f}")

    print("\n" + "=" * 72)
    if fails:
        print(f"{len(fails)} FAILURE(S): " + ", ".join(fails))
    else:
        print("All marginalisation checks passed.")
    print("=" * 72)
    return len(fails)


# ==============================================================================
# SECTION 9 — PUBLISHED TABLES
# ==============================================================================
def run_tables(m: ChainModel) -> None:
    """Reproduce every numerical table in the theory document and manuscript."""
    NM = [3, 4, 5, 6, 10, 15, 20]
    Tf, Tr, PT = np.asarray(m.T_fwd), np.asarray(m.T_rev), np.asarray(m.P_T)

    print("\n" + "=" * 72)
    print("TABLE 1 / 3  —  tautomer populations and transition matrices")
    print("=" * 72)
    print("  P(T)  " + "  ".join(f"{n}={v:.6g}" for n, v in zip(TAUT_NAMES, PT)))
    print("\n  [T_F]  (rows = from, cols = to)")
    print("        " + "".join(f"{n:>14}" for n in TAUT_NAMES))
    for i, n in enumerate(TAUT_NAMES):
        print(f"    {n}  " + "".join(f"{Tf[i, j]:>14.8g}" for j in range(4)))
    print("\n  [T_R]")
    print("        " + "".join(f"{n:>14}" for n in TAUT_NAMES))
    for i, n in enumerate(TAUT_NAMES):
        print(f"    {n}  " + "".join(f"{Tr[i, j]:>14.8g}" for j in range(4)))

    print("\n" + "=" * 72)
    print("TABLE 4  —  spectral structure and limiting distributions")
    print("=" * 72)
    for nm, T in (("T_F", Tf), ("T_R", Tr)):
        ev = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]
        lim = PT @ np.linalg.matrix_power(T, 400)
        print(f"  {nm}: eigenvalues {np.array2string(ev, precision=6)}")
        print(f"       limit -> {TAUT_NAMES[int(np.argmax(lim))]} "
              f"({lim.max():.6f})")
    print("  lam_3 = lam_4 = 0 in both: two dynamical modes only "
          "(stationary + one decay).")

    print("\n" + "=" * 72)
    print("TABLE 5  —  escape probabilities  (manuscript perspective)")
    print("=" * 72)
    for k, v in escape_probabilities(m).items():
        print(f"  {k:<30} {v:.6g}")

    print("\n" + "=" * 72)
    print("TABLE 7  —  T_c populations under the two combination rules")
    print("=" * 72)
    print(f"  {'n':>4} {'terminus':>10} {'interior min':>13} {'mean mix':>10} "
          f"{'mean product':>13}")
    for n in (6, 10, 15, 20):
        N = n - 1
        mix, pro = [], []
        for k in range(N):
            a = np.asarray(belief_at(m, k, "forward"))
            b = np.asarray(belief_at(m, N - 1 - k, "reverse"))
            gm = 0.5 * (a + b); gm /= gm.sum()
            gp = a * b / PT;    gp /= gp.sum()
            mix.append(gm[2]); pro.append(gp[2])
        mix, pro = np.array(mix), np.array(pro)
        print(f"  {n:>4} {mix[0]:>10.4f} {mix.min():>13.4f} {mix.mean():>10.4f} "
              f"{pro.mean():>13.4f}")
    print(f"  closed form gamma_1(Tc) = 0.5 P(Tc) = {0.5 * PT[2]:.4f}")

    print("\n" + "=" * 72)
    print("TABLE 8  —  chain-averaged conformational composition")
    print("=" * 72)
    print(f"  {'n':>4} {'<trans>':>9} {'<cis>':>9} {'<other>':>9}")
    for n in (3, 5, 10, 20, 50):
        tr, ci = cis_trans_profile(m, n)
        print(f"  {n:>4} {tr.mean():>9.4f} {ci.mean():>9.4f} "
              f"{1 - tr.mean() - ci.mean():>9.4f}")

    print("\n" + "=" * 72)
    print("TABLE 9  —  cumulative twist Phi_rest  (NOT Phi_sum)")
    print("=" * 72)
    print("  Phi_rest excludes the anchor angle, which is held on its own axis.")
    print("  anchor = phi_1, read forward -- the same object as Figure 4d and as")
    print("  Table 9 of the theory document.  (This printed anchor='first' until")
    print("  the v1.4 fix, so the standalone and the document disagreed by ~0.13")
    print("  nats with no cross-reference between them.)")
    print(f"  {'n':>4} {'N':>4} {'angles in Phi':>14} {'S(Phi_rest)':>12} "
          f"{'P_cis':>8} {'P_trans':>8}")
    mt, mc = np.asarray(MASK_TRANS), np.asarray(MASK_CIS)
    for n in NM:
        N = n - 1
        J = np.asarray(anchor_phirest_joint(m, n, "last", "forward", True),
                       dtype=np.float64)
        q = J.sum(0) * DPHI; q /= q.sum() * DPHI
        print(f"  {n:>4} {N:>4} {N - 1:>14} {entropy_1d(q):>12.4f} "
              f"{q[mc].sum() * DPHI:>8.4f} {q[mt].sum() * DPHI:>8.4f}")
    print("  NOTE: the cis probability is NOT monotonic -- it rises at n=4")
    print("  (0.546 -> 0.614) before decaying.  That is a real short-chain")
    print("  feature of Phi_rest, and differs from the old Phi_sum behaviour.")

    print("\n" + "=" * 72)
    print("TABLE 10  —  crossover position k*")
    print("=" * 72)
    print(f"  {'n':>4} {'k*':>9} {'(k*-1)/(N-1)':>14} {'from far end':>14}")
    for n in (6, 10, 15, 20, 30, 50):
        ks, fr = k_star(m, n)
        print(f"  {n:>4} {ks:>9.3f} {fr:>14.4f} {1 - fr:>14.4f}")
    print("  The fraction is chain-length independent; the two columns sum to")
    print("  1.0000 and describe the same physical point from opposite ends.")

    print("\n" + "=" * 72)
    print("TABLE 11  —  joint entropies S(phi_1, phi_N), nats")
    print("=" * 72)
    print(f"  {'n':>4} {'S_fwd':>9} {'S_rev':>9} {'S_avg':>9} {'JS excess':>11} "
          f"{'S_avg (no transpose)':>21}")
    sa_list, js_list = [], []
    for n in NM:
        N = n - 1
        jf = np.asarray(pairwise_joint(m, 0, N - 1, "forward", False),
                        dtype=np.float64)
        jr = np.asarray(pairwise_joint(m, 0, N - 1, "reverse", False),
                        dtype=np.float64)
        sf, sr = entropy_2d(jf), entropy_2d(jr)
        sa = entropy_2d(orientation_averaged_joint(m, 0, N - 1, False))
        sbad = entropy_2d(0.5 * (jf + jr))
        js = sa - 0.5 * (sf + sr)
        sa_list.append(sa); js_list.append(js)
        print(f"  {n:>4} {sf:>9.4f} {sr:>9.4f} {sa:>9.4f} {js:>11.4f} "
              f"{sbad:>21.4f}")
    print(f"  S_avg monotonically decreasing: {bool(np.all(np.diff(sa_list) < 0))}")
    print(f"  JS excess monotonically increasing: "
          f"{bool(np.all(np.diff(js_list) > 0))}")
    print("  The last column is the superseded convention.  Under it S_avg RISES")
    print("  from n=3 to n=4 (9.7319 -> 9.7346).  That uptick was an artefact of")
    print("  the axis labelling and does not survive the correction.")


# ==============================================================================
# SECTION 10 — PUBLISHED FIGURES
# ==============================================================================
# The block below is inserted VERBATIM from src/hipo/_figblock.py, which is
# also imported by src/hipo/plots.py.  tests/test_figure_parity.py asserts the
# two copies are identical, so a figure fix cannot land in one and miss the
# other.  Do not hand-edit this section; edit _figblock.py and rebuild.
# --- BEGIN _figblock.py ---------------------------------------------------
# ==============================================================================
# PUBLISHED FIGURE SUITE
# ==============================================================================
# One function per published figure, named for the figure it produces.  This
# block is the SINGLE SOURCE for figure code: it is inserted verbatim into both
# src/hipo/plots.py and hipo_model_standalone.py, and tests/test_figure_parity.py
# asserts the two copies are byte-identical.  Do not edit one without the other.
#
# Every function takes (outdir) and returns the path it wrote.
# ------------------------------------------------------------------------------
try:                                   # packaged use: hipo.plots imports this
    from hipo.core import *             # noqa: F401,F403
except ImportError:                     # standalone use: names already in scope
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from pathlib import Path

_BASE_FS = 10
matplotlib.rcParams.update({
    "font.size": _BASE_FS * 1.5, "axes.titlesize": _BASE_FS * 1.5,
    "axes.labelsize": _BASE_FS * 1.5, "xtick.labelsize": _BASE_FS * 1.5,
    "ytick.labelsize": _BASE_FS * 1.5, "legend.fontsize": _BASE_FS * 1.1,
    "figure.titlesize": _BASE_FS * 1.8})

ALL_NMERS = [3, 4, 5, 10, 15, 20]          # chain lengths used in S3
S4_NMERS  = [3, 4, 5, 6, 10, 15]           # chain lengths used in S4

# tautomer colours, shared by every figure
C_TA, C_TB, C_TC, C_TD = "#D32F2F", "#F9A825", "#1565C0", "#2E7D32"
C_GLOBAL = "#4A148C"
C_ANCHOR_F, C_ANCHOR_R = "#7B2D8B", "#F5A800"   # S3 panel f / Fig 4d-e accents

# display window: -90..270 so the 180 deg features sit centrally
_LO, _HI = -90, 270
_ROLL = -int(round((-90 - PHI_GRID[0]) / DPHI))
PHI_PLOT = np.where(np.roll(PHI_GRID, _ROLL) < -90,
                    np.roll(PHI_GRID, _ROLL) + 360, np.roll(PHI_GRID, _ROLL))


def _r1(y):
    return np.roll(np.asarray(y, dtype=np.float64), _ROLL)


def _r2(z):
    z = np.asarray(z, dtype=np.float64)
    return np.roll(np.roll(z, _ROLL, 0), _ROLL, 1)


_GK1 = dict(color="black", linestyle=(0, (5, 4)), linewidth=1.2, zorder=0.5)
_GK2 = dict(color="black", linestyle=(0, (5, 4)), linewidth=1.1, alpha=0.6, zorder=3)


def _guides(ax, which="v"):
    kw = _GK1 if which == "v" else _GK2
    for g in (0, 180):
        ax.axvline(g, **kw)
        if which == "vh":
            ax.axhline(g, **kw)


def _style_x(ax, step=45):
    ax.set_xlabel("φ (°)")
    ax.set_xlim(_LO, _HI)
    ax.set_xticks(range(-90, 271, step))
    ax.axhline(0, color="k", lw=0.4)


_LN_FLOOR = -25.0


def _joint_panel(fig, ax, z, title, xl, yl, cmap, ln=False):
    zz = _r2(z)
    if ln:
        zz = np.log(np.where(zz > 1e-300, zz, 1e-300))
        zz = np.where(zz < _LN_FLOOR, _LN_FLOOR, zz)
        vmin = _LN_FLOOR
    else:
        vmin = None
    cb = fig.colorbar(ax.pcolormesh(PHI_PLOT, PHI_PLOT, zz.T, shading="auto",
                                    cmap=cmap, vmin=vmin), ax=ax)
    cb.set_ticks([])
    _guides(ax, "vh")
    ax.set_title(title, pad=20)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_xlim(_LO, _HI); ax.set_ylim(_LO, _HI)
    ax.set_xticks(range(-90, 271, 90)); ax.set_yticks(range(-90, 271, 90))


def _save(fig, outdir, name):
    out = Path(outdir); out.mkdir(exist_ok=True, parents=True)
    f = out / name
    fig.savefig(f, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {f}")
    return f


# ------------------------------------------------------------------------------
# Chain-position convention for the profile figures (4f, S4)
# ------------------------------------------------------------------------------
# The model propagates from the T_a-rich end; the paper indexes from the
# T_d-rich end.  taut_profile() and cis_trans_profile() return arrays in MODEL
# order (index 0 = T_a-rich).  The published figures plot paper position
#     k_paper = N - k_model,
# i.e. the same arrays against a reversed axis, so k = 1 is the T_d/cis head and
# k = N is the T_a/trans tail.  This is the LABEL inversion declared in core,
# applied to the chain axis instead of the plot title.
# ------------------------------------------------------------------------------
def _paper_positions(N):
    """Paper dimer positions for model indices 0 .. N-1.

    Routed through core.to_paper_index so the conversion has exactly one
    definition.  It previously duplicated the arithmetic inline, which left
    to_paper_index with no caller anywhere in the repository -- a mutation that
    made it return nonsense was invisible.
    """
    return np.array([to_paper_index(N + 1, k) for k in range(N)])


def _profile_panel(m, ax, n, delta=0.10, show_legend=False, label=None):
    N = n - 1
    x = _paper_positions(N)
    g = taut_profile(m, n)
    tr, ci = cis_trans_profile(m, n)

    ax.plot(x, g[:, 0], color=C_TA, lw=1.8, marker="D", ms=6)
    ax.plot(x, g[:, 2], color=C_TC, lw=1.8, marker="^", ms=6)
    ax.plot(x, g[:, 3], color=C_TD, lw=1.8, marker="v", ms=6)
    ax.plot(x, tr, color=C_TA, lw=1.8, ls="--", marker="D", ms=6,
            markerfacecolor="none", markeredgewidth=1.2)
    ax.plot(x, ci, color=C_TD, lw=1.8, ls="--", marker="v", ms=6,
            markerfacecolor="none", markeredgewidth=1.2)

    # T_c bridge region: eq 21, |gamma(Ta) - gamma(Td)| < delta.
    # Only drawn for n >= 10; on shorter chains the criterion is either empty or
    # spans most of the chain, so shading it would overstate the case.
    reg = tc_bridge_region(m, n, delta)
    if n >= 10 and reg is not None:
        lo, hi = reg
        xa, xb = N - hi, N - lo                      # model -> paper positions
        ax.axvspan(xa - 0.5, xb + 0.5, alpha=0.12, color=C_TC, zorder=0)
        ax.annotate("Tc bridge\nregion", xy=(xa - 0.2, 0.93),
                    xycoords=("data", "axes fraction"), fontsize=_BASE_FS,
                    color=C_TC, ha="left", va="top")

    # k*: computed from the second eigenvalues, not hardcoded.
    ks_model, _ = k_star(m, n)
    ks_paper = N - ks_model + 1
    ax.axvline(ks_paper, color=C_TC, lw=1.2, ls="--", alpha=0.85)
    ax.annotate(f"k*={ks_paper:.1f}", xy=(ks_paper, 0.20),
                xycoords=("data", "axes fraction"), xytext=(4, 0),
                textcoords="offset points", fontsize=_BASE_FS * 1.1,
                color=C_TC, va="bottom", ha="left")

    ax.axhline(0, color="k", lw=0.4)
    ax.set_title(label or f"n = {n}  (N = {N} dimers)", pad=20)
    ax.set_xlabel("Dimer position k")
    ax.set_ylabel("Probability")
    ax.set_xlim(0.5, N + 0.5)
    ax.set_xticks(np.arange(1, N + 1) if N <= 10 else np.arange(1, N + 1, 2))
    ax.set_ylim(0, 1)
    # No head/tail annotations: the legend already gives Ta/Td and cis/trans,
    # and the curves themselves identify which end is which.
    if show_legend:
        h = [mlines.Line2D([], [], color=C_TA, lw=1.8, marker="D", ms=6, label="Ta"),
             mlines.Line2D([], [], color=C_TC, lw=1.8, marker="^", ms=6, label="Tc"),
             mlines.Line2D([], [], color=C_TD, lw=1.8, marker="v", ms=6, label="Td"),
             mlines.Line2D([], [], color=C_TA, lw=1.8, ls="--", marker="D", ms=6,
                           mfc="none", mew=1.2, label="trans"),
             mlines.Line2D([], [], color=C_TC, lw=1.2, ls="--", label="k* (Tc bridge)"),
             mlines.Line2D([], [], color=C_TD, lw=1.8, ls="--", marker="v", ms=6,
                           mfc="none", mew=1.2, label="cis")]
        ax.legend(handles=h, loc="upper right", ncol=2, fontsize=_BASE_FS * 0.95)


# ==============================================================================
# MAIN TEXT
# ==============================================================================
def fig_1c(m, outdir="figures"):
    """Figure 1c — relative Gibbs free energy profile of the four IPD tautomers."""
    dgphi = np.asarray(m.dG_phi, dtype=np.float64)
    dg = np.asarray(m.dG, dtype=np.float64)
    x = np.where(dgphi < -90, dgphi + 360, dgphi)
    o = np.argsort(x)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (nm, c, mk) in enumerate([("Ta", C_TA, "D"), ("Tb", C_TB, "o"),
                                     ("Tc", C_TC, "^"), ("Td", C_TD, "v")]):
        ax.plot(x[o], dg[i][o], color=c, lw=2, marker=mk, ms=7, label=nm)
    ax.plot(x[o], dg[4][o], color="0.35", lw=2, ls="--", label="Boltzmann average")
    ax.set_title("ΔG torsional profile — IPD tautomers", pad=20)
    ax.set_ylabel("ΔG (kcal mol⁻¹)")
    _style_x(ax)
    _guides(ax)
    ax.legend(ncol=2)
    fig.tight_layout()
    return _save(fig, outdir, "fig_1c_dG_torsional_profile.png")


def fig_3a(m, outdir="figures"):
    """Figure 3a — conditional P(φ|T) for the four tautomers plus global P(φ)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    step = int(15 / DPHI)
    for i, (nm, c, mk) in enumerate([("Ta", C_TA, "D"), ("Tb", C_TB, "o"),
                                     ("Tc", C_TC, "^"), ("Td", C_TD, "v")]):
        y = _r1(m.P_phi_T[i])
        ax.plot(PHI_PLOT, y, color=c, lw=2,
                label=f"{nm}  [P(T) = {float(m.P_T[i]):.4f}]")
        ax.plot(PHI_PLOT[::step], y[::step], mk, color=c, ls="none", ms=8)
    ax.plot(PHI_PLOT, _r1(global_marginal(m)), color=C_GLOBAL, lw=2.5, ls="--",
            label="P(φ) global  (eq 4)")
    ax.set_title("P(φ | T) and the global torsional distribution", pad=20)
    ax.set_ylabel("Probability density")
    _style_x(ax)
    _guides(ax)
    ax.legend()
    fig.tight_layout()
    return _save(fig, outdir, "fig_3a_conditional_and_global.png")


def fig_4abc(m, outdir="figures", n=20, ln=False, normalised=True):
    """Figure 4a-c — pairwise joint distributions for the n-mer.

    a) reverse-direction, b) forward-direction, c) orientation-averaged.
    Panel c uses orientation_averaged_joint (reverse term transposed).
    """
    N = n - 1
    jf = pairwise_joint(m, 0, N - 1, "forward", normalised)
    jr = pairwise_joint(m, 0, N - 1, "reverse", normalised)
    ja = orientation_averaged_joint(m, 0, N - 1, normalised)
    pre = "ln P" if ln else "P"

    # PANEL ORDER AND ORIENTATION -- read this before editing.
    #
    # `pairwise_joint` always places the walk's PRIOR on axis 0, in both
    # directions, so axis 0 is a different physical dimer in each:
    #     jf (code "forward",  funnels to Td): axis0 = paper phi_N, axis1 = phi_1
    #     jr (code "reverse",  funnels to Ta): axis0 = paper phi_1, axis1 = phi_N
    # The manuscript captions 4a as the reverse direction, which in its own
    # naming is [T_R] = code T_fwd = jf, i.e. the Td-funnelling panel.  So a
    # takes jf and b takes jr -- the opposite of what this function used to do.
    #
    # Every panel then plots x = paper phi_1 and y = paper phi_N, so a reader
    # sees the same physical dimer on the same axis throughout.  That needs a
    # transpose on the two panels whose axis 0 is paper phi_N (a and c); b is
    # already in that orientation.  A transpose is a horizontal flip plus a
    # 90-degree rotation, which is the manual edit that was previously applied
    # to the published figure by hand.
    a_panel = jf.T          # reverse direction, Td-funnelling
    b_panel = jr            # forward direction, Ta-funnelling
    c_panel = ja.T          # orientation-averaged, shares jf's axis convention
    fig = plt.figure(figsize=(19, 6.5))
    fig.suptitle(f"Pairwise joint distributions — {n}-mer   (N = {N} dimers)"
                 f"{'' if normalised else ', unnormalised'}"
                 f"{' , ln scale' if ln else ''}", fontweight="bold")
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.36)
    for col, (z, ttl, cm) in enumerate([
            (a_panel, f"a)  {pre}(φ1,φ{N}) — reverse direction", "viridis"),
            (b_panel, f"b)  {pre}(φ1,φ{N}) — forward direction", "viridis"),
            (c_panel, f"c)  {pre}(φ1,φ{N}) — orientation-averaged", "plasma")]):
        _joint_panel(fig, fig.add_subplot(gs[0, col]), z, ttl,
                     "φ1 (°)", f"φ{N} (°)", cm, ln)
    tag = ("norm" if normalised else "unnorm") + ("_ln" if ln else "_linear")
    return _save(fig, outdir, f"fig_4abc_pairwise_joints_{n}mer_{tag}.png")


def fig_4de(m, outdir="figures", n=20, ln=False):
    """Figure 4d-e — cumulative torsional joints P(φ_anchor, Φ_rest).

    d) forward direction, φ1 anchor.   e) reverse direction, φN anchor.
    """
    N = n - 1
    jf = anchor_phirest_joint(m, n, "last", "forward")
    jr = anchor_phirest_joint(m, n, "last", "reverse")
    # The two panels hold out DIFFERENT anchors, so their Phi sums differ:
    # d) anchors phi_1  -> Phi = phi_2 + ... + phi_N
    # e) anchors phi_N  -> Phi = phi_1 + ... + phi_{N-1}
    # A single shared label printed the d) definition on both panels.
    lab_d = f"φ2+…+φ{N}" if N > 2 else "φ2"
    lab_e = f"φ1+…+φ{N-1}" if N > 2 else "φ1"
    pre = "ln P" if ln else "P"
    fig = plt.figure(figsize=(13, 6.5))
    fig.suptitle(f"Cumulative torsional mapping — {n}-mer   (N = {N} dimers)",
                 fontweight="bold")
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32)
    for col, (z, ttl, lb, ax_lab) in enumerate([
            (jf, f"d)  {pre}(φ1, Φ) — forward, φ1 anchor", lab_d, "φ1"),
            (jr, f"e)  {pre}(φ{N}, Φ) — reverse, φ{N} anchor", lab_e, f"φ{N}")]):
        _joint_panel(fig, fig.add_subplot(gs[0, col]), z, ttl,
                     f"{ax_lab} (°)", f"Φ = {lb} (°)",
                     "plasma" if ln else "viridis", ln)
    return _save(fig, outdir,
                 f"fig_4de_cumulative_joints_{n}mer{'_ln' if ln else ''}.png")


def fig_4f(m, outdir="figures", n=20, delta=0.10):
    """Figure 4f — tautomer and cis/trans profile along the 20-mer chain,
    with the k* marker and the T_c bridge region."""
    fig, ax = plt.subplots(figsize=(11, 7))
    _profile_panel(m, ax, n, delta, show_legend=True,
                   label=f"f)  {n}-mer conformational profile  (N = {n-1} dimers)")
    fig.tight_layout()
    return _save(fig, outdir, f"fig_4f_profile_{n}mer.png")


# ==============================================================================
# SUPPORTING INFORMATION
# ==============================================================================
def fig_S1(m, outdir="figures"):
    """Figure S1 — ΔG and Boltzmann probability per tautomer, panels a-d."""
    dgphi = np.asarray(m.dG_phi, dtype=np.float64)
    dg = np.asarray(m.dG, dtype=np.float64)
    xr = np.where(dgphi < -90, dgphi + 360, dgphi)
    o = np.argsort(xr)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("ΔG and Boltzmann probability distributions per IPD tautomer",
                 fontweight="bold")
    for ax, i, nm, c, lab in zip(axes.ravel(), range(4),
                                 ["Ta", "Tb", "Tc", "Td"],
                                 [C_TA, C_TB, C_TC, C_TD], "abcd"):
        ax.plot(xr[o], dg[i][o], color=c, lw=2, marker="o", ms=6, label="ΔG")
        ax.set_ylabel("ΔG (kcal mol⁻¹)", color=c)
        ax.tick_params(axis="y", labelcolor=c)
        ax.set_title(f"{lab})  {nm}   [P(T) = {float(m.P_T[i]):.4g}]", pad=20)
        _style_x(ax, step=90)
        _guides(ax)
        ax2 = ax.twinx()
        ax2.plot(PHI_PLOT, _r1(m.P_phi_T[i]), color="0.35", lw=2, ls="--",
                 label="P(φ|T)")
        ax2.set_ylabel("P(φ | T)", color="0.35")
        ax2.tick_params(axis="y", labelcolor="0.35")
        ax2.set_ylim(bottom=0)
    fig.tight_layout()
    return _save(fig, outdir, "fig_S1_dG_and_probability.png")


def fig_S3(m, outdir="figures", n_detail=20, nmers=None):
    """
    Figure S3 — marginal distributions across chain lengths.

    a) dimer k1 (Td head)          b) dimer k_{n-1} (Ta tail)
    c) orientation-averaged        d) Φ forward (k1 anchor)
    e) Φ reverse (k_{n-1} anchor)  f) overlay for the n_detail-mer

    Every curve is a CHAIN LENGTH, so every legend entry is n=... .

    Panel c uses the manuscript eq-6 mirrored index,
        P_avg(k) = 1/2 [ P_fwd(k) + P_rev(N-1-k) ].
    The earlier standalone script for this figure used the SAME index for both
    directions.  That makes the k=0 curve equal to the prior for every chain
    length -- a whole family of curves collapsing onto one line, independent of
    n -- and it disagrees with eq 6 by 5.3e-03.  See the note in
    orientation_averaged_joint: same convention error, marginal instead of joint.
    """
    nmers = list(nmers or ALL_NMERS)
    cv = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, 19))
    cp = plt.get_cmap("plasma")(np.linspace(0.15, 0.85, 19))

    head, tail, avg_head, avg_tail = [], [], [], []
    rest_f, rest_r, anc_f, anc_r = {}, {}, {}, {}
    for n in nmers:
        N = n - 1
        head.append(np.asarray(marginal(m, N - 1, "forward")))   # Td/cis head
        tail.append(np.asarray(marginal(m, N - 1, "reverse")))   # Ta/trans tail
        avg_head.append(np.asarray(avg_marginal(m, n, N - 1)))
        avg_tail.append(np.asarray(avg_marginal(m, n, 0)))
        for dr, ra, aa in (("forward", rest_f, anc_f), ("reverse", rest_r, anc_r)):
            J = np.asarray(anchor_phirest_joint(m, n, "last", dr, True),
                           dtype=np.float64)
            a = J.sum(1) * DPHI; a /= a.sum() * DPHI
            r = J.sum(0) * DPHI; r /= r.sum() * DPHI
            aa[n] = a; ra[n] = r

    ymax = max(float(np.asarray(y).max())
               for y in head + tail + avg_head + avg_tail) * 1.62
    # 1.62 is the smallest headroom that clears panel c's two stacked legends,
    # the lower of which is a single column so it reads vertically like the ones
    # in a, b, d and e.  Going higher flattens the peaks in all three panels for
    # no gain.  Applied to
    # all three top panels so a, b and c stay on an identical y-scale and
    # peak heights remain directly comparable across them.
    Nd = n_detail - 1

    fig, axes = plt.subplots(2, 3, figsize=(19, 12))
    fig.suptitle("Marginal distributions across chain lengths", fontweight="bold")

    def _leg(ax):
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0),
                  fontsize=_BASE_FS * 1.1, framealpha=0.9,
                  columnspacing=0.8, handlelength=1.4)

    ax = axes[0, 0]
    for y, n in zip(head, nmers):
        ax.plot(PHI_PLOT, _r1(y), color=cv[n - 2], lw=2, label=f"n={n}")
    ax.set_title("a)  P(φ1) — dimer k₁ (Td/cis head)", pad=20)
    ax.set_ylabel("Probability density"); ax.set_ylim(0, ymax)
    _style_x(ax, 90); _guides(ax); _leg(ax)

    ax = axes[0, 1]
    for y, n in zip(tail, nmers):
        ax.plot(PHI_PLOT, _r1(y), color=cv[n - 2], lw=2, label=f"n={n}")
    ax.set_title("b)  P(φ_N) — dimer k_{n−1} (Ta/trans tail)", pad=20)
    ax.set_ylim(0, ymax); ax.set_yticklabels([])
    _style_x(ax, 90); _guides(ax); _leg(ax)

    # Panel c carries TWO curves per chain length, because the orientation-averaged
    # marginal is not the same at the two ends:
    #     avg(k)   = 1/2 [ fwd(k) + rev(N-1-k) ]      (manuscript eq 6)
    #     avg(0)   = 1/2 [ prior + rev converged to Ta ]
    #     avg(N-1) = 1/2 [ fwd converged to Td + prior ]
    # Those differ (1.15e-02 at n = 20) because [T_F] is not the time-reverse of
    # [T_R], so the averaged profile is NOT symmetric under k -> N-1-k.  Panels a
    # and b each show one end and need one curve each; panel c is the averaged
    # counterpart of BOTH, so it needs two.  Solid = head, dashed = tail, and the
    # second legend says so -- without it the panel looks like a duplication bug.
    ax = axes[0, 2]
    for y1, y2, n in zip(avg_head, avg_tail, nmers):
        ax.plot(PHI_PLOT, _r1(y1), color=cp[n - 2], lw=2.2, ls="-", label=f"n={n}")
        ax.plot(PHI_PLOT, _r1(y2), color=cp[n - 2], lw=2.2, ls=(0, (6, 3)))
    ax.set_title("c)  orientation-averaged, both ends", pad=20)
    ax.set_ylim(0, ymax); ax.set_yticklabels([])
    _style_x(ax, 90); _guides(ax)
    # Two stacked, centred legends: the line-style key on top (which end of the
    # chain), the chain-length key beneath it (which colour).  ax.legend()
    # replaces any previous legend, so the first is re-added as an artist.  The
    # headroom for both comes from the shared ymax set above.
    _end_legend = ax.legend(
        handles=[mlines.Line2D([], [], color="0.25", lw=2.2, ls="-",
                               label="k$_1$ (Td head)"),
                 mlines.Line2D([], [], color="0.25", lw=2.2, ls=(0, (6, 3)),
                               label="k$_{n-1}$ (Ta tail)")],
        loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=2,
        fontsize=_BASE_FS * 0.85, framealpha=0.9, handlelength=2.6,
        columnspacing=0.9, labelspacing=0.25, borderpad=0.35)
    ax.add_artist(_end_legend)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.905), ncol=1,
              fontsize=_BASE_FS * 0.95, framealpha=0.9,
              columnspacing=0.8, handlelength=1.4, labelspacing=0.18,
              borderpad=0.35)

    ax = axes[1, 0]
    for n in nmers:
        ax.plot(PHI_PLOT, _r1(rest_f[n]), color=cv[n - 2], lw=2, label=f"n={n}")
    ax.set_title("d)  Φ = φ2+…+φ_N   (forward, k₁ anchor)", pad=20)
    ax.set_ylabel("Probability density  P(Φ)"); ax.set_ylim(bottom=0)
    _style_x(ax, 90); _guides(ax); _leg(ax)

    ax = axes[1, 1]
    for n in nmers:
        ax.plot(PHI_PLOT, _r1(rest_r[n]), color=cv[n - 2], lw=2, label=f"n={n}")
    ax.set_title("e)  Φ = φ1+…+φ_{N−1}   (reverse, k_{n−1} anchor)", pad=20)
    ax.set_ylim(bottom=0); ax.set_yticklabels([])
    _style_x(ax, 90); _guides(ax); _leg(ax)

    ax = axes[1, 2]
    ax.plot(PHI_PLOT, _r1(anc_f[n_detail]), color=C_ANCHOR_F, lw=2, label="φ1")
    ax.plot(PHI_PLOT, _r1(rest_f[n_detail]), color=C_ANCHOR_F, lw=2, ls="--",
            label="Φ$_F$")
    ax.plot(PHI_PLOT, _r1(anc_r[n_detail]), color=C_ANCHOR_R, lw=2,
            label=f"φ{Nd}")
    ax.plot(PHI_PLOT, _r1(rest_r[n_detail]), color=C_ANCHOR_R, lw=2, ls="--",
            label="Φ$_R$")
    ax.set_title(f"f)  n={n_detail}  |  anchors vs cumulative Φ", pad=20)
    ax.set_ylabel("Probability density"); ax.set_ylim(bottom=0)
    _style_x(ax, 90); _guides(ax)
    # Two columns of two, symbols only: the spelled-out "Φ forward"/"Φ reverse"
    # made this legend wide enough to sit on top of the φ1 peak.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              fontsize=_BASE_FS * 1.1, framealpha=0.9,
              columnspacing=1.2, handlelength=1.6, labelspacing=0.3)

    fig.tight_layout()
    return _save(fig, outdir, "fig_S3_marginals_across_chain_lengths.png")


def fig_S4(m, outdir="figures", nmers=None, delta=0.10):
    """Figure S4 — tautomer and cis/trans profiles for n = 3, 4, 5, 6, 10, 15,
    with the k* marker and, for n >= 10, the T_c bridge region."""
    nmers = list(nmers or S4_NMERS)
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    fig.suptitle("Tautomer and cis/trans probability profiles along the chain\n"
                 "P(trans) = P(φ within ±30° of 0°)    "
                 "P(cis) = P(φ within ±30° of 180°)", fontweight="bold")
    for ax, n, lab in zip(axes.ravel(), nmers, "abcdef"):
        _profile_panel(m, ax, n, delta, show_legend=(lab == "a"),
                       label=f"{lab})  n = {n}   (N = {n-1} dimers)")
    fig.tight_layout()
    return _save(fig, outdir, "fig_S4_profiles_by_chain_length.png")


def fig_S5(m, outdir="figures", n_max=50):
    """Figure S5 — a) convergence of chain-averaged cis/trans with chain length,
    b) saturation of the terminal Ta and Td beliefs at opposing chain ends."""
    # Pure-numpy path.  avg_marginal and belief_at are jitted on static (n, k),
    # so sweeping n = 3..50 through them would trigger ~1200 recompilations.
    # The beliefs are just repeated 4x4 matrix-vector products; precompute each
    # power sequence once and reuse.  Verified against taut_profile below.
    Tf = np.asarray(m.T_fwd, dtype=np.float64)
    Tr = np.asarray(m.T_rev, dtype=np.float64)
    Pp = np.asarray(m.P_phi_T, dtype=np.float64)
    P0 = np.asarray(m.P_T, dtype=np.float64)
    mt, mc = np.asarray(MASK_TRANS), np.asarray(MASK_CIS)

    bf = [P0.copy()]
    br = [P0.copy()]
    for _ in range(n_max):
        bf.append(bf[-1] @ Tf)
        br.append(br[-1] @ Tr)

    ns = np.arange(3, n_max + 1)
    tr_avg, ci_avg, ta_end, td_end = [], [], [], []
    for n in ns:
        N = n - 1
        g = np.array([0.5 * (bf[k] + br[N - 1 - k]) for k in range(N)])
        g /= g.sum(1, keepdims=True)
        y = g @ Pp
        y /= (y.sum(1, keepdims=True) * DPHI)
        tr_avg.append(float((y[:, mt].sum(1) * DPHI).mean()))
        ci_avg.append(float((y[:, mc].sum(1) * DPHI).mean()))
        ta_end.append(float(g[0, 0]))    # model index 0     = Ta/trans tail
        td_end.append(float(g[-1, 3]))   # model index N - 1 = Td/cis head

    # cross-check the fast path against the jitted one at a single chain length
    _g = taut_profile(m, 20)
    _y0, _y1 = np.array([0.5 * (bf[k] + br[18 - k]) for k in range(19)]), _g
    _y0 /= _y0.sum(1, keepdims=True)
    assert np.abs(_y0 - _y1).max() < 1e-12, "S5 fast path disagrees with taut_profile"
    tr_avg, ci_avg = np.array(tr_avg), np.array(ci_avg)
    ta_end, td_end = np.array(ta_end), np.array(td_end)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
    fig.suptitle("Convergence with chain length", fontweight="bold")

    ax = axes[0]
    ax.plot(ns, tr_avg, "-", color=C_TA, lw=2.2, label="⟨trans⟩")
    ax.plot(ns, ci_avg, "-", color=C_TD, lw=2.2, label="⟨cis⟩")
    ax.plot(ns, 1 - tr_avg - ci_avg, ":", color="0.5", lw=1.8, label="⟨other⟩")
    ax.axhline(tr_avg[-1], color=C_TA, lw=1.0, ls="--", alpha=0.6)
    ax.axhline(ci_avg[-1], color=C_TD, lw=1.0, ls="--", alpha=0.6)
    ax.annotate(f"{ci_avg[-1]:.3f}", xy=(ns[-1], ci_avg[-1]), xytext=(-6, 6),
                textcoords="offset points", ha="right", color=C_TD,
                fontsize=_BASE_FS * 1.1)
    ax.annotate(f"{tr_avg[-1]:.3f}", xy=(ns[-1], tr_avg[-1]), xytext=(-6, -14),
                textcoords="offset points", ha="right", color=C_TA,
                fontsize=_BASE_FS * 1.1)
    ax.set_title("a)  chain-averaged cis/trans", pad=20)
    ax.set_xlabel("chain length n"); ax.set_ylabel("probability")
    ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend()

    ax = axes[1]
    ax.plot(ns, td_end, "-", color=C_TD, lw=2.2, marker="v", ms=4,
            markevery=4, label="Td at the Td/cis head")
    ax.plot(ns, ta_end, "-", color=C_TA, lw=2.2, marker="D", ms=4,
            markevery=4, label="Ta at the Ta/trans tail")
    ax.axhline(td_end[-1], color=C_TD, lw=1.0, ls="--", alpha=0.6)
    ax.axhline(ta_end[-1], color=C_TA, lw=1.0, ls="--", alpha=0.6)
    ax.annotate(f"{ta_end[-1]:.4f}", xy=(ns[-1], ta_end[-1]), xytext=(-6, 6),
                textcoords="offset points", ha="right", color=C_TA,
                fontsize=_BASE_FS * 1.1)
    ax.annotate(f"{td_end[-1]:.4f}", xy=(ns[-1], td_end[-1]), xytext=(-6, 6),
                textcoords="offset points", ha="right", color=C_TD,
                fontsize=_BASE_FS * 1.1)
    ax.set_title("b)  terminal tautomer belief", pad=20)
    ax.set_xlabel("chain length n"); ax.set_ylabel("belief γ(T)")
    ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend()

    fig.tight_layout()
    return _save(fig, outdir, "fig_S5_convergence.png")


# ------------------------------------------------------------------------------
def run_figures(m, outdir="figures", n=20):
    """Write the complete published figure set."""
    fig_1c(m, outdir)
    fig_3a(m, outdir)
    fig_4abc(m, outdir, n)
    fig_4abc(m, outdir, n, ln=True)
    fig_4de(m, outdir, n)
    fig_4de(m, outdir, n, ln=True)
    fig_4f(m, outdir, n)
    fig_S1(m, outdir)
    fig_S3(m, outdir, n_detail=n)
    fig_S4(m, outdir)
    fig_S5(m, outdir)
# --- END _figblock.py -----------------------------------------------------

# ==============================================================================
# SECTION 11 — FIGURE-TO-FUNCTION MAP
# ==============================================================================
FIGURE_MAP = """
Published item                Function in this file
----------------------------------------------------------------------------
MAIN TEXT
Fig. 1c   dG torsional        fig_1c(m)
Fig. 3a   P(phi|T) + global   fig_3a(m)
Fig. 4a-c pairwise joints     fig_4abc(m, n=20)
Fig. 4d-e cumulative joints   fig_4de(m, n=20)
Fig. 4f   20-mer profile      fig_4f(m, n=20)          k* + Tc bridge

SUPPORTING INFORMATION
Fig. S1   dG and P per taut.  fig_S1(m)
Fig. S3   marginals vs n      fig_S3(m, n_detail=20)   legends read n=...
Fig. S4   profiles vs n       fig_S4(m)                k* + Tc bridge
Fig. S5   convergence         fig_S5(m, n_max=50)

HIPO MODEL THEORY DOCUMENT  (repository only -- NOT the manuscript SI)
Tables 1,3 P(T), [T_F], [T_R]     run_tables() -> TABLE 1/3
Table 4    eigenvalues, limits    run_tables() -> TABLE 4
Table 5    escape probabilities   escape_probabilities(m)
Table 7    T_c populations        run_tables() -> TABLE 7
Table 8    chain averages         run_tables() -> TABLE 8
Table 9    Phi_rest entropy       run_tables() -> TABLE 9
Table 10   crossover k*           k_star(m, n)
Table 11   joint entropies        run_tables() -> TABLE 11

Indexing: n = MONOMERS, N = n - 1 = DIMERS.  Function arguments i, j, k are
0-based dimer indices; figure labels phi_1 .. phi_N are 1-based.  The terminal
pair of a 20-mer is pairwise_joint(m, 0, 18, ...), labelled P(phi_1, phi_19).

Direction: the paper reads from the T_d-rich end, the code propagates from the
T_a-rich end, so manuscript [T_F] == code T_rev and manuscript [T_R] == code
T_fwd.  LABEL applies the inversion to plot titles; the profile figures apply it
to the chain axis (k_paper = N - k_model).
"""


# ==============================================================================
# SECTION 12 — ENTRY POINT
# ==============================================================================
def main(argv) -> int:
    what = argv[1] if len(argv) > 1 else "all"
    if what not in ("checks", "tables", "figures", "map", "all"):
        print(__doc__.split("USAGE")[1].split("---")[0])
        return 2

    print("=" * 72)
    print("HIPO conformation model — standalone reference implementation")
    print(f"jax {jax.__version__}   x64 enabled: {jax.config.jax_enable_x64}")
    print(f"grid: {N_GRID} points, dphi = {DPHI} deg")
    print("=" * 72)

    m = build_model()
    rc = 0
    if what in ("checks", "all"):
        rc = run_checks(m)
    if what in ("tables", "all"):
        run_tables(m)
    if what in ("figures", "all"):
        print("\nFigures:")
        run_figures(m)
    if what in ("map", "all"):
        print(FIGURE_MAP)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
