"""
HIPO conformation model — corrected JAX core
============================================

Single source of truth for the imidazopyridine chain-conformation model.
Everything downstream (pairwise joints, anchor/Phi_rest joints, cis/trans
profiles) is built from this module so the four analysis scripts cannot
drift apart again.

Changes relative to the earlier scripts
---------------------------------------
1.  float64 throughout (jax_enable_x64).  The previous polymer_conformation
    scripts cast to float32 at the JAX boundary, which capped accuracy at
    ~1e-7 and made row sums come out as 1.00000003.
2.  Unnormalised pairwise joints are built from the *correlated* construction
    (tautomer transition matrix contracted between the two P(phi|T) factors),
    not from outer(marginal_i, marginal_j).  The outer product forces
    I(phi_i; phi_j) = 0 by construction.
3.  The anchor/Phi_rest accumulator no longer applies an extra transition
    before the first convolution.
4.  One orientation-average convention: the reverse belief is read from the
    mirrored dimer index (n - 1 - k), matching eq 6 of the manuscript.
5.  One labelling convention, declared once in LABEL.
6.  The orientation-averaged pairwise joint transposes the reverse term before
    averaging (orientation_averaged_joint).  Averaging without the transpose
    puts the prior on axis 0 of both terms, the mixture factorises, and the
    mutual information of the averaged joint collapses to ~0.

Author: Octavio Miranda, Department of Chemistry, Texas A&M University
Licence: <e.g. MIT>
Cite: <manuscript DOI once assigned>
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import NamedTuple
from pathlib import Path

# ---------------------------------------------------------------------------
# Labelling convention  (declare once, use everywhere — console AND figures)
# ---------------------------------------------------------------------------
# The paper reads the chain from the Td-rich end towards the Ta-rich end,
# which is the opposite of the internal head->tail propagation direction.
# Internal "forward" (T_fwd) therefore appears as "Reverse (tail->head)" in
# all printed and plotted output.  The mathematics is unaffected: this is a
# choice of which chain end is called k_1.
LABEL = {
    "forward": "Reverse (tail→head)",
    "reverse": "Forward (head→tail)",
}

N_GRID = 360
PHI_GRID = np.linspace(-180.0, 180.0, N_GRID, endpoint=False)
DPHI = float(PHI_GRID[1] - PHI_GRID[0])
TAUT_NAMES = ["Ta", "Tb", "Tc", "Td"]


class ChainModel(NamedTuple):
    phi_grid : jax.Array   # (M,)
    P_phi_T  : jax.Array   # (4, M)  P(phi | T), each row integrates to 1
    P_T      : jax.Array   # (4,)    dimer tautomer populations
    T_fwd    : jax.Array   # (4, 4)  P(T_{N+1} | T_N)
    T_rev    : jax.Array   # (4, 4)  P(T_N | T_{N+1}), Bayes at one step
    P_phi_fft: jax.Array   # (4, M)  complex, precomputed for convolutions
    scan_phi : jax.Array   # (25,)   raw 15-deg DFT scan angles
    scan_w   : jax.Array   # (4, 25) raw (unnormalised) Boltzmann weights
    dG_phi   : jax.Array   # (25,)   raw scan angles for the energy profile
    dG       : jax.Array   # (5, 25) dG(kcal/mol): Ta,Tb,Tc,Td,avg  (figs 1c, S1)


# ---------------------------------------------------------------------------
# DFT-derived input data (loaded from data/, not hardcoded)
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _load_inputs(data_dir: Path = None):
    """Read the torsional scans and tautomer parameters from data/*.csv."""
    d = Path(data_dir) if data_dir is not None else _DATA_DIR
    def _rows(path):
        """Strip comments/blank lines, return (header, list-of-rows)."""
        out = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    out.append(line.split(","))
        return out[0], out[1:]

    hdr, body = _rows(d / "torsional_scans.csv")
    col = {name: idx for idx, name in enumerate(hdr)}
    phi_raw = np.array([float(r[col["phi_deg"]]) for r in body])
    pphi_raw = {n: np.array([float(r[col[n]]) for r in body]) for n in TAUT_NAMES}

    hdr, body = _rows(d / "torsional_energies.csv")
    col = {name: idx for idx, name in enumerate(hdr)}
    dg_raw = {n: np.array([float(r[col[n]]) for r in body])
              for n in TAUT_NAMES + ["avg"]}
    dg_phi = np.array([float(r[col["phi_deg"]]) for r in body])

    _, body = _rows(d / "tautomer_parameters.csv")
    rows = {r[0]: np.array([float(v) for v in r[1:]]) for r in body}
    p_t = rows["P_T"]
    t_fwd = np.vstack([rows[f"T_fwd_from_{n}"] for n in TAUT_NAMES])
    return phi_raw, pphi_raw, p_t, t_fwd, dg_phi, dg_raw


def build_model(data_dir=None) -> ChainModel:
    phi_raw, pphi_raw, p_t_raw, t_fwd_raw, dg_phi, dg_raw = _load_inputs(data_dir)

    P_phi_T = np.zeros((4, N_GRID), dtype=np.float64)
    for i, nm in enumerate(TAUT_NAMES):
        y = np.interp(PHI_GRID, phi_raw, pphi_raw[nm], period=360.0)
        y = np.maximum(y, 0.0)
        P_phi_T[i] = y / (y.sum() * DPHI)

    P_T = p_t_raw / p_t_raw.sum()
    T_fwd = t_fwd_raw / t_fwd_raw.sum(axis=1, keepdims=True)

    # Reverse kernel by Bayes at a single step:
    #   P(T_N | T_{N+1}) = P(T_{N+1} | T_N) P(T_N) / P(T_{N+1})
    # NOTE: P_T is *not* stationary under T_fwd, so T_rev**s is not the exact
    # reverse of T_fwd**s for s >= 2.  See tests/test_bayes.py.
    P_next = T_fwd.T @ P_T
    T_rev = ((T_fwd * P_T[:, None]) / (P_next[None, :] + 1e-300)).T

    return ChainModel(
        phi_grid  = jnp.asarray(PHI_GRID),
        P_phi_T   = jnp.asarray(P_phi_T),
        P_T       = jnp.asarray(P_T),
        T_fwd     = jnp.asarray(T_fwd),
        T_rev     = jnp.asarray(T_rev),
        P_phi_fft = jnp.fft.fft(jnp.asarray(P_phi_T), axis=1),
        scan_phi  = jnp.asarray(phi_raw),
        scan_w    = jnp.asarray(np.vstack([pphi_raw[n] for n in TAUT_NAMES])),
        dG_phi    = jnp.asarray(dg_phi),
        dG        = jnp.asarray(np.vstack([dg_raw[n] for n in TAUT_NAMES + ["avg"]])),
    )


def _T(m: ChainModel, direction: str) -> jax.Array:
    return m.T_fwd if direction == "forward" else m.T_rev


# ---------------------------------------------------------------------------
# Belief propagation
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("steps", "direction"))
def belief_at(m: ChainModel, steps: int, direction: str = "forward") -> jax.Array:
    """Tautomer belief after `steps` propagations from the prior P(T)."""
    return m.P_T @ jnp.linalg.matrix_power(_T(m, direction), steps)


@partial(jax.jit, static_argnames=("steps", "direction", "normalise"))
def marginal(m: ChainModel, steps: int, direction: str = "forward",
             normalise: bool = True) -> jax.Array:
    """P(phi) at a dimer `steps` propagations along the chain."""
    y = belief_at(m, steps, direction) @ m.P_phi_T
    return y / (y.sum() * DPHI) if normalise else y


@partial(jax.jit, static_argnames=("n_monomers", "k", "normalise"))
def avg_marginal(m: ChainModel, n_monomers: int, k: int,
                 normalise: bool = True) -> jax.Array:
    """
    Orientation-averaged marginal at 0-based dimer index k, manuscript eq 6.

    Forward belief is read at k; reverse belief at the MIRRORED index
    (n - 1) - 1 - k = n - 2 - k, i.e. the same distance from the far end.
    """
    N = n_monomers - 1
    yf = marginal(m, k,           "forward", normalise=False)
    yr = marginal(m, N - 1 - k,   "reverse", normalise=False)
    y  = 0.5 * (yf + yr)
    return y / (y.sum() * DPHI) if normalise else y


# ---------------------------------------------------------------------------
# Pairwise joint  P(phi_i, phi_j)
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("i", "j", "direction", "normalise"))
def pairwise_joint(m: ChainModel, i: int, j: int,
                   direction: str = "forward",
                   normalise: bool = True) -> jax.Array:
    """
    P(phi_i, phi_j) for 0-based dimer indices i < j.

        P = sum_{Ti,Tj} P(Ti) [T^(j-i)]_{Ti,Tj} P(phi_i|Ti) P(phi_j|Tj)

    The transition matrix is contracted BETWEEN the two P(phi|T) factors, so
    the tautomer-level correlation survives.  Building this as
    outer(marginal_i, marginal_j) instead forces I(phi_i; phi_j) = 0 --
    that is the bug this function exists to avoid, and it applies equally to
    the normalised and unnormalised variants.  Pass normalise=False for the
    unnormalised figure; do NOT substitute an outer product.
    """
    assert i < j, "require i < j"
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

    THE TRANSPOSE ON THE REVERSE TERM IS NOT COSMETIC.  ``pairwise_joint``
    returns axis 0 = the dimer nearest the anchor of that direction's walk.
    Reading the chain from the other end swaps which physical dimer that is,
    so the reverse joint must be transposed before it is averaged.

    Without the transpose both terms carry the same axis-0 distribution (the
    prior).  Once the two kernels have converged the average then factorises,

        1/2 (jf + jr) ~ P_prior(i) . [1/2 P_Td(j) + 1/2 P_Ta(j)]

    which is a product, so I(phi_i; phi_j) collapses to ~0 (7.4e-07 at n = 20
    against 1.1e-02 for the correct average).  That is the same failure mode as
    building the joint from outer(marginal_i, marginal_j), reached by a
    different route.

    [T_R] is NOT the time-reverse of [T_F] -- P(T) is not stationary under
    [T_F] -- so jr != jf.T (they differ by ~75% of the peak height).  Both
    kernels are primary conditional probabilities read off the trimer topology,
    which is precisely why the axis convention has to be stated explicitly
    rather than inferred.

    Regression invariant (tests/test_marginalisation.py): marginalising this
    joint over axis 1 must reproduce ``avg_marginal(m, n, i)`` exactly.
    """
    jf = pairwise_joint(m, i, j, "forward", normalise=False)
    jr = pairwise_joint(m, i, j, "reverse", normalise=False)
    J  = jnp.maximum(0.5 * (jf + jr.T), 0.0)
    return J / (J.sum() * DPHI**2) if normalise else J


# ---------------------------------------------------------------------------
# Anchor / Phi_rest joint  P(phi_anchor, sum of the other angles)
# ---------------------------------------------------------------------------
def _accumulate(m: ChainModel, seed: jax.Array, T: jax.Array,
                n_steps: int) -> jax.Array:
    """
    FFT circular-convolution accumulator.

    seed : (4, M) joint over (T_current, phi_anchor) placed at Phi_rest = 0.
    Each of the `n_steps` iterations transitions once and convolves in the
    P(phi|T) of the dimer just moved to, so `n_steps` transitions add exactly
    `n_steps` angles.
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
    P(phi_anchor, Phi_rest) where Phi_rest is the circular sum of the N-1
    dimer angles that are NOT the anchor.  Axes never share a term.

      anchor="first" : phi_anchor = phi_1,  Phi_rest = phi_2 + ... + phi_N
      anchor="last"  : phi_anchor = phi_N,  Phi_rest = phi_1 + ... + phi_{N-1}

    FIX vs the earlier script: the "first" branch previously applied one
    transition before entering the accumulator AND another inside its first
    iteration, so the first angle folded in belonged to dimer 3 rather than
    dimer 2.  The chain was over-propagated by exactly one step.  The seed
    below is the untransitioned (T_1, phi_1) joint, and all N-1 transitions
    happen inside the accumulator.  ("last" was already correct; it is
    restated here so both branches share one code path.)
    """
    N = n_monomers - 1
    assert N >= 2, "need at least 2 dimers"

    T = _T(m, direction)
    if anchor == "first":
        seed = m.P_T[:, None] * m.P_phi_T          # (T_1, phi_1), no transition
        joint = _accumulate(m, seed, T, N - 1)
    elif anchor == "last":
        joint = _accumulate_last(m, T, N)
    else:
        raise ValueError("anchor must be 'first' or 'last'")
    joint = jnp.maximum(joint, 0.0)
    return joint / (joint.sum() * DPHI**2) if normalise else joint


# ---------------------------------------------------------------------------
# Global torsional distribution  (manuscript eqs 3-4, Figure 3a)
# ---------------------------------------------------------------------------
def global_marginal(m: ChainModel, normalise: bool = True) -> jax.Array:
    """
    P(phi) = sum_T P(T) P(phi|T) -- the single-dimer torsional distribution
    before any chain propagation.  Manuscript eq 4; plotted alongside the four
    conditional distributions in Figure 3a.
    """
    y = m.P_T @ m.P_phi_T
    return y / (y.sum() * DPHI) if normalise else y


# ---------------------------------------------------------------------------
# Tautomer belief profile along the chain  (figs 4f, S4)
# ---------------------------------------------------------------------------
def taut_profile(m: ChainModel, n_monomers: int) -> np.ndarray:
    """
    Orientation-averaged tautomer belief at every dimer of an n-mer.

    Returns (N, 4): rows are dimer index k = 0 .. N-1, columns Ta, Tb, Tc, Td.
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
    the contiguous span of dimers where |gamma(Ta) - gamma(Td)| < delta.

    Returns (k_lo, k_hi) as 0-based dimer indices, or None if no dimer
    satisfies the criterion (short chains).  delta = 0.10 is a presentational
    choice; the qualitative conclusions do not depend on it.
    """
    g = taut_profile(m, n_monomers)
    mask = np.abs(g[:, 0] - g[:, 3]) < delta
    if not mask.any():
        return None
    idx = np.flatnonzero(mask)
    return int(idx.min()), int(idx.max())


def k_star(m: ChainModel, n_monomers: int):
    """
    Crossover position where the two funnels' influence is equal
    (manuscript eqs 19-20).  Returns (k_star_1based, fractional_position).

        k* = 1 + (N-1) * ln(lam_R) / [ ln(lam_F) + ln(lam_R) ]

    lam is the second eigenvalue of each kernel.  The fraction is
    chain-length independent.  NOTE both conventions are in circulation:
    this returns 0.6281, and 1 - 0.6281 = 0.3719 is the same point measured
    from the opposite terminus.
    """
    N = n_monomers - 1
    lf = float(np.sort(np.abs(np.linalg.eigvals(np.asarray(m.T_fwd))))[-2])
    lr = float(np.sort(np.abs(np.linalg.eigvals(np.asarray(m.T_rev))))[-2])
    frac = np.log(lr) / (np.log(lf) + np.log(lr))
    return 1.0 + (N - 1) * frac, frac


def escape_probabilities(m: ChainModel) -> dict:
    """
    Per-step probability of transiting into the T_c gateway (manuscript Table 5).

    PERSPECTIVE.  The manuscript reads the chain from the T_d-rich end; this code
    propagates from the T_a-rich end, so the kernels swap names:

        manuscript [T_F] == code T_rev        manuscript [T_R] == code T_fwd

        p_F = [T_F](Tc|Td) = code T_rev[Td->Tc] = 0.40507
        p_R = [T_R](Tc|Ta) = code T_fwd[Ta->Tc] = 0.26470

    Read with CODE labels, T_fwd[Td->Tc] and T_rev[Ta->Tc] are both exactly zero;
    that is the labelling, not an error.  See LABEL at the top of this module.
    """
    Tf, Tr = np.asarray(m.T_fwd), np.asarray(m.T_rev)
    return {"p_F": float(Tr[3, 2]), "p_R": float(Tf[0, 2])}


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


# ---------------------------------------------------------------------------
# cis / trans windows
# ---------------------------------------------------------------------------
MASK_TRANS = jnp.asarray((PHI_GRID >= -30) & (PHI_GRID <= 30))
MASK_CIS   = jnp.asarray((PHI_GRID >= 150) | (PHI_GRID <= -150))


def cis_trans(p_norm: jax.Array):
    """(P_trans, P_cis) for a normalised 1-D distribution."""
    return (float(jnp.sum(jnp.where(MASK_TRANS, p_norm, 0.0)) * DPHI),
            float(jnp.sum(jnp.where(MASK_CIS,   p_norm, 0.0)) * DPHI))


def cis_trans_profile(m: ChainModel, n_monomers: int):
    """Orientation-averaged (trans, cis) at every dimer of an n-mer."""
    N = n_monomers - 1
    out = [cis_trans(avg_marginal(m, n_monomers, k)) for k in range(N)]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


MODEL = build_model()


__all__ = [
    "ChainModel", "build_model", "MODEL", "LABEL", "TAUT_NAMES",
    "PHI_GRID", "DPHI", "N_GRID", "Path", "np", "jnp",
    "belief_at", "marginal", "avg_marginal",
    "pairwise_joint", "orientation_averaged_joint",
    "anchor_phirest_joint", "global_marginal",
    "taut_profile", "tc_bridge_region", "k_star", "escape_probabilities",
    "mean_cos2",
    "ManuscriptView", "manuscript_view", "to_paper_index", "to_model_index",
    "paper_profile",
    "cis_trans", "cis_trans_profile", "MASK_TRANS", "MASK_CIS",
]
