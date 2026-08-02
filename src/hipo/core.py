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

    _, body = _rows(d / "tautomer_parameters.csv")
    rows = {r[0]: np.array([float(v) for v in r[1:]]) for r in body}
    p_t = rows["P_T"]
    t_fwd = np.vstack([rows[f"T_fwd_from_{n}"] for n in TAUT_NAMES])
    return phi_raw, pphi_raw, p_t, t_fwd


def build_model(data_dir=None) -> ChainModel:
    phi_raw, pphi_raw, p_t_raw, t_fwd_raw = _load_inputs(data_dir)

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

    if anchor == "first":
        T = _T(m, direction)
        seed = m.P_T[:, None] * m.P_phi_T          # (T_1, phi_1), no transition
    elif anchor == "last":
        T_to   = _T(m, direction)
        T_walk = _T(m, "reverse" if direction == "forward" else "forward")
        belief = belief_at(m, N - 1, direction)    # belief at the last dimer
        seed   = belief[:, None] * m.P_phi_T       # (T_N, phi_N)
        T      = T_walk
    else:
        raise ValueError("anchor must be 'first' or 'last'")

    joint = _accumulate(m, seed, T, N - 1)
    joint = jnp.maximum(joint, 0.0)
    return joint / (joint.sum() * DPHI**2) if normalise else joint


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
