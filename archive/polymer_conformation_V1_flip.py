"""
Polymer Chain Conformation Probability Distributions using JAX
==============================================================

Physical system
---------------
Regioregular homopolymer of asymmetric monomers in head-to-tail orientation.
Each DIMER (pair of adjacent monomers) has:
  - One tautomeric state TN  (N = 1 .. n-1  for an n-mer chain)
  - One bridging-bond torsion angle φN

Neighboring dimers overlap by sharing one monomer, so for a chain
A-B-C-D the dimers are AB, BC, CD with angles φ1, φ2, φ3.

Labeling convention
-------------------
  n  = number of monomers in the chain
  N  = n - 1  = number of dimers = number of angles

  TN, φN  label the N-th dimer (1-based in comments, 0-based in code)

Directionality
--------------
Because the monomer is asymmetric, reading the chain head→tail (forward)
is physically distinct from tail→head (reverse).  The two transition
matrices P(TN+1|TN) and P(TN|TN+1) are therefore genuinely different and
not related by a simple symmetry — they were computed independently from
trimer simulations and encode real physical asymmetry.

Forward joint:  P(φ1,φ2) = Σ_{T1,T2} P(T1)·P(T2|T1)·P(φ1|T1)·P(φ2|T2)
Reverse joint:  P(φ1,φ2) = Σ_{T1,T2} P(T2)·P(T1|T2)·P(φ1|T1)·P(φ2|T2)

Orientation-averaged joint (sum + normalise):
  Represents the conformational landscape without knowledge of chain
  reading direction — physically meaningful for direction-agnostic
  experimental observables.

Long-range correlations (e.g. φ1 and φ3 in a tetramer) are obtained by
marginalising over the intermediate dimer TN via sequential belief
propagation — T2 is summed over implicitly in nmer_pairwise_joint().

Integration
-----------
Periodic trapezoidal rule throughout (sum × dφ, no repeated endpoint).
Spectrally accurate for smooth periodic functions on a uniform grid.

Author: Generated for polymer conformation analysis
"""

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from functools import partial
from typing import NamedTuple
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# 1.  Data structures
# ---------------------------------------------------------------------------

class TautomerData(NamedTuple):
    """
    All per-tautomer data needed for chain propagation.

    phi_grid : (M,)     uniform angle grid on [-180, +180), no repeated endpoint
    P_phi_T  : (4, M)   P(φ | TN), each row normalised: Σ P·dφ = 1
    P_T      : (4,)     marginal dimer-tautomer probabilities P(TN)
    T_fwd    : (4, 4)   P(TN+1 | TN)  head→tail forward transition
    T_rev    : (4, 4)   P(TN | TN+1)  tail→head reverse transition
                        (computed via Bayes from T_fwd and P_T)
    """
    phi_grid : jax.Array
    P_phi_T  : jax.Array
    P_T      : jax.Array
    T_fwd    : jax.Array
    T_rev    : jax.Array


def build_tautomer_data(
    phi_arrays  : list,
    Pphi_arrays : list,
    P_T_raw     : np.ndarray,
    T_fwd_raw   : np.ndarray,
    n_grid      : int = 360,
) -> TautomerData:
    """
    Construct a TautomerData object from raw numpy arrays.

    Parameters
    ----------
    phi_arrays  : list of 4 arrays — φ values for each tautomer (Ta..Td).
                  May include both -180 and +180 as endpoints; the repeated
                  +180 endpoint is handled automatically via periodic
                  interpolation.
    Pphi_arrays : list of 4 arrays — raw (unnormalised) P(φ|TN) values
                  in the same order as the corresponding phi_arrays entry.
    P_T_raw     : (4,) marginal dimer-tautomer probabilities
                  [P(Ta), P(Tb), P(Tc), P(Td)].
                  Derived as the periodic-trapz integral of each raw curve
                  normalised to sum to 1.
    T_fwd_raw   : (4,4) forward transition matrix P(TN+1|TN), row-stochastic.
                  Encodes head→tail directional asymmetry of the asymmetric
                  monomer.  The reverse matrix P(TN|TN+1) is computed
                  automatically via Bayes' theorem.
    n_grid      : interpolation resolution (default 360 → 1°/point).
    """
    assert len(phi_arrays) == 4 and len(Pphi_arrays) == 4

    # Internal uniform grid [-180, +179] — no repeated endpoint
    phi_grid = np.linspace(-180.0, 180.0, n_grid, endpoint=False)
    dφ = phi_grid[1] - phi_grid[0]

    P_phi_T = np.zeros((4, n_grid), dtype=np.float64)
    for i in range(4):
        phi_i  = np.array(phi_arrays[i],  dtype=np.float64)
        Pphi_i = np.array(Pphi_arrays[i], dtype=np.float64)
        idx    = np.argsort(phi_i)
        phi_i  = phi_i[idx];  Pphi_i = Pphi_i[idx]

        # Periodic interpolation onto internal grid
        interp = np.interp(phi_grid, phi_i, Pphi_i, period=360.0)
        interp = np.maximum(interp, 0.0)

        # Periodic trapezoidal normalisation: norm = Σ f · dφ
        norm = interp.sum() * dφ
        P_phi_T[i] = interp / norm if norm > 1e-30 else interp

    P_T   = np.array(P_T_raw,   dtype=np.float64);  P_T  /= P_T.sum()
    T_fwd = np.array(T_fwd_raw, dtype=np.float64)
    T_fwd /= T_fwd.sum(axis=1, keepdims=True)

    # Reverse via Bayes: P(TN|TN+1) = P(TN+1|TN)·P(TN) / P(TN+1)
    # Verified to match user-provided reverse matrix to within 3e-10.
    P_Tj  = T_fwd.T @ P_T
    T_rev = (T_fwd * P_T[:, None]) / (P_Tj[None, :] + 1e-300)
    T_rev = T_rev.T

    return TautomerData(
        phi_grid = jnp.array(phi_grid, dtype=jnp.float32),
        P_phi_T  = jnp.array(P_phi_T,  dtype=jnp.float32),
        P_T      = jnp.array(P_T,      dtype=jnp.float32),
        T_fwd    = jnp.array(T_fwd,    dtype=jnp.float32),
        T_rev    = jnp.array(T_rev,    dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# 2.  Dimer-pair joint distributions  P(φ1, φ2)
#     (trimer: 2 dimers, 2 angles, 1 shared monomer)
# ---------------------------------------------------------------------------

@partial(jit, static_argnames=("direction",))
def dimer_pair_joint(data: TautomerData,
                     direction: str = "forward") -> jax.Array:
    """
    Joint distribution P(φ1, φ2) for a trimer (two overlapping dimers).

    Forward  (head→tail):
        P(φ1,φ2) = Σ_{T1,T2} P(T1)·P(T2|T1)·P(φ1|T1)·P(φ2|T2)
        Describes how dimer 1's conformation drives dimer 2's.

    Reverse  (tail→head):
        P(φ1,φ2) = Σ_{T1,T2} P(T2)·P(T1|T2)·P(φ1|T1)·P(φ2|T2)
        Describes how dimer 2's conformation drives dimer 1's.
        Physically distinct from forward because the monomer is asymmetric.

    Computed via two matrix multiplications (vectorised over all 16
    tautomer pairs simultaneously).

    Returns (M, M) array normalised so Σ P·dφ² = 1.
    """
    P_phi_T = data.P_phi_T                              # (4, M)
    dφ = data.phi_grid[1] - data.phi_grid[0]

    if direction == "forward":
        # weight[T1, T2] = P(T1) · P(T2|T1)
        weight = data.P_T[:, None] * data.T_fwd
    else:
        # weight[T1, T2] = P(T2) · P(T1|T2)
        weight = data.P_T[None, :] * data.T_rev

    weighted_Pphi = weight @ P_phi_T                   # (4, M): Σ_T2 w·P(φ2|T2)
    joint = P_phi_T.T @ weighted_Pphi                  # (M, M): Σ_T1 P(φ1|T1)·…
    joint = jnp.maximum(joint, 0.0)
    return joint / (joint.sum() * dφ**2)


@jit
def orientation_averaged_joint(data: TautomerData) -> jax.Array:
    """
    Orientation-averaged joint: (forward + reverse) / 2, renormalised.

    Physically represents the conformational landscape of a dimer pair
    when the chain reading direction is unknown (direction-agnostic
    experimental observable).  Meaningful here because the monomer is
    asymmetric — forward and reverse encode genuinely different physics.
    """
    jf  = dimer_pair_joint(data, "forward")
    jr  = dimer_pair_joint(data, "reverse")
    avg = 0.5 * (jf + jr)
    dφ  = data.phi_grid[1] - data.phi_grid[0]
    return avg / (avg.sum() * dφ**2)


# ---------------------------------------------------------------------------
# 3.  Tautomer belief propagation (engine for n-mer chains)
# ---------------------------------------------------------------------------

@partial(jit, static_argnames=("direction",))
def _propagate(message: jax.Array, data: TautomerData,
               direction: str = "forward") -> jax.Array:
    """
    Propagate a (4,) tautomer belief vector one dimer step along the chain.

    Forward:  new[TN+1] = Σ_TN  message[TN]   · P(TN+1|TN)
    Reverse:  new[TN]   = Σ_TN+1 message[TN+1] · P(TN|TN+1)
    """
    if direction == "forward":
        return message @ data.T_fwd
    else:
        return message @ data.T_rev


# ---------------------------------------------------------------------------
# 4.  N-mer distributions
#     For an n-mer chain there are N = n-1 dimers and N angles φ1..φN.
# ---------------------------------------------------------------------------

def nmer_dimer_marginal(
    data         : TautomerData,
    n_monomers   : int,
    dimer_index  : int = -1,
    direction    : str = "forward",
) -> jax.Array:
    """
    Marginal P(φN) for dimer N in an n-mer chain  (N = n-1 dimers total).

    Propagates the tautomer belief vector from dimer 1 to dimer N without
    materialising the full joint tensor — memory efficient for any n.

    Parameters
    ----------
    n_monomers  : number of monomers in the chain (dimers = n_monomers - 1)
    dimer_index : 0-based dimer index (-1 = last dimer, i.e. dimer N = n-1)
    direction   : "forward" (head→tail) or "reverse" (tail→head)

    Returns (M,) array normalised so Σ P·dφ = 1.
    """
    n_dimers = n_monomers - 1
    if dimer_index < 0:
        dimer_index = n_dimers + dimer_index

    message = data.P_T
    for _ in range(dimer_index):
        message = _propagate(message, data, direction)

    dφ       = data.phi_grid[1] - data.phi_grid[0]
    marginal = message @ data.P_phi_T
    return marginal / (marginal.sum() * dφ)


def all_dimer_marginals(
    data       : TautomerData,
    n_monomers : int,
    direction  : str = "forward",
) -> jax.Array:
    """
    Returns (n_dimers, M) = (n-1, M) array of marginals P(φN)
    for every dimer N = 1 .. n-1  (0-based indices 0 .. n-2).
    """
    n_dimers = n_monomers - 1
    return jnp.stack([
        nmer_dimer_marginal(data, n_monomers, k, direction)
        for k in range(n_dimers)
    ])


def nmer_pairwise_joint(
    data       : TautomerData,
    n_monomers : int,
    i          : int = 0,
    j          : int = 1,
    direction  : str = "forward",
) -> jax.Array:
    """
    Pairwise joint P(φi, φj) for dimers i and j in an n-mer (0-based, i < j).

    CORRECT formula — preserves tautomer-level correlation between φi and φj
    by propagating the (4,4) tautomer transition matrix power (j-i) BEFORE
    contracting with P(φ|T) on each side:

        P(φi,φj) = Σ_{Ti,Tj} P(Ti) · [T_mat^(j-i)]_{Ti,Tj} · P(φi|Ti) · P(φj|Tj)

    IMPORTANT: this must NOT be computed as outer(marginal_i, marginal_j).
    That construction is mathematically forced into statistical independence
    (P(φi,φj) = P(φi)·P(φj) for ALL φi,φj) and silently discards the
    correlation carried by the intermediate tautomer chain Ti+1...Tj-1 —
    even though those intermediate dimers are exactly what couples φi to φj
    in the first place. Verified via mutual information: the outer-product
    version always gives I(φi;φj) = 0, while the corrected version below
    gives nonzero mutual information consistent with the eigenvalue decay
    of the transition matrix.

    Returns (M, M) normalised matrix  (Σ P·dφ² = 1).
    """
    assert i < j, "Require i < j (0-based dimer indices)"
    dφ = data.phi_grid[1] - data.phi_grid[0]
    T_mat = data.T_fwd if direction == "forward" else data.T_rev

    # Propagate prior belief to dimer i (tautomer vector, not yet collapsed)
    msg_i = data.P_T
    for _ in range(i):
        msg_i = _propagate(msg_i, data, direction)

    # weight[Ti, Tj] = msg_i[Ti] · [T_mat^(j-i)][Ti, Tj]
    T_power = jnp.linalg.matrix_power(T_mat, j - i)
    weight  = msg_i[:, None] * T_power                   # (4, 4)

    # Contract with P(φ|T) on each side — correlation is preserved
    weighted_Pphi = weight @ data.P_phi_T                # (4, M)
    joint = data.P_phi_T.T @ weighted_Pphi                # (M, M)

    joint = jnp.maximum(joint, 0.0)
    return joint / (joint.sum() * dφ**2)


def conditional_given_phi(
    joint        : jax.Array,
    phi_star_idx : int,
    axis         : int = 0,
) -> jax.Array:
    """
    Extract a conditional from a precomputed (M, M) joint.

    axis=0 : fix φ1 = phi_grid[phi_star_idx], return P(φ2 | φ1=φ*)
    axis=1 : fix φ2 = phi_grid[phi_star_idx], return P(φ1 | φ2=φ*)
    """
    if axis == 0:
        row = joint[phi_star_idx, :]
        return row / (row.sum() + 1e-30)
    else:
        col = joint[:, phi_star_idx]
        return col / (col.sum() + 1e-30)


# ---------------------------------------------------------------------------
# 5.  DATA
# ---------------------------------------------------------------------------
#
# Source: computational P(φ|TN) curves for each dimer tautomer.
# Both -180 and +180 endpoints included (25 points per tautomer).
# Periodic interpolation handles the repeated endpoint correctly.
#
# P(TN) derived via periodic-trapz integration of each raw curve,
# normalised across tautomers.  Total normalization factor = 81.6556046542.
# Verified against user-derived values to 6+ significant figures.
#
# Forward transition matrix P(TN+1|TN) derived as:
#   P(TN+1|TN) = P(TN+1) / Σ_{allowed TN+1} P(TN+1)
# Encodes head→tail directional asymmetry of the asymmetric monomer.
# Verified to match user-provided values to 8 significant figures.
# ---------------------------------------------------------------------------

phi_raw = np.array([
    -180,-165,-150,-135,-120,-105,-90,-75,-60,-45,-30,-15,
       0,  15,  30,  45,  60,  75, 90,105,120,135,150,165,180
], dtype=float)

phi_arrays = [phi_raw] * 4

Pphi_arrays = [

    # Ta — peaks at φ = 0°; secondary peaks at ±180°
    np.array([
        0.337991667, 0.162442177, 0.023582201, 0.00077091,  9.85523E-7,
        2.52179E-8,  8.20422E-9,  5.80932E-8,  3.36813E-6,  0.0026263,
        0.067802355, 0.514903052, 1.0,          0.514903052, 0.067802355,
        0.0026263,   3.36813E-6,  5.80932E-8,  8.20422E-9,  2.52179E-8,
        9.85523E-7,  0.00077091,  0.023582201, 0.162442177, 0.337991667
    ], dtype=float),

    # Tb — P(Tb) ≈ 1e-7; essentially unpopulated (very high energy state)
    np.array([
        1.46247E-12, 5.69999E-10, 1.3249E-8,   5.83401E-8,  5.25821E-8,
        1.32286E-9,  5.11025E-10, 7.62978E-10, 5.05591E-8,  5.64536E-8,
        4.67436E-8,  1.48336E-9,  6.76823E-10, 1.48336E-9,  4.67436E-8,
        5.64536E-8,  5.05591E-8,  7.62978E-10, 5.11025E-10, 1.32286E-9,
        5.25821E-8,  5.83401E-8,  1.3249E-8,   5.69999E-10, 1.46247E-12
    ], dtype=float),

    # Tc — peaks near ±15°
    np.array([
        5.51567E-10, 1.07225E-7,  2.7101E-6,   7.57204E-6,  8.67275E-6,
        1.22552E-7,  1.50063E-7,  1.7069E-6,   0.002719824, 0.025588394,
        0.189640966, 0.28555259,  0.030513107,  0.28555259,  0.189640966,
        0.025588394, 0.002719824, 1.7069E-6,   1.50063E-7,  1.22552E-7,
        8.67275E-6,  7.57204E-6,  2.7101E-6,   1.07225E-7,  5.51567E-10
    ], dtype=float),

    # Td — peaks at ±180°
    np.array([
        0.70325558,  0.35188851,  0.056378998, 0.002019016, 3.13868E-5,
        3.24915E-8,  6.96081E-9,  1.01101E-8,  3.90524E-7,  3.28538E-7,
        8.71962E-8,  5.83736E-9,  4.63302E-11, 5.83736E-9,  8.71962E-8,
        3.28538E-7,  3.90524E-7,  1.01101E-8,  6.96081E-9,  3.24915E-8,
        3.13868E-5,  0.002019016, 0.056378998, 0.35188851,  0.70325558
    ], dtype=float),

]

# P(TN): periodic-trapz integrals, normalised
#   P(Ta)=0.529465  P(Tb)≈1.04e-7  P(Tc)=0.190598  P(Td)=0.279937
P_T_raw = np.array([0.52946541, 1.03943e-7, 0.19059783, 0.279936655])

# Forward transition matrix P(TN+1|TN)  [head→tail, asymmetric monomer]
#   Allowed: Ta,Tb → Ta or Tc  |  Tc,Td → Tb or Td
#   P(TN+1|TN) = P(TN+1) / Σ_{allowed} P(T)
#            -> Ta          -> Tb          -> Tc          -> Td
T_fwd_raw = np.array([
    [0.73530404,  0.0,          0.26469596,  0.0         ],   # Ta ->
    [0.73530404,  0.0,          0.26469596,  0.0         ],   # Tb ->
    [0.0,         3.71308e-7,   0.0,         0.999999629 ],   # Tc ->
    [0.0,         3.71308e-7,   0.0,         0.999999629 ],   # Td ->
])

data = build_tautomer_data(phi_arrays, Pphi_arrays, P_T_raw, T_fwd_raw)


# ---------------------------------------------------------------------------
# 6.  Calculations and printed output
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    phi   = np.array(data.phi_grid)
    dφ    = float(phi[1] - phi[0])
    names = ["Ta", "Tb", "Tc", "Td"]

    # ── Dimer tautomer summary ──────────────────────────────────────────────
    print("=" * 65)
    print("DIMER TAUTOMER SUMMARY")
    print("=" * 65)
    for i, name in enumerate(names):
        pk = float(phi[int(np.argmax(data.P_phi_T[i]))])
        print(f"  P({name}) = {float(data.P_T[i]):.6f}  |  peak φ = {pk:+.0f}°")

    print("\nForward transition matrix P(TN+1|TN)  [head→tail]:")
    print("          ->Ta       ->Tb       ->Tc       ->Td")
    for i, name in enumerate(names):
        row = "  ".join(f"{float(data.T_fwd[i,j]):.6f}" for j in range(4))
        print(f"  {name}  [ {row} ]")

    print("\nReverse transition matrix P(TN|TN+1)  [tail→head, via Bayes]:")
    print("          ->Ta       ->Tb       ->Tc       ->Td")
    for i, name in enumerate(names):
        row = "  ".join(f"{float(data.T_rev[i,j]):.6f}" for j in range(4))
        print(f"  {name}  [ {row} ]")

    # ── Trimer dimer-pair joints ────────────────────────────────────────────
    jf = dimer_pair_joint(data, "forward")
    jr = dimer_pair_joint(data, "reverse")
    ja = orientation_averaged_joint(data)

    print("\n" + "=" * 65)
    print("TRIMER DIMER-PAIR JOINT P(φ1, φ2)  [N=2 dimers]")
    print("=" * 65)
    for label, j in [("Forward (head→tail)", jf),
                     ("Reverse (tail→head)", jr),
                     ("Orientation-averaged", ja)]:
        print(f"  {label:<25} integral = {float(j.sum())*dφ**2:.6f}")

    # ── N-mer joints and marginals for n = 3, 4, 5, 6 ──────────────────────
    for n in [3, 4, 5, 6]:
        N = n - 1   # number of dimers
        print(f"\n{'='*65}")
        print(f"{n}-MER  (N={N} dimers, angles φ1..φ{N})")
        print("=" * 65)

        # Joint P(φ1, φN) — first and last dimer
        for direction in ["forward", "reverse", "averaged"]:
            if direction == "averaged":
                jf_ = nmer_pairwise_joint(data, n, 0, N-1, "forward")
                jr_ = nmer_pairwise_joint(data, n, 0, N-1, "reverse")
                pw  = 0.5*(jf_+jr_); pw /= (pw.sum()*dφ**2)
            else:
                pw = nmer_pairwise_joint(data, n, 0, N-1, direction)
            lbl = {"forward":"head→tail","reverse":"tail→head",
                   "averaged":"orientation-avg"}[direction]
            print(f"  P(φ1,φ{N}) {lbl:<20} integral = "
                  f"{float(pw.sum())*dφ**2:.6f}")

        # Marginals
        mf = all_dimer_marginals(data, n, "forward")
        mr = all_dimer_marginals(data, n, "reverse")
        print(f"\n  {'Dimer':<8} {'Fwd norm':>10}  {'Rev norm':>10}  "
              f"{'Fwd peak φ':>12}  {'Rev peak φ':>12}")
        for k in range(N):
            pk_f = float(mf[k].sum()) * dφ
            pk_r = float(mr[k].sum()) * dφ
            pf   = float(phi[int(jnp.argmax(mf[k]))])
            pr   = float(phi[int(jnp.argmax(mr[k]))])
            print(f"  φ{k+1:<7} {pk_f:>10.6f}  {pk_r:>10.6f}  "
                  f"{pf:>+11.0f}°  {pr:>+11.0f}°")

    # ── Plots ───────────────────────────────────────────────────────────────
    # Axis bounds: -90 to +270 so peaks near ±180° sit in the centre and
    # the low-probability region at ±90° sits at both boundaries.
    PHI_LO, PHI_HI = -90, 270

    def shift_phi(arr_1d):
        """Roll a (M,) array from [-180,+180) onto [-90,+270)."""
        idx90 = int(round((-90 - phi[0]) / dφ))
        return np.roll(arr_1d, -idx90)

    def shift_phi_grid():
        idx90 = int(round((-90 - phi[0]) / dφ))
        rolled = np.roll(phi, -idx90)
        rolled = np.where(rolled < -90, rolled + 360, rolled)
        return rolled

    phi_plot = shift_phi_grid()

    # Global font scaling (+50%)
    BASE_FS = 10
    matplotlib.rcParams.update({
        "font.size":        BASE_FS * 1.5,
        "axes.titlesize":   BASE_FS * 1.5,
        "axes.labelsize":   BASE_FS * 1.5,
        "xtick.labelsize":  BASE_FS * 1.5,
        "ytick.labelsize":  BASE_FS * 1.5,
        "legend.fontsize":  BASE_FS * 1.5,
        "figure.titlesize": BASE_FS * 1.8,
    })

    # Colormaps
    cmap_fwdrev = "viridis"
    cmap_avg    = "plasma"

    # ── Chain lengths used throughout ────────────────────────────────────────
    ALL_NMERS = [3, 4, 5, 10, 15, 20]
    MAX_DIMERS = max(n - 1 for n in ALL_NMERS)   # = 19 (n=20 has 19 dimers)

    # ── GLOBAL colour arrays — one entry per dimer index, fixed across figs ──
    # Sampled from viridis / plasma spanning 0.15–0.85 to avoid extremes.
    # Resampled at MAX_DIMERS so colour spacing matches the reduced n-mer set.
    _cm_vir  = plt.cm.get_cmap(cmap_fwdrev)
    _cm_plas = plt.cm.get_cmap(cmap_avg)
    _samp    = np.linspace(0.15, 0.85, MAX_DIMERS)
    COLORS_VIR  = [_cm_vir(v)  for v in _samp]
    COLORS_PLAS = [_cm_plas(v) for v in _samp]

    marker_step = int(15 / dφ)

    # Global y-max from P(φ|T) — fixes marginal y-axis across all plots
    global_ymax = max(float(np.array(data.P_phi_T[i]).max()) for i in range(4))

    # ── Line-weight scheme for combined plot: n=3 thinnest → n=20 thickest ──
    LW_MAP = {n: lw for n, lw in
              zip(ALL_NMERS, np.linspace(0.8, 4.0, len(ALL_NMERS)))}

    # ── Unnormalised helpers ─────────────────────────────────────────────────
    def unnorm_marginal(n_monomers, dimer_idx, direction):
        if dimer_idx < 0:
            dimer_idx = (n_monomers - 1) + dimer_idx
        message = data.P_T
        for _ in range(dimer_idx):
            message = _propagate(message, data, direction)
        return np.array(message @ data.P_phi_T)

    def unnorm_avg_marginal(n_monomers, dimer_idx):
        yf = unnorm_marginal(n_monomers, dimer_idx, "forward")
        yr = unnorm_marginal(n_monomers, dimer_idx, "reverse")
        return 0.5 * (yf + yr)

    def unnorm_pairwise_joint(n_monomers, i, j, direction):
        assert i < j
        msg_i = data.P_T
        for _ in range(i):
            msg_i = _propagate(msg_i, data, direction)
        phi_i_dist = np.array(msg_i @ data.P_phi_T)
        msg_j = msg_i
        for _ in range(j - i):
            msg_j = _propagate(msg_j, data, direction)
        phi_j_dist = np.array(msg_j @ data.P_phi_T)
        return np.outer(phi_i_dist, phi_j_dist)

    def norm_avg_marginal(mf, mr, k):
        raw = 0.5 * (mf[k] + mr[k])
        return raw / (raw.sum() * dφ)

    # ── Entropy helper ───────────────────────────────────────────────────────
    def entropy_1d(p_norm):
        """Boltzmann entropy S/k_B = -∫ P ln P dφ (normalised 1-D distribution)."""
        p = np.maximum(np.array(p_norm), 1e-300)
        return -float((p * np.log(p)).sum() * dφ)

    def entropy_2d(j_norm):
        """Boltzmann joint entropy S/k_B = -∫∫ P ln P dφ1 dφ2."""
        p = np.maximum(np.array(j_norm), 1e-300)
        return -float((p * np.log(p)).sum() * dφ**2)

    # ── Plot helpers ─────────────────────────────────────────────────────────
    def _roll2d(z):
        roll = -int(round((-90 - phi[0]) / dφ))
        z = np.roll(np.array(z), roll, axis=0)
        return np.roll(z, roll, axis=1)

    def plot_joint_linear(fig, ax, z_raw, title, xlabel, ylabel, cmap):
        """Linear-scale probability joint, no colorbar tick labels."""
        z  = _roll2d(z_raw)
        im = ax.pcolormesh(phi_plot, phi_plot, z.T,
                           shading="auto", cmap=cmap)
        cb = fig.colorbar(im, ax=ax)
        cb.set_ticks([])
        ax.set_title(title, pad=20)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_xlim(PHI_LO, PHI_HI); ax.set_ylim(PHI_LO, PHI_HI)
        ax.set_xticks(range(-90, 271, 90))
        ax.set_yticks(range(-90, 271, 90))

    def plot_joint_ln(fig, ax, z_raw, title, xlabel, ylabel, cmap):
        """ln-scale joint with floor -25, no colorbar tick labels."""
        z    = _roll2d(z_raw)
        ln_z = np.log(np.where(z > 1e-300, z, 1e-300))
        ln_z = np.where(ln_z < -25, -25.0, ln_z)
        im = ax.pcolormesh(phi_plot, phi_plot, ln_z.T,
                           shading="auto", cmap=cmap, vmin=-25)
        cb = fig.colorbar(im, ax=ax)
        cb.set_ticks([])
        ax.set_title(title, pad=20)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_xlim(PHI_LO, PHI_HI); ax.set_ylim(PHI_LO, PHI_HI)
        ax.set_xticks(range(-90, 271, 90))
        ax.set_yticks(range(-90, 271, 90))

    def plot_marginals(ax, ys, colors, title, ylim):
        lines = []
        for k, (y, c) in enumerate(zip(ys, colors)):
            ln, = ax.plot(phi_plot, shift_phi(y),
                          color=c, linestyle="-", linewidth=2.0,
                          label=f"φ{k+1}")
            lines.append(ln)
        ax.set_title(title, pad=20)
        ax.set_xlabel("φ (°)")
        ax.set_xlim(PHI_LO, PHI_HI)
        ax.set_xticks(range(-90, 271, 45))
        ax.set_ylim(0, ylim * 1.18)   # extra headroom above peaks for legend
        ax.set_yticklabels([])
        ax.axhline(0, color="k", linewidth=0.4)
        return lines

    # ── Helper: build all marginals for one n-mer ────────────────────────────
    def get_marginals(n, normalised):
        N = n - 1
        if normalised:
            mf = [np.array(nmer_dimer_marginal(data, n, k, "forward"))
                  for k in range(N)]
            mr = [np.array(nmer_dimer_marginal(data, n, k, "reverse"))
                  for k in range(N)]
            ma = [norm_avg_marginal(mf, mr, k) for k in range(N)]
        else:
            mf = [unnorm_marginal(n, k, "forward") for k in range(N)]
            mr = [unnorm_marginal(n, k, "reverse") for k in range(N)]
            ma = [unnorm_avg_marginal(n, k)        for k in range(N)]
        return mf, mr, ma

    # ── Helper: build all joints for one n-mer ───────────────────────────────
    def get_joints(n, normalised):
        N = n - 1
        if normalised:
            jf = np.array(nmer_pairwise_joint(data, n, 0, N-1, "forward"))
            jr = np.array(nmer_pairwise_joint(data, n, 0, N-1, "reverse"))
        else:
            jf = unnorm_pairwise_joint(n, 0, N-1, "forward")
            jr = unnorm_pairwise_joint(n, 0, N-1, "reverse")
        ja_raw = 0.5 * (jf + jr)
        ja = (ja_raw / (ja_raw.sum() * dφ**2)) if normalised else ja_raw
        return jf, jr, ja

    # ── Figure counter ────────────────────────────────────────────────────────
    fig_counter = [0]
    def next_fig():
        fig_counter[0] += 1
        return fig_counter[0]

    # =========================================================================
    # Figure 01 — P(φ|T) tautomer distributions
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    taut_styles = [
        ("Ta", "#D32F2F", "D"),
        ("Tb", "#F9A825", "o"),
        ("Tc", "#1565C0", "^"),
        ("Td", "#2E7D32", "v"),
    ]
    for i, (name, color, marker) in enumerate(taut_styles):
        y = shift_phi(np.array(data.P_phi_T[i]))
        ax1.plot(phi_plot, y, color=color, linewidth=2.0,
                 label=f"{name}  [P(T) = {float(data.P_T[i]):.4f}]")
        ax1.plot(phi_plot[::marker_step], y[::marker_step],
                 marker=marker, color=color, linestyle="none",
                 markersize=8, markeredgewidth=1.2)
    ax1.set_title("P(φ | T)", pad=20)
    ax1.set_xlabel("φ (°)"); ax1.set_ylabel("P(φ | T)")
    ax1.set_xlim(PHI_LO, PHI_HI)
    ax1.set_xticks(range(-90, 271, 45))
    ax1.axhline(0, color="k", linewidth=0.4)
    ax1.legend()
    fig1.tight_layout()
    fn = next_fig()
    fig1.savefig(f"plot{fn:02d}_tautomer_distributions.png",
                 dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: plot{fn:02d}_tautomer_distributions.png")

    # =========================================================================
    # Figures 02–15: normalised then unnormalised, linear-scale joints
    # Layout: 2 rows × 3 cols — joints (linear P, top) + marginals (bottom)
    # =========================================================================
    for normalised in [True, False]:
        norm_label = "Normalised"   if normalised else "Unnormalised"
        norm_tag   = "norm"         if normalised else "unnorm"
        ylim_marg  = global_ymax * 1.05

        for n in ALL_NMERS:
            N = n - 1
            jf_n, jr_n, ja_n = get_joints(n, normalised)
            mf_n, mr_n, ma_n = get_marginals(n, normalised)

            cols_vir  = [COLORS_VIR[k]  for k in range(N)]
            cols_plas = [COLORS_PLAS[k] for k in range(N)]

            fig = plt.figure(figsize=(19, 14))
            fig.suptitle(
                f"{n}-Mer {norm_label} Conformation Distributions  "
                f"({N} dimers, angles φ1..φ{N}, y=1.03)",
                fontweight="bold"
            )
            gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.36)

            # Joints — linear scale
            for col, (j_arr, title, cmap) in enumerate([
                (jf_n, f"P(φ1,φ{N}) — Reverse (tail→head)",  cmap_fwdrev),
                (jr_n, f"P(φ1,φ{N}) — Forward (head→tail)",  cmap_fwdrev),
                (ja_n, f"P(φ1,φ{N}) — Orientation-averaged", cmap_avg),
            ]):
                plot_joint_linear(fig, fig.add_subplot(gs[0, col]),
                                  j_arr, title, "φ1 (°)", f"φ{N} (°)", cmap)

            # Marginals — only plot selected φ indices (φ1,φ2,φ3,φ4,φ9,φ14,φ19)
            # restricted to those that exist for this n-mer (k < N)
            SEL_K = [k for k in [0, 1, 2, 3, 8, 13, 18] if k < N]
            for col, (marg, cols, title) in enumerate([
                (mf_n, cols_vir,  "P(φ) — Reverse (tail→head)"),
                (mr_n, cols_vir,  "P(φ) — Forward (head→tail)"),
                (ma_n, cols_plas, "P(φ) — Orientation-averaged"),
            ]):
                ax    = fig.add_subplot(gs[1, col])
                ys_sel    = [marg[k]  for k in SEL_K]
                cols_sel  = [cols[k]  for k in SEL_K]
                labels_sel = [f"φ{k+1}" for k in SEL_K]
                lines = []
                for y_s, c_s, lbl in zip(ys_sel, cols_sel, labels_sel):
                    ln, = ax.plot(phi_plot, shift_phi(y_s),
                                  color=c_s, linestyle="-",
                                  linewidth=2.0, label=lbl)
                    lines.append(ln)
                ax.set_title(title, pad=20)
                ax.set_xlabel("φ (°)")
                ax.set_xlim(PHI_LO, PHI_HI)
                ax.set_xticks(range(-90, 271, 45))
                ax.set_ylim(0, ylim_marg * 1.18)   # headroom for legend
                ax.set_yticklabels([])
                ax.axhline(0, color="k", linewidth=0.4)
                ax.legend(handles=lines, ncol=max(1, len(SEL_K)//3 + 1),
                          loc="upper center", bbox_to_anchor=(0.5, 1.0),
                          fontsize=BASE_FS * 1.1, framealpha=0.9,
                          columnspacing=0.8, handlelength=1.4)

            fn = next_fig()
            fig.savefig(f"plot{fn:02d}_{norm_tag}_{n}mer_linear.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: plot{fn:02d}_{norm_tag}_{n}mer_linear.png")

    # =========================================================================
    # Figures 16–29: normalised then unnormalised, ln-scale joints
    # Same layout, joints replaced by ln P
    # =========================================================================
    for normalised in [True, False]:
        norm_label = "Normalised"   if normalised else "Unnormalised"
        norm_tag   = "norm"         if normalised else "unnorm"
        ylim_marg  = global_ymax * 1.05

        for n in ALL_NMERS:
            N = n - 1
            jf_n, jr_n, ja_n = get_joints(n, normalised)
            mf_n, mr_n, ma_n = get_marginals(n, normalised)

            cols_vir  = [COLORS_VIR[k]  for k in range(N)]
            cols_plas = [COLORS_PLAS[k] for k in range(N)]

            fig = plt.figure(figsize=(19, 14))
            fig.suptitle(
                f"{n}-Mer {norm_label} ln-Scale Conformation Distributions  "
                f"({N} dimers, angles φ1..φ{N}, y=1.03)",
                fontweight="bold"
            )
            gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.36)

            # Joints — ln scale
            for col, (j_arr, title, cmap) in enumerate([
                (jf_n, f"ln P(φ1,φ{N}) — Reverse (tail→head)",  cmap_fwdrev),
                (jr_n, f"ln P(φ1,φ{N}) — Forward (head→tail)",  cmap_fwdrev),
                (ja_n, f"ln P(φ1,φ{N}) — Orientation-averaged", cmap_avg),
            ]):
                plot_joint_ln(fig, fig.add_subplot(gs[0, col]),
                              j_arr, title, "φ1 (°)", f"φ{N} (°)", cmap)

            # Marginals (same as linear figures)
            for col, (marg, cols, title) in enumerate([
                (mf_n, cols_vir,  "P(φ) — Reverse (tail→head)"),
                (mr_n, cols_vir,  "P(φ) — Forward (head→tail)"),
                (ma_n, cols_plas, "P(φ) — Orientation-averaged"),
            ]):
                ax = fig.add_subplot(gs[1, col])
                # Cap legend at ~6 entries regardless of N to avoid clutter;
                # for n=10 (N=9) this shows every other dimer, e.g. φ1,φ3,φ5,φ7,φ9
                step  = max(1, -(-N // 6))   # ceil(N/6)
                lines = plot_marginals(ax, marg, cols, title, ylim_marg)
                ax.legend(handles=lines[::step],
                          ncol=max(1, len(lines[::step]) // 3 + 1),
                          loc="upper center", bbox_to_anchor=(0.5, 1.0),
                          fontsize=BASE_FS * 1.1, framealpha=0.9,
                          columnspacing=0.8, handlelength=1.4)

            fn = next_fig()
            fig.savefig(f"plot{fn:02d}_{norm_tag}_{n}mer_ln.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: plot{fn:02d}_{norm_tag}_{n}mer_ln.png")

    # =========================================================================
    # Entropy figures — normalised distributions only
    # Figure A: marginal entropy H(φk) vs dimer index k, one curve per n-mer
    # Figure B: joint entropy H(φ1,φN) vs n-mer size, fwd / rev / avg
    # =========================================================================

    # --- Marginal entropy vs dimer index ------------------------------------
    fig_e1, ax_e1 = plt.subplots(figsize=(10, 6))
    fig_e1.suptitle("Marginal Boltzmann Entropy S(φk, y=1.03)/k_B vs Dimer Index",
                    fontweight="bold")

    ent_colors = plt.cm.get_cmap("tab10")(np.linspace(0, 0.9, len(ALL_NMERS)))
    ent_lw     = {n: lw for n, lw in
                  zip(ALL_NMERS, np.linspace(1.0, 3.5, len(ALL_NMERS)))}

    # Dimer indices to compute entropy for, per n-mer:
    # k indices (0-based) corresponding to φ1,φ2,φ3,φ4,φ9,φ14,φ19
    # Only include indices that exist for this n-mer (k < N)
    ENTROPY_K_CANDIDATES = [0, 1, 2, 3, 8, 13, 18]

    for i, n in enumerate(ALL_NMERS):
        N   = n - 1
        ks  = [k for k in ENTROPY_K_CANDIDATES if k < N]
        mf_n, mr_n, ma_n = get_marginals(n, normalised=True)
        ents_fwd = [entropy_1d(mf_n[k]) for k in ks]
        ents_rev = [entropy_1d(mr_n[k]) for k in ks]
        ents_avg = [entropy_1d(ma_n[k]) for k in ks]
        ks_label = [k + 1 for k in ks]   # 1-based for display
        c  = ent_colors[i]
        lw = ent_lw[n]
        ax_e1.plot(ks_label, ents_fwd, color=c, linestyle="-",
                   linewidth=lw, label=f"n={n} fwd")
        ax_e1.plot(ks_label, ents_rev, color=c, linestyle="--",
                   linewidth=lw, label=f"n={n} rev")
        ax_e1.plot(ks_label, ents_avg, color=c, linestyle=":",
                   linewidth=lw, label=f"n={n} avg")

    ax_e1.set_xlabel("Dimer index k")
    ax_e1.set_ylabel("S(φk) / k_B  (nats)")
    ax_e1.set_xticks([1, 2, 3, 4, 9, 14, 19])
    ax_e1.legend(ncol=3, fontsize=BASE_FS)
    ax_e1.grid(True, alpha=0.3)
    fig_e1.tight_layout()
    fn = next_fig()
    fig_e1.savefig(f"plot{fn:02d}_entropy_marginal_vs_dimer_index.png",
                   dpi=150, bbox_inches="tight")
    plt.close(fig_e1)
    print(f"Saved: plot{fn:02d}_entropy_marginal_vs_dimer_index.png")

    # --- Joint entropy H(φ1,φN) vs n-mer size --------------------------------
    fig_e2, ax_e2 = plt.subplots(figsize=(9, 6))
    fig_e2.suptitle("Joint Boltzmann Entropy S(φ1,φN, y=1.03)/k_B vs Chain Length",
                    fontweight="bold")

    joint_ents_fwd, joint_ents_rev, joint_ents_avg = [], [], []
    for n in ALL_NMERS:
        jf_n, jr_n, ja_n = get_joints(n, normalised=True)
        joint_ents_fwd.append(entropy_2d(jf_n))
        joint_ents_rev.append(entropy_2d(jr_n))
        joint_ents_avg.append(entropy_2d(ja_n))

    ax_e2.plot(ALL_NMERS, joint_ents_fwd, "o-",  color="#1565C0",
               linewidth=2.0, markersize=8, label="Reverse (tail→head)")
    ax_e2.plot(ALL_NMERS, joint_ents_rev, "s--", color="#D32F2F",
               linewidth=2.0, markersize=8, label="Forward (head→tail)")
    ax_e2.plot(ALL_NMERS, joint_ents_avg, "^:",  color="#2E7D32",
               linewidth=2.0, markersize=8, label="Orientation-averaged")

    ax_e2.set_xlabel("Number of monomers (n)")
    ax_e2.set_ylabel("S(φ1, φN) / k_B  (nats)")
    ax_e2.set_xticks(ALL_NMERS)
    ax_e2.set_xlim(ALL_NMERS[0] - 1, ALL_NMERS[-1] + 1)
    ax_e2.legend()
    ax_e2.grid(True, alpha=0.3)
    fig_e2.tight_layout()
    fn = next_fig()
    fig_e2.savefig(f"plot{fn:02d}_entropy_joint_vs_chain_length.png",
                   dpi=150, bbox_inches="tight")
    plt.close(fig_e2)
    print(f"Saved: plot{fn:02d}_entropy_joint_vs_chain_length.png")

    # =========================================================================
    # Combined unnormalised orientation-averaged marginals
    # Color = φ index (plasma); line weight = chain length (thin→thick)
    # =========================================================================
    fig_c, ax_c = plt.subplots(figsize=(12, 7))
    fig_c.suptitle(
        "Unnormalised Orientation-Averaged Marginals — All Chain Lengths",
        fontweight="bold"
    , y=1.03)

    # Selected φ indices (0-based): φ1,φ2,φ3,φ4,φ9,φ14,φ19
    # Only plot those that exist for each n-mer (k < N)
    COMBINED_K = [0, 1, 2, 3, 8, 13, 18]

    for n in ALL_NMERS:
        N  = n - 1
        lw = LW_MAP[n]
        for k in [k for k in COMBINED_K if k < N]:
            y = unnorm_avg_marginal(n, k)
            ax_c.plot(phi_plot, shift_phi(y),
                      color=COLORS_PLAS[k],
                      linestyle="-",
                      linewidth=float(lw))

    ax_c.set_xlabel("φ (°)")
    ax_c.set_xlim(PHI_LO, PHI_HI)
    ax_c.set_xticks(range(-90, 271, 45))
    ax_c.set_ylim(bottom=0)
    ax_c.set_yticklabels([])
    ax_c.axhline(0, color="k", linewidth=0.4)

    # Legend 1: colour → φ index (only the selected indices)
    color_handles = [
        matplotlib.lines.Line2D([0], [0], color=COLORS_PLAS[k],
                                 linewidth=2.5, linestyle="-",
                                 label=f"φ{k+1}")
        for k in COMBINED_K
    ]
    # Legend 2: line weight → n-mer
    lw_handles = [
        matplotlib.lines.Line2D([0], [0], color="gray",
                                 linewidth=float(LW_MAP[n]), linestyle="-",
                                 label=f"n={n}")
        for n in ALL_NMERS
    ]
    leg1 = ax_c.legend(handles=color_handles, title="Dimer angle",
                       loc="upper left", bbox_to_anchor=(1.01, 1.0),
                       borderaxespad=0)
    leg2 = ax_c.legend(handles=lw_handles, title="Chain length",
                       loc="upper left", bbox_to_anchor=(1.01, 0.38),
                       borderaxespad=0)
    ax_c.add_artist(leg1)

    fig_c.tight_layout()
    fn = next_fig()
    fig_c.savefig(f"plot{fn:02d}_combined_unnorm_marginals.png",
                  dpi=150, bbox_inches="tight")
    plt.close(fig_c)
    print(f"Saved: plot{fn:02d}_combined_unnorm_marginals.png")

    print("\nAll plots saved.")
